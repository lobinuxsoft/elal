//! Agent loop — orchestrates user ↔ LLM ↔ tools within a single turn.
//!
//! This is the "brain" layer. It consumes [`StreamEvent`](elal_protocol::StreamEvent)s
//! from an [`elal_provider::Provider`], routes them to [`AgentEvent`]s on a
//! consumer-facing channel, assembles tool calls, dispatches them through
//! [`elal_tools::ToolRegistry`], feeds results back as `Message::tool`, and
//! loops until the model produces plain text with no pending tool calls.
//!
//! Lives in `elal_core` (not a standalone crate) because it orchestrates
//! existing pieces — Provider, ToolRegistry, config, context — without a
//! meaningful encapsulation boundary of its own.
//!
//! Chunk 4 wires the approval handshake: the dispatcher consults
//! [`elal_tools::approval::resolve`], emits
//! [`AgentEvent::ApprovalRequired`] when the user needs to decide, and
//! waits for the matching [`UserAction::ApprovalResponse`] before
//! executing. Session-scoped and tool-scoped approvals persist across
//! turns on the [`Agent`]; turn-scoped approvals live in a local set
//! created fresh at the start of every `run_turn`.

use std::collections::HashSet;
use std::path::PathBuf;

use elal_protocol::{
    ApprovalMode, Message, Role, SessionId, SessionRecord, StreamEvent, TurnStatus,
};
use elal_provider::{CompletionRequest, Provider, SamplingControls};
use elal_tools::ToolRegistry;
use tokio::sync::mpsc;

mod event;
mod history;
mod persist;
mod stream_router;
mod tool_exec;
mod turn;

pub use event::{AgentEvent, ApprovalRequest, UserAction};
pub use turn::{TURN_STEP_CAP, TurnError, TurnSummary};

use persist::Persistence;
use stream_router::{StreamRouter, StreamSummary};

use crate::session::{RolloutStore, TokenBudget, compact_messages};

/// Stateful agent tied to a loaded [`Provider`] and a [`ToolRegistry`].
pub struct Agent<'p, P: Provider> {
    provider: &'p P,
    tools: &'p ToolRegistry,
    history: Vec<Message>,
    system_prompt: String,
    approval_mode: ApprovalMode,
    working_dir: PathBuf,
    sampling: SamplingControls,
    /// Per-turn session id — fresh per `run_turn`, passed into every
    /// `ToolContext` so tools can correlate their own telemetry.
    session_id: SessionId,
    /// Tool names pre-approved at session scope. Persists across turns
    /// for the lifetime of the agent; cleared by [`Agent::reset`].
    approved_for_session: HashSet<String>,
    /// Optional rollout-journal sink. `None` means session persistence is
    /// disabled (tests, ephemeral one-shot invocations).
    persistence: Option<Persistence>,
    /// Token budget driving compaction. Defaults to a conservative window;
    /// the CLI override-fits this to the loaded model's `n_ctx`.
    token_budget: TokenBudget,
    /// Input tokens reported by the most recent turn. Drives the compaction
    /// trigger via [`TokenBudget::should_compact`].
    last_input_tokens: usize,
    /// Optional KV-cache snapshot path threaded into every
    /// [`CompletionRequest`]. `Some(path)` opts the session into the
    /// `state_save_file` / `state_load_file` round-trip; `None` falls back
    /// to full prompt evaluation each turn.
    kv_cache_path: Option<PathBuf>,
}

impl<'p, P: Provider> Agent<'p, P> {
    pub fn new(
        provider: &'p P,
        tools: &'p ToolRegistry,
        system_prompt: impl Into<String>,
        approval_mode: ApprovalMode,
        working_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            provider,
            tools,
            history: Vec::new(),
            system_prompt: system_prompt.into(),
            approval_mode,
            working_dir: working_dir.into(),
            sampling: SamplingControls::default(),
            session_id: SessionId::new(),
            approved_for_session: HashSet::new(),
            persistence: None,
            token_budget: TokenBudget::default(),
            last_input_tokens: 0,
            kv_cache_path: None,
        }
    }

    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingControls) -> Self {
        self.sampling = sampling;
        self
    }

    /// Override the token budget that drives compaction. Defaults to
    /// [`TokenBudget::default`] (32K window / 4K output) — callers should
    /// align this with the loaded model's `n_ctx` after auto-tune.
    #[must_use]
    pub fn with_token_budget(mut self, budget: TokenBudget) -> Self {
        self.token_budget = budget;
        self
    }

    /// Opt the session into KV-cache snapshots — every turn loads the file
    /// at `path` (when present) and atomically rewrites it after generation.
    /// Threaded into every [`CompletionRequest`] this agent emits.
    #[must_use]
    pub fn with_kv_cache_path(mut self, path: PathBuf) -> Self {
        self.kv_cache_path = Some(path);
        self
    }

    /// Attach a rollout store + the canonical session record for a fresh
    /// session. The `SessionMeta` line is emitted on the first turn; this
    /// builder does not perform any I/O.
    ///
    /// The agent's runtime [`SessionId`] is synced to `record.id` so tool
    /// dispatch correlates with the persisted session.
    #[must_use]
    pub fn with_persistence(mut self, store: RolloutStore, record: SessionRecord) -> Self {
        self.session_id = record.id;
        self.persistence = Some(Persistence::for_new_session(store, record));
        self
    }

    /// Resume a persisted session — rehydrates `history`, `last_input_tokens`,
    /// `session_id`, and wires the rollout store with continuity over the
    /// existing turn / item sequence counters so subsequent appends do not
    /// collide with the prior journal.
    #[must_use]
    pub fn resume_session(
        mut self,
        store: RolloutStore,
        loaded: crate::session::LoadedSession,
    ) -> Self {
        self.session_id = loaded.record.id;
        self.history = loaded.state.messages;
        self.last_input_tokens = loaded.state.last_input_tokens;
        self.persistence = Some(Persistence::for_resumed_session(
            store,
            loaded.record,
            loaded.last_turn_seq,
            loaded.last_item_seq,
        ));
        self
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Most recent input-token count reported by the provider — used by
    /// compaction policy and exposed for diagnostics.
    pub fn last_input_tokens(&self) -> usize {
        self.last_input_tokens
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.session_id = SessionId::new();
        self.approved_for_session.clear();
        self.last_input_tokens = 0;
        // Persistence is intentionally not reset — it is married to a
        // specific rollout file. Callers that want to start a new session
        // must rebuild the agent.
    }

    /// Run a single turn against the provider.
    ///
    /// Iterates provider → tool dispatch → provider until the model
    /// returns plain text with no pending tool calls, or the step cap is
    /// hit.
    pub async fn run_turn(
        &mut self,
        user_input: &str,
        events_tx: mpsc::Sender<AgentEvent>,
        actions_rx: &mut mpsc::Receiver<UserAction>,
    ) -> Result<TurnSummary, TurnError> {
        self.ensure_system_prompt();

        // Compaction runs before the new user message is enqueued so the
        // ContextCompaction marker lands in the journal ahead of the new
        // UserMessage and the model never sees a runaway prompt.
        let compaction_summary = self.maybe_compact();
        self.history.push(Message::user(user_input));

        if let Some(p) = self.persistence.as_mut() {
            persist::log_persist_err(p.begin_turn(self.session_id));
            if let Some(text) = compaction_summary {
                persist::log_persist_err(p.append_item(persist::compaction_item(text)));
            }
            persist::log_persist_err(p.append_item(persist::user_item(user_input)));
            p.record_first_user_message(user_input);
        }

        send(&events_tx, AgentEvent::TurnStart).await?;

        let mut total = TurnSummary {
            stop_reason: elal_protocol::StopReason::EndTurn,
            tool_calls_executed: 0,
            usage: elal_protocol::Usage::default(),
            cancelled: false,
        };
        let mut approved_for_turn: HashSet<String> = HashSet::new();
        let mut hit_step_cap = false;

        for round in 0..TURN_STEP_CAP {
            let budget = (self.provider.context_length() as f64 * 0.9) as usize;
            history::fit_to_budget(&mut self.history, budget);

            let request = CompletionRequest {
                messages: self.history.clone(),
                tools: self.tools.definitions(),
                sampling: self.sampling.clone(),
                max_tokens: None,
                kv_cache_path: self.kv_cache_path.clone(),
            };

            let stream_summary = self.pump_round(request, &events_tx).await?;
            total.accumulate(&stream_summary.usage);
            total.stop_reason = stream_summary.stop_reason;

            // Record the assistant's output in history so the next round
            // (and future turns) see what the model just said.
            self.history.push(stream_summary.assistant_message.clone());
            self.persist_assistant_round(&stream_summary);

            if stream_summary.pending_tool_calls.is_empty() {
                // Model finished without requesting more work.
                break;
            }

            let dispatch_ctx = tool_exec::DispatchContext {
                tools: self.tools,
                working_dir: &self.working_dir,
                session_id: self.session_id,
                mode: self.approval_mode,
                approved_for_session: &mut self.approved_for_session,
                approved_for_turn: &mut approved_for_turn,
                events_tx: &events_tx,
                actions_rx,
            };
            let dispatch =
                tool_exec::dispatch(stream_summary.pending_tool_calls, dispatch_ctx).await?;

            self.persist_tool_results(&dispatch.events);
            for event in dispatch.events {
                send(&events_tx, event).await?;
            }
            self.history.extend(dispatch.tool_messages);
            total.tool_calls_executed = total.tool_calls_executed.saturating_add(dispatch.executed);

            if dispatch.cancelled {
                total.cancelled = true;
                break;
            }

            // If the cap is one away, and we still need another round,
            // the next iteration will abort — flag it so the persistence
            // close sees the failure status before the error unwinds.
            if round + 1 == TURN_STEP_CAP {
                hit_step_cap = true;
                break;
            }
        }

        self.last_input_tokens = total.usage.prompt_tokens as usize;
        let status = if total.cancelled {
            TurnStatus::Cancelled
        } else if hit_step_cap {
            TurnStatus::Failed
        } else {
            TurnStatus::Completed
        };
        if let Some(p) = self.persistence.as_mut() {
            persist::log_persist_err(
                p.end_turn(status, Some(persist::usage_from_round(&total.usage))),
            );
        }
        if hit_step_cap {
            return Err(TurnError::TurnStepCapExceeded { cap: TURN_STEP_CAP });
        }

        send(&events_tx, AgentEvent::TurnComplete(total.clone())).await?;
        Ok(total)
    }

    /// Compacts `history` if the last turn's prompt size crossed the budget
    /// threshold. Preserves system messages and the most recent user prompt.
    /// Returns a short human-readable summary when something was dropped, so
    /// the agent loop can emit a `ContextCompaction` marker into the journal.
    fn maybe_compact(&mut self) -> Option<String> {
        if !self.token_budget.should_compact(self.last_input_tokens) {
            return None;
        }
        let removed = compact_messages(
            &mut self.history,
            self.last_input_tokens,
            &self.token_budget,
        );
        (removed > 0).then(|| {
            format!(
                "context compacted: {removed} oldest non-system message(s) dropped to fit the {} token budget",
                self.token_budget.input_budget(),
            )
        })
    }

    fn persist_assistant_round(&mut self, summary: &StreamSummary) {
        let Some(p) = self.persistence.as_mut() else {
            return;
        };
        if let Some(reasoning) = &summary.assistant_message.reasoning_content
            && !reasoning.is_empty()
        {
            persist::log_persist_err(p.append_item(persist::reasoning_item(reasoning.clone())));
        }
        if let Some(text) = &summary.assistant_message.content
            && !text.is_empty()
        {
            persist::log_persist_err(p.append_item(persist::agent_item(text.clone())));
        }
        for pc in &summary.pending_tool_calls {
            persist::log_persist_err(p.append_item(persist::tool_call_item(
                pc.id.clone(),
                pc.name.clone(),
                &pc.arguments,
            )));
        }
    }

    fn persist_tool_results(&mut self, events: &[AgentEvent]) {
        let Some(p) = self.persistence.as_mut() else {
            return;
        };
        for event in events {
            if let AgentEvent::ToolResult {
                id,
                content,
                is_error,
            } = event
            {
                persist::log_persist_err(p.append_item(persist::tool_result_item(
                    id.clone(),
                    content,
                    *is_error,
                )));
            }
        }
    }

    /// Drive one provider call to completion, returning the accumulated
    /// stream summary (text + reasoning + pending tool calls).
    async fn pump_round(
        &self,
        request: CompletionRequest,
        events_tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<StreamSummary, TurnError> {
        let (stream_tx, mut stream_rx) = mpsc::channel::<StreamEvent>(128);
        let collector_tx = events_tx.clone();

        let collector = tokio::spawn(async move {
            let mut router = StreamRouter::new();
            while let Some(event) = stream_rx.recv().await {
                for agent_event in router.on_event(event) {
                    if collector_tx.send(agent_event).await.is_err() {
                        break;
                    }
                }
            }
            router.into_summary()
        });

        self.provider
            .chat_completion_stream(request, stream_tx)
            .await?;
        let summary = collector.await.map_err(|_| TurnError::ChannelClosed)?;
        Ok(summary)
    }

    fn ensure_system_prompt(&mut self) {
        let has_system = self
            .history
            .first()
            .map(|m| matches!(m.role, Role::System))
            .unwrap_or(false);
        if !has_system {
            self.history.insert(0, Message::system(&self.system_prompt));
        }
    }
}

async fn send(tx: &mpsc::Sender<AgentEvent>, event: AgentEvent) -> Result<(), TurnError> {
    tx.send(event).await.map_err(|_| TurnError::ChannelClosed)
}

#[cfg(test)]
mod test_helpers;
#[cfg(test)]
mod tests;
#[cfg(test)]
#[path = "tests_persistence.rs"]
mod tests_persistence;
