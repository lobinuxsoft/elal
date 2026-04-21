//! Agent loop — orchestrates user ↔ LLM ↔ tools within a single turn.
//!
//! This is the "brain" layer. It consumes [`StreamEvent`](oma_protocol::StreamEvent)s
//! from an [`oma_provider::Provider`], routes them to [`AgentEvent`]s on a
//! consumer-facing channel, assembles tool calls, dispatches them through
//! [`oma_tools::ToolRegistry`], feeds results back as `Message::tool`, and
//! loops until the model produces plain text with no pending tool calls.
//!
//! Lives in `oma_core` (not a standalone crate) because it orchestrates
//! existing pieces — Provider, ToolRegistry, config, context — without a
//! meaningful encapsulation boundary of its own.
//!
//! Chunk 3 (this) wires tool dispatch and the multi-round loop. Approval
//! is bypassed — every tool executes, equivalent to `ApprovalMode::Never`.
//! Chunk 4 inserts the approval handshake around the `execute` call.

use std::path::PathBuf;

use oma_protocol::{ApprovalMode, Message, Role, SessionId, StreamEvent};
use oma_provider::{CompletionRequest, Provider, SamplingControls};
use oma_tools::ToolRegistry;
use tokio::sync::mpsc;

mod event;
mod history;
mod stream_router;
mod tool_exec;
mod turn;

pub use event::{AgentEvent, ApprovalRequest, UserAction};
pub use turn::{TURN_STEP_CAP, TurnError, TurnSummary};

use stream_router::{StreamRouter, StreamSummary};

/// Stateful agent tied to a loaded [`Provider`] and a [`ToolRegistry`].
pub struct Agent<'p, P: Provider> {
    provider: &'p P,
    tools: &'p ToolRegistry,
    history: Vec<Message>,
    system_prompt: String,
    #[allow(dead_code)] // chunk 4 wires the approval resolver
    approval_mode: ApprovalMode,
    working_dir: PathBuf,
    sampling: SamplingControls,
    /// Per-turn session id — fresh per `run_turn`, passed into every
    /// `ToolContext` so tools can correlate their own telemetry.
    session_id: SessionId,
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
        }
    }

    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingControls) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.session_id = SessionId::new();
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
        _actions_rx: &mut mpsc::Receiver<UserAction>,
    ) -> Result<TurnSummary, TurnError> {
        self.ensure_system_prompt();
        self.history.push(Message::user(user_input));

        send(&events_tx, AgentEvent::TurnStart).await?;

        let mut total = TurnSummary {
            stop_reason: oma_protocol::StopReason::EndTurn,
            tool_calls_executed: 0,
            usage: oma_protocol::Usage::default(),
            cancelled: false,
        };

        for round in 0..TURN_STEP_CAP {
            let budget = (self.provider.context_length() as f64 * 0.9) as usize;
            history::fit_to_budget(&mut self.history, budget);

            let request = CompletionRequest {
                messages: self.history.clone(),
                tools: self.tools.definitions(),
                sampling: self.sampling.clone(),
                max_tokens: None,
            };

            let stream_summary = self.pump_round(request, &events_tx).await?;
            total.accumulate(&stream_summary.usage);
            total.stop_reason = stream_summary.stop_reason;

            // Record the assistant's output in history so the next round
            // (and future turns) see what the model just said.
            self.history.push(stream_summary.assistant_message.clone());

            if stream_summary.pending_tool_calls.is_empty() {
                // Model finished without requesting more work.
                break;
            }

            let dispatch = tool_exec::dispatch(
                stream_summary.pending_tool_calls,
                self.tools,
                &self.working_dir,
                self.session_id,
            )
            .await?;

            for event in dispatch.events {
                send(&events_tx, event).await?;
            }
            self.history.extend(dispatch.tool_messages);
            total.tool_calls_executed = total.tool_calls_executed.saturating_add(dispatch.executed);

            // If the cap is one away, and we still need another round,
            // the next iteration will abort — fall through and let the
            // loop condition do its thing.
            if round + 1 == TURN_STEP_CAP {
                return Err(TurnError::TurnStepCapExceeded { cap: TURN_STEP_CAP });
            }
        }

        send(&events_tx, AgentEvent::TurnComplete(total.clone())).await?;
        Ok(total)
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
mod tests;
