//! Agent loop — orchestrates user ↔ LLM ↔ tools within a single turn.
//!
//! This is the "brain" layer. It consumes [`StreamEvent`](oma_protocol::StreamEvent)s
//! from an [`oma_provider::Provider`], routes them to [`AgentEvent`]s on a
//! consumer-facing channel, assembles tool calls, dispatches them through
//! [`oma_tools::ToolRegistry`] with approval mediated by
//! [`oma_tools::approval::resolve`], feeds results back as
//! `Message::tool`, and loops until the model produces plain text with no
//! pending tool calls.
//!
//! Lives in `oma_core` (not a standalone crate) because it orchestrates
//! existing pieces — Provider, ToolRegistry, config, context — and adding
//! a new crate would force downstream consumers to import yet another
//! path without any encapsulation benefit.
//!
//! Chunk 2 (this) wires the plain path only: no tools are advertised to
//! the provider, and the collector task drops tool-call fences on the
//! floor. Tool dispatch arrives in chunk 3.

use std::path::PathBuf;

use oma_protocol::{ApprovalMode, Message, Role, StreamEvent};
use oma_provider::{CompletionRequest, Provider, SamplingControls};
use oma_tools::ToolRegistry;
use tokio::sync::mpsc;

mod event;
mod history;
mod stream_router;
mod turn;

pub use event::{AgentEvent, ApprovalRequest, UserAction};
pub use turn::{TURN_STEP_CAP, TurnError, TurnSummary};

use stream_router::StreamRouter;

/// Stateful agent tied to a loaded [`Provider`] and a [`ToolRegistry`].
///
/// The lifetime parameter binds the agent to its borrowed pieces —
/// providers own GPU state that is expensive to re-create, so the typical
/// usage is to construct one `Agent` per REPL session and call
/// [`run_turn`](Self::run_turn) repeatedly.
pub struct Agent<'p, P: Provider> {
    provider: &'p P,
    #[allow(dead_code)] // chunk 3 wires the tools path
    tools: &'p ToolRegistry,
    history: Vec<Message>,
    system_prompt: String,
    #[allow(dead_code)] // chunk 4 wires the approval resolver
    approval_mode: ApprovalMode,
    #[allow(dead_code)] // chunk 3 passes this into ToolContext
    working_dir: PathBuf,
    sampling: SamplingControls,
}

impl<'p, P: Provider> Agent<'p, P> {
    /// Construct an agent bound to `provider` + `tools`. The `system_prompt`
    /// becomes the first message on every turn; history is empty until the
    /// first `run_turn` is called.
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
        }
    }

    /// Override the sampling controls. Builder-style for ergonomic setup.
    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingControls) -> Self {
        self.sampling = sampling;
        self
    }

    /// Snapshot of the current conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Drop all history; the system prompt is reinserted on the next turn.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Run a single turn against the provider.
    ///
    /// Chunk 2: plain path only — the provider sees an empty tool list,
    /// so no tool calls are expected. A future chunk advertises the
    /// registry's definitions and runs the loop until the model settles.
    pub async fn run_turn(
        &mut self,
        user_input: &str,
        events_tx: mpsc::Sender<AgentEvent>,
        _actions_rx: &mut mpsc::Receiver<UserAction>,
    ) -> Result<TurnSummary, TurnError> {
        self.ensure_system_prompt();
        self.history.push(Message::user(user_input));
        // Leave 10% headroom for generation.
        let budget = (self.provider.context_length() as f64 * 0.9) as usize;
        history::fit_to_budget(&mut self.history, budget);

        send(&events_tx, AgentEvent::TurnStart).await?;

        let request = CompletionRequest {
            messages: self.history.clone(),
            // Chunk 2: no tools advertised. Chunk 3 swaps this for
            // `self.tools.definitions()`.
            tools: Vec::new(),
            sampling: self.sampling.clone(),
            max_tokens: None,
        };

        let summary = self.pump_round(request, &events_tx).await?;

        let turn_summary = TurnSummary {
            stop_reason: summary.stop_reason,
            tool_calls_executed: 0,
            usage: summary.usage,
            cancelled: false,
        };
        send(&events_tx, AgentEvent::TurnComplete(turn_summary.clone())).await?;
        Ok(turn_summary)
    }

    /// Drive one provider call to completion and forward routed events.
    async fn pump_round(
        &self,
        request: CompletionRequest,
        events_tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<oma_provider::CompletionSummary, TurnError> {
        let (stream_tx, mut stream_rx) = mpsc::channel::<StreamEvent>(128);
        let collector_tx = events_tx.clone();

        // Collector runs on a separate task — only touches Send-safe
        // types. The Provider future stays on the main task because its
        // `chat_completion_stream` is `?Send`.
        let collector = tokio::spawn(async move {
            let mut router = StreamRouter::new();
            while let Some(event) = stream_rx.recv().await {
                for agent_event in router.on_event(event) {
                    if collector_tx.send(agent_event).await.is_err() {
                        // Consumer dropped the receiver — nothing more to
                        // emit, but keep draining so the provider's sender
                        // doesn't back-pressure.
                        break;
                    }
                }
            }
            router
        });

        let summary = self
            .provider
            .chat_completion_stream(request, stream_tx)
            .await?;
        let _router = collector.await.map_err(|_| TurnError::ChannelClosed)?;
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

/// Send an event, converting a closed channel into [`TurnError::ChannelClosed`].
async fn send(tx: &mpsc::Sender<AgentEvent>, event: AgentEvent) -> Result<(), TurnError> {
    tx.send(event).await.map_err(|_| TurnError::ChannelClosed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use oma_protocol::{StopReason, Usage};
    use oma_provider::{CompletionSummary, LlmError, ModelCapabilities};

    struct MockProvider {
        context_len: usize,
        events: Vec<StreamEvent>,
        stop_reason: StopReason,
        capabilities: ModelCapabilities,
    }

    impl MockProvider {
        fn new(events: Vec<StreamEvent>) -> Self {
            Self {
                context_len: 8192,
                events,
                stop_reason: StopReason::EndTurn,
                capabilities: ModelCapabilities::default(),
            }
        }
    }

    #[async_trait(?Send)]
    impl Provider for MockProvider {
        async fn chat_completion_stream(
            &self,
            _request: CompletionRequest,
            events: mpsc::Sender<StreamEvent>,
        ) -> Result<CompletionSummary, LlmError> {
            for ev in &self.events {
                events
                    .send(ev.clone())
                    .await
                    .map_err(|_| LlmError::ChannelClosed)?;
            }
            Ok(CompletionSummary {
                stop_reason: self.stop_reason,
                usage: Usage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    reasoning_tokens: 0,
                    prompt_eval_ms: 10,
                    generation_ms: 20,
                },
            })
        }
        fn model_name(&self) -> &str {
            "mock"
        }
        fn capabilities(&self) -> &ModelCapabilities {
            &self.capabilities
        }
        fn context_length(&self) -> usize {
            self.context_len
        }
    }

    fn new_registry() -> ToolRegistry {
        ToolRegistry::new()
    }

    #[tokio::test]
    async fn plain_path_emits_text_deltas_and_turn_complete() {
        let provider = MockProvider::new(vec![
            StreamEvent::TextStart,
            StreamEvent::TextDelta("hello ".into()),
            StreamEvent::TextDelta("world".into()),
            StreamEvent::TextEnd,
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
                usage: Usage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    reasoning_tokens: 0,
                    prompt_eval_ms: 10,
                    generation_ms: 20,
                },
            },
        ]);
        let tools = new_registry();
        let mut agent = Agent::new(&provider, &tools, "sys", ApprovalMode::Never, "/tmp");

        let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
        let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
        let summary = agent.run_turn("hi", ev_tx, &mut act_rx).await.unwrap();

        assert_eq!(summary.tool_calls_executed, 0);
        assert!(!summary.cancelled);

        let mut text_pieces = Vec::new();
        let mut saw_start = false;
        let mut saw_complete = false;
        while let Some(event) = ev_rx.recv().await {
            match event {
                AgentEvent::TurnStart => saw_start = true,
                AgentEvent::TextDelta(s) => text_pieces.push(s),
                AgentEvent::TurnComplete(_) => saw_complete = true,
                _ => {}
            }
        }
        assert!(saw_start);
        assert!(saw_complete);
        assert_eq!(text_pieces.join(""), "hello world");
    }

    #[tokio::test]
    async fn reasoning_deltas_pass_through() {
        let provider = MockProvider::new(vec![
            StreamEvent::ReasoningStart,
            StreamEvent::ReasoningDelta("thinking…".into()),
            StreamEvent::ReasoningEnd,
            StreamEvent::TextDelta("answer".into()),
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            },
        ]);
        let tools = new_registry();
        let mut agent = Agent::new(&provider, &tools, "sys", ApprovalMode::Never, "/tmp");
        let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
        let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
        agent.run_turn("?", ev_tx, &mut act_rx).await.unwrap();

        let mut reasoning = String::new();
        let mut text = String::new();
        while let Some(event) = ev_rx.recv().await {
            match event {
                AgentEvent::ReasoningDelta(s) => reasoning.push_str(&s),
                AgentEvent::TextDelta(s) => text.push_str(&s),
                _ => {}
            }
        }
        assert_eq!(reasoning, "thinking…");
        assert_eq!(text, "answer");
    }

    #[tokio::test]
    async fn ensures_system_prompt_is_first_message() {
        let provider = MockProvider::new(vec![StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }]);
        let tools = new_registry();
        let mut agent = Agent::new(&provider, &tools, "be helpful", ApprovalMode::Never, "/tmp");
        let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
        let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
        agent.run_turn("hi", ev_tx, &mut act_rx).await.unwrap();
        while ev_rx.recv().await.is_some() {}

        let hist = agent.history();
        assert_eq!(hist.len(), 2);
        assert!(matches!(hist[0].role, Role::System));
        assert_eq!(hist[0].content.as_deref(), Some("be helpful"));
        assert!(matches!(hist[1].role, Role::User));
    }
}
