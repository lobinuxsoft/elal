//! Integration-style tests for `Agent::run_turn` against a scripted
//! mock provider and a registry of scripted mock tools.

use async_trait::async_trait;
use oma_protocol::{StopReason, StreamEvent, ToolDefinition, Usage};
use oma_provider::{CompletionRequest, CompletionSummary, LlmError, ModelCapabilities, Provider};
use oma_tools::{
    ApprovalHint, SideEffects, Tool, ToolContext, ToolError, ToolRegistry, ToolResult, ToolSpec,
    ToolTier,
};
use std::cell::RefCell;
use std::sync::Arc;
use tokio::sync::mpsc;

use super::{Agent, AgentEvent, UserAction};
use oma_protocol::ApprovalMode;

/// Mock provider that replays a per-round script of events. Each call to
/// `chat_completion_stream` consumes the next script entry.
struct ScriptedProvider {
    rounds: RefCell<Vec<Vec<StreamEvent>>>,
    capabilities: ModelCapabilities,
}

impl ScriptedProvider {
    fn new(rounds: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            rounds: RefCell::new(rounds),
            capabilities: ModelCapabilities::default(),
        }
    }
}

#[async_trait(?Send)]
impl Provider for ScriptedProvider {
    async fn chat_completion_stream(
        &self,
        _request: CompletionRequest,
        events: mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionSummary, LlmError> {
        let next = self
            .rounds
            .borrow_mut()
            .pop()
            .ok_or_else(|| LlmError::Decode("no more scripted rounds".into()))?;
        for ev in next {
            events.send(ev).await.map_err(|_| LlmError::ChannelClosed)?;
        }
        Ok(CompletionSummary {
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                reasoning_tokens: 0,
                prompt_eval_ms: 1,
                generation_ms: 1,
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
        8192
    }
}

impl ScriptedProvider {
    fn new_last_first(mut rounds: Vec<Vec<StreamEvent>>) -> Self {
        // Pop consumes from the end — reverse so the first script entry
        // is what the first call sees.
        rounds.reverse();
        Self::new(rounds)
    }
}

/// Scripted tool whose behaviour is fixed at construction.
struct ScriptedTool {
    name: &'static str,
    output: String,
}

#[async_trait]
impl Tool for ScriptedTool {
    fn name(&self) -> &str {
        self.name
    }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            self.name,
            "scripted",
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name,
            tier: ToolTier::Read,
            approval_hint: ApprovalHint::Never,
            side_effects: SideEffects::None,
        }
    }
    fn describe_action(&self, _args: &serde_json::Value) -> String {
        format!("run {}", self.name)
    }
    async fn execute(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::ok(self.output.clone()))
    }
}

fn done(stop: StopReason) -> StreamEvent {
    StreamEvent::Done {
        stop_reason: stop,
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            reasoning_tokens: 0,
            prompt_eval_ms: 1,
            generation_ms: 1,
        },
    }
}

#[tokio::test]
async fn plain_path_emits_text_deltas_and_turn_complete() {
    let provider = ScriptedProvider::new(vec![vec![
        StreamEvent::TextStart,
        StreamEvent::TextDelta("hello ".into()),
        StreamEvent::TextDelta("world".into()),
        StreamEvent::TextEnd,
        done(StopReason::EndTurn),
    ]]);
    let tools = ToolRegistry::new();
    let mut agent = Agent::new(&provider, &tools, "sys", ApprovalMode::Never, "/tmp");

    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
    let summary = agent.run_turn("hi", ev_tx, &mut act_rx).await.unwrap();

    assert_eq!(summary.tool_calls_executed, 0);
    let mut text = String::new();
    let mut saw_complete = false;
    while let Some(ev) = ev_rx.recv().await {
        match ev {
            AgentEvent::TextDelta(s) => text.push_str(&s),
            AgentEvent::TurnComplete(_) => saw_complete = true,
            _ => {}
        }
    }
    assert!(saw_complete);
    assert_eq!(text, "hello world");
}

#[tokio::test]
async fn tool_call_loop_dispatches_and_feeds_back_result() {
    // Round 1: model emits a tool call. Round 2: model produces text.
    let provider = ScriptedProvider::new_last_first(vec![
        vec![
            StreamEvent::ToolCallStart {
                id: "c1".into(),
                name: "echo".into(),
            },
            StreamEvent::ToolCallInputDelta("{}".into()),
            StreamEvent::ToolCallEnd,
            done(StopReason::EndTurn),
        ],
        vec![
            StreamEvent::TextDelta("final answer".into()),
            done(StopReason::EndTurn),
        ],
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(ScriptedTool {
        name: "echo",
        output: "tool said hi".into(),
    }));
    let mut agent = Agent::new(&provider, &tools, "sys", ApprovalMode::Never, "/tmp");

    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(128);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
    let summary = agent
        .run_turn("use tools", ev_tx, &mut act_rx)
        .await
        .unwrap();

    assert_eq!(summary.tool_calls_executed, 1);
    let mut saw_start = false;
    let mut saw_result = false;
    let mut text = String::new();
    while let Some(ev) = ev_rx.recv().await {
        match ev {
            AgentEvent::ToolCallStart { name, .. } if name == "echo" => saw_start = true,
            AgentEvent::ToolResult {
                content, is_error, ..
            } => {
                saw_result = true;
                assert!(!is_error);
                assert!(content.contains("tool said hi"));
            }
            AgentEvent::TextDelta(s) => text.push_str(&s),
            _ => {}
        }
    }
    assert!(saw_start);
    assert!(saw_result);
    assert_eq!(text, "final answer");
    // History must reflect the full round-trip: system, user, assistant
    // (with tool_calls), tool, assistant (final text).
    let h = agent.history();
    assert_eq!(h.len(), 5);
}

#[tokio::test]
async fn unknown_tool_becomes_soft_error_and_loop_continues() {
    let provider = ScriptedProvider::new_last_first(vec![
        vec![
            StreamEvent::ToolCallStart {
                id: "c1".into(),
                name: "ghost".into(),
            },
            StreamEvent::ToolCallInputDelta("{}".into()),
            StreamEvent::ToolCallEnd,
            done(StopReason::EndTurn),
        ],
        vec![
            StreamEvent::TextDelta("sorry".into()),
            done(StopReason::EndTurn),
        ],
    ]);
    let tools = ToolRegistry::new();
    let mut agent = Agent::new(&provider, &tools, "sys", ApprovalMode::Never, "/tmp");
    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
    let summary = agent
        .run_turn("call ghost", ev_tx, &mut act_rx)
        .await
        .unwrap();
    assert_eq!(summary.tool_calls_executed, 0);
    let mut saw_error = false;
    while let Some(ev) = ev_rx.recv().await {
        if let AgentEvent::ToolResult {
            is_error, content, ..
        } = ev
        {
            if is_error && content.contains("not registered") {
                saw_error = true;
            }
        }
    }
    assert!(saw_error);
}

#[tokio::test]
async fn ensures_system_prompt_is_first_message() {
    let provider = ScriptedProvider::new(vec![vec![done(StopReason::EndTurn)]]);
    let tools = ToolRegistry::new();
    let mut agent = Agent::new(&provider, &tools, "be helpful", ApprovalMode::Never, "/tmp");
    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
    agent.run_turn("hi", ev_tx, &mut act_rx).await.unwrap();
    while ev_rx.recv().await.is_some() {}

    let hist = agent.history();
    assert!(matches!(hist[0].role, oma_protocol::Role::System));
    assert_eq!(hist[0].content.as_deref(), Some("be helpful"));
}
