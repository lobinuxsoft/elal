//! Unit tests for the tool dispatcher + approval flow.
//!
//! Each test spawns a tiny "responder" task that listens on the events
//! channel for [`AgentEvent::ApprovalRequired`] and replies on the
//! actions channel. This keeps the main task free to drive `dispatch`
//! to completion without borrow-checker gymnastics.

use super::*;
use async_trait::async_trait;
use oma_protocol::{ApprovalDecisionValue, ApprovalScopeValue, ToolDefinition};
use oma_tools::{ApprovalHint, SideEffects, Tool, ToolError, ToolResult, ToolSpec, ToolTier};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::mpsc;

struct ScriptedTool {
    name: &'static str,
    hint: ApprovalHint,
    output: Result<String, String>,
}

#[async_trait]
impl Tool for ScriptedTool {
    fn name(&self) -> &str {
        self.name
    }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            self.name,
            "scripted tool",
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name,
            tier: ToolTier::Read,
            approval_hint: self.hint,
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
        match &self.output {
            Ok(s) => Ok(ToolResult::ok(s.clone())),
            Err(e) => Err(ToolError::Execution(e.clone())),
        }
    }
}

fn tool_always(name: &'static str) -> ScriptedTool {
    ScriptedTool {
        name,
        hint: ApprovalHint::Always,
        output: Ok("done".into()),
    }
}

fn tool_never(name: &'static str) -> ScriptedTool {
    ScriptedTool {
        name,
        hint: ApprovalHint::Never,
        output: Ok("done".into()),
    }
}

fn registry_with(tool: ScriptedTool) -> ToolRegistry {
    let mut r = ToolRegistry::new();
    r.register(Arc::new(tool));
    r
}

fn pending(name: &str, id: &str) -> PendingToolCall {
    PendingToolCall {
        id: id.into(),
        name: name.into(),
        arguments: "{}".into(),
    }
}

/// Spawn a responder that auto-replies to every `ApprovalRequired` event
/// with the given decision + scope. Returns a JoinHandle the caller can
/// await after dropping the events_tx.
fn spawn_responder(
    mut events_rx: mpsc::Receiver<AgentEvent>,
    actions_tx: mpsc::Sender<UserAction>,
    decision: ApprovalDecisionValue,
    scope: ApprovalScopeValue,
) -> tokio::task::JoinHandle<u32> {
    tokio::spawn(async move {
        let mut prompts_seen = 0u32;
        while let Some(event) = events_rx.recv().await {
            if let AgentEvent::ApprovalRequired(req) = event {
                prompts_seen += 1;
                let _ = actions_tx
                    .send(UserAction::ApprovalResponse {
                        request_id: req.request_id,
                        decision,
                        scope,
                    })
                    .await;
            }
        }
        prompts_seen
    })
}

#[tokio::test]
async fn never_hint_in_smart_mode_executes_without_prompt() {
    let (events_tx, mut events_rx) = mpsc::channel::<AgentEvent>(64);
    let (_actions_tx, mut actions_rx) = mpsc::channel::<UserAction>(16);
    let wd = TempDir::new().unwrap();
    let reg = registry_with(tool_never("read"));
    let mut approved_session = HashSet::new();
    let mut approved_turn = HashSet::new();

    let out = dispatch(
        vec![pending("read", "c1")],
        DispatchContext {
            tools: &reg,
            working_dir: wd.path(),
            session_id: SessionId::new(),
            mode: ApprovalMode::Smart,
            approved_for_session: &mut approved_session,
            approved_for_turn: &mut approved_turn,
            events_tx: &events_tx,
            actions_rx: &mut actions_rx,
        },
    )
    .await
    .unwrap();

    assert_eq!(out.executed, 1);
    assert!(!out.cancelled);

    drop(events_tx);
    // No ApprovalRequired should have been emitted.
    while let Some(ev) = events_rx.recv().await {
        assert!(!matches!(ev, AgentEvent::ApprovalRequired(_)));
    }
}

#[tokio::test]
async fn always_hint_prompts_and_respects_approve() {
    let (events_tx, events_rx) = mpsc::channel::<AgentEvent>(64);
    let (actions_tx, mut actions_rx) = mpsc::channel::<UserAction>(16);
    let responder = spawn_responder(
        events_rx,
        actions_tx,
        ApprovalDecisionValue::Approve,
        ApprovalScopeValue::Once,
    );
    let wd = TempDir::new().unwrap();
    let reg = registry_with(tool_always("write"));
    let mut approved_session = HashSet::new();
    let mut approved_turn = HashSet::new();

    let out = dispatch(
        vec![pending("write", "c1")],
        DispatchContext {
            tools: &reg,
            working_dir: wd.path(),
            session_id: SessionId::new(),
            mode: ApprovalMode::Smart,
            approved_for_session: &mut approved_session,
            approved_for_turn: &mut approved_turn,
            events_tx: &events_tx,
            actions_rx: &mut actions_rx,
        },
    )
    .await
    .unwrap();

    drop(events_tx);
    let prompts = responder.await.unwrap();
    assert_eq!(prompts, 1);
    assert_eq!(out.executed, 1);
    assert!(!out.cancelled);
}

#[tokio::test]
async fn deny_becomes_soft_error_and_skips_execute() {
    let (events_tx, events_rx) = mpsc::channel::<AgentEvent>(64);
    let (actions_tx, mut actions_rx) = mpsc::channel::<UserAction>(16);
    let responder = spawn_responder(
        events_rx,
        actions_tx,
        ApprovalDecisionValue::Deny,
        ApprovalScopeValue::Once,
    );
    let wd = TempDir::new().unwrap();
    let reg = registry_with(tool_always("write"));
    let mut approved_session = HashSet::new();
    let mut approved_turn = HashSet::new();

    let out = dispatch(
        vec![pending("write", "c1")],
        DispatchContext {
            tools: &reg,
            working_dir: wd.path(),
            session_id: SessionId::new(),
            mode: ApprovalMode::Smart,
            approved_for_session: &mut approved_session,
            approved_for_turn: &mut approved_turn,
            events_tx: &events_tx,
            actions_rx: &mut actions_rx,
        },
    )
    .await
    .unwrap();

    drop(events_tx);
    responder.await.unwrap();
    assert_eq!(out.executed, 0);
    assert!(!out.cancelled);
    assert!(
        out.tool_messages[0]
            .content
            .as_deref()
            .unwrap()
            .contains("denied by user")
    );
}

#[tokio::test]
async fn cancel_ends_the_turn_early() {
    let (events_tx, events_rx) = mpsc::channel::<AgentEvent>(64);
    let (actions_tx, mut actions_rx) = mpsc::channel::<UserAction>(16);
    let responder = spawn_responder(
        events_rx,
        actions_tx,
        ApprovalDecisionValue::Cancel,
        ApprovalScopeValue::Once,
    );
    let wd = TempDir::new().unwrap();
    let reg = registry_with(tool_always("write"));
    let mut approved_session = HashSet::new();
    let mut approved_turn = HashSet::new();

    let out = dispatch(
        vec![pending("write", "c1"), pending("write", "c2")],
        DispatchContext {
            tools: &reg,
            working_dir: wd.path(),
            session_id: SessionId::new(),
            mode: ApprovalMode::Smart,
            approved_for_session: &mut approved_session,
            approved_for_turn: &mut approved_turn,
            events_tx: &events_tx,
            actions_rx: &mut actions_rx,
        },
    )
    .await
    .unwrap();

    drop(events_tx);
    responder.await.unwrap();
    assert_eq!(out.executed, 0);
    assert!(out.cancelled);
}

#[tokio::test]
async fn session_scope_prevents_second_prompt_for_same_tool() {
    let (events_tx, events_rx) = mpsc::channel::<AgentEvent>(64);
    let (actions_tx, mut actions_rx) = mpsc::channel::<UserAction>(16);
    let responder = spawn_responder(
        events_rx,
        actions_tx,
        ApprovalDecisionValue::Approve,
        ApprovalScopeValue::Session,
    );
    let wd = TempDir::new().unwrap();
    let reg = registry_with(tool_always("write"));
    let mut approved_session = HashSet::new();
    let mut approved_turn = HashSet::new();

    // First call — prompt + approve with Session scope.
    dispatch(
        vec![pending("write", "c1")],
        DispatchContext {
            tools: &reg,
            working_dir: wd.path(),
            session_id: SessionId::new(),
            mode: ApprovalMode::Smart,
            approved_for_session: &mut approved_session,
            approved_for_turn: &mut approved_turn,
            events_tx: &events_tx,
            actions_rx: &mut actions_rx,
        },
    )
    .await
    .unwrap();
    assert!(approved_session.contains("write"));

    // Second call — should skip the prompt entirely because the tool is
    // in the session allowlist.
    let out2 = dispatch(
        vec![pending("write", "c2")],
        DispatchContext {
            tools: &reg,
            working_dir: wd.path(),
            session_id: SessionId::new(),
            mode: ApprovalMode::Smart,
            approved_for_session: &mut approved_session,
            approved_for_turn: &mut approved_turn,
            events_tx: &events_tx,
            actions_rx: &mut actions_rx,
        },
    )
    .await
    .unwrap();

    drop(events_tx);
    let prompts = responder.await.unwrap();
    // Exactly one prompt total — the second call bypassed it.
    assert_eq!(prompts, 1);
    assert_eq!(out2.executed, 1);
}

#[tokio::test]
async fn always_mode_prompts_even_for_never_hint() {
    let (events_tx, events_rx) = mpsc::channel::<AgentEvent>(64);
    let (actions_tx, mut actions_rx) = mpsc::channel::<UserAction>(16);
    let responder = spawn_responder(
        events_rx,
        actions_tx,
        ApprovalDecisionValue::Approve,
        ApprovalScopeValue::Once,
    );
    let wd = TempDir::new().unwrap();
    let reg = registry_with(tool_never("read"));
    let mut approved_session = HashSet::new();
    let mut approved_turn = HashSet::new();

    let out = dispatch(
        vec![pending("read", "c1")],
        DispatchContext {
            tools: &reg,
            working_dir: wd.path(),
            session_id: SessionId::new(),
            mode: ApprovalMode::Always,
            approved_for_session: &mut approved_session,
            approved_for_turn: &mut approved_turn,
            events_tx: &events_tx,
            actions_rx: &mut actions_rx,
        },
    )
    .await
    .unwrap();

    drop(events_tx);
    let prompts = responder.await.unwrap();
    assert_eq!(prompts, 1);
    assert_eq!(out.executed, 1);
}

#[tokio::test]
async fn unknown_tool_is_soft_error_before_approval() {
    let (events_tx, mut events_rx) = mpsc::channel::<AgentEvent>(64);
    let (_actions_tx, mut actions_rx) = mpsc::channel::<UserAction>(16);
    let wd = TempDir::new().unwrap();
    let reg = ToolRegistry::new();
    let mut approved_session = HashSet::new();
    let mut approved_turn = HashSet::new();

    let out = dispatch(
        vec![pending("ghost", "c1")],
        DispatchContext {
            tools: &reg,
            working_dir: wd.path(),
            session_id: SessionId::new(),
            mode: ApprovalMode::Always,
            approved_for_session: &mut approved_session,
            approved_for_turn: &mut approved_turn,
            events_tx: &events_tx,
            actions_rx: &mut actions_rx,
        },
    )
    .await
    .unwrap();

    assert_eq!(out.executed, 0);
    assert!(!out.cancelled);
    assert!(
        out.tool_messages[0]
            .content
            .as_deref()
            .unwrap()
            .contains("not registered")
    );

    drop(events_tx);
    while let Some(ev) = events_rx.recv().await {
        assert!(!matches!(ev, AgentEvent::ApprovalRequired(_)));
    }
}
