//! Tool dispatch — runs a batch of [`PendingToolCall`]s against the
//! [`ToolRegistry`], emitting [`AgentEvent::ToolResult`]s and returning
//! the `Message::tool` payloads the next provider round needs.
//!
//! Chunk 3 runs every tool without consulting the approval resolver —
//! equivalent to `ApprovalMode::Never`. Chunk 4 will insert the approval
//! handshake around the `execute` call.

use std::path::Path;

use oma_protocol::{Message, SessionId};
use oma_tools::{ToolContext, ToolRegistry};
use tokio::sync::mpsc;

use super::event::AgentEvent;
use super::stream_router::PendingToolCall;
use super::turn::TurnError;

/// Outcome of dispatching a batch of pending tool calls.
pub(crate) struct DispatchOutcome {
    /// Consumer-facing events produced during dispatch — one
    /// [`AgentEvent::ToolResult`] per call, successful or soft-error.
    pub events: Vec<AgentEvent>,
    /// Messages to append to history so the next provider round sees the
    /// tool results in context.
    pub tool_messages: Vec<Message>,
    /// How many tools actually ran (vs lookup misses / arg parse fails /
    /// hard errors we synthesised into soft results).
    pub executed: u32,
}

/// Dispatch the batch. Unknown tools, invalid argument JSON, and any
/// [`oma_tools::ToolError`] returned by `execute` are converted into soft
/// `ToolResult { is_error: true }` payloads so the model can recover —
/// hard `TurnError` only surfaces for unrecoverable issues (none today).
pub(crate) async fn dispatch(
    pending: Vec<PendingToolCall>,
    tools: &ToolRegistry,
    working_dir: &Path,
    session_id: SessionId,
) -> Result<DispatchOutcome, TurnError> {
    let mut events = Vec::with_capacity(pending.len());
    let mut tool_messages = Vec::with_capacity(pending.len());
    let mut executed = 0u32;

    for pc in pending {
        let tool = match tools.get(&pc.name) {
            Some(t) => t.clone(),
            None => {
                let msg = format!("tool '{}' is not registered", pc.name);
                push_soft_error(&pc.id, msg, &mut events, &mut tool_messages);
                continue;
            }
        };

        let args: serde_json::Value = match serde_json::from_str(&pc.arguments) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("arguments for '{}' were not valid JSON: {e}", pc.name);
                push_soft_error(&pc.id, msg, &mut events, &mut tool_messages);
                continue;
            }
        };

        // Build a ToolContext for the invocation. The progress channel is
        // currently dropped — chunk 3 doesn't forward `ToolEvent::Progress`
        // to AgentEvent. Forwarding is a later follow-up once the TUI
        // actually renders it.
        let (tool_events_tx, _tool_events_rx) = mpsc::channel(16);
        let ctx = ToolContext {
            working_dir,
            project_root: None,
            session_id,
            events: tool_events_tx,
        };

        match tool.execute(args, &ctx).await {
            Ok(result) => {
                events.push(AgentEvent::ToolResult {
                    id: pc.id.clone(),
                    content: result.content.clone(),
                    is_error: result.is_error,
                });
                tool_messages.push(Message::tool(pc.id, result.content));
                executed += 1;
            }
            Err(err) => {
                // Convert any tool error into a soft failure so the model
                // can see the failure and decide how to recover.
                let msg = format!("tool '{}' failed: {err}", pc.name);
                push_soft_error(&pc.id, msg, &mut events, &mut tool_messages);
            }
        }
    }

    Ok(DispatchOutcome {
        events,
        tool_messages,
        executed,
    })
}

fn push_soft_error(
    id: &str,
    message: String,
    events: &mut Vec<AgentEvent>,
    tool_messages: &mut Vec<Message>,
) {
    events.push(AgentEvent::ToolResult {
        id: id.to_string(),
        content: message.clone(),
        is_error: true,
    });
    tool_messages.push(Message::tool(id.to_string(), message));
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use oma_protocol::ToolDefinition;
    use oma_tools::{ApprovalHint, SideEffects, Tool, ToolError, ToolResult, ToolSpec, ToolTier};
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Dummy tool whose behaviour is scripted per test — echoes args or
    /// returns a configured error.
    struct ScriptedTool {
        name: &'static str,
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
                serde_json::json!({
                    "type": "object",
                    "properties": {"arg": {"type": "string"}}
                }),
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
            match &self.output {
                Ok(s) => Ok(ToolResult::ok(s.clone())),
                Err(e) => Err(ToolError::Execution(e.clone())),
            }
        }
    }

    fn registry_with(tool: ScriptedTool) -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(tool));
        r
    }

    #[tokio::test]
    async fn successful_tool_produces_result_and_tool_message() {
        let reg = registry_with(ScriptedTool {
            name: "echo",
            output: Ok("hello".into()),
        });
        let wd = TempDir::new().unwrap();
        let out = dispatch(
            vec![PendingToolCall {
                id: "c1".into(),
                name: "echo".into(),
                arguments: "{}".into(),
            }],
            &reg,
            wd.path(),
            SessionId::new(),
        )
        .await
        .unwrap();
        assert_eq!(out.executed, 1);
        assert_eq!(out.events.len(), 1);
        match &out.events[0] {
            AgentEvent::ToolResult {
                content, is_error, ..
            } => {
                assert_eq!(content, "hello");
                assert!(!is_error);
            }
            _ => panic!("expected ToolResult"),
        }
        assert_eq!(out.tool_messages.len(), 1);
        assert_eq!(out.tool_messages[0].content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn unknown_tool_becomes_soft_error_not_hard_error() {
        let reg = ToolRegistry::new();
        let wd = TempDir::new().unwrap();
        let out = dispatch(
            vec![PendingToolCall {
                id: "c1".into(),
                name: "ghost".into(),
                arguments: "{}".into(),
            }],
            &reg,
            wd.path(),
            SessionId::new(),
        )
        .await
        .unwrap();
        assert_eq!(out.executed, 0);
        assert!(matches!(
            &out.events[0],
            AgentEvent::ToolResult { is_error: true, content, .. } if content.contains("not registered")
        ));
    }

    #[tokio::test]
    async fn invalid_args_become_soft_error() {
        let reg = registry_with(ScriptedTool {
            name: "echo",
            output: Ok("ok".into()),
        });
        let wd = TempDir::new().unwrap();
        let out = dispatch(
            vec![PendingToolCall {
                id: "c1".into(),
                name: "echo".into(),
                arguments: "not json".into(),
            }],
            &reg,
            wd.path(),
            SessionId::new(),
        )
        .await
        .unwrap();
        assert_eq!(out.executed, 0);
        assert!(matches!(
            &out.events[0],
            AgentEvent::ToolResult { is_error: true, content, .. } if content.contains("not valid JSON")
        ));
    }

    #[tokio::test]
    async fn tool_execution_error_becomes_soft_error() {
        let reg = registry_with(ScriptedTool {
            name: "boom",
            output: Err("disk full".into()),
        });
        let wd = TempDir::new().unwrap();
        let out = dispatch(
            vec![PendingToolCall {
                id: "c1".into(),
                name: "boom".into(),
                arguments: "{}".into(),
            }],
            &reg,
            wd.path(),
            SessionId::new(),
        )
        .await
        .unwrap();
        assert_eq!(out.executed, 0);
        assert!(matches!(
            &out.events[0],
            AgentEvent::ToolResult { is_error: true, content, .. } if content.contains("disk full")
        ));
    }
}
