//! `todo_write` built-in — model-facing scratchpad for tracking task progress.
//!
//! The tool itself is a pure echo: the model emits a list of `{step, status}`
//! items, the tool validates that at most one item is `in_progress`, and the
//! same payload is returned both as text content (for the model's next round)
//! and as a structured JSON value (for UI layers that want to render a live
//! checklist).

use async_trait::async_trait;
use oma_protocol::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = "Create and update a structured task list for the current coding session. \
                           Each item has a `step` description and a `status` (pending, in_progress, completed). \
                           At most one step may be in_progress at a time. \
                           Use this to track progress on multi-step requests so the user can see what is left.";

/// Allowed status values for a todo item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// One entry in the todo list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub step: String,
    pub status: TodoStatus,
}

/// Tool: lets the model maintain a visible task list.
pub struct TodoWriteTool;

#[async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &str {
        "todo_write"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "todo_write",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "Ordered list of todo items.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": { "type": "string" },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"]
                                }
                            },
                            "required": ["step", "status"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["todos"],
                "additionalProperties": false
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "todo_write",
            tier: ToolTier::Read,
            approval_hint: ApprovalHint::Never,
            side_effects: SideEffects::None,
        }
    }

    fn describe_action(&self, args: &serde_json::Value) -> String {
        let count = args
            .get("todos")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        format!("Update todo list ({count} items)")
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let todos: Vec<TodoItem> = serde_json::from_value(
            args.get("todos").cloned().unwrap_or(json!([])),
        )
        .map_err(|e| ToolError::InvalidArgs {
            name: "todo_write".into(),
            reason: format!("`todos` did not match the schema: {e}"),
        })?;

        let in_progress = todos
            .iter()
            .filter(|t| t.status == TodoStatus::InProgress)
            .count();
        if in_progress > 1 {
            return Ok(ToolResult::soft_error(format!(
                "{in_progress} items are in_progress; at most one is allowed at a time."
            )));
        }

        let rendered = render_todos(&todos);
        let structured =
            serde_json::to_value(&todos).map_err(|e| ToolError::Execution(e.to_string()))?;
        Ok(ToolResult::ok(rendered).with_structured(structured))
    }
}

fn render_todos(todos: &[TodoItem]) -> String {
    if todos.is_empty() {
        return "(empty todo list)".into();
    }
    let mut out = String::new();
    for (i, t) in todos.iter().enumerate() {
        let mark = match t.status {
            TodoStatus::Pending => "[ ]",
            TodoStatus::InProgress => "[~]",
            TodoStatus::Completed => "[x]",
        };
        out.push_str(&format!("{}. {} {}\n", i + 1, mark, t.step));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use tokio::sync::mpsc;

    fn ctx<'a>(wd: &'a Path) -> (ToolContext<'a>, mpsc::Receiver<crate::context::ToolEvent>) {
        let (tx, rx) = mpsc::channel(4);
        (
            ToolContext {
                working_dir: wd,
                project_root: None,
                session_id: oma_protocol::SessionId::new(),
                events: tx,
            },
            rx,
        )
    }

    #[tokio::test]
    async fn echoes_todos_and_renders_checkboxes() {
        let wd = PathBuf::from("/tmp");
        let (cx, _rx) = ctx(&wd);
        let args = json!({
            "todos": [
                { "step": "scaffold module", "status": "completed" },
                { "step": "wire registry",  "status": "in_progress" },
                { "step": "tests",          "status": "pending" }
            ]
        });
        let out = TodoWriteTool.execute(args, &cx).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("[x] scaffold module"));
        assert!(out.content.contains("[~] wire registry"));
        assert!(out.content.contains("[ ] tests"));
        assert!(out.structured.is_some());
    }

    #[tokio::test]
    async fn empty_todos_render_placeholder() {
        let wd = PathBuf::from("/tmp");
        let (cx, _rx) = ctx(&wd);
        let out = TodoWriteTool
            .execute(json!({ "todos": [] }), &cx)
            .await
            .unwrap();
        assert_eq!(out.content, "(empty todo list)");
    }

    #[tokio::test]
    async fn rejects_two_in_progress() {
        let wd = PathBuf::from("/tmp");
        let (cx, _rx) = ctx(&wd);
        let args = json!({
            "todos": [
                { "step": "a", "status": "in_progress" },
                { "step": "b", "status": "in_progress" }
            ]
        });
        let out = TodoWriteTool.execute(args, &cx).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("at most one"));
    }

    #[tokio::test]
    async fn invalid_status_is_hard_error() {
        let wd = PathBuf::from("/tmp");
        let (cx, _rx) = ctx(&wd);
        let args = json!({
            "todos": [{ "step": "x", "status": "blocked" }]
        });
        let err = TodoWriteTool.execute(args, &cx).await.unwrap_err();
        match err {
            ToolError::InvalidArgs { name, .. } => assert_eq!(name, "todo_write"),
            other => panic!("expected InvalidArgs, got {other:?}"),
        }
    }

    #[test]
    fn describe_action_reports_count() {
        let args = json!({ "todos": [{"step":"a","status":"pending"}] });
        assert_eq!(
            TodoWriteTool.describe_action(&args),
            "Update todo list (1 items)"
        );
    }

    #[test]
    fn spec_is_read_only_pure() {
        let s = TodoWriteTool.spec();
        assert_eq!(s.name, "todo_write");
        assert_eq!(s.tier, ToolTier::Read);
        assert_eq!(s.approval_hint, ApprovalHint::Never);
        assert_eq!(s.side_effects, SideEffects::None);
    }
}
