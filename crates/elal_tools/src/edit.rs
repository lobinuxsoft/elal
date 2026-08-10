//! `edit` built-in — single-hunk convenience wrapper over `apply_patch`.

use std::fs;

use async_trait::async_trait;
use elal_protocol::ToolDefinition;
use serde_json::json;

use crate::apply_patch::{
    Hunk, apply_error_to_tool_error, apply_hunks, resolve_path, unified_diff,
};
use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("edit.txt");

/// Ergonomic single-hunk edit tool — reuses `apply_patch`'s core logic.
pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "edit",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit. Must exist."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact substring to replace. Must appear exactly once."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement substring."
                    }
                },
                "required": ["path", "old_text", "new_text"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit",
            tier: ToolTier::Write,
            approval_hint: ApprovalHint::Always,
            side_effects: SideEffects::Local,
        }
    }

    fn describe_action(&self, args: &serde_json::Value) -> String {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing path>");
        format!("Edit {path} (1 hunk)")
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let path_str =
            args.get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs {
                    name: "edit".into(),
                    reason: "missing 'path' field".into(),
                })?;
        let old_text = args
            .get("old_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                name: "edit".into(),
                reason: "missing 'old_text' field".into(),
            })?;
        let new_text = args
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                name: "edit".into(),
                reason: "missing 'new_text' field".into(),
            })?;

        let path = resolve_path(path_str, ctx.working_dir);
        if !path.exists() {
            return Ok(ToolResult::soft_error(format!(
                "File not found: {}",
                path.display()
            )));
        }
        if !path.is_file() {
            return Ok(ToolResult::soft_error(format!(
                "Path is not a file: {}",
                path.display()
            )));
        }

        let hunks = [Hunk {
            old_text: old_text.to_string(),
            new_text: new_text.to_string(),
        }];
        let outcome =
            apply_hunks(&path, &hunks).map_err(|e| apply_error_to_tool_error(e, "edit"))?;

        fs::write(&path, &outcome.updated)
            .map_err(|e| ToolError::Execution(format!("failed to write: {e}")))?;

        let diff = unified_diff(&path, &outcome.original, &outcome.updated);
        let content = format!("<path>{}</path>\n<diff>\n{}\n</diff>", path.display(), diff,);
        Ok(ToolResult::ok(content).with_structured(json!({
            "bytes_written": outcome.updated.len(),
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;
    use tokio::sync::mpsc;

    fn ctx<'a>(wd: &'a Path) -> (ToolContext<'a>, mpsc::Receiver<crate::context::ToolEvent>) {
        let (tx, rx) = mpsc::channel(4);
        (
            ToolContext {
                working_dir: wd,
                project_root: None,
                session_id: elal_protocol::SessionId::new(),
                events: tx,
            },
            rx,
        )
    }

    #[tokio::test]
    async fn replaces_unique_occurrence() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "greeting: hello\n").unwrap();
        let tool = EditTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({
                    "path": path.to_string_lossy(),
                    "old_text": "hello",
                    "new_text": "goodbye",
                }),
                &c,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(fs::read_to_string(&path).unwrap(), "greeting: goodbye\n");
    }

    #[tokio::test]
    async fn ambiguous_old_text_is_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "foo\nfoo\n").unwrap();
        let tool = EditTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(
                json!({
                    "path": path.to_string_lossy(),
                    "old_text": "foo",
                    "new_text": "bar",
                }),
                &c,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
        assert_eq!(fs::read_to_string(&path).unwrap(), "foo\nfoo\n");
    }

    #[tokio::test]
    async fn missing_old_text_is_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "hello\n").unwrap();
        let tool = EditTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(
                json!({
                    "path": path.to_string_lossy(),
                    "old_text": "not present",
                    "new_text": "x",
                }),
                &c,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn missing_path_arg_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = EditTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(json!({"old_text": "a", "new_text": "b"}), &c)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[test]
    fn spec_declares_always_approval() {
        let spec = EditTool.spec();
        assert_eq!(spec.tier, ToolTier::Write);
        assert_eq!(spec.approval_hint, ApprovalHint::Always);
    }

    #[test]
    fn describe_action_mentions_single_hunk() {
        let tool = EditTool;
        assert_eq!(
            tool.describe_action(&json!({"path": "/x", "old_text": "a", "new_text": "b"})),
            "Edit /x (1 hunk)"
        );
    }
}
