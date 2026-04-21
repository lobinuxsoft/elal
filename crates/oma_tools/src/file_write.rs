//! `write` built-in (file `file_write.rs`) — create or fully overwrite a file.

use std::fs;

use async_trait::async_trait;
use oma_protocol::ToolDefinition;
use serde_json::json;

use crate::apply_patch::{resolve_path, unified_diff};
use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("write.txt");

/// Writes the full contents of a file, creating or overwriting.
pub struct FileWriteTool;

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "write",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target file path. Parent directory must exist."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full UTF-8 content to write."
                    }
                },
                "required": ["path", "content"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write",
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
        let size = args
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.len())
            .unwrap_or(0);
        format!("Write {path} ({size} bytes)")
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
                    name: "write".into(),
                    reason: "missing 'path' field".into(),
                })?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                name: "write".into(),
                reason: "missing 'content' field".into(),
            })?;

        let path = resolve_path(path_str, ctx.working_dir);
        if path.is_dir() {
            return Ok(ToolResult::soft_error(format!(
                "Path is a directory: {}",
                path.display()
            )));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Ok(ToolResult::soft_error(format!(
                    "Parent directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        let is_new = !path.exists();
        let original = if is_new {
            String::new()
        } else {
            fs::read_to_string(&path)
                .map_err(|e| ToolError::Execution(format!("failed to read existing file: {e}")))?
        };

        fs::write(&path, content)
            .map_err(|e| ToolError::Execution(format!("failed to write: {e}")))?;

        let body = if is_new {
            format!(
                "<path>{}</path>\n<created>true</created>\n<bytes>{}</bytes>",
                path.display(),
                content.len(),
            )
        } else {
            let diff = unified_diff(&path, &original, content);
            format!(
                "<path>{}</path>\n<created>false</created>\n<bytes>{}</bytes>\n<diff>\n{}\n</diff>",
                path.display(),
                content.len(),
                diff,
            )
        };
        Ok(ToolResult::ok(body).with_structured(json!({
            "created": is_new,
            "bytes_written": content.len(),
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
                session_id: oma_protocol::SessionId::new(),
                events: tx,
            },
            rx,
        )
    }

    #[tokio::test]
    async fn creates_new_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("new.txt");
        let tool = FileWriteTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({"path": path.to_string_lossy(), "content": "hello\n"}),
                &c,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello\n");
        assert!(out.content.contains("<created>true</created>"));
    }

    #[tokio::test]
    async fn overwrites_existing_file_and_emits_diff() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "old\n").unwrap();
        let tool = FileWriteTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({"path": path.to_string_lossy(), "content": "new\n"}),
                &c,
            )
            .await
            .unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "new\n");
        assert!(out.content.contains("<created>false</created>"));
        assert!(out.content.contains("<diff>"));
    }

    #[tokio::test]
    async fn directory_path_returns_soft_error() {
        let dir = TempDir::new().unwrap();
        let tool = FileWriteTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({
                    "path": dir.path().to_string_lossy(),
                    "content": "x",
                }),
                &c,
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("directory"));
    }

    #[tokio::test]
    async fn missing_parent_returns_soft_error() {
        let dir = TempDir::new().unwrap();
        let tool = FileWriteTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({
                    "path": dir.path().join("nope").join("a.txt").to_string_lossy(),
                    "content": "x",
                }),
                &c,
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("Parent directory"));
    }

    #[tokio::test]
    async fn missing_path_arg_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = FileWriteTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({"content": "x"}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn missing_content_arg_is_invalid() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        let tool = FileWriteTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(json!({"path": path.to_string_lossy()}), &c)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[test]
    fn spec_declares_always_approval_and_write_tier() {
        let spec = FileWriteTool.spec();
        assert_eq!(spec.name, "write");
        assert_eq!(spec.tier, ToolTier::Write);
        assert_eq!(spec.approval_hint, ApprovalHint::Always);
    }

    #[test]
    fn describe_action_reports_size() {
        let tool = FileWriteTool;
        assert_eq!(
            tool.describe_action(&json!({"path": "/x", "content": "12345"})),
            "Write /x (5 bytes)"
        );
    }
}
