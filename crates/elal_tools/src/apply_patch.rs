//! `apply_patch` built-in — transactional multi-hunk edits over a single file.
//!
//! Also exposes the shared [`apply_hunks`] helper and [`Hunk`] type consumed
//! by the `edit` built-in so both tools share a single apply implementation.

use std::fs;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use elal_protocol::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use similar::TextDiff;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("apply_patch.txt");

/// One contextual substitution used by both `apply_patch` and `edit`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Hunk {
    pub old_text: String,
    pub new_text: String,
}

/// Outcome of applying a sequence of hunks in memory.
pub(crate) struct ApplyOutcome {
    pub original: String,
    pub updated: String,
}

/// Errors produced by [`apply_hunks`] before any disk write happens.
#[derive(Debug)]
pub(crate) enum ApplyError {
    Io(std::io::Error),
    EmptyHunks,
    EmptyOldText { index: usize },
    NoMatch { index: usize },
    Ambiguous { index: usize, occurrences: usize },
    Noop { index: usize },
}

/// Applies hunks in memory against `path`'s current content.
///
/// Returns the original and updated buffers without touching disk. Callers
/// persist the result themselves so callers can also render diffs before
/// committing.
pub(crate) fn apply_hunks(path: &Path, hunks: &[Hunk]) -> Result<ApplyOutcome, ApplyError> {
    if hunks.is_empty() {
        return Err(ApplyError::EmptyHunks);
    }
    let original = fs::read_to_string(path).map_err(ApplyError::Io)?;
    let mut buffer = original.clone();
    for (i, hunk) in hunks.iter().enumerate() {
        if hunk.old_text.is_empty() {
            return Err(ApplyError::EmptyOldText { index: i });
        }
        if hunk.old_text == hunk.new_text {
            return Err(ApplyError::Noop { index: i });
        }
        let occurrences = buffer.matches(&hunk.old_text).count();
        match occurrences {
            0 => return Err(ApplyError::NoMatch { index: i }),
            1 => {
                buffer = buffer.replacen(&hunk.old_text, &hunk.new_text, 1);
            }
            n => {
                return Err(ApplyError::Ambiguous {
                    index: i,
                    occurrences: n,
                });
            }
        }
    }
    Ok(ApplyOutcome {
        original,
        updated: buffer,
    })
}

/// Renders a unified diff for a single file's before/after pair.
pub(crate) fn unified_diff(path: &Path, original: &str, updated: &str) -> String {
    let display = path.display().to_string();
    TextDiff::from_lines(original, updated)
        .unified_diff()
        .header(&display, &display)
        .to_string()
}

/// Converts [`ApplyError`] into our tool-surface error model.
///
/// `tool_name` names the surface so the error message mentions the right
/// tool (`apply_patch` vs `edit`).
pub(crate) fn apply_error_to_tool_error(err: ApplyError, tool_name: &str) -> ToolError {
    match err {
        ApplyError::Io(e) => ToolError::Execution(e.to_string()),
        ApplyError::EmptyHunks => ToolError::InvalidArgs {
            name: tool_name.into(),
            reason: "patch must contain at least one hunk".into(),
        },
        ApplyError::EmptyOldText { index } => ToolError::InvalidArgs {
            name: tool_name.into(),
            reason: format!("hunk {index} has empty 'old_text'"),
        },
        ApplyError::NoMatch { index } => ToolError::InvalidArgs {
            name: tool_name.into(),
            reason: format!("hunk {index} 'old_text' not found in file"),
        },
        ApplyError::Ambiguous { index, occurrences } => ToolError::InvalidArgs {
            name: tool_name.into(),
            reason: format!(
                "hunk {index} 'old_text' appears {occurrences} times; add context to disambiguate"
            ),
        },
        ApplyError::Noop { index } => ToolError::InvalidArgs {
            name: tool_name.into(),
            reason: format!("hunk {index} has identical 'old_text' and 'new_text'"),
        },
    }
}

/// Resolves a possibly-relative path against the tool's working directory.
pub(crate) fn resolve_path(path_str: &str, working_dir: &Path) -> PathBuf {
    let p = Path::new(path_str);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        working_dir.join(p)
    }
}

/// Multi-hunk transactional file editor.
pub struct ApplyPatchTool;

#[async_trait]
impl Tool for ApplyPatchTool {
    fn name(&self) -> &str {
        "apply_patch"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "apply_patch",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit. Must exist."
                    },
                    "patch": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": { "type": "string" },
                                "new_text": { "type": "string" }
                            },
                            "required": ["old_text", "new_text"]
                        }
                    }
                },
                "required": ["path", "patch"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "apply_patch",
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
        let count = args
            .get("patch")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        format!(
            "Apply patch to {path} ({count} hunk{})",
            if count == 1 { "" } else { "s" }
        )
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
                    name: "apply_patch".into(),
                    reason: "missing 'path' field".into(),
                })?;
        let hunks_raw = args.get("patch").ok_or_else(|| ToolError::InvalidArgs {
            name: "apply_patch".into(),
            reason: "missing 'patch' field".into(),
        })?;
        let hunks: Vec<Hunk> =
            serde_json::from_value(hunks_raw.clone()).map_err(|e| ToolError::InvalidArgs {
                name: "apply_patch".into(),
                reason: format!("invalid 'patch' payload: {e}"),
            })?;
        if hunks.is_empty() {
            return Err(ToolError::InvalidArgs {
                name: "apply_patch".into(),
                reason: "'patch' must contain at least one hunk".into(),
            });
        }

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

        let outcome =
            apply_hunks(&path, &hunks).map_err(|e| apply_error_to_tool_error(e, "apply_patch"))?;

        fs::write(&path, &outcome.updated)
            .map_err(|e| ToolError::Execution(format!("failed to write: {e}")))?;

        let diff = unified_diff(&path, &outcome.original, &outcome.updated);
        let content = format!(
            "<path>{}</path>\n<hunks_applied>{}</hunks_applied>\n<diff>\n{}\n</diff>",
            path.display(),
            hunks.len(),
            diff,
        );
        Ok(ToolResult::ok(content).with_structured(json!({
            "hunks_applied": hunks.len(),
            "bytes_written": outcome.updated.len(),
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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
    async fn single_hunk_edit_succeeds() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "hello world\n").unwrap();
        let tool = ApplyPatchTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({
                    "path": path.to_string_lossy(),
                    "patch": [{"old_text": "hello", "new_text": "goodbye"}],
                }),
                &c,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(fs::read_to_string(&path).unwrap(), "goodbye world\n");
        assert!(out.content.contains("<hunks_applied>1</hunks_applied>"));
    }

    #[tokio::test]
    async fn multiple_hunks_apply_sequentially() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "aaa bbb ccc\n").unwrap();
        let tool = ApplyPatchTool;
        let (c, _rx) = ctx(dir.path());
        tool.execute(
            json!({
                "path": path.to_string_lossy(),
                "patch": [
                    {"old_text": "aaa", "new_text": "AAA"},
                    {"old_text": "ccc", "new_text": "CCC"},
                ],
            }),
            &c,
        )
        .await
        .unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "AAA bbb CCC\n");
    }

    #[tokio::test]
    async fn ambiguous_hunk_rejects_without_writing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "foo\nfoo\n").unwrap();
        let tool = ApplyPatchTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(
                json!({
                    "path": path.to_string_lossy(),
                    "patch": [{"old_text": "foo", "new_text": "bar"}],
                }),
                &c,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
        assert_eq!(fs::read_to_string(&path).unwrap(), "foo\nfoo\n");
    }

    #[tokio::test]
    async fn transaction_aborts_on_any_hunk_failure() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "aaa\n").unwrap();
        let tool = ApplyPatchTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(
                json!({
                    "path": path.to_string_lossy(),
                    "patch": [
                        {"old_text": "aaa", "new_text": "AAA"},
                        {"old_text": "zzz", "new_text": "ZZZ"},
                    ],
                }),
                &c,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "aaa\n",
            "file must be unchanged when any hunk fails"
        );
    }

    #[tokio::test]
    async fn missing_file_returns_soft_error() {
        let dir = TempDir::new().unwrap();
        let tool = ApplyPatchTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({
                    "path": "nope.txt",
                    "patch": [{"old_text": "a", "new_text": "b"}],
                }),
                &c,
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("File not found"));
    }

    #[tokio::test]
    async fn noop_hunk_is_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "same\n").unwrap();
        let tool = ApplyPatchTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(
                json!({
                    "path": path.to_string_lossy(),
                    "patch": [{"old_text": "same", "new_text": "same"}],
                }),
                &c,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn empty_patch_is_invalid() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        fs::write(&path, "x\n").unwrap();
        let tool = ApplyPatchTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(
                json!({
                    "path": path.to_string_lossy(),
                    "patch": [],
                }),
                &c,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[test]
    fn spec_declares_always_approval_and_write_tier() {
        let spec = ApplyPatchTool.spec();
        assert_eq!(spec.tier, ToolTier::Write);
        assert_eq!(spec.approval_hint, ApprovalHint::Always);
        assert_eq!(spec.side_effects, SideEffects::Local);
    }

    #[test]
    fn describe_action_includes_hunk_count() {
        let tool = ApplyPatchTool;
        assert_eq!(
            tool.describe_action(&json!({
                "path": "/x",
                "patch": [{"old_text": "a", "new_text": "b"}],
            })),
            "Apply patch to /x (1 hunk)"
        );
        assert_eq!(
            tool.describe_action(&json!({
                "path": "/x",
                "patch": [
                    {"old_text": "a", "new_text": "b"},
                    {"old_text": "c", "new_text": "d"},
                ],
            })),
            "Apply patch to /x (2 hunks)"
        );
    }
}
