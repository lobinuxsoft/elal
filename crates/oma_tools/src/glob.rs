//! `glob` built-in — find files by glob pattern, sorted by mtime.

use std::cmp::Reverse;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use async_trait::async_trait;
use oma_protocol::ToolDefinition;
use serde_json::json;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("glob.txt");
const MAX_RESULTS: usize = 200;

/// Walks the filesystem applying a glob pattern.
pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "glob",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. `**/*.rs`). Relative to `path`."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional base directory. Defaults to the current working directory."
                    }
                },
                "required": ["pattern"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "glob",
            tier: ToolTier::Read,
            approval_hint: ApprovalHint::Never,
            side_effects: SideEffects::Local,
        }
    }

    fn describe_action(&self, args: &serde_json::Value) -> String {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing pattern>");
        match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => format!("Glob `{pattern}` under {p}"),
            None => format!("Glob `{pattern}`"),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                name: "glob".into(),
                reason: "missing 'pattern' field".into(),
            })?;
        if pattern.trim().is_empty() {
            return Err(ToolError::InvalidArgs {
                name: "glob".into(),
                reason: "'pattern' must not be empty".into(),
            });
        }

        let base = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => {
                let candidate = Path::new(p);
                if candidate.is_absolute() {
                    candidate.to_path_buf()
                } else {
                    ctx.working_dir.join(candidate)
                }
            }
            None => ctx.working_dir.to_path_buf(),
        };
        if !base.exists() {
            return Ok(ToolResult::soft_error(format!(
                "Base path not found: {}",
                base.display()
            )));
        }
        if !base.is_dir() {
            return Ok(ToolResult::soft_error(format!(
                "Base path is not a directory: {}",
                base.display()
            )));
        }

        let walker = globwalk::GlobWalkerBuilder::from_patterns(&base, &[pattern])
            .follow_links(false)
            .build()
            .map_err(|e| ToolError::InvalidArgs {
                name: "glob".into(),
                reason: format!("invalid glob pattern: {e}"),
            })?;

        let mut hits: Vec<(PathBuf, SystemTime)> = Vec::new();
        for entry in walker.flatten() {
            let path = entry.path().to_path_buf();
            if path.is_dir() {
                continue;
            }
            if file_is_hidden(&path, &base) {
                continue;
            }
            let mtime = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            hits.push((path, mtime));
        }
        hits.sort_by_key(|(_, mtime)| Reverse(*mtime));

        let total = hits.len();
        let truncated = total > MAX_RESULTS;
        let shown: Vec<&PathBuf> = hits.iter().take(MAX_RESULTS).map(|(p, _)| p).collect();

        let mut content = format!(
            "<base>{}</base>\n<pattern>{}</pattern>\n<matches>\n",
            base.display(),
            pattern,
        );
        for p in &shown {
            content.push_str(&p.display().to_string());
            content.push('\n');
        }
        if truncated {
            content.push_str(&format!(
                "\n(Showing {} of {} matches. Refine the pattern to see more.)",
                shown.len(),
                total,
            ));
        } else {
            content.push_str(&format!("\n({} matches)", total));
        }
        content.push_str("\n</matches>");

        Ok(ToolResult::ok(content).with_structured(json!({
            "total": total,
            "returned": shown.len(),
            "truncated": truncated,
        })))
    }
}

fn file_is_hidden(path: &Path, base: &Path) -> bool {
    let Ok(rel) = path.strip_prefix(base) else {
        return false;
    };
    rel.components()
        .any(|c| matches!(c.as_os_str().to_str(), Some(s) if s.starts_with('.')))
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
                session_id: oma_protocol::SessionId::new(),
                events: tx,
            },
            rx,
        )
    }

    fn scaffold(root: &Path) {
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src").join("lib.rs"), "").unwrap();
        fs::write(root.join("src").join("main.rs"), "").unwrap();
        fs::create_dir_all(root.join("tests")).unwrap();
        fs::write(root.join("tests").join("it.rs"), "").unwrap();
        fs::write(root.join("README.md"), "").unwrap();
        fs::create_dir_all(root.join(".hidden")).unwrap();
        fs::write(root.join(".hidden").join("x.rs"), "").unwrap();
    }

    #[tokio::test]
    async fn matches_rust_files_recursively() {
        let dir = TempDir::new().unwrap();
        scaffold(dir.path());
        let tool = GlobTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"pattern": "**/*.rs"}), &c)
            .await
            .unwrap();
        assert!(out.content.contains("lib.rs"));
        assert!(out.content.contains("main.rs"));
        assert!(out.content.contains("it.rs"));
        assert!(!out.content.contains(".hidden"));
    }

    #[tokio::test]
    async fn uses_explicit_base_path() {
        let dir = TempDir::new().unwrap();
        scaffold(dir.path());
        let tool = GlobTool;
        let (c, _rx) = ctx(Path::new("/"));
        let out = tool
            .execute(
                json!({
                    "pattern": "*.rs",
                    "path": dir.path().join("src").to_string_lossy(),
                }),
                &c,
            )
            .await
            .unwrap();
        assert!(out.content.contains("lib.rs"));
        assert!(!out.content.contains("it.rs"));
    }

    #[tokio::test]
    async fn missing_pattern_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = GlobTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn empty_pattern_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = GlobTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(json!({"pattern": "   "}), &c)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn missing_base_path_returns_soft_error() {
        let dir = TempDir::new().unwrap();
        let tool = GlobTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"pattern": "*", "path": "does_not_exist"}), &c)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("not found"));
    }

    #[test]
    fn spec_declares_never_approval() {
        let spec = GlobTool.spec();
        assert_eq!(spec.tier, ToolTier::Read);
        assert_eq!(spec.approval_hint, ApprovalHint::Never);
    }
}
