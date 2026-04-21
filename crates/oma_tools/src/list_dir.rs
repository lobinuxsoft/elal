//! `list_dir` built-in — directory listing, optionally recursive.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use oma_protocol::ToolDefinition;
use serde_json::json;
use walkdir::WalkDir;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("list_dir.txt");
const DEFAULT_MAX_DEPTH: usize = 3;
const MAX_ENTRIES: usize = 500;

/// Lists directory contents with optional recursion.
pub struct ListDirTool;

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_dir",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory. Relative paths are resolved against the working directory."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, walk subdirectories up to max_depth. Default false."
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum recursion depth when recursive is true. Default 3."
                    }
                },
                "required": ["path"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir",
            tier: ToolTier::Read,
            approval_hint: ApprovalHint::Never,
            side_effects: SideEffects::Local,
        }
    }

    fn describe_action(&self, args: &serde_json::Value) -> String {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing path>");
        let recursive = args
            .get("recursive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if recursive {
            let d = args
                .get("max_depth")
                .and_then(|v| v.as_u64())
                .unwrap_or(DEFAULT_MAX_DEPTH as u64);
            format!("List {path} (recursive, depth {d})")
        } else {
            format!("List {path}")
        }
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
                    name: "list_dir".into(),
                    reason: "missing 'path' field".into(),
                })?;
        let recursive = args
            .get("recursive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max_depth = match args.get("max_depth") {
            Some(v) if v.is_null() => DEFAULT_MAX_DEPTH,
            Some(v) => {
                let n = v.as_u64().ok_or_else(|| ToolError::InvalidArgs {
                    name: "list_dir".into(),
                    reason: "'max_depth' must be a positive integer".into(),
                })?;
                if n == 0 {
                    return Err(ToolError::InvalidArgs {
                        name: "list_dir".into(),
                        reason: "'max_depth' must be >= 1".into(),
                    });
                }
                n as usize
            }
            None => DEFAULT_MAX_DEPTH,
        };

        let path = resolve_path(path_str, ctx.working_dir);
        if !path.exists() {
            return Ok(ToolResult::soft_error(format!(
                "Path not found: {}",
                path.display()
            )));
        }
        if !path.is_dir() {
            return Ok(ToolResult::soft_error(format!(
                "Path is not a directory: {}",
                path.display()
            )));
        }

        let entries = collect_entries(&path, recursive, max_depth)?;
        let total = entries.len();
        let truncated = total > MAX_ENTRIES;
        let shown: Vec<&Entry> = entries.iter().take(MAX_ENTRIES).collect();

        let mut content = format!(
            "<path>{}</path>\n<type>directory</type>\n<entries>\n",
            path.display()
        );
        for e in &shown {
            content.push_str(&format!("{} {:>10}  {}\n", e.kind, e.size, e.display));
        }
        if truncated {
            content.push_str(&format!(
                "\n(Showing {} of {} entries. Use glob for pattern-based lookup across larger trees.)",
                shown.len(),
                total,
            ));
        } else {
            content.push_str(&format!("\n({} entries)", total));
        }
        content.push_str("\n</entries>");

        Ok(ToolResult::ok(content).with_structured(json!({
            "total": total,
            "returned": shown.len(),
            "truncated": truncated,
            "recursive": recursive,
        })))
    }
}

struct Entry {
    kind: char,
    size: u64,
    display: String,
}

fn resolve_path(path_str: &str, working_dir: &Path) -> PathBuf {
    let p = Path::new(path_str);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        working_dir.join(p)
    }
}

fn collect_entries(
    root: &Path,
    recursive: bool,
    max_depth: usize,
) -> Result<Vec<Entry>, ToolError> {
    let walker = if recursive {
        WalkDir::new(root)
            .min_depth(1)
            .max_depth(max_depth)
            .sort_by_file_name()
    } else {
        WalkDir::new(root)
            .min_depth(1)
            .max_depth(1)
            .sort_by_file_name()
    };

    let mut entries = Vec::new();
    for result in walker {
        let entry = result.map_err(|e| ToolError::Execution(e.to_string()))?;
        let ft = entry.file_type();
        let kind = if ft.is_dir() {
            'd'
        } else if ft.is_symlink() {
            'l'
        } else {
            'f'
        };
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
        let mut display = rel.to_string_lossy().to_string();
        if ft.is_dir() {
            display.push('/');
        }
        entries.push(Entry {
            kind,
            size,
            display,
        });
    }
    entries.sort_by(|a, b| a.display.to_lowercase().cmp(&b.display.to_lowercase()));
    Ok(entries)
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
        fs::write(root.join("a.txt"), "hello").unwrap();
        fs::write(root.join("b.txt"), "world").unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        fs::write(root.join("sub").join("c.txt"), "nested").unwrap();
    }

    #[tokio::test]
    async fn non_recursive_lists_top_level_only() {
        let dir = TempDir::new().unwrap();
        scaffold(dir.path());
        let tool = ListDirTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"path": dir.path().to_string_lossy()}), &c)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("a.txt"));
        assert!(out.content.contains("b.txt"));
        assert!(out.content.contains("sub/"));
        assert!(!out.content.contains("c.txt"));
        assert!(out.content.contains("(3 entries)"));
    }

    #[tokio::test]
    async fn recursive_traverses_subdirectories() {
        let dir = TempDir::new().unwrap();
        scaffold(dir.path());
        let tool = ListDirTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({"path": dir.path().to_string_lossy(), "recursive": true}),
                &c,
            )
            .await
            .unwrap();
        assert!(out.content.contains("c.txt"));
    }

    #[tokio::test]
    async fn max_depth_caps_recursion() {
        let dir = TempDir::new().unwrap();
        let level1 = dir.path().join("l1");
        let level2 = level1.join("l2");
        let level3 = level2.join("l3");
        fs::create_dir_all(&level3).unwrap();
        fs::write(level3.join("deep.txt"), "x").unwrap();

        let tool = ListDirTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({
                    "path": dir.path().to_string_lossy(),
                    "recursive": true,
                    "max_depth": 1
                }),
                &c,
            )
            .await
            .unwrap();
        assert!(out.content.contains("l1/"));
        assert!(!out.content.contains("deep.txt"));
    }

    #[tokio::test]
    async fn missing_path_returns_soft_error() {
        let dir = TempDir::new().unwrap();
        let tool = ListDirTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool.execute(json!({"path": "nope"}), &c).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("not found"));
    }

    #[tokio::test]
    async fn file_path_returns_soft_error() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("a.txt");
        fs::write(&file, "x").unwrap();
        let tool = ListDirTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"path": file.to_string_lossy()}), &c)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("not a directory"));
    }

    #[tokio::test]
    async fn missing_path_arg_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = ListDirTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[test]
    fn describe_action_renders_flags() {
        let tool = ListDirTool;
        assert_eq!(tool.describe_action(&json!({"path": "/p"})), "List /p");
        assert_eq!(
            tool.describe_action(&json!({"path": "/p", "recursive": true})),
            "List /p (recursive, depth 3)"
        );
        assert_eq!(
            tool.describe_action(&json!({"path": "/p", "recursive": true, "max_depth": 5})),
            "List /p (recursive, depth 5)"
        );
    }

    #[test]
    fn spec_declares_never_approval() {
        let spec = ListDirTool.spec();
        assert_eq!(spec.name, "list_dir");
        assert_eq!(spec.tier, ToolTier::Read);
        assert_eq!(spec.approval_hint, ApprovalHint::Never);
    }
}
