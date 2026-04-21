//! `grep` built-in — regex search across files using the ripgrep building
//! blocks (`grep-searcher` + `grep-regex`).

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use grep_matcher::Matcher;
use grep_regex::RegexMatcher;
use grep_searcher::{Searcher, Sink, SinkMatch};
use oma_protocol::ToolDefinition;
use serde_json::json;
use walkdir::WalkDir;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("grep.txt");
const MAX_MATCHES: usize = 500;

/// Searches file contents for a regex pattern.
pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "grep",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern in Rust regex syntax."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional base directory. Defaults to the current working directory."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional filename glob filter (e.g. `**/*.rs`)."
                    }
                },
                "required": ["pattern"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep",
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
        let base = args.get("path").and_then(|v| v.as_str()).unwrap_or("<cwd>");
        match args.get("glob").and_then(|v| v.as_str()) {
            Some(g) => format!("Grep `{pattern}` in {base} (glob: {g})"),
            None => format!("Grep `{pattern}` in {base}"),
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
                name: "grep".into(),
                reason: "missing 'pattern' field".into(),
            })?;
        if pattern.trim().is_empty() {
            return Err(ToolError::InvalidArgs {
                name: "grep".into(),
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

        let matcher = RegexMatcher::new(pattern).map_err(|e| ToolError::InvalidArgs {
            name: "grep".into(),
            reason: format!("invalid regex: {e}"),
        })?;

        let glob_filter = args
            .get("glob")
            .and_then(|v| v.as_str())
            .map(|g| {
                globwalk::GlobWalkerBuilder::from_patterns(&base, &[g])
                    .follow_links(false)
                    .build()
                    .map(|w| {
                        w.flatten()
                            .map(|e| e.path().to_path_buf())
                            .collect::<Vec<_>>()
                    })
                    .map_err(|e| ToolError::InvalidArgs {
                        name: "grep".into(),
                        reason: format!("invalid glob: {e}"),
                    })
            })
            .transpose()?;

        let candidates: Vec<PathBuf> = match glob_filter {
            Some(files) => files.into_iter().filter(|p| p.is_file()).collect(),
            None => {
                let walker = if base.is_file() {
                    WalkDir::new(&base).into_iter().collect::<Vec<_>>()
                } else {
                    WalkDir::new(&base)
                        .into_iter()
                        .filter_entry(|e| !is_hidden(e.path(), &base))
                        .collect()
                };
                walker
                    .into_iter()
                    .flatten()
                    .filter(|e| e.file_type().is_file())
                    .map(|e| e.path().to_path_buf())
                    .collect()
            }
        };

        let mut matches: Vec<Match> = Vec::new();
        let mut truncated = false;
        for file in &candidates {
            if matches.len() >= MAX_MATCHES {
                truncated = true;
                break;
            }
            let before = matches.len();
            let remaining = MAX_MATCHES - before;
            let mut sink = CollectorSink {
                file: file.clone(),
                base: &base,
                out: &mut matches,
                remaining,
            };
            let mut searcher = Searcher::new();
            if searcher.search_path(&matcher, file, &mut sink).is_err() {
                // Silently skip files we can't read (binary, permission denied, etc.)
                continue;
            }
        }
        matches.sort_by(|a, b| a.path.cmp(&b.path).then(a.line.cmp(&b.line)));

        let mut content = format!(
            "<base>{}</base>\n<pattern>{}</pattern>\n<matches>\n",
            base.display(),
            pattern,
        );
        for m in &matches {
            content.push_str(&format!("{}:{}:{}\n", m.path.display(), m.line, m.text));
        }
        if truncated {
            content.push_str(&format!(
                "\n(Output capped at {} matches. Refine the pattern or narrow the path.)",
                MAX_MATCHES,
            ));
        } else {
            content.push_str(&format!("\n({} matches)", matches.len()));
        }
        content.push_str("\n</matches>");

        Ok(ToolResult::ok(content).with_structured(json!({
            "total": matches.len(),
            "truncated": truncated,
        })))
    }
}

struct Match {
    path: PathBuf,
    line: u64,
    text: String,
}

struct CollectorSink<'a> {
    file: PathBuf,
    base: &'a Path,
    out: &'a mut Vec<Match>,
    remaining: usize,
}

impl<'a> Sink for CollectorSink<'a> {
    type Error = std::io::Error;

    fn matched(&mut self, _searcher: &Searcher, mat: &SinkMatch<'_>) -> Result<bool, Self::Error> {
        if self.remaining == 0 {
            return Ok(false);
        }
        let line_number = mat.line_number().unwrap_or(0);
        let text = String::from_utf8_lossy(mat.bytes())
            .trim_end_matches('\n')
            .trim_end_matches('\r')
            .to_string();
        let rel = self.file.strip_prefix(self.base).unwrap_or(&self.file);
        self.out.push(Match {
            path: rel.to_path_buf(),
            line: line_number,
            text,
        });
        self.remaining -= 1;
        Ok(true)
    }
}

fn is_hidden(path: &Path, base: &Path) -> bool {
    let Ok(rel) = path.strip_prefix(base) else {
        return false;
    };
    rel.components()
        .any(|c| matches!(c.as_os_str().to_str(), Some(s) if s != "." && s.starts_with('.')))
}

// Silences the unused-parameter lint: `Matcher` is needed for the trait
// bound; the import is otherwise only used via its method call through
// `RegexMatcher`.
#[allow(dead_code)]
fn _matcher_trait_bound<M: Matcher>(_: &M) {}

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
        fs::write(
            root.join("src").join("lib.rs"),
            "fn foo() {}\nfn bar() {}\n// TODO: wire\n",
        )
        .unwrap();
        fs::write(
            root.join("src").join("main.rs"),
            "fn main() {\n    println!(\"TODO\");\n}\n",
        )
        .unwrap();
        fs::write(root.join("README.md"), "# project\n\nTODO\n").unwrap();
        fs::create_dir_all(root.join(".hidden")).unwrap();
        fs::write(root.join(".hidden").join("secret"), "TODO hidden\n").unwrap();
    }

    #[tokio::test]
    async fn finds_matches_across_tree() {
        let dir = TempDir::new().unwrap();
        scaffold(dir.path());
        let tool = GrepTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool.execute(json!({"pattern": "TODO"}), &c).await.unwrap();
        assert!(out.content.contains("src/lib.rs"));
        assert!(out.content.contains("src/main.rs"));
        assert!(out.content.contains("README.md"));
        assert!(!out.content.contains(".hidden"));
    }

    #[tokio::test]
    async fn glob_filter_narrows_search() {
        let dir = TempDir::new().unwrap();
        scaffold(dir.path());
        let tool = GrepTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"pattern": "TODO", "glob": "**/*.rs"}), &c)
            .await
            .unwrap();
        assert!(out.content.contains("lib.rs"));
        assert!(out.content.contains("main.rs"));
        assert!(!out.content.contains("README.md"));
    }

    #[tokio::test]
    async fn path_narrows_to_subtree() {
        let dir = TempDir::new().unwrap();
        scaffold(dir.path());
        let tool = GrepTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({
                    "pattern": "fn",
                    "path": dir.path().join("src").to_string_lossy(),
                }),
                &c,
            )
            .await
            .unwrap();
        assert!(out.content.contains("lib.rs"));
        assert!(out.content.contains("main.rs"));
        assert!(!out.content.contains("README.md"));
    }

    #[tokio::test]
    async fn invalid_regex_is_invalid_args() {
        let dir = TempDir::new().unwrap();
        let tool = GrepTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({"pattern": "["}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn missing_pattern_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = GrepTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn missing_base_returns_soft_error() {
        let dir = TempDir::new().unwrap();
        let tool = GrepTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"pattern": "x", "path": "does_not_exist"}), &c)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("not found"));
    }

    #[test]
    fn spec_declares_never_approval() {
        let spec = GrepTool.spec();
        assert_eq!(spec.tier, ToolTier::Read);
        assert_eq!(spec.approval_hint, ApprovalHint::Never);
    }
}
