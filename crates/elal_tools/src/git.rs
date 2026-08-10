//! `git` built-in — structured git operations with args-aware approval.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use elal_protocol::ToolDefinition;
use serde_json::json;
use tokio::process::Command;
use tokio::time::timeout;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("git.txt");
const DEFAULT_TIMEOUT_SECS: u64 = 60;
const MAX_STREAM_BYTES: usize = 50 * 1024;

/// Runs structured git operations.
pub struct GitTool;

#[async_trait]
impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "git",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "add", "commit", "branch", "checkout"]
                    },
                    "path": {"type": "string"},
                    "cached": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1},
                    "message": {"type": "string"},
                    "name": {"type": "string"},
                    "delete": {"type": "boolean"},
                    "working_dir": {"type": "string"}
                },
                "required": ["operation"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "git",
            tier: ToolTier::Exec,
            approval_hint: ApprovalHint::Maybe,
            side_effects: SideEffects::Process,
        }
    }

    fn describe_action(&self, args: &serde_json::Value) -> String {
        let op = args
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing operation>");
        let detail = match op {
            "diff" => args
                .get("path")
                .and_then(|v| v.as_str())
                .map(|p| format!(" {p}"))
                .unwrap_or_default(),
            "add" | "checkout" => args
                .get(if op == "add" { "path" } else { "name" })
                .and_then(|v| v.as_str())
                .map(|s| format!(" {s}"))
                .unwrap_or_default(),
            "commit" => String::from(" -m …"),
            "branch" => {
                let delete = args
                    .get("delete")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                match args.get("name").and_then(|v| v.as_str()) {
                    Some(n) if delete => format!(" -D {n}"),
                    Some(n) => format!(" {n}"),
                    None => String::new(),
                }
            }
            _ => String::new(),
        };
        format!("git {op}{detail}")
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let op = args
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                name: "git".into(),
                reason: "missing 'operation' field".into(),
            })?;
        let argv = build_argv(op, &args)?;
        let working_dir = resolve_working_dir(&args, ctx.working_dir);

        let mut cmd = Command::new("git");
        cmd.args(&argv)
            .current_dir(&working_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let child = cmd
            .spawn()
            .map_err(|e| ToolError::Execution(format!("failed to spawn git: {e}")))?;
        let output_fut = child.wait_with_output();
        let output = match timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS), output_fut).await {
            Ok(r) => r.map_err(|e| ToolError::Execution(e.to_string()))?,
            Err(_) => {
                return Err(ToolError::Timeout {
                    timeout_secs: DEFAULT_TIMEOUT_SECS,
                });
            }
        };

        let (stdout, stdout_truncated) = cap_stream(&output.stdout);
        let (stderr, stderr_truncated) = cap_stream(&output.stderr);
        let exit_code = output.status.code();

        let mut content = format!(
            "<git>{}</git>\n<cwd>{}</cwd>\n<exit_code>{}</exit_code>\n<stdout>\n{stdout}",
            argv.join(" "),
            working_dir.display(),
            exit_code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string()),
        );
        if stdout_truncated {
            content.push_str(&format!(
                "\n... (truncated at {} KB)",
                MAX_STREAM_BYTES / 1024
            ));
        }
        content.push_str("\n</stdout>\n<stderr>\n");
        content.push_str(&stderr);
        if stderr_truncated {
            content.push_str(&format!(
                "\n... (truncated at {} KB)",
                MAX_STREAM_BYTES / 1024
            ));
        }
        content.push_str("\n</stderr>");

        let is_error = !output.status.success();
        let result = if is_error {
            ToolResult::soft_error(content)
        } else {
            ToolResult::ok(content)
        };
        Ok(result.with_structured(json!({
            "operation": op,
            "exit_code": exit_code,
            "argv": argv,
        })))
    }
}

fn build_argv(op: &str, args: &serde_json::Value) -> Result<Vec<String>, ToolError> {
    let invalid = |msg: &str| ToolError::InvalidArgs {
        name: "git".into(),
        reason: msg.into(),
    };
    let required_str = |key: &str| -> Result<String, ToolError> {
        args.get(key)
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .map(String::from)
            .ok_or_else(|| invalid(&format!("'{op}' requires non-empty '{key}'")))
    };
    let argv: Vec<String> = match op {
        "status" => vec!["status".into(), "--short".into()],
        "diff" => {
            let mut v = vec!["diff".into()];
            if args
                .get("cached")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                v.push("--cached".into());
            }
            if let Some(p) = args.get("path").and_then(|v| v.as_str()) {
                v.push("--".into());
                v.push(p.to_string());
            }
            v
        }
        "log" => {
            let limit = args
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(10)
                .max(1);
            vec!["log".into(), "--oneline".into(), format!("-n{limit}")]
        }
        "add" => {
            let path = required_str("path")?;
            vec!["add".into(), "--".into(), path]
        }
        "commit" => {
            let message = required_str("message")?;
            vec!["commit".into(), "-m".into(), message]
        }
        "branch" => {
            let delete = args
                .get("delete")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            match args.get("name").and_then(|v| v.as_str()) {
                Some(name) if delete => vec!["branch".into(), "-D".into(), name.to_string()],
                Some(name) => vec!["branch".into(), name.to_string()],
                None => vec!["branch".into(), "--list".into()],
            }
        }
        "checkout" => {
            let name = required_str("name")?;
            vec!["checkout".into(), name]
        }
        other => {
            return Err(invalid(&format!("unknown operation: {other}")));
        }
    };
    Ok(argv)
}

fn resolve_working_dir(args: &serde_json::Value, working_dir: &Path) -> PathBuf {
    match args.get("working_dir").and_then(|v| v.as_str()) {
        Some(s) => {
            let p = Path::new(s);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                working_dir.join(p)
            }
        }
        None => working_dir.to_path_buf(),
    }
}

fn cap_stream(bytes: &[u8]) -> (String, bool) {
    if bytes.len() <= MAX_STREAM_BYTES {
        (String::from_utf8_lossy(bytes).into_owned(), false)
    } else {
        (
            String::from_utf8_lossy(&bytes[..MAX_STREAM_BYTES]).into_owned(),
            true,
        )
    }
}

/// Classifies a git operation as read-only (safe to auto-allow) or write.
///
/// Consumed by [`crate::approval::resolve_maybe`] when the tool name is `git`.
pub(crate) fn is_read_op(args: &serde_json::Value) -> bool {
    matches!(
        args.get("operation").and_then(|v| v.as_str()),
        Some("status") | Some("diff") | Some("log")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use tokio::process::Command as TokioCommand;
    use tokio::sync::mpsc;

    async fn init_repo(dir: &Path) {
        let run = |args: &[&str]| {
            let mut cmd = TokioCommand::new("git");
            cmd.args(args).current_dir(dir);
            cmd
        };
        run(&["init", "-q", "-b", "main"])
            .spawn()
            .unwrap()
            .wait()
            .await
            .unwrap();
        run(&["config", "user.email", "t@t.local"])
            .spawn()
            .unwrap()
            .wait()
            .await
            .unwrap();
        run(&["config", "user.name", "tester"])
            .spawn()
            .unwrap()
            .wait()
            .await
            .unwrap();
        run(&["commit", "--allow-empty", "-m", "init", "-q"])
            .spawn()
            .unwrap()
            .wait()
            .await
            .unwrap();
    }

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
    async fn status_in_clean_repo_succeeds() {
        let dir = TempDir::new().unwrap();
        init_repo(dir.path()).await;
        let tool = GitTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"operation": "status"}), &c)
            .await
            .unwrap();
        assert!(!out.is_error, "stdout was: {}", out.content);
        assert!(out.content.contains("<git>status --short</git>"));
    }

    #[tokio::test]
    async fn log_reports_init_commit() {
        let dir = TempDir::new().unwrap();
        init_repo(dir.path()).await;
        let tool = GitTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"operation": "log", "limit": 1}), &c)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("init"));
    }

    #[tokio::test]
    async fn add_and_commit_flow() {
        let dir = TempDir::new().unwrap();
        init_repo(dir.path()).await;
        fs::write(dir.path().join("a.txt"), "hello\n").unwrap();
        let tool = GitTool;
        let (c1, _rx1) = ctx(dir.path());
        let add = tool
            .execute(json!({"operation": "add", "path": "a.txt"}), &c1)
            .await
            .unwrap();
        assert!(!add.is_error, "add stdout: {}", add.content);

        let (c2, _rx2) = ctx(dir.path());
        let commit = tool
            .execute(json!({"operation": "commit", "message": "add a"}), &c2)
            .await
            .unwrap();
        assert!(!commit.is_error, "commit output: {}", commit.content);
    }

    #[tokio::test]
    async fn add_requires_path() {
        let dir = TempDir::new().unwrap();
        init_repo(dir.path()).await;
        let tool = GitTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(json!({"operation": "add"}), &c)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn unknown_operation_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = GitTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(json!({"operation": "rm"}), &c)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[test]
    fn is_read_op_classifies_correctly() {
        assert!(is_read_op(&json!({"operation": "status"})));
        assert!(is_read_op(&json!({"operation": "diff"})));
        assert!(is_read_op(&json!({"operation": "log"})));
        assert!(!is_read_op(&json!({"operation": "add"})));
        assert!(!is_read_op(&json!({"operation": "commit"})));
        assert!(!is_read_op(&json!({"operation": "branch"})));
        assert!(!is_read_op(&json!({"operation": "checkout"})));
        assert!(!is_read_op(&json!({})));
    }

    #[test]
    fn spec_declares_maybe_approval() {
        let spec = GitTool.spec();
        assert_eq!(spec.name, "git");
        assert_eq!(spec.tier, ToolTier::Exec);
        assert_eq!(spec.approval_hint, ApprovalHint::Maybe);
    }

    #[test]
    fn describe_action_formats_per_operation() {
        let tool = GitTool;
        assert_eq!(
            tool.describe_action(&json!({"operation": "status"})),
            "git status"
        );
        assert_eq!(
            tool.describe_action(&json!({"operation": "diff", "path": "src/lib.rs"})),
            "git diff src/lib.rs"
        );
        assert_eq!(
            tool.describe_action(&json!({"operation": "branch", "name": "feat/x", "delete": true})),
            "git branch -D feat/x"
        );
        assert_eq!(
            tool.describe_action(&json!({"operation": "commit", "message": "hi"})),
            "git commit -m …"
        );
    }
}
