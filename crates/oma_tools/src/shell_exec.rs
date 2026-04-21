//! `shell_exec` built-in — runs a command through `sh -c` with timeout and
//! bounded output capture.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use oma_protocol::ToolDefinition;
use serde_json::json;
use tokio::process::Command;
use tokio::time::timeout;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("shell_exec.txt");
const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_STREAM_BYTES: usize = 50 * 1024;

/// Runs shell commands through `sh -c` with a configurable timeout.
pub struct ShellExecTool;

#[async_trait]
impl Tool for ShellExecTool {
    fn name(&self) -> &str {
        "shell_exec"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "shell_exec",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Full shell command string interpreted by `sh -c`."
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Optional working directory. Relative paths resolve against the current working directory."
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Timeout in seconds (default 120). The command is killed if it exceeds this."
                    }
                },
                "required": ["command"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "shell_exec",
            tier: ToolTier::Exec,
            approval_hint: ApprovalHint::Always,
            side_effects: SideEffects::Process,
        }
    }

    fn describe_action(&self, args: &serde_json::Value) -> String {
        let cmd = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing command>");
        let preview = if cmd.len() > 80 {
            format!("{}…", &cmd[..80])
        } else {
            cmd.to_string()
        };
        match args.get("working_dir").and_then(|v| v.as_str()) {
            Some(wd) => format!("Run `{preview}` (cwd: {wd})"),
            None => format!("Run `{preview}`"),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                name: "shell_exec".into(),
                reason: "missing 'command' field".into(),
            })?;
        if command.trim().is_empty() {
            return Err(ToolError::InvalidArgs {
                name: "shell_exec".into(),
                reason: "'command' must not be empty".into(),
            });
        }

        let timeout_secs = parse_positive_u64(&args, "timeout")?.unwrap_or(DEFAULT_TIMEOUT_SECS);
        let working_dir = resolve_working_dir(&args, ctx.working_dir);

        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(command)
            .current_dir(&working_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let child = cmd
            .spawn()
            .map_err(|e| ToolError::Execution(format!("failed to spawn: {e}")))?;

        let output_fut = child.wait_with_output();
        let output = match timeout(Duration::from_secs(timeout_secs), output_fut).await {
            Ok(r) => r.map_err(|e| ToolError::Execution(e.to_string()))?,
            Err(_) => return Err(ToolError::Timeout { timeout_secs }),
        };

        let (stdout, stdout_truncated) = cap_stream(&output.stdout);
        let (stderr, stderr_truncated) = cap_stream(&output.stderr);
        let exit_code = output.status.code();

        let mut content = format!(
            "<command>{}</command>\n<cwd>{}</cwd>\n<exit_code>{}</exit_code>\n",
            command,
            working_dir.display(),
            exit_code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string()),
        );
        content.push_str(&format!("<stdout>\n{stdout}"));
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
            "exit_code": exit_code,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        })))
    }
}

fn parse_positive_u64(args: &serde_json::Value, key: &str) -> Result<Option<u64>, ToolError> {
    let Some(v) = args.get(key) else {
        return Ok(None);
    };
    if v.is_null() {
        return Ok(None);
    }
    let n = v.as_u64().ok_or_else(|| ToolError::InvalidArgs {
        name: "shell_exec".into(),
        reason: format!("'{key}' must be a positive integer"),
    })?;
    if n == 0 {
        return Err(ToolError::InvalidArgs {
            name: "shell_exec".into(),
            reason: format!("'{key}' must be >= 1"),
        });
    }
    Ok(Some(n))
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

#[cfg(test)]
mod tests {
    use super::*;
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
    async fn captures_stdout_of_successful_command() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"command": "printf hello"}), &c)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("hello"));
        assert!(out.content.contains("<exit_code>0</exit_code>"));
    }

    #[tokio::test]
    async fn captures_stderr_separately() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"command": "printf err 1>&2"}), &c)
            .await
            .unwrap();
        assert!(out.content.contains("<stderr>\nerr"));
    }

    #[tokio::test]
    async fn nonzero_exit_is_soft_error() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"command": "exit 3"}), &c)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("<exit_code>3</exit_code>"));
    }

    #[tokio::test]
    async fn timeout_kills_the_command() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(json!({"command": "sleep 10", "timeout": 1}), &c)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout { timeout_secs: 1 }));
    }

    #[tokio::test]
    async fn working_dir_is_applied() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let (c, _rx) = ctx(Path::new("/")); // absolute root, irrelevant
        let out = tool
            .execute(
                json!({
                    "command": "pwd",
                    "working_dir": dir.path().to_string_lossy(),
                }),
                &c,
            )
            .await
            .unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        assert!(
            out.content
                .contains(&canonical.to_string_lossy().to_string()),
            "expected pwd output to contain {}",
            canonical.display()
        );
    }

    #[tokio::test]
    async fn truncates_stdout_past_cap() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let (c, _rx) = ctx(dir.path());
        // Produce ~100 KB of 'a'.
        let out = tool
            .execute(json!({"command": "yes a | head -c 102400"}), &c)
            .await
            .unwrap();
        assert!(out.content.contains("truncated at 50 KB"));
    }

    #[tokio::test]
    async fn missing_command_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn empty_command_is_invalid() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(json!({"command": "   "}), &c)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[test]
    fn describe_action_trims_long_commands() {
        let tool = ShellExecTool;
        let long = "a".repeat(200);
        let preview = tool.describe_action(&json!({"command": long}));
        assert!(preview.contains('…'));
        assert!(preview.len() < 200);
    }

    #[test]
    fn spec_declares_always_approval_and_exec_tier() {
        let spec = ShellExecTool.spec();
        assert_eq!(spec.tier, ToolTier::Exec);
        assert_eq!(spec.approval_hint, ApprovalHint::Always);
        assert_eq!(spec.side_effects, SideEffects::Process);
    }
}
