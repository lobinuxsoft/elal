//! `read` built-in — returns file contents with 1-indexed line numbers.

use std::fs::File;
use std::io::{BufRead, BufReader, Read as _};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use oma_protocol::ToolDefinition;
use serde_json::json;

use crate::context::ToolContext;
use crate::result::{ToolError, ToolResult};
use crate::spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
use crate::trait_def::Tool;

const DESCRIPTION: &str = include_str!("read.txt");
const MAX_LINE_CHARS: usize = 2000;
const MAX_OUTPUT_BYTES: usize = 50 * 1024;
const DEFAULT_LIMIT: usize = 2000;

/// Reads UTF-8 files with offset/limit paging and binary-file rejection.
///
/// Directory listing is intentionally delegated to the `list_dir` tool so
/// each built-in stays small and focused.
pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read",
            DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read. Relative paths are resolved against the working directory."
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-indexed line number to start reading from (default 1)."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum number of lines to return (default 2000)."
                    }
                },
                "required": ["path"]
            }),
        )
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read",
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
        match (
            args.get("offset").and_then(|v| v.as_u64()),
            args.get("limit").and_then(|v| v.as_u64()),
        ) {
            (Some(o), Some(l)) => format!("Read {path} (lines {o}-{})", o + l - 1),
            (Some(o), None) => format!("Read {path} (from line {o})"),
            (None, Some(l)) => format!("Read {path} (first {l} lines)"),
            (None, None) => format!("Read {path}"),
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
                    name: "read".into(),
                    reason: "missing 'path' field".into(),
                })?;

        let offset = parse_positive(&args, "offset")?.unwrap_or(1);
        let limit = parse_positive(&args, "limit")?.unwrap_or(DEFAULT_LIMIT);

        let path = resolve_path(path_str, ctx.working_dir);
        if !path.exists() {
            return Ok(ToolResult::soft_error(format!(
                "File not found: {}",
                path.display()
            )));
        }
        if path.is_dir() {
            return Ok(ToolResult::soft_error(format!(
                "Path is a directory (use list_dir instead): {}",
                path.display()
            )));
        }
        if is_binary_file(&path).map_err(io_err)? {
            return Ok(ToolResult::soft_error(format!(
                "Cannot read binary file: {}",
                path.display()
            )));
        }

        read_file(&path, offset, limit).map_err(io_err)
    }
}

fn io_err(err: std::io::Error) -> ToolError {
    ToolError::Execution(err.to_string())
}

fn parse_positive(args: &serde_json::Value, key: &str) -> Result<Option<usize>, ToolError> {
    let Some(value) = args.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let n = value.as_u64().ok_or_else(|| ToolError::InvalidArgs {
        name: "read".into(),
        reason: format!("'{key}' must be a positive integer"),
    })?;
    if n == 0 {
        return Err(ToolError::InvalidArgs {
            name: "read".into(),
            reason: format!("'{key}' must be >= 1"),
        });
    }
    Ok(Some(n as usize))
}

fn resolve_path(path_str: &str, working_dir: &Path) -> PathBuf {
    let p = Path::new(path_str);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        working_dir.join(p)
    }
}

fn read_file(path: &Path, offset: usize, limit: usize) -> std::io::Result<ToolResult> {
    let reader = BufReader::new(File::open(path)?);
    let start = offset.saturating_sub(1);
    let mut emitted: Vec<String> = Vec::new();
    let mut total_lines = 0usize;
    let mut bytes = 0usize;
    let mut capped = false;
    let mut more = false;

    for line in reader.lines() {
        let mut line = line?;
        total_lines += 1;
        if total_lines <= start {
            continue;
        }
        if emitted.len() >= limit {
            more = true;
            continue;
        }
        if line.len() > MAX_LINE_CHARS {
            line.truncate(MAX_LINE_CHARS);
            line.push_str("... (line truncated)");
        }
        let line_bytes = line.len() + 1;
        if bytes + line_bytes > MAX_OUTPUT_BYTES {
            capped = true;
            more = true;
            break;
        }
        emitted.push(line);
        bytes += line_bytes;
    }

    if total_lines < offset && !(total_lines == 0 && offset == 1) {
        return Ok(ToolResult::soft_error(format!(
            "Offset {offset} is out of range for this file ({total_lines} lines)."
        )));
    }

    let mut content = format!(
        "<path>{}</path>\n<type>file</type>\n<content>\n",
        path.display()
    );
    for (i, line) in emitted.iter().enumerate() {
        content.push_str(&format!("{}: {}\n", offset + i, line));
    }
    let last_line = offset + emitted.len().saturating_sub(1);
    let next = last_line + 1;
    if capped {
        content.push_str(&format!(
            "\n(Output capped at {} KB. Showing lines {}-{}. Use offset={} to continue.)",
            MAX_OUTPUT_BYTES / 1024,
            offset,
            last_line,
            next,
        ));
    } else if more {
        content.push_str(&format!(
            "\n(Showing lines {}-{} of {}. Use offset={} to continue.)",
            offset, last_line, total_lines, next,
        ));
    } else {
        content.push_str(&format!("\n(End of file - total {} lines)", total_lines));
    }
    content.push_str("\n</content>");

    Ok(ToolResult::ok(content).with_structured(json!({
        "lines_returned": emitted.len(),
        "total_lines": total_lines,
        "truncated": capped || more,
    })))
}

fn is_binary_file(path: &Path) -> std::io::Result<bool> {
    const BINARY_EXTENSIONS: &[&str] = &[
        "zip", "tar", "gz", "xz", "bz2", "7z", "rar", "exe", "dll", "so", "dylib", "class", "jar",
        "war", "bin", "dat", "obj", "o", "a", "lib", "wasm", "pyc", "pyo", "png", "jpg", "jpeg",
        "gif", "bmp", "webp", "tiff", "ico", "pdf", "mp3", "mp4", "avi", "mov", "flac", "ogg",
    ];
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if BINARY_EXTENSIONS.contains(&ext.as_str()) {
        return Ok(true);
    }
    let mut file = File::open(path)?;
    let size = file.metadata()?.len() as usize;
    if size == 0 {
        return Ok(false);
    }
    let sample = size.min(4096);
    let mut buf = vec![0u8; sample];
    let read = file.read(&mut buf)?;
    if read == 0 {
        return Ok(false);
    }
    let mut non_printable = 0usize;
    for byte in &buf[..read] {
        if *byte == 0 {
            return Ok(true);
        }
        if *byte < 9 || (*byte > 13 && *byte < 32) {
            non_printable += 1;
        }
    }
    Ok((non_printable as f64) / (read as f64) > 0.3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
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

    fn write_lines(path: &Path, lines: &[&str]) {
        let mut f = fs::File::create(path).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
    }

    #[tokio::test]
    async fn reads_entire_file_by_default() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        write_lines(&path, &["foo", "bar", "baz"]);
        let tool = ReadTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"path": path.to_string_lossy()}), &c)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("1: foo"));
        assert!(out.content.contains("3: baz"));
        assert!(out.content.contains("End of file - total 3 lines"));
    }

    #[tokio::test]
    async fn honors_offset_and_limit() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        write_lines(&path, &["l1", "l2", "l3", "l4", "l5"]);
        let tool = ReadTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(
                json!({"path": path.to_string_lossy(), "offset": 2, "limit": 2}),
                &c,
            )
            .await
            .unwrap();
        assert!(out.content.contains("2: l2"));
        assert!(out.content.contains("3: l3"));
        assert!(!out.content.contains("4: l4"));
        assert!(out.content.contains("Use offset=4 to continue"));
    }

    #[tokio::test]
    async fn relative_path_resolves_against_working_dir() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rel.txt");
        write_lines(&path, &["hello"]);
        let tool = ReadTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool.execute(json!({"path": "rel.txt"}), &c).await.unwrap();
        assert!(out.content.contains("1: hello"));
    }

    #[tokio::test]
    async fn missing_file_returns_soft_error() {
        let dir = TempDir::new().unwrap();
        let tool = ReadTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool.execute(json!({"path": "nope.txt"}), &c).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("File not found"));
    }

    #[tokio::test]
    async fn directory_path_returns_soft_error_pointing_to_list_dir() {
        let dir = TempDir::new().unwrap();
        let tool = ReadTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"path": dir.path().to_string_lossy()}), &c)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("list_dir"));
    }

    #[tokio::test]
    async fn binary_file_is_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("blob.bin");
        fs::write(&path, [0u8, 1, 2, 3]).unwrap();
        let tool = ReadTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"path": path.to_string_lossy()}), &c)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("binary"));
    }

    #[tokio::test]
    async fn missing_path_is_invalid_args() {
        let dir = TempDir::new().unwrap();
        let tool = ReadTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool.execute(json!({}), &c).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn zero_offset_is_invalid_args() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        write_lines(&path, &["x"]);
        let tool = ReadTool;
        let (c, _rx) = ctx(dir.path());
        let err = tool
            .execute(json!({"path": path.to_string_lossy(), "offset": 0}), &c)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs { .. }));
    }

    #[tokio::test]
    async fn long_lines_are_truncated() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("long.txt");
        let long = "x".repeat(3000);
        fs::write(&path, format!("{long}\n")).unwrap();
        let tool = ReadTool;
        let (c, _rx) = ctx(dir.path());
        let out = tool
            .execute(json!({"path": path.to_string_lossy()}), &c)
            .await
            .unwrap();
        assert!(out.content.contains("line truncated"));
    }

    #[test]
    fn describe_action_is_human_readable() {
        let tool = ReadTool;
        assert_eq!(tool.describe_action(&json!({"path": "/x"})), "Read /x");
        assert_eq!(
            tool.describe_action(&json!({"path": "/x", "offset": 10})),
            "Read /x (from line 10)"
        );
        assert_eq!(
            tool.describe_action(&json!({"path": "/x", "limit": 20})),
            "Read /x (first 20 lines)"
        );
        assert_eq!(
            tool.describe_action(&json!({"path": "/x", "offset": 10, "limit": 5})),
            "Read /x (lines 10-14)"
        );
    }

    #[test]
    fn spec_declares_never_approval_and_read_tier() {
        let spec = ReadTool.spec();
        assert_eq!(spec.name, "read");
        assert_eq!(spec.tier, ToolTier::Read);
        assert_eq!(spec.approval_hint, ApprovalHint::Never);
        assert_eq!(spec.side_effects, SideEffects::Local);
    }
}
