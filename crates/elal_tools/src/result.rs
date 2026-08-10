//! Output and error types returned by [`Tool::execute`](crate::Tool::execute).

use serde::{Deserialize, Serialize};

/// Payload returned by a successful tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Primary text content shown back to the model.
    pub content: String,
    /// Whether the tool considers this run a soft failure. The model sees the
    /// content regardless; this flag lets the runtime tag it as an error
    /// result without promoting it to a hard [`ToolError`].
    pub is_error: bool,
    /// Optional structured data for runtime consumers that want more than
    /// the text payload (e.g. UI layers rendering diffs or tables).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured: Option<serde_json::Value>,
}

impl ToolResult {
    /// Builds a successful text-only result.
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            structured: None,
        }
    }

    /// Builds a soft-failure text result (still returned to the model).
    pub fn soft_error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            structured: None,
        }
    }

    /// Attaches a structured JSON payload to the result.
    pub fn with_structured(mut self, value: serde_json::Value) -> Self {
        self.structured = Some(value);
        self
    }
}

/// Hard errors produced by the tool subsystem itself.
///
/// Tools return this when the invocation could not complete. Soft failures
/// the model should still see as content belong in [`ToolResult::soft_error`].
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The registry had no tool registered under this name.
    #[error("tool '{name}' not found")]
    NotFound {
        /// The name that was looked up.
        name: String,
    },
    /// Input JSON failed schema or semantic validation.
    #[error("invalid args for '{name}': {reason}")]
    InvalidArgs {
        /// The tool that rejected the arguments.
        name: String,
        /// Human-readable reason shown in logs and UI.
        reason: String,
    },
    /// The user declined the approval prompt.
    #[error("approval denied by user")]
    ApprovalDenied,
    /// The tool ran but failed with a runtime error.
    #[error("execution failed: {0}")]
    Execution(String),
    /// The tool exceeded its configured timeout.
    #[error("timeout after {timeout_secs}s")]
    Timeout {
        /// How many seconds elapsed before cancellation.
        timeout_secs: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_result_has_no_error_flag() {
        let r = ToolResult::ok("done");
        assert_eq!(r.content, "done");
        assert!(!r.is_error);
        assert!(r.structured.is_none());
    }

    #[test]
    fn soft_error_sets_flag() {
        let r = ToolResult::soft_error("file missing");
        assert!(r.is_error);
    }

    #[test]
    fn with_structured_attaches_payload() {
        let r = ToolResult::ok("x").with_structured(serde_json::json!({"k": 1}));
        assert_eq!(r.structured.unwrap()["k"], 1);
    }

    #[test]
    fn serde_skips_structured_when_none() {
        let json = serde_json::to_string(&ToolResult::ok("x")).unwrap();
        assert!(!json.contains("structured"));
    }

    #[test]
    fn tool_error_display_matches_variant() {
        let err = ToolError::NotFound {
            name: "grep".into(),
        };
        assert_eq!(err.to_string(), "tool 'grep' not found");

        let err = ToolError::Timeout { timeout_secs: 30 };
        assert_eq!(err.to_string(), "timeout after 30s");

        let err = ToolError::InvalidArgs {
            name: "read".into(),
            reason: "missing path".into(),
        };
        assert_eq!(err.to_string(), "invalid args for 'read': missing path");
    }
}
