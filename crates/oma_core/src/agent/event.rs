//! Events flowing between the agent and its consumer (CLI, TUI, tests).
//!
//! Two channels:
//!
//! - **agent → consumer**: [`AgentEvent`]s describe what the turn is
//!   doing — text deltas, reasoning, tool calls, approval prompts, final
//!   summary. They are emitted on a `tokio::sync::mpsc::Sender`.
//! - **consumer → agent**: [`UserAction`]s carry decisions from the
//!   consumer back into the loop. Today only approval responses; cancel
//!   semantics are achieved by dropping the receiver.

use oma_protocol::{ApprovalDecisionValue, ApprovalScopeValue};

use super::turn::TurnSummary;

/// Everything the agent can tell a consumer during a turn.
///
/// Errors intentionally have no variant here — they are returned as the
/// `Result` of `Agent::run_turn` so the consumer can't confuse an in-band
/// error event with a real hard failure.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// The turn has been accepted and generation is starting.
    TurnStart,

    /// Plain assistant text, streamed.
    TextDelta(String),

    /// Reasoning / thinking content for models that emit it (Qwen3,
    /// DeepSeek-R1 distills). Consumers typically render these muted or
    /// hidden by default.
    ReasoningDelta(String),

    /// A tool call started streaming. The `describe_action` string is the
    /// tool's human-readable preview, ready for display in an approval
    /// modal or a progress line.
    ToolCallStart {
        id: String,
        name: String,
        describe_action: String,
    },

    /// Incremental tool-call argument chunk (JSON fragment). Consumers
    /// generally don't need to render this unless they want a live view
    /// of what the model is asking for.
    ToolCallArgs(String),

    /// The agent needs a user decision before executing this tool call.
    /// Blocks on the consumer sending back a [`UserAction::ApprovalResponse`]
    /// with the matching `request_id`.
    ApprovalRequired(ApprovalRequest),

    /// A tool call finished — either successfully, as a soft error, or as
    /// a denial. The content is the string that will be fed back to the
    /// model as a `Message::tool`, so consumers can preview what the model
    /// will see next.
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },

    /// The turn reached its natural end (or was cancelled / capped).
    /// Always the last event on the channel for a given turn.
    TurnComplete(TurnSummary),
}

/// Payload the agent emits when it needs a decision for a tool call.
///
/// The consumer displays `describe_action` (and optionally `args_preview`
/// for power users), collects a decision + scope from the user, and sends
/// back a matching [`UserAction::ApprovalResponse`].
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// Stable id so the response can be matched to this request — matters
    /// when multiple approvals can be in flight conceptually, even though
    /// today we serialise them within a turn.
    pub request_id: String,
    /// The tool's stable name (`"read"`, `"git"`, …).
    pub tool_name: String,
    /// Human-readable one-liner of what the tool is about to do, produced
    /// by the tool's `describe_action`.
    pub describe_action: String,
    /// Pretty-printed JSON arguments the tool is about to receive. Useful
    /// when the user wants to audit before approving.
    pub args_preview: String,
}

/// Messages the consumer sends back to the agent.
///
/// Cancel semantics are achieved by dropping the events receiver — the
/// agent's `send` hits `ChannelClosed` and unwinds cleanly. No explicit
/// `Cancel` variant is needed for MVP.
#[derive(Debug, Clone)]
pub enum UserAction {
    /// Response to an [`AgentEvent::ApprovalRequired`] with the matching
    /// `request_id`.
    ApprovalResponse {
        request_id: String,
        decision: ApprovalDecisionValue,
        scope: ApprovalScopeValue,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approval_request_is_clone() {
        let req = ApprovalRequest {
            request_id: "a".into(),
            tool_name: "read".into(),
            describe_action: "Read /tmp/x".into(),
            args_preview: "{\"path\": \"/tmp/x\"}".into(),
        };
        let copy = req.clone();
        assert_eq!(req.request_id, copy.request_id);
    }

    #[test]
    fn user_action_approval_roundtrips() {
        let a = UserAction::ApprovalResponse {
            request_id: "r1".into(),
            decision: ApprovalDecisionValue::Approve,
            scope: ApprovalScopeValue::Once,
        };
        // Clone ensures Debug / Clone derive is sound on the nested enums.
        let _ = a.clone();
    }
}
