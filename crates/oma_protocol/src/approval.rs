//! Wire-format vocabulary for the approval handshake.
//!
//! These are pure data enums the TUI, agent loop, and protocol layers share
//! to talk about tool approvals. The actual resolver lives in `oma_tools`
//! because it couples with the [`Tool`](crate::ToolDefinition) trait; the
//! types below just define the contract.
//!
//! Mirrors the shape in `claw-code-rust/crates/protocol/src/approval.rs`.

use serde::{Deserialize, Serialize};

/// Client response to a pending approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecisionValue {
    /// User approved the invocation.
    Approve,
    /// User denied the invocation.
    Deny,
    /// User cancelled the prompt without deciding.
    Cancel,
}

/// Scope over which the user's decision applies.
///
/// Enables UX flows such as "approve for the whole session" or "always
/// approve writes under this path prefix" without hardcoding per-tool logic
/// in the resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalScopeValue {
    /// The decision applies only to this single invocation.
    Once,
    /// The decision persists for the current turn.
    Turn,
    /// The decision persists for the current session.
    Session,
    /// The decision persists for all paths sharing a prefix.
    PathPrefix,
    /// The decision persists for all requests to a given host.
    Host,
    /// The decision persists for the entire tool across the session.
    Tool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_serializes_as_snake_case() {
        assert_eq!(
            serde_json::to_string(&ApprovalDecisionValue::Approve).unwrap(),
            "\"approve\""
        );
        let d: ApprovalDecisionValue = serde_json::from_str("\"deny\"").unwrap();
        assert_eq!(d, ApprovalDecisionValue::Deny);
    }

    #[test]
    fn scope_serializes_as_snake_case() {
        assert_eq!(
            serde_json::to_string(&ApprovalScopeValue::PathPrefix).unwrap(),
            "\"path_prefix\""
        );
        let s: ApprovalScopeValue = serde_json::from_str("\"session\"").unwrap();
        assert_eq!(s, ApprovalScopeValue::Session);
    }

    #[test]
    fn all_decision_variants_roundtrip() {
        for d in [
            ApprovalDecisionValue::Approve,
            ApprovalDecisionValue::Deny,
            ApprovalDecisionValue::Cancel,
        ] {
            let json = serde_json::to_string(&d).unwrap();
            let back: ApprovalDecisionValue = serde_json::from_str(&json).unwrap();
            assert_eq!(d, back);
        }
    }

    #[test]
    fn all_scope_variants_roundtrip() {
        for s in [
            ApprovalScopeValue::Once,
            ApprovalScopeValue::Turn,
            ApprovalScopeValue::Session,
            ApprovalScopeValue::PathPrefix,
            ApprovalScopeValue::Host,
            ApprovalScopeValue::Tool,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: ApprovalScopeValue = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }
}
