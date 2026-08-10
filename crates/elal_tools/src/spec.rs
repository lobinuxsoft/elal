//! Static descriptors attached to every [`Tool`](crate::Tool) implementation.
//!
//! These types are plain serializable data. Runtime decisions (approval,
//! scheduling) are made by inspecting them without invoking any tool logic.

use serde::{Deserialize, Serialize};

/// Static descriptor for one tool.
///
/// Held by every [`Tool`](crate::Tool) and returned from `Tool::spec()`. Used
/// by the registry, approval resolver, and UI layers to reason about tools
/// without invoking them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    /// Stable lowercase name used by the LLM and the registry.
    pub name: &'static str,
    /// Coarse classification of what kind of work the tool performs.
    pub tier: ToolTier,
    /// How the approval layer should treat invocations of this tool.
    pub approval_hint: ApprovalHint,
    /// What kind of side effects the tool produces.
    pub side_effects: SideEffects,
}

/// Coarse classification of the tool's workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolTier {
    /// Pure reads (file reads, searches, web fetches).
    Read,
    /// Writes to local state (file writes, patches, edits).
    Write,
    /// Process or network execution beyond simple reads.
    Exec,
}

/// Plain serializable approval hint.
///
/// Mirrors `claw-code-rust`'s `ApprovalHint` pattern: the hint is pure data
/// and the "smart" decision for [`ApprovalHint::Maybe`] lives in the
/// [`approval`](crate::approval) resolver, not inside a function pointer on
/// the spec. This keeps [`ToolSpec`] trivially `Clone`/`PartialEq`/serde-ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalHint {
    /// Tool is read-only or fully safe — never prompt.
    Never,
    /// Decision depends on arguments plus runtime policy.
    Maybe,
    /// Side-effect critical — always prompt.
    Always,
}

/// Kind of side effects produced by a tool invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideEffects {
    /// Pure computation, no external observable effect.
    None,
    /// Touches the local filesystem.
    Local,
    /// Spawns a subprocess.
    Process,
    /// Performs network I/O.
    Network,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spec() -> ToolSpec {
        ToolSpec {
            name: "read_file",
            tier: ToolTier::Read,
            approval_hint: ApprovalHint::Never,
            side_effects: SideEffects::Local,
        }
    }

    #[test]
    fn spec_is_clone_and_eq() {
        let a = sample_spec();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn approval_hint_serializes_as_snake_case() {
        let json = serde_json::to_string(&ApprovalHint::Always).unwrap();
        assert_eq!(json, "\"always\"");
        let back: ApprovalHint = serde_json::from_str("\"maybe\"").unwrap();
        assert_eq!(back, ApprovalHint::Maybe);
    }

    #[test]
    fn tool_tier_serializes_as_snake_case() {
        let json = serde_json::to_string(&ToolTier::Exec).unwrap();
        assert_eq!(json, "\"exec\"");
    }

    #[test]
    fn side_effects_serializes_as_snake_case() {
        let json = serde_json::to_string(&SideEffects::Network).unwrap();
        assert_eq!(json, "\"network\"");
        let back: SideEffects = serde_json::from_str("\"none\"").unwrap();
        assert_eq!(back, SideEffects::None);
    }
}
