//! Approval resolver — decides whether a tool invocation should prompt the
//! user, given the active [`ApprovalMode`] and the tool's [`ApprovalHint`].
//!
//! The resolver is intentionally tiny and data-driven: no closures, no
//! function pointers, no per-tool handlers wired in at compile time. When
//! the hint is [`ApprovalHint::Maybe`], it dispatches by tool name to a
//! small table that grows as [`#5`] (built-ins) introduces concrete cases.
//! Unknown names always fall back to [`ApprovalDecision::Prompt`] — failing
//! safe is the only acceptable default for a yet-unknown side-effect.
//!
//! Mirrors the placement-by-responsibility of `claw-code-rust`'s resolver:
//! they split it into `server::approval`, we keep it in `oma_tools` because
//! oh-my-agent runs single-process.
//!
//! [`#5`]: https://github.com/lobinuxsoft/oh-my-agent/issues/5

use oma_core::ApprovalMode;

use crate::spec::ApprovalHint;

/// Outcome produced by [`resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// The tool may execute immediately without prompting the user.
    Allow,
    /// The user must approve (via TUI or agent channel) before execution.
    Prompt,
}

/// Resolves the full `ApprovalMode × ApprovalHint` matrix.
///
/// For `ApprovalHint::Maybe`, the decision is delegated to
/// [`resolve_maybe`] which pattern-matches on `tool_name` and inspects
/// `args`. Tools not yet wired there fall back to `Prompt` (fail-safe).
pub fn resolve(
    mode: ApprovalMode,
    hint: ApprovalHint,
    tool_name: &str,
    args: &serde_json::Value,
) -> ApprovalDecision {
    match mode {
        // Force-always overrides any per-tool hint.
        ApprovalMode::Always => ApprovalDecision::Prompt,
        // YOLO: never prompt regardless of hint.
        ApprovalMode::Never => ApprovalDecision::Allow,
        // Smart: honor per-tool hint, delegating `Maybe` to per-tool matcher.
        ApprovalMode::Smart => match hint {
            ApprovalHint::Never => ApprovalDecision::Allow,
            ApprovalHint::Always => ApprovalDecision::Prompt,
            ApprovalHint::Maybe => resolve_maybe(tool_name, args),
        },
    }
}

/// Per-tool matcher for [`ApprovalHint::Maybe`].
///
/// Unknown tools fail safe to `Prompt`. Known tools delegate to a per-tool
/// classifier: for `git`, read operations (`status`, `diff`, `log`) are
/// auto-allowed, write operations are prompted.
fn resolve_maybe(tool_name: &str, args: &serde_json::Value) -> ApprovalDecision {
    match tool_name {
        "git" if crate::git::is_read_op(args) => ApprovalDecision::Allow,
        _ => ApprovalDecision::Prompt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> serde_json::Value {
        serde_json::json!({})
    }

    #[test]
    fn always_mode_always_prompts() {
        for hint in [
            ApprovalHint::Never,
            ApprovalHint::Maybe,
            ApprovalHint::Always,
        ] {
            assert_eq!(
                resolve(ApprovalMode::Always, hint, "any", &args()),
                ApprovalDecision::Prompt,
                "Always mode must prompt for hint {hint:?}"
            );
        }
    }

    #[test]
    fn never_mode_never_prompts() {
        for hint in [
            ApprovalHint::Never,
            ApprovalHint::Maybe,
            ApprovalHint::Always,
        ] {
            assert_eq!(
                resolve(ApprovalMode::Never, hint, "any", &args()),
                ApprovalDecision::Allow,
                "Never mode must allow for hint {hint:?}"
            );
        }
    }

    #[test]
    fn smart_mode_honors_never_hint() {
        assert_eq!(
            resolve(
                ApprovalMode::Smart,
                ApprovalHint::Never,
                "read_file",
                &args()
            ),
            ApprovalDecision::Allow,
        );
    }

    #[test]
    fn smart_mode_honors_always_hint() {
        assert_eq!(
            resolve(
                ApprovalMode::Smart,
                ApprovalHint::Always,
                "apply_patch",
                &args(),
            ),
            ApprovalDecision::Prompt,
        );
    }

    #[test]
    fn smart_mode_maybe_defaults_to_prompt_for_unknown_tool() {
        assert_eq!(
            resolve(
                ApprovalMode::Smart,
                ApprovalHint::Maybe,
                "some_unregistered_tool",
                &args(),
            ),
            ApprovalDecision::Prompt,
            "unknown Maybe tool must fail safe to Prompt",
        );
    }

    #[test]
    fn smart_mode_maybe_allows_git_read_operations() {
        for op in ["status", "diff", "log"] {
            assert_eq!(
                resolve(
                    ApprovalMode::Smart,
                    ApprovalHint::Maybe,
                    "git",
                    &serde_json::json!({ "operation": op }),
                ),
                ApprovalDecision::Allow,
                "git {op} should auto-allow under Smart mode"
            );
        }
    }

    #[test]
    fn smart_mode_maybe_prompts_git_write_operations() {
        for op in ["add", "commit", "branch", "checkout"] {
            assert_eq!(
                resolve(
                    ApprovalMode::Smart,
                    ApprovalHint::Maybe,
                    "git",
                    &serde_json::json!({ "operation": op }),
                ),
                ApprovalDecision::Prompt,
                "git {op} should prompt under Smart mode"
            );
        }
    }

    #[test]
    fn always_mode_overrides_git_read_auto_allow() {
        assert_eq!(
            resolve(
                ApprovalMode::Always,
                ApprovalHint::Maybe,
                "git",
                &serde_json::json!({ "operation": "status" }),
            ),
            ApprovalDecision::Prompt,
            "Always mode should force Prompt even for git read ops",
        );
    }

    #[test]
    fn full_matrix_is_exhaustive() {
        // Sanity: explicitly cover every combination once to protect against
        // accidental regressions in the match arms.
        let cases = [
            (
                ApprovalMode::Always,
                ApprovalHint::Never,
                ApprovalDecision::Prompt,
            ),
            (
                ApprovalMode::Always,
                ApprovalHint::Maybe,
                ApprovalDecision::Prompt,
            ),
            (
                ApprovalMode::Always,
                ApprovalHint::Always,
                ApprovalDecision::Prompt,
            ),
            (
                ApprovalMode::Smart,
                ApprovalHint::Never,
                ApprovalDecision::Allow,
            ),
            (
                ApprovalMode::Smart,
                ApprovalHint::Maybe,
                ApprovalDecision::Prompt,
            ),
            (
                ApprovalMode::Smart,
                ApprovalHint::Always,
                ApprovalDecision::Prompt,
            ),
            (
                ApprovalMode::Never,
                ApprovalHint::Never,
                ApprovalDecision::Allow,
            ),
            (
                ApprovalMode::Never,
                ApprovalHint::Maybe,
                ApprovalDecision::Allow,
            ),
            (
                ApprovalMode::Never,
                ApprovalHint::Always,
                ApprovalDecision::Allow,
            ),
        ];
        for (mode, hint, expected) in cases {
            assert_eq!(
                resolve(mode, hint, "t", &args()),
                expected,
                "mode={mode:?} hint={hint:?} expected={expected:?}"
            );
        }
    }
}
