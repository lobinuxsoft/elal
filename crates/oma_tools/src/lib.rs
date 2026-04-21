//! `oma_tools` — extensible tool subsystem for oh-my-agent.
//!
//! Exposes the [`Tool`] trait every capability implements, the
//! [`ToolRegistry`] that holds them, and the [`approval`] resolver that
//! decides whether an invocation should prompt the user.
//!
//! Architectural alignment with `claw-code-rust` (ADR-7): the approval
//! model uses a plain [`ApprovalHint`] enum plus an external resolver,
//! not function pointers embedded in the spec. See issue #4 for the full
//! rationale.

pub mod approval;
mod context;
mod registry;
mod result;
mod spec;
mod trait_def;

pub use approval::{ApprovalDecision, resolve};
pub use context::{ToolContext, ToolEvent};
pub use registry::ToolRegistry;
pub use result::{ToolError, ToolResult};
pub use spec::{ApprovalHint, SideEffects, ToolSpec, ToolTier};
pub use trait_def::Tool;
