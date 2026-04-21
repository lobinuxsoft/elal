//! `oma_core` — runtime state and coordination.
//!
//! Houses configuration loading, agent execution context, the agent loop,
//! and the top-level error type. Wire-format types live in `oma_protocol`.

pub mod agent;
pub mod config;
pub mod context;
pub mod error;

pub use config::{
    AgentOverrides, ContextConfig, GlobalConfig, ProjectConfig, ToolsConfig, load_effective,
};
// Re-export from oma_protocol for backward compat — ApprovalMode moved to the
// protocol crate in Phase 3 prep so it can be shared between oma_tools and
// oma_core without a circular dependency.
pub use context::{AgentContext, GitInfo, OsInfo};
pub use error::{OmaError, Result};
pub use oma_protocol::ApprovalMode;
