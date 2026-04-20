//! `oma_core` — runtime state and coordination.
//!
//! Houses configuration loading, agent execution context, and the top-level
//! error type. Wire-format types live in `oma_protocol`.

pub mod config;
pub mod context;
pub mod error;

pub use config::{
    AgentOverrides, ApprovalMode, ContextConfig, GlobalConfig, ProjectConfig, ToolsConfig,
    load_effective,
};
pub use context::{AgentContext, GitInfo, OsInfo};
pub use error::{OmaError, Result};
