//! `elal_core` — runtime state and coordination.
//!
//! Houses configuration loading, agent execution context, the agent loop,
//! and the top-level error type. Wire-format types live in `elal_protocol`.

pub mod agent;
pub mod config;
pub mod context;
pub mod error;
pub mod session;

pub use agent::{
    Agent, AgentEvent, ApprovalRequest, TURN_STEP_CAP, TurnError, TurnSummary, UserAction,
};
pub use config::{
    AgentOverrides, ContextConfig, GlobalConfig, ProjectConfig, ToolsConfig, load_effective,
};
// Re-export from elal_protocol for backward compat — ApprovalMode moved to the
// protocol crate in Phase 3 prep so it can be shared between elal_tools and
// elal_core without a circular dependency.
pub use context::{AgentContext, GitInfo, OsInfo};
pub use elal_protocol::{ApprovalMode, SessionId};
pub use error::{ElalError, Result};
pub use session::{
    ListedSession, LoadedSession, RolloutStore, SessionConfig, SessionState, TokenBudget,
    find_latest, list_sessions, load_session, locate,
};
