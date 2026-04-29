//! `oma_protocol` — wire-format types shared across the workspace.
//!
//! Pure data types only: no runtime state, no I/O, no async.
//! Everything here is serde-serializable and OpenAI-Chat-Completions-compatible
//! where applicable.

mod approval;
mod event;
mod identifiers;
mod message;
mod response;
mod role;
mod session;
mod tool_schema;

pub use approval::{ApprovalDecisionValue, ApprovalMode, ApprovalScopeValue};
pub use event::StreamEvent;
pub use identifiers::{ItemId, MessageId, SessionId, ToolCallId, TurnId};
pub use message::{FunctionCall, Message, ToolCall};
pub use response::{ResponseContent, StopReason, Usage};
pub use role::Role;
pub use session::{
    ItemLine, ItemRecord, ReasoningItem, RolloutLine, SCHEMA_VERSION, SessionMetaLine,
    SessionRecord, SessionTitleUpdatedLine, TextItem, ToolCallItem, ToolResultItem, TurnItem,
    TurnLine, TurnRecord, TurnStatus, TurnUsage,
};
pub use tool_schema::{FunctionDefinition, ToolDefinition};
