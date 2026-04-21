//! Agent loop — orchestrates user ↔ LLM ↔ tools within a single turn.
//!
//! This is the "brain" layer. It consumes [`StreamEvent`](oma_protocol::StreamEvent)s
//! from an [`oma_provider::Provider`], routes them to [`AgentEvent`]s on a
//! consumer-facing channel, assembles tool calls, dispatches them through
//! [`oma_tools::ToolRegistry`] with approval mediated by
//! [`oma_tools::approval::resolve`], feeds results back as
//! `Message::tool`, and loops until the model produces plain text with no
//! pending tool calls.
//!
//! Lives in `oma_core` (not a standalone crate) because it is the
//! orchestrator of existing pieces — Provider, ToolRegistry, config,
//! context — and adding a new crate would force downstream consumers to
//! import yet another path without any encapsulation benefit.
//!
//! Types and error surfaces land here; the actual loop implementation
//! lives in sibling modules and is wired in chunk 2 and beyond.

mod event;
mod turn;

pub use event::{AgentEvent, ApprovalRequest, UserAction};
pub use turn::{TURN_STEP_CAP, TurnError, TurnSummary};
