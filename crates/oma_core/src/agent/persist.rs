//! Persistence wiring for [`super::Agent::run_turn`].
//!
//! Translates in-flight turn events (assistant text, reasoning, tool calls,
//! tool results, compactions) into canonical [`TurnItem`] variants and
//! delegates writes to the [`RolloutStore`]. Lives next to the agent loop
//! because the mapping from internal state to wire-format items is the
//! agent's responsibility, not the rollout store's.
//!
//! ## Lifecycle per turn
//!
//! 1. [`Persistence::begin_turn`] — assigns a fresh [`TurnId`], appends the
//!    `Turn(Running)` line, and (the first time only) the mandatory
//!    `SessionMeta` header.
//! 2. [`Persistence::append_item`] — one `Item` line per [`TurnItem`]. Each
//!    record carries a strictly increasing per-session `seq`.
//! 3. [`Persistence::end_turn`] — appends the terminal `Turn(<status>)` line
//!    with usage and atomically rewrites the meta sidecar.
//!
//! Resumed sessions construct via [`Persistence::for_resumed_session`] so
//! the header is treated as already written and the seq counters pick up
//! where replay left off.

use chrono::{DateTime, Utc};
use oma_protocol::{
    ItemId, ItemRecord, ReasoningItem, SCHEMA_VERSION, SessionId, SessionRecord, TextItem,
    ToolCallId, ToolCallItem, ToolResultItem, TurnId, TurnItem, TurnRecord, TurnStatus, TurnUsage,
    Usage,
};
use serde_json::Value;

use crate::error::Result;
use crate::session::RolloutStore;

/// Bookkeeping for an in-flight turn — opaque to the agent loop.
struct TurnHandle {
    id: TurnId,
    session_id: SessionId,
    started_at: DateTime<Utc>,
}

/// Persistence wiring for [`super::Agent`]. Holds a mutable [`SessionRecord`]
/// copy that mirrors disk state — `total_*_tokens`, `updated_at`, `title`,
/// and `first_user_message` mutate as the session progresses.
pub(super) struct Persistence {
    store: RolloutStore,
    record: SessionRecord,
    turn_seq: u32,
    item_seq: u64,
    /// `true` once the `SessionMeta` line has been emitted for this rollout.
    meta_written: bool,
    current_turn: Option<TurnHandle>,
}

impl Persistence {
    /// New session — meta line will be written on the first `begin_turn`.
    pub(super) fn for_new_session(store: RolloutStore, record: SessionRecord) -> Self {
        Self {
            store,
            record,
            turn_seq: 0,
            item_seq: 0,
            meta_written: false,
            current_turn: None,
        }
    }

    /// Resumed session — caller supplies the highest-seen sequence counters
    /// so subsequent appends do not collide with the prior journal.
    pub(super) fn for_resumed_session(
        store: RolloutStore,
        record: SessionRecord,
        last_turn_seq: u32,
        last_item_seq: u64,
    ) -> Self {
        Self {
            store,
            record,
            turn_seq: last_turn_seq,
            item_seq: last_item_seq,
            meta_written: true,
            current_turn: None,
        }
    }

    #[allow(dead_code, reason = "exposed for chunk 6 telemetry")]
    pub(super) fn record(&self) -> &SessionRecord {
        &self.record
    }

    /// Captures the first user message verbatim — no-op once set.
    pub(super) fn record_first_user_message(&mut self, text: &str) {
        if self.record.first_user_message.is_none() {
            self.record.first_user_message = Some(text.to_string());
        }
    }

    /// Writes the `SessionMeta` line (idempotent), assigns a new turn id,
    /// and emits the `Turn(Running)` line.
    pub(super) fn begin_turn(&mut self, session_id: SessionId) -> Result<()> {
        if !self.meta_written {
            self.store.append_session_meta(&self.record)?;
            self.meta_written = true;
        }
        self.turn_seq = self.turn_seq.saturating_add(1);
        let id = TurnId::new();
        let started_at = Utc::now();
        self.store.append_turn(
            &self.record,
            TurnRecord {
                id,
                session_id,
                sequence: self.turn_seq,
                started_at,
                completed_at: None,
                status: TurnStatus::Running,
                model_path: self.record.model_path.clone(),
                usage: None,
                schema_version: SCHEMA_VERSION,
            },
        )?;
        self.current_turn = Some(TurnHandle {
            id,
            session_id,
            started_at,
        });
        Ok(())
    }

    /// Appends a single-item `ItemRecord` for `item`. No-op when no turn is
    /// currently open — the caller is in an inconsistent state but we would
    /// rather drop the line than panic mid-turn.
    pub(super) fn append_item(&mut self, item: TurnItem) -> Result<()> {
        let Some(handle) = self.current_turn.as_ref() else {
            return Ok(());
        };
        self.item_seq = self.item_seq.saturating_add(1);
        self.store.append_item(
            &self.record,
            ItemRecord {
                id: ItemId::new(),
                session_id: handle.session_id,
                turn_id: handle.id,
                seq: self.item_seq,
                timestamp: Utc::now(),
                items: vec![item],
                schema_version: SCHEMA_VERSION,
            },
        )
    }

    /// Closes the turn — appends the terminal `Turn(<status>)` line, mirrors
    /// the running totals back to the meta sidecar, updates `updated_at`.
    pub(super) fn end_turn(&mut self, status: TurnStatus, usage: Option<TurnUsage>) -> Result<()> {
        let Some(handle) = self.current_turn.take() else {
            return Ok(());
        };
        if let Some(u) = usage {
            self.record.total_input_tokens = self
                .record
                .total_input_tokens
                .saturating_add(u.input_tokens);
            self.record.total_output_tokens = self
                .record
                .total_output_tokens
                .saturating_add(u.output_tokens);
        }
        let now = Utc::now();
        self.record.updated_at = now;
        self.store.append_turn(
            &self.record,
            TurnRecord {
                id: handle.id,
                session_id: handle.session_id,
                sequence: self.turn_seq,
                started_at: handle.started_at,
                completed_at: Some(now),
                status,
                model_path: self.record.model_path.clone(),
                usage,
                schema_version: SCHEMA_VERSION,
            },
        )?;
        self.store.write_meta_sidecar(&self.record)?;
        Ok(())
    }
}

/// Logs a persistence error without aborting the turn. Local sessions would
/// rather keep working with degraded continuity than fail the whole turn
/// because the rollout file briefly couldn't be appended to.
pub(super) fn log_persist_err<T>(result: Result<T>) {
    if let Err(err) = result {
        tracing::error!(error = %err, "session persistence write failed");
    }
}

pub(super) fn user_item(text: impl Into<String>) -> TurnItem {
    TurnItem::UserMessage(TextItem { text: text.into() })
}

pub(super) fn agent_item(text: impl Into<String>) -> TurnItem {
    TurnItem::AgentMessage(TextItem { text: text.into() })
}

pub(super) fn reasoning_item(text: impl Into<String>) -> TurnItem {
    TurnItem::Reasoning(ReasoningItem { text: text.into() })
}

pub(super) fn compaction_item(text: impl Into<String>) -> TurnItem {
    TurnItem::ContextCompaction(TextItem { text: text.into() })
}

/// Builds a `ToolCall` item, parsing `arguments` as JSON when possible. An
/// invalid-JSON arguments string (the model misbehaving) is preserved as a
/// `Value::String` so the rollout still records what the model attempted.
pub(super) fn tool_call_item(
    id: impl Into<String>,
    name: impl Into<String>,
    arguments: &str,
) -> TurnItem {
    TurnItem::ToolCall(ToolCallItem {
        tool_call_id: ToolCallId(id.into()),
        tool_name: name.into(),
        input: parse_json_or_string(arguments),
    })
}

/// Builds a `ToolResult` item. Tool output is a free-form string from the
/// tool layer — we try to parse it as JSON for richer downstream rendering
/// and fall back to a JSON string when it is not valid.
pub(super) fn tool_result_item(id: impl Into<String>, content: &str, is_error: bool) -> TurnItem {
    TurnItem::ToolResult(ToolResultItem {
        tool_call_id: ToolCallId(id.into()),
        output: parse_json_or_string(content),
        is_error,
    })
}

/// Maps a per-round agent [`Usage`] to the persisted [`TurnUsage`]. Cache
/// fields are zero — embedded local inference does not report cache hits.
pub(super) fn usage_from_round(usage: &Usage) -> TurnUsage {
    TurnUsage {
        input_tokens: u64::from(usage.prompt_tokens),
        output_tokens: u64::from(usage.completion_tokens),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
    }
}

fn parse_json_or_string(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

#[cfg(test)]
#[path = "persist_tests.rs"]
mod tests;
