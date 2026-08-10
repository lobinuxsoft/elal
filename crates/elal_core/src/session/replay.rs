//! Rehydrates a [`SessionState`] from its append-only JSONL rollout.
//!
//! Counterpart to [`super::rollout::RolloutStore`]. Reads the JSONL line by
//! line, applies each [`RolloutLine`] in order, and produces a
//! [`LoadedSession`] containing both the canonical [`SessionRecord`] and an
//! in-memory [`SessionState`] ready for the next turn.
//!
//! Tolerance contract:
//! - Empty file → `Err(ElalError::Session)`.
//! - First non-empty line not [`RolloutLine::SessionMeta`] → `Err`.
//! - **Last** line malformed (truncated by a mid-append crash) → silently
//!   ignored, prior state preserved. This matches `claw-code-rust`'s replay
//!   semantics so a hard kill can never lock the user out of their session.
//! - Any other malformed line → `Err`. Append-only is supposed to leave a
//!   parseable journal; non-tail corruption is a signal we shouldn't paper
//!   over.
//! - Duplicate [`RolloutLine::SessionMeta`] lines after the first are
//!   tolerated (no-op). They are not expected — the writer never emits them
//!   — but a third-party tool concatenating rollouts shouldn't surprise the
//!   replayer.
//!
//! `SessionState.config` is left at [`SessionConfig::default`]; the loader
//! cannot know the active model's `n_ctx` (and therefore the real
//! [`super::TokenBudget`]) until the provider has been initialised. The CLI
//! glue (chunk 6) overrides `state.config` after loading the model.

use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::path::Path;

use elal_protocol::{Message, Role, RolloutLine, SessionRecord, ToolCall, TurnItem};

use crate::error::{ElalError, Result};
use crate::session::{SessionConfig, SessionState};

/// Output of a successful replay.
#[derive(Debug)]
pub struct LoadedSession {
    /// Canonical session metadata. `title` and `updated_at` reflect the
    /// most recent [`RolloutLine::SessionTitleUpdated`] when one is present.
    pub record: SessionRecord,
    /// In-memory state with `messages`, `turn_count`, and the cumulative
    /// `total_*_tokens` populated from the rollout. `config` is the default;
    /// the caller overrides it once the model is loaded.
    pub state: SessionState,
    /// Highest `sequence` seen in any `Turn` line — used by resumed sessions
    /// so the next `Persistence::begin_turn` allocates a fresh, monotonic id.
    pub last_turn_seq: u32,
    /// Highest `seq` seen in any `Item` line — used by resumed sessions to
    /// continue assigning monotonic seq values.
    pub last_item_seq: u64,
}

/// Replays the rollout at `rollout_path` into a [`LoadedSession`].
pub fn load_session(rollout_path: &Path) -> Result<LoadedSession> {
    let file = File::open(rollout_path)?;
    let reader = BufReader::new(file);
    let mut iter = reader.lines();

    let mut record = read_session_meta(&mut iter)?;
    let mut state = SessionState::with_id(record.id, SessionConfig::default(), record.cwd.clone());
    let mut last_turn_seq: u32 = 0;
    let mut last_item_seq: u64 = 0;

    // Buffer the next raw line so we can detect "is this the last line"
    // before parsing — needed to tolerate truncated tail writes.
    let mut pending = next_non_empty(&mut iter)?;
    while let Some(raw) = pending.take() {
        let lookahead = next_non_empty(&mut iter)?;
        match serde_json::from_str::<RolloutLine>(&raw) {
            Ok(parsed) => apply_line(
                &mut record,
                &mut state,
                &mut last_turn_seq,
                &mut last_item_seq,
                parsed,
            ),
            Err(err) => {
                if lookahead.is_none() {
                    // Truncated tail — silently drop and finish.
                    break;
                }
                return Err(err.into());
            }
        }
        pending = lookahead;
    }

    Ok(LoadedSession {
        record,
        state,
        last_turn_seq,
        last_item_seq,
    })
}

/// Reads the next non-empty line from `iter`, returning `Ok(None)` at EOF.
fn next_non_empty(iter: &mut Lines<BufReader<File>>) -> Result<Option<String>> {
    for line in iter.by_ref() {
        let raw = line?;
        if !raw.trim().is_empty() {
            return Ok(Some(raw));
        }
    }
    Ok(None)
}

fn read_session_meta(iter: &mut Lines<BufReader<File>>) -> Result<SessionRecord> {
    match next_non_empty(iter)? {
        None => Err(ElalError::Session("rollout file is empty".into())),
        Some(raw) => match serde_json::from_str::<RolloutLine>(&raw)? {
            RolloutLine::SessionMeta(meta) => Ok(meta.session),
            other => Err(ElalError::Session(format!(
                "first rollout line must be session_meta, got {}",
                rollout_line_kind(&other)
            ))),
        },
    }
}

fn rollout_line_kind(line: &RolloutLine) -> &'static str {
    match line {
        RolloutLine::SessionMeta(_) => "session_meta",
        RolloutLine::Turn(_) => "turn",
        RolloutLine::Item(_) => "item",
        RolloutLine::SessionTitleUpdated(_) => "session_title_updated",
    }
}

fn apply_line(
    record: &mut SessionRecord,
    state: &mut SessionState,
    last_turn_seq: &mut u32,
    last_item_seq: &mut u64,
    line: RolloutLine,
) {
    match line {
        // Duplicate meta after the first — accept silently for tooling robustness.
        RolloutLine::SessionMeta(_) => {}
        RolloutLine::Turn(turn_line) => {
            let turn = turn_line.turn;
            *last_turn_seq = (*last_turn_seq).max(turn.sequence);
            state.turn_count = state.turn_count.max(turn.sequence as usize);
            if let Some(usage) = turn.usage {
                state.total_input_tokens =
                    state.total_input_tokens.saturating_add(usage.input_tokens);
                state.total_output_tokens = state
                    .total_output_tokens
                    .saturating_add(usage.output_tokens);
                state.last_input_tokens = usage.input_tokens as usize;
            }
        }
        RolloutLine::Item(item_line) => {
            *last_item_seq = (*last_item_seq).max(item_line.item.seq);
            for item in item_line.item.items {
                state.push_message(item_to_message(item));
            }
        }
        RolloutLine::SessionTitleUpdated(title_line) => {
            record.title = Some(title_line.title);
            record.updated_at = title_line.timestamp;
        }
    }
}

/// Maps a persisted [`TurnItem`] back to an `elal_protocol::Message`.
///
/// The mapping is intentionally 1:1 — a single ItemRecord with N items
/// rehydrates to N messages in order. Grouping reasoning + tool_call +
/// final-text into one assistant message is the agent loop's responsibility
/// (chunk 5); the replayer only decodes what was written.
fn item_to_message(item: TurnItem) -> Message {
    match item {
        TurnItem::UserMessage(text) => Message::user(text.text),
        TurnItem::AgentMessage(text) => Message::assistant(text.text),
        TurnItem::Reasoning(reasoning) => Message {
            role: Role::Assistant,
            content: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: Some(reasoning.text),
        },
        TurnItem::ToolCall(call) => {
            let arguments = call.input.to_string();
            Message {
                role: Role::Assistant,
                content: None,
                tool_calls: Some(vec![ToolCall::function(
                    call.tool_call_id.0,
                    call.tool_name,
                    arguments,
                )]),
                tool_call_id: None,
                reasoning_content: None,
            }
        }
        TurnItem::ToolResult(result) => {
            let content = result.output.to_string();
            Message::tool(result.tool_call_id.0, content)
        }
        TurnItem::ContextCompaction(text) => {
            Message::system(format!("[compacted history]\n{}", text.text))
        }
    }
}

#[cfg(test)]
#[path = "replay_tests.rs"]
mod tests;
