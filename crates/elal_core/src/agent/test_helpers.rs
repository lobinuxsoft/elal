//! Shared mock infrastructure for the agent test suites: scripted provider,
//! scripted tool, rollout helpers. Kept separate from `tests.rs` and
//! `tests_persistence.rs` so each test file stays focused on assertions.
//!
//! Mounted via `#[cfg(test)] mod test_helpers;` in `mod.rs` so this file
//! never enters non-test builds.

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use elal_protocol::{RolloutLine, StopReason, StreamEvent, ToolDefinition, Usage};
use elal_provider::{CompletionRequest, CompletionSummary, LlmError, ModelCapabilities, Provider};
use elal_tools::{
    ApprovalHint, SideEffects, Tool, ToolContext, ToolError, ToolResult, ToolSpec, ToolTier,
};
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tokio::sync::mpsc;

use super::AgentEvent;
use crate::session::RolloutStore;

/// Mock provider that replays a pre-baked per-round event script.
///
/// Each call to [`Provider::chat_completion_stream`] consumes one round
/// from the back of the queue — use [`Self::new_last_first`] when you want
/// the script entries to fire in declaration order.
pub(super) struct ScriptedProvider {
    rounds: RefCell<Vec<Vec<StreamEvent>>>,
    capabilities: ModelCapabilities,
    summary_usage: Usage,
}

impl ScriptedProvider {
    pub(super) fn new_last_first(mut rounds: Vec<Vec<StreamEvent>>) -> Self {
        rounds.reverse();
        Self {
            rounds: RefCell::new(rounds),
            capabilities: ModelCapabilities::default(),
            summary_usage: trivial_usage(),
        }
    }
}

#[async_trait(?Send)]
impl Provider for ScriptedProvider {
    async fn chat_completion_stream(
        &self,
        _request: CompletionRequest,
        events: mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionSummary, LlmError> {
        let next = self
            .rounds
            .borrow_mut()
            .pop()
            .ok_or_else(|| LlmError::Decode("no more scripted rounds".into()))?;
        for ev in next {
            events.send(ev).await.map_err(|_| LlmError::ChannelClosed)?;
        }
        Ok(CompletionSummary {
            stop_reason: StopReason::EndTurn,
            usage: self.summary_usage.clone(),
        })
    }
    fn model_name(&self) -> &str {
        "mock"
    }
    fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }
    fn context_length(&self) -> usize {
        8192
    }
}

/// Provider variant whose first round reports a huge `prompt_tokens` count
/// in its `Done` event so the next turn trips the compaction threshold.
pub(super) struct CompactionProvider {
    rounds: RefCell<Vec<Vec<StreamEvent>>>,
    capabilities: ModelCapabilities,
}

impl CompactionProvider {
    pub(super) fn new() -> Self {
        let big_done = StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                prompt_tokens: 100_000,
                completion_tokens: 1,
                reasoning_tokens: 0,
                prompt_eval_ms: 1,
                generation_ms: 1,
            },
        };
        let rounds = vec![
            vec![
                StreamEvent::TextDelta("third".into()),
                done(StopReason::EndTurn),
            ],
            vec![
                StreamEvent::TextDelta("second".into()),
                done(StopReason::EndTurn),
            ],
            vec![StreamEvent::TextDelta("first".into()), big_done],
        ];
        Self {
            rounds: RefCell::new(rounds),
            capabilities: ModelCapabilities::default(),
        }
    }
}

#[async_trait(?Send)]
impl Provider for CompactionProvider {
    async fn chat_completion_stream(
        &self,
        _request: CompletionRequest,
        events: mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionSummary, LlmError> {
        let next = self
            .rounds
            .borrow_mut()
            .pop()
            .ok_or_else(|| LlmError::Decode("no more scripted rounds".into()))?;
        for ev in next {
            events.send(ev).await.map_err(|_| LlmError::ChannelClosed)?;
        }
        Ok(CompletionSummary {
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }
    fn model_name(&self) -> &str {
        "mock"
    }
    fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }
    fn context_length(&self) -> usize {
        8192
    }
}

/// Trivial scripted tool that always returns a fixed echoed string.
pub(super) struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "echo",
            "echoes",
            serde_json::json!({"type":"object","properties":{}}),
        )
    }
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo",
            tier: ToolTier::Read,
            approval_hint: ApprovalHint::Never,
            side_effects: SideEffects::None,
        }
    }
    fn describe_action(&self, _args: &serde_json::Value) -> String {
        "echo".into()
    }
    async fn execute(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::ok("echoed"))
    }
}

pub(super) fn trivial_usage() -> Usage {
    Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        reasoning_tokens: 0,
        prompt_eval_ms: 1,
        generation_ms: 1,
    }
}

pub(super) fn done(stop: StopReason) -> StreamEvent {
    StreamEvent::Done {
        stop_reason: stop,
        usage: trivial_usage(),
    }
}

pub(super) fn make_store(dir: &Path) -> RolloutStore {
    RolloutStore::new(dir.to_path_buf())
}

pub(super) fn fixed_ts() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 29, 14, 0, 0).unwrap()
}

pub(super) fn read_lines(path: &Path) -> Vec<RolloutLine> {
    let file = File::open(path).expect("open rollout");
    BufReader::new(file)
        .lines()
        .map(|l| serde_json::from_str::<RolloutLine>(&l.unwrap()).expect("parse"))
        .collect()
}

pub(super) async fn drain(events_rx: &mut mpsc::Receiver<AgentEvent>) {
    while events_rx.recv().await.is_some() {}
}
