//! Turn-level outputs and error surface.

use elal_protocol::{StopReason, Usage};

/// Safety cap on the number of tool-call rounds per turn. Prevents a
/// runaway model from pinning GPU or disk via an endless loop of
/// self-correcting tool calls.
pub const TURN_STEP_CAP: u32 = 10;

/// Summary returned from a completed turn. Always accompanied by a final
/// [`super::AgentEvent::TurnComplete(summary)`].
#[derive(Debug, Clone)]
pub struct TurnSummary {
    /// Reason the provider stopped generating on the final round.
    pub stop_reason: StopReason,
    /// How many tool calls the agent executed across all rounds.
    pub tool_calls_executed: u32,
    /// Aggregate token / timing usage across all rounds.
    pub usage: Usage,
    /// Whether the turn ended because the user cancelled an approval.
    pub cancelled: bool,
}

impl TurnSummary {
    /// Sum another round's usage into this running total. Saturating
    /// arithmetic — we would rather cap than panic if a pathological turn
    /// ever ran long enough to overflow a `u32` of tokens.
    //
    // Allowed dead_code until chunk 2 wires the loop — exercising this in
    // tests already proves correctness.
    #[allow(dead_code)]
    pub(crate) fn accumulate(&mut self, round: &Usage) {
        self.usage.prompt_tokens = self.usage.prompt_tokens.saturating_add(round.prompt_tokens);
        self.usage.completion_tokens = self
            .usage
            .completion_tokens
            .saturating_add(round.completion_tokens);
        self.usage.reasoning_tokens = self
            .usage
            .reasoning_tokens
            .saturating_add(round.reasoning_tokens);
        self.usage.prompt_eval_ms = self
            .usage
            .prompt_eval_ms
            .saturating_add(round.prompt_eval_ms);
        self.usage.generation_ms = self.usage.generation_ms.saturating_add(round.generation_ms);
    }
}

/// Hard errors that abort a turn. Soft per-tool failures live in
/// `AgentEvent::ToolResult { is_error: true }` and do NOT surface here —
/// they are part of normal agent flow.
#[derive(Debug, thiserror::Error)]
pub enum TurnError {
    /// The underlying provider failed (model loading, decode, sampling).
    #[error("provider error: {0}")]
    Provider(#[from] elal_provider::LlmError),

    /// Tool registry lookup miss is a soft error (goes back to the model as
    /// a ToolResult). This variant is for genuine tool execution failures
    /// that can't be fed back meaningfully.
    #[error("tool dispatch failed: {0}")]
    ToolDispatch(String),

    /// The model asked for a tool call but the arguments weren't valid JSON.
    #[error("invalid tool arguments: {0}")]
    InvalidArgs(String),

    /// The step cap was hit — likely a model looping on tool calls without
    /// making progress.
    #[error("turn step cap exceeded ({cap} rounds without the model finishing)")]
    TurnStepCapExceeded { cap: u32 },

    /// The consumer dropped the events receiver mid-turn. We treat this as
    /// a cooperative cancel but still surface the error so the caller can
    /// tell the channel died.
    #[error("event channel closed before the turn finished")]
    ChannelClosed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_sums_all_fields() {
        let mut s = TurnSummary {
            stop_reason: StopReason::EndTurn,
            tool_calls_executed: 0,
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                reasoning_tokens: 5,
                prompt_eval_ms: 100,
                generation_ms: 200,
            },
            cancelled: false,
        };
        s.accumulate(&Usage {
            prompt_tokens: 3,
            completion_tokens: 7,
            reasoning_tokens: 1,
            prompt_eval_ms: 50,
            generation_ms: 150,
        });
        assert_eq!(s.usage.prompt_tokens, 13);
        assert_eq!(s.usage.completion_tokens, 27);
        assert_eq!(s.usage.reasoning_tokens, 6);
        assert_eq!(s.usage.prompt_eval_ms, 150);
        assert_eq!(s.usage.generation_ms, 350);
    }

    #[test]
    fn turn_step_cap_is_reasonable() {
        let cap = TURN_STEP_CAP;
        assert!((3..=20).contains(&cap));
    }
}
