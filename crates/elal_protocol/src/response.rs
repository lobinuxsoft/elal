use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    Cancelled,
    Error,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Tokens spent on reasoning output (QwQ, DeepSeek-R1). Included in
    /// `completion_tokens` per OpenAI convention but surfaced separately for
    /// cost/latency analysis.
    pub reasoning_tokens: u32,
    pub prompt_eval_ms: u64,
    pub generation_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContent {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Model-internal thinking. Not shown by default.
    Reasoning(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reason_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&StopReason::EndTurn).unwrap(),
            "\"end_turn\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::ToolUse).unwrap(),
            "\"tool_use\""
        );
    }

    #[test]
    fn stop_reason_roundtrip() {
        for r in [
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::StopSequence,
            StopReason::ToolUse,
            StopReason::Cancelled,
            StopReason::Error,
        ] {
            let json = serde_json::to_string(&r).unwrap();
            let back: StopReason = serde_json::from_str(&json).unwrap();
            assert_eq!(r, back);
        }
    }

    #[test]
    fn usage_default_is_zero() {
        let u = Usage::default();
        assert_eq!(u.prompt_tokens, 0);
        assert_eq!(u.completion_tokens, 0);
        assert_eq!(u.reasoning_tokens, 0);
    }

    #[test]
    fn response_content_tool_use_roundtrip() {
        let c = ResponseContent::ToolUse {
            id: "call_1".into(),
            name: "read".into(),
            input: serde_json::json!({"path": "a.rs"}),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: ResponseContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
