use serde::{Deserialize, Serialize};

use crate::response::{StopReason, Usage};

/// Structured stream events emitted by a `Provider` during chat completion.
///
/// Lives in `oma_protocol` because these cross crate boundaries (agent loop,
/// TUI) without pulling in `oma_provider`'s dependency tree.
///
/// Errors are carried out-of-band via `Result<StreamEvent, LlmError>` on the
/// consumer side; there is intentionally no `Error` variant here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum StreamEvent {
    TextStart,
    TextDelta(String),
    TextEnd,

    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallInputDelta(String),
    ToolCallEnd,

    /// Model-internal thinking block (QwQ, DeepSeek-R1). Not shown by default.
    ReasoningStart,
    ReasoningDelta(String),
    ReasoningEnd,

    Done {
        stop_reason: StopReason,
        usage: Usage,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_delta_roundtrip() {
        let e = StreamEvent::TextDelta("hello".into());
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn tool_call_start_roundtrip() {
        let e = StreamEvent::ToolCallStart {
            id: "call_1".into(),
            name: "read".into(),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn done_carries_reason_and_usage() {
        let e = StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                reasoning_tokens: 0,
                prompt_eval_ms: 100,
                generation_ms: 500,
            },
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn serializes_snake_case_kind() {
        let e = StreamEvent::TextStart;
        let value = serde_json::to_value(&e).unwrap();
        assert_eq!(value["kind"], "text_start");
    }
}
