// This module is consumed by `embedded/*` once chunk 4 wires it in; until
// then clippy flags everything as dead code.
#![allow(dead_code)]

//! Adapter layer between our protocol types and llama-cpp-2's
//! OpenAI-compatible layer.
//!
//! Handles two directions:
//!
//! - **Outgoing** (our types → JSON): [`messages_to_json`] and
//!   [`tools_to_json`] produce the string payloads that
//!   `apply_chat_template_with_tools_oaicompat` expects.
//! - **Incoming** (JSON deltas → our events): [`DeltaClassifier`] consumes
//!   the JSON strings returned by
//!   `ChatParseStateOaicompat::update()` and fans them out to
//!   `StreamEvent::TextDelta`, `ReasoningDelta`, `ToolCallStart`,
//!   `ToolCallInputDelta`, `ToolCallEnd` — in the right order and with
//!   start/end bookends generated from state transitions.

use oma_protocol::{Message, Role, StreamEvent, ToolDefinition};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::LlmError;

/// Minimal typed view of the JSON deltas emitted by
/// `ChatParseStateOaicompat::update()`. Only the fields we care about are
/// listed; serde's `#[serde(default)]` tolerates their absence and missing
/// fields stay as `None` rather than erroring, which matches the streaming
/// partial-update semantics.
#[derive(Debug, Default, Deserialize)]
struct OaiDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiToolCallDelta>>,
}

#[derive(Debug, Default, Deserialize)]
struct OaiToolCallDelta {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OaiFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
struct OaiFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Serialize our `Message` slice to the OpenAI-compatible JSON array that
/// `apply_chat_template_with_tools_oaicompat` consumes as `messages_json`.
pub(crate) fn messages_to_json(messages: &[Message]) -> Result<String, LlmError> {
    let arr: Vec<Value> = messages.iter().map(message_to_json).collect();
    serde_json::to_string(&arr).map_err(|e| LlmError::Serialize(e.to_string()))
}

fn message_to_json(m: &Message) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut obj = json!({ "role": role });

    if let Some(content) = &m.content {
        obj["content"] = Value::String(content.clone());
    }
    if let Some(id) = &m.tool_call_id {
        obj["tool_call_id"] = Value::String(id.clone());
    }
    if let Some(calls) = &m.tool_calls {
        obj["tool_calls"] = Value::Array(
            calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": tc.kind,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })
                })
                .collect(),
        );
    }
    if let Some(reasoning) = &m.reasoning_content {
        obj["reasoning_content"] = Value::String(reasoning.clone());
    }
    obj
}

/// Serialize our `ToolDefinition` slice to the JSON array
/// `apply_chat_template_with_tools_oaicompat` consumes as `tools_json`.
pub(crate) fn tools_to_json(tools: &[ToolDefinition]) -> Result<String, LlmError> {
    let arr: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": t.kind,
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
                }
            })
        })
        .collect();
    serde_json::to_string(&arr).map_err(|e| LlmError::Serialize(e.to_string()))
}

/// Incrementally turn JSON deltas from `ChatParseStateOaicompat::update()`
/// into `StreamEvent`s.
///
/// The classifier tracks whether we are currently inside a reasoning block
/// or a tool call so it can emit the matching Start/End bookends that our
/// protocol expects. Call [`classify`](Self::classify) for each delta string
/// and [`flush`](Self::flush) once the model reaches EOS so in-flight
/// sections are properly closed.
#[derive(Debug, Default)]
pub(crate) struct DeltaClassifier {
    in_reasoning: bool,
    current_tool_id: Option<String>,
}

impl DeltaClassifier {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Classify one JSON delta.
    ///
    /// Unknown fields are ignored so forward-compatibility with llama.cpp
    /// extensions doesn't break us.
    pub(crate) fn classify(&mut self, delta_json: &str) -> Result<Vec<StreamEvent>, LlmError> {
        if delta_json.trim().is_empty() {
            return Ok(Vec::new());
        }
        let delta: OaiDelta = serde_json::from_str(delta_json)
            .map_err(|e| LlmError::Serialize(format!("invalid delta JSON: {e}")))?;

        let mut out = Vec::new();

        // Reasoning content — emit Start on first sight, Delta per chunk.
        if let Some(r) = delta.reasoning_content.as_deref() {
            if !self.in_reasoning {
                out.push(StreamEvent::ReasoningStart);
                self.in_reasoning = true;
            }
            if !r.is_empty() {
                out.push(StreamEvent::ReasoningDelta(r.to_string()));
            }
        } else if self.in_reasoning && delta.content.is_some() {
            // Reasoning block ended the moment we see normal content.
            out.push(StreamEvent::ReasoningEnd);
            self.in_reasoning = false;
        }

        // Plain text content.
        if let Some(c) = delta.content.as_deref() {
            if !c.is_empty() {
                out.push(StreamEvent::TextDelta(c.to_string()));
            }
        }

        // Tool call deltas — llama.cpp may emit a single-element array per
        // frame, with either `id + function.name` (start) or only
        // `function.arguments` (continuation).
        if let Some(calls) = delta.tool_calls {
            for call in calls {
                self.classify_tool_call_delta(call, &mut out);
            }
        }

        Ok(out)
    }

    /// Emit closing bookends for any section still open. Call once the
    /// model has reached EOS or the channel is about to close.
    pub(crate) fn flush(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if self.current_tool_id.take().is_some() {
            out.push(StreamEvent::ToolCallEnd);
        }
        if self.in_reasoning {
            out.push(StreamEvent::ReasoningEnd);
            self.in_reasoning = false;
        }
        out
    }

    fn classify_tool_call_delta(&mut self, call: OaiToolCallDelta, out: &mut Vec<StreamEvent>) {
        let function = call.function.unwrap_or_default();

        // A fresh `id`+`name` pair starts a new tool call. If we were
        // already inside one, close it first — consecutive tool calls
        // without an explicit end marker are allowed.
        if let (Some(id), Some(name)) = (call.id, function.name) {
            if self.current_tool_id.is_some() {
                out.push(StreamEvent::ToolCallEnd);
            }
            out.push(StreamEvent::ToolCallStart {
                id: id.clone(),
                name,
            });
            self.current_tool_id = Some(id);
        }

        if let Some(chunk) = function.arguments {
            if !chunk.is_empty() {
                out.push(StreamEvent::ToolCallInputDelta(chunk));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oma_protocol::ToolDefinition;

    #[test]
    fn messages_serialize_to_expected_shape() {
        let msgs = vec![Message::system("be helpful"), Message::user("read /tmp/x")];
        let json = messages_to_json(&msgs).unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["role"], "system");
        assert_eq!(arr[0]["content"], "be helpful");
        assert_eq!(arr[1]["role"], "user");
        assert_eq!(arr[1]["content"], "read /tmp/x");
    }

    #[test]
    fn tool_messages_carry_call_id() {
        let msgs = vec![Message::tool("call_abc", "42")];
        let json = messages_to_json(&msgs).unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed[0]["role"], "tool");
        assert_eq!(parsed[0]["tool_call_id"], "call_abc");
        assert_eq!(parsed[0]["content"], "42");
    }

    #[test]
    fn tools_serialize_to_expected_shape() {
        let tools = vec![ToolDefinition::function(
            "read",
            "Read a file",
            json!({"type": "object", "properties": {}}),
        )];
        let json = tools_to_json(&tools).unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed[0]["type"], "function");
        assert_eq!(parsed[0]["function"]["name"], "read");
        assert_eq!(parsed[0]["function"]["description"], "Read a file");
    }

    #[test]
    fn empty_delta_yields_nothing() {
        let mut c = DeltaClassifier::new();
        assert!(c.classify("").unwrap().is_empty());
        assert!(c.classify("   ").unwrap().is_empty());
    }

    #[test]
    fn plain_content_becomes_text_delta() {
        let mut c = DeltaClassifier::new();
        let out = c.classify(r#"{"content": "Hello"}"#).unwrap();
        assert_eq!(out, vec![StreamEvent::TextDelta("Hello".into())]);
    }

    #[test]
    fn reasoning_emits_start_then_delta() {
        let mut c = DeltaClassifier::new();
        let first = c.classify(r#"{"reasoning_content": "thinking"}"#).unwrap();
        assert_eq!(
            first,
            vec![
                StreamEvent::ReasoningStart,
                StreamEvent::ReasoningDelta("thinking".into())
            ]
        );
        let second = c.classify(r#"{"reasoning_content": "..."}"#).unwrap();
        assert_eq!(second, vec![StreamEvent::ReasoningDelta("...".into())]);
    }

    #[test]
    fn reasoning_ends_when_content_appears() {
        let mut c = DeltaClassifier::new();
        c.classify(r#"{"reasoning_content": "thinking"}"#).unwrap();
        let out = c.classify(r#"{"content": "Answer"}"#).unwrap();
        assert_eq!(
            out,
            vec![
                StreamEvent::ReasoningEnd,
                StreamEvent::TextDelta("Answer".into())
            ]
        );
    }

    #[test]
    fn flush_closes_open_reasoning() {
        let mut c = DeltaClassifier::new();
        c.classify(r#"{"reasoning_content": "x"}"#).unwrap();
        assert_eq!(c.flush(), vec![StreamEvent::ReasoningEnd]);
        // Idempotent: a second flush is a no-op.
        assert!(c.flush().is_empty());
    }

    #[test]
    fn tool_call_start_then_arg_chunks() {
        let mut c = DeltaClassifier::new();
        let start = c
            .classify(
                r#"{"tool_calls":[{"id":"call_1","function":{"name":"read","arguments":"{"}}]}"#,
            )
            .unwrap();
        assert_eq!(
            start,
            vec![
                StreamEvent::ToolCallStart {
                    id: "call_1".into(),
                    name: "read".into(),
                },
                StreamEvent::ToolCallInputDelta("{".into()),
            ]
        );
        let mid = c
            .classify(r#"{"tool_calls":[{"function":{"arguments":"\"path\":\"/x\""}}]}"#)
            .unwrap();
        assert_eq!(
            mid,
            vec![StreamEvent::ToolCallInputDelta("\"path\":\"/x\"".into())]
        );
        let end = c
            .classify(r#"{"tool_calls":[{"function":{"arguments":"}"}}]}"#)
            .unwrap();
        assert_eq!(end, vec![StreamEvent::ToolCallInputDelta("}".into())]);
    }

    #[test]
    fn consecutive_tool_calls_get_end_between_them() {
        let mut c = DeltaClassifier::new();
        c.classify(
            r#"{"tool_calls":[{"id":"call_1","function":{"name":"read","arguments":"{}"}}]}"#,
        )
        .unwrap();
        let second = c
            .classify(
                r#"{"tool_calls":[{"id":"call_2","function":{"name":"write","arguments":"{}"}}]}"#,
            )
            .unwrap();
        assert_eq!(
            second,
            vec![
                StreamEvent::ToolCallEnd,
                StreamEvent::ToolCallStart {
                    id: "call_2".into(),
                    name: "write".into(),
                },
                StreamEvent::ToolCallInputDelta("{}".into()),
            ]
        );
    }

    #[test]
    fn flush_closes_open_tool_call() {
        let mut c = DeltaClassifier::new();
        c.classify(
            r#"{"tool_calls":[{"id":"call_x","function":{"name":"read","arguments":"{}"}}]}"#,
        )
        .unwrap();
        assert_eq!(c.flush(), vec![StreamEvent::ToolCallEnd]);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let mut c = DeltaClassifier::new();
        let out = c
            .classify(r#"{"weird_new_field": "hi", "content": "hello"}"#)
            .unwrap();
        assert_eq!(out, vec![StreamEvent::TextDelta("hello".into())]);
    }

    #[test]
    fn invalid_json_is_an_error() {
        let mut c = DeltaClassifier::new();
        assert!(c.classify("not json at all").is_err());
    }
}
