//! Route provider [`StreamEvent`]s to consumer-facing [`AgentEvent`]s and
//! accumulate enough state to reconstruct the assistant message + pending
//! tool calls after the stream closes.
//!
//! The router is a small state machine running inside the collector task.
//! It tracks the open tool-call fence (`id` + `name` + args chunks), the
//! accumulated assistant text + reasoning (for history reconstruction),
//! and the final `StopReason` + `Usage` from the provider. Everything
//! else is passed through to the consumer channel.

use elal_protocol::{FunctionCall, Message, Role, StopReason, StreamEvent, ToolCall, Usage};

use super::event::AgentEvent;

/// One tool call assembled from the provider's streaming fence.
#[derive(Debug, Clone)]
pub(crate) struct PendingToolCall {
    pub id: String,
    pub name: String,
    /// Full JSON arguments string as emitted by the provider (not yet
    /// parsed — the dispatcher parses and surfaces soft `ToolResult`s
    /// with an error message when the JSON is bad).
    pub arguments: String,
}

/// Everything the dispatcher needs once the provider stream has closed.
#[derive(Debug)]
pub(crate) struct StreamSummary {
    /// Reconstructed `assistant` message ready to push into history —
    /// carries text, reasoning, and tool-call metadata so the next
    /// provider round sees exactly what the model just said.
    pub assistant_message: Message,
    pub pending_tool_calls: Vec<PendingToolCall>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

/// Incremental router used by the agent loop's collector task.
#[derive(Debug, Default)]
pub(crate) struct StreamRouter {
    current_id: Option<String>,
    current_name: Option<String>,
    current_args: String,
    pending: Vec<PendingToolCall>,
    stop_reason: Option<StopReason>,
    usage: Option<Usage>,
    /// Plain assistant text accumulated across all `TextDelta`s.
    assistant_text: String,
    /// Reasoning content accumulated across all `ReasoningDelta`s.
    assistant_reasoning: String,
}

impl StreamRouter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feed one [`StreamEvent`] from the provider. Returns the
    /// consumer-facing events the caller should forward on `events_tx`.
    ///
    /// `TextStart` / `TextEnd` / `ReasoningStart` / `ReasoningEnd` are
    /// dropped silently — the top-level `TurnStart` / `TurnComplete`
    /// already bracket the turn and section changes are visible from the
    /// kind of delta being emitted.
    ///
    /// Tool-call fences emit `ToolCallStart` + `ToolCallArgs` as they
    /// stream, and the full call is accumulated internally so the
    /// dispatcher can drain it from [`StreamRouter::into_summary`].
    pub(crate) fn on_event(&mut self, event: StreamEvent) -> Vec<AgentEvent> {
        match event {
            StreamEvent::TextStart | StreamEvent::TextEnd => Vec::new(),
            StreamEvent::TextDelta(s) => {
                self.assistant_text.push_str(&s);
                vec![AgentEvent::TextDelta(s)]
            }

            StreamEvent::ReasoningStart | StreamEvent::ReasoningEnd => Vec::new(),
            StreamEvent::ReasoningDelta(s) => {
                self.assistant_reasoning.push_str(&s);
                vec![AgentEvent::ReasoningDelta(s)]
            }

            StreamEvent::ToolCallStart { id, name } => {
                self.current_id = Some(id.clone());
                self.current_name = Some(name.clone());
                self.current_args.clear();
                vec![AgentEvent::ToolCallStart { id, name }]
            }
            StreamEvent::ToolCallInputDelta(chunk) => {
                self.current_args.push_str(&chunk);
                vec![AgentEvent::ToolCallArgs(chunk)]
            }
            StreamEvent::ToolCallEnd => {
                if let (Some(id), Some(name)) = (self.current_id.take(), self.current_name.take()) {
                    self.pending.push(PendingToolCall {
                        id,
                        name,
                        arguments: std::mem::take(&mut self.current_args),
                    });
                }
                Vec::new()
            }

            StreamEvent::Done { stop_reason, usage } => {
                self.stop_reason = Some(stop_reason);
                self.usage = Some(usage);
                Vec::new()
            }
        }
    }

    /// Consume the router and produce everything the agent needs.
    pub(crate) fn into_summary(self) -> StreamSummary {
        let has_text = !self.assistant_text.is_empty();
        let has_reasoning = !self.assistant_reasoning.is_empty();
        let has_calls = !self.pending.is_empty();

        let tool_calls = if has_calls {
            Some(
                self.pending
                    .iter()
                    .map(|pc| ToolCall {
                        id: pc.id.clone(),
                        kind: "function".into(),
                        function: FunctionCall {
                            name: pc.name.clone(),
                            arguments: pc.arguments.clone(),
                        },
                    })
                    .collect(),
            )
        } else {
            None
        };

        let assistant_message = Message {
            role: Role::Assistant,
            content: if has_text {
                Some(self.assistant_text)
            } else {
                None
            },
            tool_calls,
            tool_call_id: None,
            reasoning_content: if has_reasoning {
                Some(self.assistant_reasoning)
            } else {
                None
            },
        };

        StreamSummary {
            assistant_message,
            pending_tool_calls: self.pending,
            stop_reason: self.stop_reason.unwrap_or(StopReason::EndTurn),
            usage: self.usage.unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_delta_is_routed_and_accumulated() {
        let mut r = StreamRouter::new();
        let out = r.on_event(StreamEvent::TextDelta("hi ".into()));
        assert!(matches!(out.as_slice(), [AgentEvent::TextDelta(s)] if s == "hi "));
        r.on_event(StreamEvent::TextDelta("world".into()));
        let summary = r.into_summary();
        assert_eq!(
            summary.assistant_message.content.as_deref(),
            Some("hi world")
        );
    }

    #[test]
    fn reasoning_is_routed_and_accumulated() {
        let mut r = StreamRouter::new();
        r.on_event(StreamEvent::ReasoningDelta("think".into()));
        let summary = r.into_summary();
        assert_eq!(
            summary.assistant_message.reasoning_content.as_deref(),
            Some("think")
        );
    }

    #[test]
    fn tool_call_fence_accumulates_and_populates_message() {
        let mut r = StreamRouter::new();
        r.on_event(StreamEvent::ToolCallStart {
            id: "c1".into(),
            name: "read".into(),
        });
        r.on_event(StreamEvent::ToolCallInputDelta(
            "{\"path\":\"/tmp\"}".into(),
        ));
        r.on_event(StreamEvent::ToolCallEnd);

        let summary = r.into_summary();
        assert_eq!(summary.pending_tool_calls.len(), 1);
        assert_eq!(summary.pending_tool_calls[0].id, "c1");
        assert_eq!(
            summary.pending_tool_calls[0].arguments,
            "{\"path\":\"/tmp\"}"
        );

        let calls = summary.assistant_message.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
    }

    #[test]
    fn done_event_records_reason_and_usage() {
        let mut r = StreamRouter::new();
        r.on_event(StreamEvent::Done {
            stop_reason: StopReason::MaxTokens,
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 10,
                reasoning_tokens: 0,
                prompt_eval_ms: 1,
                generation_ms: 2,
            },
        });
        let s = r.into_summary();
        assert_eq!(s.stop_reason, StopReason::MaxTokens);
        assert_eq!(s.usage.completion_tokens, 10);
    }

    #[test]
    fn empty_stream_yields_message_with_no_content() {
        let r = StreamRouter::new();
        let s = r.into_summary();
        assert!(s.assistant_message.content.is_none());
        assert!(s.assistant_message.tool_calls.is_none());
        assert!(s.assistant_message.reasoning_content.is_none());
        assert!(s.pending_tool_calls.is_empty());
    }

    #[test]
    fn bookends_are_dropped() {
        let mut r = StreamRouter::new();
        assert!(r.on_event(StreamEvent::TextStart).is_empty());
        assert!(r.on_event(StreamEvent::TextEnd).is_empty());
        assert!(r.on_event(StreamEvent::ReasoningStart).is_empty());
        assert!(r.on_event(StreamEvent::ReasoningEnd).is_empty());
    }
}
