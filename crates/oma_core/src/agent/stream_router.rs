//! Route provider [`StreamEvent`]s to consumer-facing [`AgentEvent`]s and
//! accumulate tool-call fences for the agent to dispatch after the stream
//! closes.
//!
//! The router is a small state machine — it holds exactly the state
//! needed to assemble a tool call as it streams (`id` + `name` + args
//! chunks) and records the final `StopReason` + `Usage` from the provider.
//! Everything else is passed through to the consumer.
//!
//! Chunk 2 wires only the text + reasoning passes. Tool-call accumulation
//! lives here so chunk 3 can plug dispatch into `take_pending_tool_calls`
//! without touching this module's wiring again.

use oma_protocol::{StopReason, StreamEvent, Usage};

use super::event::AgentEvent;

/// One tool call assembled from the provider's streaming fence.
///
/// Consumed by the dispatcher in chunk 3; fields are held `pub(crate)` and
/// `#[allow(dead_code)]` until then so the router can land without a
/// downstream caller.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct PendingToolCall {
    pub id: String,
    pub name: String,
    /// Full JSON arguments string as emitted by the provider (not yet
    /// parsed — the dispatcher does that and surfaces errors as
    /// `InvalidArgs` or soft `ToolResult`s).
    pub arguments: String,
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
}

impl StreamRouter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feed one [`StreamEvent`] from the provider. Returns the
    /// consumer-facing events the caller should forward on `events_tx`.
    ///
    /// When a tool-call fence closes, the accumulated call is stored
    /// internally — drain it with [`take_pending_tool_calls`] after the
    /// stream has finished.
    pub(crate) fn on_event(&mut self, event: StreamEvent) -> Vec<AgentEvent> {
        match event {
            // TextStart / TextEnd are internal provider bookends; consumers
            // don't need them — AgentEvent::TurnStart + TurnComplete bracket
            // the whole turn and chunks of text arrive as TextDelta.
            StreamEvent::TextStart | StreamEvent::TextEnd => Vec::new(),
            StreamEvent::TextDelta(s) => vec![AgentEvent::TextDelta(s)],

            StreamEvent::ReasoningStart | StreamEvent::ReasoningEnd => Vec::new(),
            StreamEvent::ReasoningDelta(s) => vec![AgentEvent::ReasoningDelta(s)],

            StreamEvent::ToolCallStart { id, name } => {
                self.current_id = Some(id.clone());
                self.current_name = Some(name.clone());
                self.current_args.clear();
                // describe_action is filled by the agent after lookup; we
                // emit a placeholder here so consumers can render the call
                // the moment it starts. Chunk 3 upgrades this to the real
                // description before emitting.
                vec![AgentEvent::ToolCallStart {
                    id,
                    name,
                    describe_action: String::new(),
                }]
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
                // No AgentEvent emitted at fence close — the dispatcher
                // emits `ToolResult` once it finishes the call.
                Vec::new()
            }

            StreamEvent::Done { stop_reason, usage } => {
                self.stop_reason = Some(stop_reason);
                self.usage = Some(usage);
                Vec::new()
            }
        }
    }

    // Drain + inspection helpers consumed by the dispatcher in chunk 3;
    // gated until then.
    #[allow(dead_code)]
    pub(crate) fn take_pending_tool_calls(&mut self) -> Vec<PendingToolCall> {
        std::mem::take(&mut self.pending)
    }

    #[allow(dead_code)]
    pub(crate) fn stop_reason(&self) -> StopReason {
        self.stop_reason.unwrap_or(StopReason::EndTurn)
    }

    #[allow(dead_code)]
    pub(crate) fn usage(&self) -> Usage {
        self.usage.clone().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_delta_routes_to_agent_event() {
        let mut r = StreamRouter::new();
        let out = r.on_event(StreamEvent::TextDelta("hi".into()));
        assert!(matches!(out.as_slice(), [AgentEvent::TextDelta(s)] if s == "hi"));
    }

    #[test]
    fn text_bookends_are_dropped() {
        let mut r = StreamRouter::new();
        assert!(r.on_event(StreamEvent::TextStart).is_empty());
        assert!(r.on_event(StreamEvent::TextEnd).is_empty());
    }

    #[test]
    fn reasoning_delta_routes_to_agent_event() {
        let mut r = StreamRouter::new();
        let out = r.on_event(StreamEvent::ReasoningDelta("think".into()));
        assert!(matches!(out.as_slice(), [AgentEvent::ReasoningDelta(s)] if s == "think"));
    }

    #[test]
    fn tool_call_fence_accumulates_and_drains() {
        let mut r = StreamRouter::new();
        let start = r.on_event(StreamEvent::ToolCallStart {
            id: "c1".into(),
            name: "read".into(),
        });
        assert_eq!(start.len(), 1);
        r.on_event(StreamEvent::ToolCallInputDelta("{\"path\":".into()));
        r.on_event(StreamEvent::ToolCallInputDelta("\"/tmp\"}".into()));
        let end = r.on_event(StreamEvent::ToolCallEnd);
        assert!(end.is_empty());

        let pending = r.take_pending_tool_calls();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, "c1");
        assert_eq!(pending[0].name, "read");
        assert_eq!(pending[0].arguments, r#"{"path":"/tmp"}"#);
    }

    #[test]
    fn done_records_stop_reason_and_usage() {
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
        assert_eq!(r.stop_reason(), StopReason::MaxTokens);
        assert_eq!(r.usage().completion_tokens, 10);
    }

    #[test]
    fn stop_reason_defaults_to_end_turn_when_absent() {
        let r = StreamRouter::new();
        assert_eq!(r.stop_reason(), StopReason::EndTurn);
    }

    #[test]
    fn take_pending_returns_empty_when_no_calls() {
        let mut r = StreamRouter::new();
        r.on_event(StreamEvent::TextDelta("hi".into()));
        assert!(r.take_pending_tool_calls().is_empty());
    }
}
