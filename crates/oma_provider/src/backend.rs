use async_trait::async_trait;
use tokio::sync::mpsc::Sender;

use oma_protocol::{Message, StopReason, StreamEvent, ToolDefinition, Usage};

use crate::capabilities::ModelCapabilities;
use crate::error::LlmError;
use crate::sampling::SamplingControls;

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub sampling: SamplingControls,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct CompletionSummary {
    pub stop_reason: StopReason,
    pub usage: Usage,
}

/// Provider abstraction over an LLM backend.
///
/// Note: `?Send` because `llama-cpp-2`'s `LlamaBatch` and `LlamaContext` are
/// not `Send`. The agent loop runs inference single-threaded; moving the
/// future across threads would be incorrect anyway since GPU state is bound
/// to the thread that initialised it. Advanced concurrency (multiple
/// concurrent inferences) can layer a `spawn_blocking` adapter later.
#[async_trait(?Send)]
pub trait Provider {
    /// Run chat completion. Emits structured events as the model generates.
    async fn chat_completion_stream(
        &self,
        request: CompletionRequest,
        events: Sender<StreamEvent>,
    ) -> Result<CompletionSummary, LlmError>;

    fn model_name(&self) -> &str;

    fn capabilities(&self) -> &ModelCapabilities;

    fn context_length(&self) -> usize;
}
