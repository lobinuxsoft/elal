//! Embedded `llama-cpp-2`-backed provider.
//!
//! The crate-root `EmbeddedProvider` lives here. The heavier details —
//! prompt construction, sampler building, and the inference loop — are
//! peeled off into sibling modules so no single file concentrates too much
//! of the provider's responsibilities.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaChatTemplate;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use oma_protocol::StreamEvent;
use tokio::sync::mpsc::Sender;

use crate::backend::{CompletionRequest, CompletionSummary, Provider};
use crate::capabilities::{ModelCapabilities, resolve_by_filename};
use crate::capabilities_resolver;
use crate::compute::ComputeBackend;
use crate::error::LlmError;

mod prompt;
mod sampler;
mod streaming;

pub(super) const DEFAULT_MAX_TOKENS: u32 = 2048;

#[derive(Debug, Clone)]
pub struct ModelLoadParams {
    /// GPU layers to offload. Negative or very large values mean "all layers".
    pub n_gpu_layers: i32,
    /// Context window size. 0 means "model's native max".
    pub n_ctx: u32,
    pub n_batch: u32,
    pub capabilities: Option<ModelCapabilities>,
}

impl Default for ModelLoadParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: -1,
            n_ctx: 0,
            // 4096 fits agent prompts that include the full tool catalog
            // (11 tool JSON schemas + system + first user message). The
            // upstream llama.cpp default of 512 fails GGML_ASSERT(n_tokens
            // <= n_batch) on the very first decode in agent mode.
            n_batch: 4096,
            capabilities: None,
        }
    }
}

pub struct EmbeddedProvider {
    pub(super) backend: LlamaBackend,
    pub(super) model: LlamaModel,
    pub(super) model_name: String,
    pub(super) model_path: PathBuf,
    pub(super) capabilities: ModelCapabilities,
    pub(super) compute: ComputeBackend,
    pub(super) context_length: usize,
    pub(super) n_batch: u32,
    pub(super) chat_template: Option<LlamaChatTemplate>,
}

impl EmbeddedProvider {
    pub fn load(model_path: &Path, params: &ModelLoadParams) -> Result<Self, LlmError> {
        let backend = LlamaBackend::init().map_err(|e| LlmError::Load(e.to_string()))?;

        let n_gpu_layers = if params.n_gpu_layers < 0 {
            u32::MAX
        } else {
            params.n_gpu_layers as u32
        };

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| LlmError::Load(format!("{}: {e}", model_path.display())))?;

        let native_ctx = model.n_ctx_train();
        let context_length = if params.n_ctx == 0 {
            native_ctx as usize
        } else {
            params.n_ctx.min(native_ctx) as usize
        };

        // Start with either an explicit override or the filename heuristic,
        // then upgrade with whatever GGUF metadata has to say about the
        // model family. Filename-based resolution stays in effect when the
        // metadata is silent or unrecognised.
        let filename_fallback = params
            .capabilities
            .clone()
            .unwrap_or_else(|| resolve_by_filename(model_path));
        let capabilities = capabilities_resolver::resolve_from_model(&model, filename_fallback);

        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let chat_template = model.chat_template(None).ok();

        Ok(Self {
            backend,
            model,
            model_name,
            model_path: model_path.to_path_buf(),
            capabilities,
            compute: ComputeBackend::compiled(),
            context_length,
            n_batch: params.n_batch,
            chat_template,
        })
    }

    pub fn compute(&self) -> ComputeBackend {
        self.compute
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}

#[async_trait(?Send)]
impl Provider for EmbeddedProvider {
    async fn chat_completion_stream(
        &self,
        request: CompletionRequest,
        events: Sender<StreamEvent>,
    ) -> Result<CompletionSummary, LlmError> {
        streaming::run(self, &request, &events).await
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    fn context_length(&self) -> usize {
        self.context_length
    }
}

/// Send a stream event or convert a closed receiver into a [`LlmError`].
pub(super) async fn send(events: &Sender<StreamEvent>, event: StreamEvent) -> Result<(), LlmError> {
    events
        .send(event)
        .await
        .map_err(|_| LlmError::ChannelClosed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_load_params_use_all_gpu_layers() {
        let p = ModelLoadParams::default();
        assert_eq!(p.n_gpu_layers, -1);
    }
}
