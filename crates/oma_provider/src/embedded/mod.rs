use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Instant;

use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use oma_protocol::{Message, Role, StopReason, StreamEvent, Usage};
use tokio::sync::mpsc::Sender;

use crate::backend::{CompletionRequest, CompletionSummary, Provider};
use crate::capabilities::{ModelCapabilities, resolve_by_filename};
use crate::compute::ComputeBackend;
use crate::error::LlmError;
use crate::sampling::SamplingControls;

const DEFAULT_MAX_TOKENS: u32 = 2048;

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
            n_batch: 512,
            capabilities: None,
        }
    }
}

pub struct EmbeddedProvider {
    backend: LlamaBackend,
    model: LlamaModel,
    model_name: String,
    model_path: PathBuf,
    capabilities: ModelCapabilities,
    compute: ComputeBackend,
    context_length: usize,
    n_batch: u32,
    chat_template: Option<LlamaChatTemplate>,
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

        let capabilities = params
            .capabilities
            .clone()
            .unwrap_or_else(|| resolve_by_filename(model_path));

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

    fn build_prompt(&self, messages: &[Message]) -> Result<String, LlmError> {
        let tmpl = self
            .chat_template
            .as_ref()
            .ok_or_else(|| LlmError::ChatTemplate("model has no embedded chat template".into()))?;

        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| {
                let role = role_to_string(m.role);
                let content = m.content.clone().unwrap_or_default();
                LlamaChatMessage::new(role, content)
                    .map_err(|e| LlmError::ChatTemplate(e.to_string()))
            })
            .collect::<Result<_, _>>()?;

        self.model
            .apply_chat_template(tmpl, &chat, true)
            .map_err(|e| LlmError::ChatTemplate(e.to_string()))
    }

    async fn stream_completion(
        &self,
        request: &CompletionRequest,
        events: &Sender<StreamEvent>,
    ) -> Result<CompletionSummary, LlmError> {
        let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.context_length as u32))
            .with_n_batch(self.n_batch);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| LlmError::Context(e.to_string()))?;

        let prompt = self.build_prompt(&request.messages)?;
        let prompt_eval_start = Instant::now();

        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| LlmError::Tokenize(e.to_string()))?;

        if tokens.is_empty() {
            return Err(LlmError::Tokenize("prompt yielded zero tokens".into()));
        }

        let prompt_tokens = tokens.len() as u32;
        if prompt_tokens as usize >= self.context_length {
            return Err(LlmError::Decode(format!(
                "prompt ({prompt_tokens} tokens) exceeds context window ({})",
                self.context_length
            )));
        }

        let mut batch = LlamaBatch::new(self.context_length, 1);
        let last_idx = tokens.len() - 1;
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == last_idx;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| LlmError::Decode(e.to_string()))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Decode(e.to_string()))?;

        let prompt_eval_ms = prompt_eval_start.elapsed().as_millis() as u64;
        let mut sampler = build_sampler(&request.sampling);

        send(events, StreamEvent::TextStart).await?;

        let generation_start = Instant::now();
        let mut n_generated: u32 = 0;
        let mut cur_pos = tokens.len() as i32;
        let stop_reason;

        loop {
            if n_generated >= max_tokens {
                stop_reason = StopReason::MaxTokens;
                break;
            }

            let next = sampler.sample(&ctx, -1);
            sampler.accept(next);

            if self.model.is_eog_token(next) {
                stop_reason = StopReason::EndTurn;
                break;
            }

            #[allow(deprecated)]
            let piece = self
                .model
                .token_to_str(next, Special::Tokenize)
                .map_err(|e| LlmError::Sampling(e.to_string()))?;

            if !piece.is_empty() {
                send(events, StreamEvent::TextDelta(piece)).await?;
            }

            batch.clear();
            batch
                .add(next, cur_pos, &[0], true)
                .map_err(|e| LlmError::Decode(e.to_string()))?;
            ctx.decode(&mut batch)
                .map_err(|e| LlmError::Decode(e.to_string()))?;

            n_generated += 1;
            cur_pos += 1;
        }

        send(events, StreamEvent::TextEnd).await?;

        let generation_ms = generation_start.elapsed().as_millis() as u64;
        let usage = Usage {
            prompt_tokens,
            completion_tokens: n_generated,
            reasoning_tokens: 0,
            prompt_eval_ms,
            generation_ms,
        };

        send(
            events,
            StreamEvent::Done {
                stop_reason,
                usage: usage.clone(),
            },
        )
        .await?;

        Ok(CompletionSummary { stop_reason, usage })
    }
}

#[async_trait(?Send)]
impl Provider for EmbeddedProvider {
    async fn chat_completion_stream(
        &self,
        request: CompletionRequest,
        events: Sender<StreamEvent>,
    ) -> Result<CompletionSummary, LlmError> {
        self.stream_completion(&request, &events).await
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

fn role_to_string(role: Role) -> String {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
    .to_string()
}

fn build_sampler(controls: &SamplingControls) -> LlamaSampler {
    let mut chain: Vec<LlamaSampler> = Vec::new();
    if controls.top_k > 0 {
        chain.push(LlamaSampler::top_k(controls.top_k));
    }
    if controls.top_p < 1.0 {
        chain.push(LlamaSampler::top_p(controls.top_p, 1));
    }
    if controls.min_p > 0.0 {
        chain.push(LlamaSampler::min_p(controls.min_p, 1));
    }
    if controls.temperature > 0.0 {
        chain.push(LlamaSampler::temp(controls.temperature));
        chain.push(LlamaSampler::dist(controls.seed));
    } else {
        chain.push(LlamaSampler::greedy());
    }
    LlamaSampler::chain_simple(chain)
}

async fn send(events: &Sender<StreamEvent>, event: StreamEvent) -> Result<(), LlmError> {
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

    #[test]
    fn role_mapping_matches_openai_wire_format() {
        assert_eq!(role_to_string(Role::System), "system");
        assert_eq!(role_to_string(Role::User), "user");
        assert_eq!(role_to_string(Role::Assistant), "assistant");
        assert_eq!(role_to_string(Role::Tool), "tool");
    }
}
