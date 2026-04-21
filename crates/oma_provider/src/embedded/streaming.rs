//! Inference loop — takes a loaded provider, a request, and a sender, then
//! drives token sampling until end-of-generation or the max-token budget.

use std::num::NonZeroU32;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use oma_protocol::{StopReason, StreamEvent, Usage};
use tokio::sync::mpsc::Sender;

use crate::backend::{CompletionRequest, CompletionSummary};
use crate::error::LlmError;

use super::prompt::build_plain_prompt;
use super::sampler::build_sampler;
use super::{DEFAULT_MAX_TOKENS, EmbeddedProvider, send};

pub(super) async fn run(
    provider: &EmbeddedProvider,
    request: &CompletionRequest,
    events: &Sender<StreamEvent>,
) -> Result<CompletionSummary, LlmError> {
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(provider.context_length as u32))
        .with_n_batch(provider.n_batch);

    let mut ctx = provider
        .model
        .new_context(&provider.backend, ctx_params)
        .map_err(|e| LlmError::Context(e.to_string()))?;

    let prompt = build_plain_prompt(
        &provider.model,
        provider.chat_template.as_ref(),
        &request.messages,
    )?;
    let prompt_eval_start = Instant::now();

    let tokens = provider
        .model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| LlmError::Tokenize(e.to_string()))?;

    if tokens.is_empty() {
        return Err(LlmError::Tokenize("prompt yielded zero tokens".into()));
    }

    let prompt_tokens = tokens.len() as u32;
    if prompt_tokens as usize >= provider.context_length {
        return Err(LlmError::Decode(format!(
            "prompt ({prompt_tokens} tokens) exceeds context window ({})",
            provider.context_length
        )));
    }

    let mut batch = LlamaBatch::new(provider.context_length, 1);
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

        if provider.model.is_eog_token(next) {
            stop_reason = StopReason::EndTurn;
            break;
        }

        #[allow(deprecated)]
        let piece = provider
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
