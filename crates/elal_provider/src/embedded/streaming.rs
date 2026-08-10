//! Inference loop — takes a loaded provider, a request, and a sender, then
//! drives token sampling until end-of-generation or the max-token budget.
//!
//! Two configurations feed into the same loop:
//!
//! - **Plain path** (no tools requested): each piece becomes a
//!   `StreamEvent::TextDelta`. No grammar, no oaicompat parser.
//! - **Tools path** (tools present): the prompt is built via
//!   `apply_chat_template_with_tools_oaicompat`, the sampler is wrapped
//!   with a grammar sampler when the template generated one, and each
//!   piece is fed through a [`ChatParseStateOaicompat`] whose JSON deltas
//!   are classified into `StreamEvent::{ToolCall*, Reasoning*,
//!   TextDelta}` by a [`DeltaClassifier`].

use std::num::NonZeroU32;
use std::time::Instant;

use elal_protocol::{StopReason, StreamEvent, Usage};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use llama_cpp_2::openai::ChatParseStateOaicompat;
use llama_cpp_2::token::LlamaToken;
use tokio::sync::mpsc::Sender;

use crate::backend::{CompletionRequest, CompletionSummary};
use crate::error::LlmError;
use crate::kv_snapshot;
use crate::oaicompat::DeltaClassifier;

use super::prompt::{ToolPromptSetup, build_oaicompat_prompt, build_plain_prompt};
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
        .new_context(&provider.backend, ctx_params.clone())
        .map_err(|e| LlmError::Context(e.to_string()))?;

    // Try to load a prior KV snapshot if the caller opted in. Failures are
    // logged and swallowed — the snapshot is an optimisation, never the
    // contract; we always have the full prompt to fall back on.
    let mut loaded_tokens: Vec<LlamaToken> = Vec::new();
    if let Some(kv_path) = request.kv_cache_path.as_ref()
        && kv_path.exists()
    {
        match kv_snapshot::load(&mut ctx, kv_path, provider.context_length) {
            Ok(loaded) => {
                tracing::debug!(
                    kv_path = %kv_path.display(),
                    loaded_count = loaded.len(),
                    "kv cache loaded"
                );
                loaded_tokens = loaded;
            }
            Err(err) => {
                tracing::warn!(
                    kv_path = %kv_path.display(),
                    error = %err,
                    "kv cache load failed; falling back to full prompt eval"
                );
            }
        }
    }

    // Build prompt + sink — plain path on empty tools, oaicompat otherwise.
    let use_tools = !request.tools.is_empty();
    let (prompt, mut sink) = if use_tools {
        let setup = build_oaicompat_prompt(
            &provider.model,
            provider.chat_template.as_ref(),
            &request.messages,
            &request.tools,
        )?;
        let prompt = setup.prompt.clone();
        let sink = PieceSink::tools(setup);
        (prompt, sink)
    } else {
        let prompt = build_plain_prompt(
            &provider.model,
            provider.chat_template.as_ref(),
            &request.messages,
        )?;
        (prompt, PieceSink::Plain)
    };

    let grammar_ref = sink.grammar_config();
    let mut sampler = build_sampler(&request.sampling, grammar_ref, &provider.model)?;

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

    // Compute the prefix the loaded snapshot already covers, falling back
    // to a fresh context when the prefix doesn't match the live prompt.
    let prefix_len = if !loaded_tokens.is_empty() && tokens.starts_with(&loaded_tokens) {
        // Always re-decode the last token so logits are fresh for the
        // sampler — saves us reasoning about whether `state_load_file`
        // restored the logits buffer in addition to the KV cache.
        loaded_tokens.len().min(tokens.len().saturating_sub(1))
    } else {
        if !loaded_tokens.is_empty() {
            tracing::warn!(
                loaded = loaded_tokens.len(),
                live = tokens.len(),
                "kv prefix mismatch — recreating context for full prompt eval"
            );
            ctx = provider
                .model
                .new_context(&provider.backend, ctx_params.clone())
                .map_err(|e| LlmError::Context(e.to_string()))?;
        }
        0
    };

    let mut batch = LlamaBatch::new(provider.context_length, 1);
    let to_feed = &tokens[prefix_len..];
    let last_idx = to_feed.len() - 1;
    for (i, token) in to_feed.iter().enumerate() {
        let pos = (prefix_len + i) as i32;
        batch
            .add(*token, pos, &[0], i == last_idx)
            .map_err(|e| LlmError::Decode(e.to_string()))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| LlmError::Decode(e.to_string()))?;

    let prompt_eval_ms = prompt_eval_start.elapsed().as_millis() as u64;
    if prefix_len > 0 {
        tracing::info!(
            kv_prefix = prefix_len,
            decoded = to_feed.len(),
            prompt_eval_ms,
            "kv-cache hit — skipped prompt-eval over prefix"
        );
    }

    // Only emit a top-level TextStart for the plain path — tool runs emit
    // their own bookends per-section via the classifier.
    if !use_tools {
        send(events, StreamEvent::TextStart).await?;
    }

    let generation_start = Instant::now();
    let mut n_generated: u32 = 0;
    let mut cur_pos = tokens.len() as i32;
    let mut generated_tokens: Vec<LlamaToken> = Vec::new();
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

        for event in sink.on_piece(&piece)? {
            send(events, event).await?;
        }

        generated_tokens.push(next);

        batch.clear();
        batch
            .add(next, cur_pos, &[0], true)
            .map_err(|e| LlmError::Decode(e.to_string()))?;
        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Decode(e.to_string()))?;

        n_generated += 1;
        cur_pos += 1;
    }

    // Save the KV state alongside the rollout if the caller opted in.
    // Failures are non-fatal — the next turn falls back to a full eval and
    // logs the problem.
    if let Some(kv_path) = request.kv_cache_path.as_ref() {
        let mut full = tokens.clone();
        full.extend(&generated_tokens);
        match kv_snapshot::save(&ctx, &full, kv_path) {
            Ok(()) => tracing::debug!(
                kv_path = %kv_path.display(),
                tokens = full.len(),
                "kv cache saved"
            ),
            Err(err) => tracing::warn!(
                kv_path = %kv_path.display(),
                error = %err,
                "kv cache save failed"
            ),
        }
    }

    // Drain any events the sink was still holding onto.
    for event in sink.flush()? {
        send(events, event).await?;
    }

    if !use_tools {
        send(events, StreamEvent::TextEnd).await?;
    }

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

/// Per-piece sink that hides the plain vs oaicompat branching from the main
/// loop. The loop only knows `on_piece(&str) -> Vec<StreamEvent>` and
/// `flush() -> Vec<StreamEvent>`; the variant decides what actually happens.
enum PieceSink {
    Plain,
    Tools {
        setup: ToolPromptSetup,
        parse_state: Option<ChatParseStateOaicompat>,
        classifier: DeltaClassifier,
    },
}

impl PieceSink {
    fn tools(mut setup: ToolPromptSetup) -> Self {
        let parse_state = setup.parse_state.take();
        Self::Tools {
            setup,
            parse_state,
            classifier: DeltaClassifier::new(),
        }
    }

    fn grammar_config(&self) -> Option<&super::prompt::GrammarConfig> {
        match self {
            PieceSink::Plain => None,
            PieceSink::Tools { setup, .. } => setup.grammar.as_ref(),
        }
    }

    fn on_piece(&mut self, piece: &str) -> Result<Vec<StreamEvent>, LlmError> {
        match self {
            PieceSink::Plain => {
                if piece.is_empty() {
                    Ok(Vec::new())
                } else {
                    Ok(vec![StreamEvent::TextDelta(piece.to_string())])
                }
            }
            PieceSink::Tools {
                parse_state,
                classifier,
                ..
            } => {
                let mut out = Vec::new();
                if let Some(state) = parse_state.as_mut() {
                    let deltas = state
                        .update(piece, true)
                        .map_err(|e| LlmError::Llama(format!("oaicompat parse: {e}")))?;
                    for delta in deltas {
                        out.extend(classifier.classify(&delta)?);
                    }
                } else if !piece.is_empty() {
                    // Template asked us not to parse tool calls — stream
                    // raw text through as-is.
                    out.push(StreamEvent::TextDelta(piece.to_string()));
                }
                Ok(out)
            }
        }
    }

    fn flush(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        match self {
            PieceSink::Plain => Ok(Vec::new()),
            PieceSink::Tools {
                parse_state,
                classifier,
                ..
            } => {
                let mut out = Vec::new();
                if let Some(state) = parse_state.as_mut() {
                    let deltas = state
                        .update("", false)
                        .map_err(|e| LlmError::Llama(format!("oaicompat flush: {e}")))?;
                    for delta in deltas {
                        out.extend(classifier.classify(&delta)?);
                    }
                }
                out.extend(classifier.flush());
                Ok(out)
            }
        }
    }
}
