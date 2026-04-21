//! Build a [`LlamaSampler`] chain from our [`SamplingControls`], optionally
//! constrained by a grammar produced by the tool-aware chat template.

use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use crate::error::LlmError;
use crate::sampling::SamplingControls;

use super::prompt::GrammarConfig;

/// Build the token sampler chain. When a grammar is supplied, the grammar
/// sampler is prepended so token distributions are masked before
/// `top_k`/`top_p`/etc. take effect. Lazy grammars are activated only once
/// the trigger patterns or tokens match, so non-tool output remains
/// unconstrained.
pub(super) fn build_sampler(
    controls: &SamplingControls,
    grammar: Option<&GrammarConfig>,
    model: &LlamaModel,
) -> Result<LlamaSampler, LlmError> {
    let mut chain: Vec<LlamaSampler> = Vec::new();

    if let Some(cfg) = grammar {
        let g = build_grammar_sampler(cfg, model)?;
        chain.push(g);
    }

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
    Ok(LlamaSampler::chain_simple(chain))
}

fn build_grammar_sampler(
    cfg: &GrammarConfig,
    model: &LlamaModel,
) -> Result<LlamaSampler, LlmError> {
    if !cfg.lazy || (cfg.trigger_patterns.is_empty() && cfg.trigger_tokens.is_empty()) {
        // Non-lazy — or lazy without any triggers, which collapses to plain
        // grammar behaviour — apply the grammar to every token.
        LlamaSampler::grammar(model, &cfg.grammar, "root")
            .map_err(|e| LlmError::Sampling(format!("grammar init failed: {e}")))
    } else {
        LlamaSampler::grammar_lazy_patterns(
            model,
            &cfg.grammar,
            "root",
            &cfg.trigger_patterns,
            &cfg.trigger_tokens,
        )
        .map_err(|e| LlmError::Sampling(format!("lazy grammar init failed: {e}")))
    }
}
