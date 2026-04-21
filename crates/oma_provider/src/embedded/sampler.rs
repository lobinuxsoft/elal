//! Build a [`LlamaSampler`] chain from our [`SamplingControls`].

use llama_cpp_2::sampling::LlamaSampler;

use crate::sampling::SamplingControls;

/// Build the token sampler chain. The chain always ends in a distribution or
/// greedy sampler so the caller can invoke `sample(&ctx, -1)` directly.
pub(super) fn build_sampler(controls: &SamplingControls) -> LlamaSampler {
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
