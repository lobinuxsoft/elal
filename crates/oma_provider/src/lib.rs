//! `oma_provider` — embedded LLM inference via `llama-cpp-2`.
//!
//! 100% local. No HTTP, no cloud LLMs (ADR-2).
//! Vulkan is the default backend; ROCm is available behind the `rocm` feature.
//!
//! Phase 1b-a scope (#3): baseline text streaming. Tool-call parsing, grammar
//! constraints, and reasoning-tag handling arrive in #3b.

mod backend;
mod capabilities;
mod compute;
mod embedded;
mod error;
mod sampling;

pub use backend::{CompletionRequest, CompletionSummary, Provider};
pub use capabilities::{
    ChatTemplateOverride, ModelCapabilities, ModelFamily, SpecialTokens, ToolCallingTier,
    resolve_by_filename,
};
pub use compute::ComputeBackend;
pub use embedded::{EmbeddedProvider, ModelLoadParams};
pub use error::LlmError;
pub use sampling::SamplingControls;

/// Which backend this build activated (compile-time via feature flags).
pub const BACKEND: &str = match ComputeBackend::compiled() {
    ComputeBackend::Vulkan => "vulkan",
    ComputeBackend::Rocm => "rocm",
    ComputeBackend::Dual => "dual",
    ComputeBackend::Cpu => "cpu",
};
