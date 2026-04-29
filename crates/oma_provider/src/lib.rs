//! `oma_provider` — embedded LLM inference via `llama-cpp-2`.
//!
//! 100% local. No HTTP, no cloud LLMs (ADR-2).
//! Vulkan is the default backend; ROCm is available behind the `rocm` feature.
//!
//! Phase 1b-a scope (#3): baseline text streaming. Tool-call parsing, grammar
//! constraints, and reasoning-tag handling arrive in #3b.

mod backend;
mod capabilities;
mod capabilities_resolver;
mod compute;
mod embedded;
mod error;
pub mod kv_snapshot;
mod oaicompat;
mod sampling;

pub use backend::{CompletionRequest, CompletionSummary, Provider};
pub use capabilities::{
    ChatTemplateOverride, ModelCapabilities, ModelFamily, SpecialTokens, ToolCallingTier,
    resolve_by_filename,
};
pub use capabilities_resolver::resolve_from_model;
pub use compute::ComputeBackend;
pub use embedded::{EmbeddedProvider, ModelLoadParams};
pub use error::LlmError;
pub use kv_snapshot::{KvSnapshotError, compute_model_sha256, kv_path, validate_compatible};
pub use sampling::SamplingControls;

/// Which backend this build activated (compile-time via feature flags).
pub const BACKEND: &str = match ComputeBackend::compiled() {
    ComputeBackend::Vulkan => "vulkan",
    ComputeBackend::Rocm => "rocm",
    ComputeBackend::Dual => "dual",
    ComputeBackend::Cpu => "cpu",
};
