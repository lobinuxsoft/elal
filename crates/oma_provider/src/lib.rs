//! `oma_provider` — embedded LLM inference via `llama-cpp-2`.
//!
//! 100% local. No HTTP, no cloud LLMs (ADR-2).
//! Vulkan is the default backend; ROCm is available behind the `rocm` feature.
//!
//! See GitHub issue #3 for the implementation roadmap.

#[cfg(all(feature = "vulkan", not(feature = "dual")))]
pub const BACKEND: &str = "vulkan";

#[cfg(all(feature = "rocm", not(feature = "dual")))]
pub const BACKEND: &str = "rocm";

#[cfg(feature = "dual")]
pub const BACKEND: &str = "dual";
