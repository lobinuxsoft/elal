//! `oma_models` — model lifecycle for the embedded `llama-cpp-2`
//! provider.
//!
//! Slim deadline-cut from #8 + #26: a hardcoded catalog of GGUFs known
//! to live on this dev box, plus a single-slot synchronous
//! [`ModelManager`] that loads and unloads [`oma_provider::EmbeddedProvider`]
//! instances on demand. Multi-slot loading, HF Hub downloads, GGUF
//! metadata inspection, and storage management are explicitly deferred
//! to follow-up issues.
//!
//! The 80%-of-total VRAM budget rule is enforced upstream by
//! [`oma_provider::auto_tune_n_ctx`]; this crate just propagates any
//! resulting [`oma_provider::LlmError`] as a typed [`ModelError`].

mod catalog;
mod error;
mod manager;

pub use catalog::{BUILTIN_MODELS_DIR, ModelCatalog, ModelEntry};
pub use error::ModelError;
pub use manager::ModelManager;
