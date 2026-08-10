//! `ModelManager` — single-slot, synchronous, VRAM-bounded model
//! lifecycle.
//!
//! The agent loop holds a `&mut ModelManager` and asks for a specific
//! model by id whenever it needs one. The manager keeps at most one
//! [`EmbeddedProvider`] alive at a time; switching to a different id
//! drops the previous provider before loading the new GGUF.
//!
//! The 80%-of-total VRAM rule is **not** re-implemented here — the
//! provider's [`elal_provider::auto_tune_n_ctx`] already enforces it.
//! `ModelManager` simply maps any [`elal_provider::LlmError`] from the
//! load attempt into [`ModelError::LoadFailed`] so the caller sees a
//! crisp typed error rather than a string-wrapped panic.

use std::path::PathBuf;

use elal_provider::{EmbeddedProvider, ModelLoadParams, Provider};

use crate::catalog::{ModelCatalog, ModelEntry};
use crate::error::ModelError;

/// Holds the single live [`EmbeddedProvider`] plus the bookkeeping
/// the manager needs to flush KV snapshots cleanly across swaps.
struct LoadedModel {
    id: String,
    provider: EmbeddedProvider,
    /// When the agent has opted into KV snapshots, the rollout's
    /// `<rollout>.kv` path is registered here so the manager can log
    /// what is about to be unloaded. The actual file write happens
    /// inside `elal_provider::embedded::streaming::run` at the end of
    /// the prior turn — this field is purely a hint for logging /
    /// invariant checking.
    pending_kv_path: Option<PathBuf>,
}

/// Single-slot model lifecycle manager. Cheap to construct — no I/O
/// happens until [`Self::use_model`] is called.
pub struct ModelManager {
    catalog: ModelCatalog,
    current: Option<LoadedModel>,
}

impl ModelManager {
    /// Build a manager over `catalog`. No model is loaded yet — the
    /// first [`Self::use_model`] call performs the load.
    pub fn new(catalog: ModelCatalog) -> Self {
        Self {
            catalog,
            current: None,
        }
    }

    /// Resolve `id` against the catalog and return a borrow of the
    /// matching loaded provider. Behaviour:
    ///
    /// - Same id as the currently-loaded model → no I/O, returns the
    ///   existing borrow.
    /// - Different id → drops the current provider (releasing its
    ///   VRAM), then loads the new one. The 80%-of-total VRAM budget
    ///   is enforced by [`EmbeddedProvider::load`]; an over-budget
    ///   request surfaces as [`ModelError::LoadFailed`].
    /// - Unknown id → [`ModelError::UnknownId`].
    /// - Path missing on disk → [`ModelError::MissingFile`].
    pub fn use_model(
        &mut self,
        id: &str,
        load_params: &ModelLoadParams,
    ) -> Result<&EmbeddedProvider, ModelError> {
        if self.current.as_ref().is_some_and(|c| c.id == id) {
            // Idempotent path: the borrow re-resolution below is
            // unconditional so the same code services hot and cold
            // hits.
            return Ok(&self.current.as_ref().expect("just checked").provider);
        }

        let entry = self
            .catalog
            .by_id(id)
            .ok_or_else(|| ModelError::UnknownId(id.to_string(), self.catalog.ids()))?
            .clone();

        Self::ensure_path_exists(&entry)?;

        if let Some(prev) = self.current.take() {
            tracing::info!(
                from = %prev.id,
                to = id,
                pending_kv = ?prev.pending_kv_path,
                "model swap: dropping previous provider"
            );
            drop(prev);
        }

        let provider = EmbeddedProvider::load(&entry.path, load_params).map_err(|source| {
            ModelError::LoadFailed {
                id: id.to_string(),
                source,
            }
        })?;

        tracing::info!(
            id = %entry.id,
            display = %entry.display_name,
            path = %entry.path.display(),
            context_length = provider.context_length(),
            auto_tune = %provider.auto_tune().reason,
            "model loaded"
        );

        self.current = Some(LoadedModel {
            id: entry.id,
            provider,
            pending_kv_path: None,
        });

        Ok(&self.current.as_ref().expect("just inserted").provider)
    }

    /// Currently-loaded model id, if any. `None` until the first
    /// successful [`Self::use_model`].
    pub fn current_id(&self) -> Option<&str> {
        self.current.as_ref().map(|c| c.id.as_str())
    }

    /// Currently-loaded provider borrow, if any. Useful when the
    /// caller already knows it just loaded the right model and wants
    /// to skip the catalog round-trip.
    pub fn current_provider(&self) -> Option<&EmbeddedProvider> {
        self.current.as_ref().map(|c| &c.provider)
    }

    /// Tell the manager that a KV snapshot is in flight for the
    /// currently-loaded model. The actual file write happens inside
    /// the streaming layer; this hook is purely for observability —
    /// when the next [`Self::use_model`] swaps the model out, the log
    /// line records what is pending.
    ///
    /// Calling this before any model is loaded is a no-op (we have
    /// no slot to attach the path to).
    pub fn register_pending_kv(&mut self, path: PathBuf) {
        if let Some(current) = self.current.as_mut() {
            current.pending_kv_path = Some(path);
        }
    }

    /// Drop the currently-loaded provider, if any. Equivalent to
    /// swapping to "nothing" — the next [`Self::use_model`] does a
    /// cold load. Used by tests and by orderly shutdown paths.
    pub fn unload(&mut self) {
        if let Some(prev) = self.current.take() {
            tracing::info!(
                id = %prev.id,
                pending_kv = ?prev.pending_kv_path,
                "model manager: explicit unload"
            );
        }
    }

    /// Read-only access to the underlying catalog.
    pub fn catalog(&self) -> &ModelCatalog {
        &self.catalog
    }

    fn ensure_path_exists(entry: &ModelEntry) -> Result<(), ModelError> {
        if entry.path.is_file() {
            Ok(())
        } else {
            Err(ModelError::MissingFile {
                id: entry.id.clone(),
                path: entry.path.clone(),
            })
        }
    }
}

impl Drop for ModelManager {
    fn drop(&mut self) {
        if let Some(current) = self.current.as_ref() {
            tracing::debug!(
                id = %current.id,
                pending_kv = ?current.pending_kv_path,
                "ModelManager dropped while a model was loaded"
            );
        }
    }
}

#[cfg(test)]
#[path = "manager_tests.rs"]
mod tests;
