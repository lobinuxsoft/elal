//! Hardcoded catalog of GGUFs known to live on disk for this project.
//!
//! The original #8 plan envisioned an HF-Hub-backed downloader,
//! GGUF metadata inspection, and per-VRAM recommendation. All of that
//! is deferred — for the deadline cut we only need a stable mapping
//! from `id → on-disk path` so `ModelManager::use_model` has something
//! to resolve.
//!
//! Adding a new model is a one-line edit to [`ModelCatalog::builtin`].
//! The catalog deliberately rejects duplicate ids at construction time
//! so a typo can't silently shadow an existing entry.

use std::path::{Path, PathBuf};

/// One catalog row. The minimum the manager needs to swap models on
/// demand: a stable id, a filesystem path, and human-facing metadata
/// for logs / `elal sessions` output.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Stable, kebab-case slug (e.g. `qwen3-1.7b-q5`).
    pub id: String,
    /// Display name for logs and CLI output.
    pub display_name: String,
    /// Absolute path to the GGUF on disk. The manager fails fast if
    /// this does not exist at lookup time — better a clear error than
    /// llama.cpp's cryptic load-failure path.
    pub path: PathBuf,
    /// Approximate file size in bytes. Stored verbatim from the
    /// catalog declaration; not re-read on every lookup.
    pub size_bytes: u64,
    /// Free-form one-line description shown in CLI listings.
    pub description: String,
}

impl ModelEntry {
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        path: impl Into<PathBuf>,
        size_bytes: u64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            path: path.into(),
            size_bytes,
            description: description.into(),
        }
    }
}

/// Read-only catalog of [`ModelEntry`]. Lookups are O(n) — the catalog
/// is tiny and lookup happens once per model swap.
#[derive(Debug, Clone)]
pub struct ModelCatalog {
    entries: Vec<ModelEntry>,
}

impl ModelCatalog {
    /// Construct an empty catalog. Useful for tests that inject their
    /// own entries via [`Self::with_entry`].
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Construct a catalog from an explicit list. Panics on duplicate
    /// ids — silently shadowing entries is exactly the kind of bug
    /// that survives months unnoticed, so we crash early.
    pub fn from_entries(entries: Vec<ModelEntry>) -> Self {
        let mut seen = std::collections::HashSet::new();
        for e in &entries {
            assert!(seen.insert(e.id.clone()), "duplicate catalog id: {}", e.id);
        }
        Self { entries }
    }

    /// Builder helper for tests / dynamic catalogs.
    #[must_use]
    pub fn with_entry(mut self, entry: ModelEntry) -> Self {
        assert!(
            self.entries.iter().all(|e| e.id != entry.id),
            "duplicate catalog id: {}",
            entry.id
        );
        self.entries.push(entry);
        self
    }

    /// Hardcoded catalog of GGUFs known to live on this dev box.
    /// Update [`crate::catalog::BUILTIN_MODELS_DIR`] if the layout
    /// moves. Adding a new model is a one-line edit here.
    pub fn builtin() -> Self {
        let dir = Path::new(BUILTIN_MODELS_DIR);
        Self::from_entries(vec![
            ModelEntry::new(
                "qwen3-1.7b-q5",
                "Qwen3-1.7B Q5_K_M",
                dir.join("build/linux/models/Qwen3-1.7B.Q5_K_M.gguf"),
                1_400_000_000,
                "Compact chat model. Native tool format Qwen3, no tool training.",
            ),
            ModelEntry::new(
                "qwen3-coder-30b-a3b-q3_k_s",
                "Qwen3-Coder-30B-A3B Q3_K_S",
                dir.join("models/Qwen3-Coder-30B-A3B-Instruct-Q3_K_S.gguf"),
                13_292_471_456,
                "30B MoE, 3B active. Native tool calling, fits 16 GiB at n_ctx 4K under the 80% rule.",
            ),
        ])
    }

    pub fn entries(&self) -> &[ModelEntry] {
        &self.entries
    }

    pub fn by_id(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn ids(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.id.clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::empty()
    }
}

/// Root directory the [`ModelCatalog::builtin`] entries are anchored
/// at. Hardcoded for the deadline cut — auto-discovery / config
/// override land in a follow-up issue.
pub const BUILTIN_MODELS_DIR: &str = "/var/mnt/DATA/Repos/OhMyDialogSystem";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_has_known_ids() {
        let cat = ModelCatalog::builtin();
        assert!(cat.by_id("qwen3-1.7b-q5").is_some());
        assert!(cat.by_id("qwen3-coder-30b-a3b-q3_k_s").is_some());
    }

    #[test]
    fn builtin_paths_are_absolute() {
        for e in ModelCatalog::builtin().entries() {
            assert!(e.path.is_absolute(), "path must be absolute: {:?}", e.path);
        }
    }

    #[test]
    fn unknown_id_returns_none() {
        let cat = ModelCatalog::builtin();
        assert!(cat.by_id("does-not-exist").is_none());
    }

    #[test]
    fn ids_lists_all_entries() {
        let cat = ModelCatalog::builtin();
        let ids = cat.ids();
        assert_eq!(ids.len(), cat.len());
        assert!(ids.iter().any(|i| i == "qwen3-1.7b-q5"));
    }

    #[test]
    #[should_panic(expected = "duplicate catalog id")]
    fn duplicate_ids_panic_at_construction() {
        let entry = ModelEntry::new("dup", "A", "/tmp/a", 1, "");
        let dup = ModelEntry::new("dup", "B", "/tmp/b", 1, "");
        let _ = ModelCatalog::from_entries(vec![entry, dup]);
    }

    #[test]
    fn empty_catalog_round_trip() {
        let cat = ModelCatalog::empty();
        assert!(cat.is_empty());
        assert_eq!(cat.len(), 0);
        assert!(cat.ids().is_empty());
    }

    #[test]
    fn with_entry_extends_catalog() {
        let cat = ModelCatalog::empty()
            .with_entry(ModelEntry::new("a", "A", "/tmp/a", 1, ""))
            .with_entry(ModelEntry::new("b", "B", "/tmp/b", 2, ""));
        assert_eq!(cat.len(), 2);
        assert!(cat.by_id("a").is_some());
        assert!(cat.by_id("b").is_some());
    }
}
