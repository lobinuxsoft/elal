//! Unit tests for `ModelManager` that do NOT require a real GGUF.
//!
//! Anything exercising the actual load path (idempotency, drop on
//! swap, real VRAM accounting) belongs in the smoke test — those
//! cases need a real `EmbeddedProvider::load`, which means a real
//! GGUF on disk and a real GPU backend.

use super::*;
use crate::catalog::{ModelCatalog, ModelEntry};

fn empty_load_params() -> ModelLoadParams {
    ModelLoadParams::default()
}

#[test]
fn current_id_is_none_until_first_load() {
    let mgr = ModelManager::new(ModelCatalog::empty());
    assert!(mgr.current_id().is_none());
    assert!(mgr.current_provider().is_none());
}

#[test]
fn use_model_unknown_id_returns_typed_error() {
    let mut mgr = ModelManager::new(ModelCatalog::empty());
    match mgr.use_model("nope", &empty_load_params()) {
        Err(ModelError::UnknownId(id, _known)) => assert_eq!(id, "nope"),
        Err(other) => panic!("expected UnknownId, got {other:?}"),
        Ok(_) => panic!("unknown id must not load"),
    }
}

#[test]
fn unknown_id_error_includes_known_ids_for_diagnostics() {
    let cat = ModelCatalog::empty()
        .with_entry(ModelEntry::new("a", "A", "/tmp/a.gguf", 1, ""))
        .with_entry(ModelEntry::new("b", "B", "/tmp/b.gguf", 2, ""));
    let mut mgr = ModelManager::new(cat);
    let msg = match mgr.use_model("c", &empty_load_params()) {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("unknown id must not load"),
    };
    assert!(msg.contains("\"a\""), "{msg}");
    assert!(msg.contains("\"b\""), "{msg}");
}

#[test]
fn use_model_missing_file_returns_typed_error() {
    let cat = ModelCatalog::empty().with_entry(ModelEntry::new(
        "ghost",
        "Ghost",
        "/var/oma-test/this-file-does-not-exist.gguf",
        0,
        "",
    ));
    let mut mgr = ModelManager::new(cat);
    match mgr.use_model("ghost", &empty_load_params()) {
        Err(ModelError::MissingFile { id, path }) => {
            assert_eq!(id, "ghost");
            assert_eq!(
                path,
                std::path::PathBuf::from("/var/oma-test/this-file-does-not-exist.gguf")
            );
        }
        Err(other) => panic!("expected MissingFile, got {other:?}"),
        Ok(_) => panic!("missing file must not load"),
    }
    assert!(mgr.current_id().is_none());
}

#[test]
fn register_pending_kv_is_noop_without_current() {
    let mut mgr = ModelManager::new(ModelCatalog::empty());
    mgr.register_pending_kv(std::path::PathBuf::from("/tmp/whatever.kv"));
    // No panic, still no current model.
    assert!(mgr.current_id().is_none());
}

#[test]
fn unload_without_current_is_noop() {
    let mut mgr = ModelManager::new(ModelCatalog::empty());
    mgr.unload();
    mgr.unload();
    assert!(mgr.current_id().is_none());
}

#[test]
fn catalog_accessor_returns_underlying_entries() {
    let cat = ModelCatalog::empty()
        .with_entry(ModelEntry::new("a", "A", "/tmp/a.gguf", 1, ""))
        .with_entry(ModelEntry::new("b", "B", "/tmp/b.gguf", 2, ""));
    let mgr = ModelManager::new(cat);
    let ids = mgr.catalog().ids();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&"a".to_string()));
    assert!(ids.contains(&"b".to_string()));
}
