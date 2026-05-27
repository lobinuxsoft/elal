//! Error type for `oma_models`.

use oma_provider::LlmError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    /// `use_model(id)` was called with an id not present in the catalog.
    #[error("unknown model id: {0} (catalog ids: {known:?})", known = .1.iter().take(8).collect::<Vec<_>>())]
    UnknownId(String, Vec<String>),

    /// Catalog entry resolved but its `path` does not exist on disk.
    #[error("catalog entry {id} points at {path}, which does not exist")]
    MissingFile {
        id: String,
        path: std::path::PathBuf,
    },

    /// `EmbeddedProvider::load` returned an error — typically the 80%
    /// VRAM budget rule rejecting the load. The underlying message
    /// already explains the cause.
    #[error("failed to load model {id}: {source}")]
    LoadFailed {
        id: String,
        #[source]
        source: LlmError,
    },
}
