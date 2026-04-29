//! Resolves the on-disk data root for `oh-my-agent`.
//!
//! Honors XDG via `dirs::data_dir()`:
//! - Linux: `$XDG_DATA_HOME/oh-my-agent` (default `~/.local/share/oh-my-agent`).
//! - macOS: `~/Library/Application Support/oh-my-agent`.
//! - Windows: `%APPDATA%\oh-my-agent`.
//!
//! The `OMA_DATA_ROOT` environment variable overrides the resolved path —
//! useful for integration tests and ephemeral sandboxes.

use std::path::PathBuf;

use anyhow::{Context, Result};

const APP_DIR_NAME: &str = "oh-my-agent";

/// Returns the configured data root, creating intermediate directories on first
/// use. Errors out when neither `OMA_DATA_ROOT` nor the platform's data dir is
/// available — without one we have nowhere to persist sessions.
pub fn resolve() -> Result<PathBuf> {
    if let Some(env_override) = std::env::var_os("OMA_DATA_ROOT") {
        return Ok(PathBuf::from(env_override));
    }
    let base = dirs::data_dir().context(
        "could not resolve a platform data directory; set OMA_DATA_ROOT to a writable path",
    )?;
    Ok(base.join(APP_DIR_NAME))
}
