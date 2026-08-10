//! Resolves the on-disk data root for `elal`.
//!
//! Honors XDG via `dirs::data_dir()`:
//! - Linux: `$XDG_DATA_HOME/elal` (default `~/.local/share/elal`).
//! - macOS: `~/Library/Application Support/elal`.
//! - Windows: `%APPDATA%\elal`.
//!
//! The `ELAL_DATA_ROOT` environment variable overrides the resolved path —
//! useful for integration tests and ephemeral sandboxes.

use std::path::PathBuf;

use anyhow::{Context, Result};

const APP_DIR_NAME: &str = "elal";

/// Directory used before the rename to `elal`. Honored when it exists and the
/// current one does not: downloaded models are gigabytes, and silently pointing
/// at an empty root would look like the install lost them.
const LEGACY_DIR_NAME: &str = "oh-my-agent";

/// Returns the configured data root, creating intermediate directories on first
/// use. Errors out when neither `ELAL_DATA_ROOT` nor the platform's data dir is
/// available — without one we have nowhere to persist sessions.
pub fn resolve() -> Result<PathBuf> {
    if let Some(env_override) = std::env::var_os("ELAL_DATA_ROOT") {
        return Ok(PathBuf::from(env_override));
    }
    let base = dirs::data_dir().context(
        "could not resolve a platform data directory; set ELAL_DATA_ROOT to a writable path",
    )?;
    let current = base.join(APP_DIR_NAME);
    let legacy = base.join(LEGACY_DIR_NAME);
    if !current.exists() && legacy.exists() {
        return Ok(legacy);
    }
    Ok(current)
}
