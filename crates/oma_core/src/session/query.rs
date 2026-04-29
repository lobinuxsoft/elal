//! Discovery over the `<data_root>/sessions/` tree.
//!
//! Three operations the CLI needs:
//! - [`list_sessions`] — every session, optionally filtered by cwd, sorted
//!   by most recent activity (rollout-file mtime).
//! - [`find_latest`] — single most-recent session matching a cwd, for
//!   `oma agent --continue`.
//! - [`locate`] — find a rollout path by [`SessionId`] without opening
//!   every file, for `oma agent --resume <id>`.
//!
//! Reading order: prefer the sidecar `.meta.json` (cheap, single small
//! JSON file) and fall back to a full JSONL replay only when the sidecar
//! is missing. A rollout that fails both paths is silently skipped so a
//! single corrupted file cannot block `oma sessions list`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::{DateTime, Utc};
use oma_protocol::{SessionId, SessionRecord};
use uuid::Uuid;

use crate::error::Result;
use crate::session::replay::load_session;
use crate::session::rollout::RolloutStore;

/// One entry in a session listing — the persisted record plus the
/// rollout-file mtime, which the CLI uses for "last touched" sorting.
#[derive(Debug, Clone)]
pub struct ListedSession {
    /// Canonical session metadata (from sidecar when present, otherwise
    /// reconstructed from the JSONL).
    pub record: SessionRecord,
    /// `mtime` of the rollout file. Every append touches it, so it tracks
    /// real activity even when [`SessionRecord::updated_at`] hasn't moved
    /// (the only writes that bump `updated_at` are title changes).
    pub last_activity: DateTime<Utc>,
}

/// Lists every readable session under `store`, sorted by [`last_activity`]
/// descending (most recent first).
///
/// `cwd_filter` restricts the result to sessions whose
/// [`SessionRecord::cwd`] equals the supplied path. `None` returns every
/// session in the tree.
///
/// Rollouts that cannot be parsed (no sidecar AND replay fails) are
/// silently skipped — corruption of one rollout must never block listing
/// the rest.
pub fn list_sessions(
    store: &RolloutStore,
    cwd_filter: Option<&Path>,
) -> Result<Vec<ListedSession>> {
    let mut entries = Vec::new();
    for rollout in rollout_paths(store)? {
        let Some(record) = read_record(store, &rollout) else {
            continue;
        };
        if let Some(want_cwd) = cwd_filter {
            if record.cwd != want_cwd {
                continue;
            }
        }
        let last_activity = file_mtime_utc(&rollout).unwrap_or(record.updated_at);
        entries.push(ListedSession {
            record,
            last_activity,
        });
    }
    entries.sort_by(|a, b| b.last_activity.cmp(&a.last_activity));
    Ok(entries)
}

/// Returns the most recently active session whose `cwd` equals `cwd`.
/// Used by `oma agent --continue`.
pub fn find_latest(store: &RolloutStore, cwd: &Path) -> Result<Option<ListedSession>> {
    Ok(list_sessions(store, Some(cwd))?.into_iter().next())
}

/// Resolves a [`SessionId`] to its rollout path without opening every
/// file. Walks the tree once and parses the trailing UUID from each
/// rollout filename. Returns `Ok(None)` when no match is found; never
/// errors on a malformed rollout filename, only on directory I/O.
pub fn locate(store: &RolloutStore, id: SessionId) -> Result<Option<PathBuf>> {
    for rollout in rollout_paths(store)? {
        if parse_session_id_from_filename(&rollout) == Some(id) {
            return Ok(Some(rollout));
        }
    }
    Ok(None)
}

/// Walks `<data_root>/sessions/` and returns every `*.jsonl` rollout, in
/// no particular order. Returns an empty vector when the directory does
/// not yet exist.
fn rollout_paths(store: &RolloutStore) -> Result<Vec<PathBuf>> {
    let root = store.data_root().join("sessions");
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    collect_rollouts(&root, &mut out)?;
    Ok(out)
}

fn collect_rollouts(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_rollouts(&path, out)?;
        } else if file_type.is_file() && path.extension().and_then(|e| e.to_str()) == Some("jsonl")
        {
            out.push(path);
        }
    }
    Ok(())
}

/// Sidecar-first record reader. Returns `None` when neither the sidecar
/// nor a JSONL replay yields a usable record.
fn read_record(store: &RolloutStore, rollout: &Path) -> Option<SessionRecord> {
    if let Ok(Some(record)) = store.read_meta_sidecar(rollout) {
        return Some(record);
    }
    load_session(rollout).ok().map(|loaded| loaded.record)
}

fn file_mtime_utc(path: &Path) -> Option<DateTime<Utc>> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    let duration = modified.duration_since(SystemTime::UNIX_EPOCH).ok()?;
    DateTime::<Utc>::from_timestamp(duration.as_secs() as i64, duration.subsec_nanos())
}

/// Extracts the trailing 36-char UUID from a `rollout-<ts>-<uuid>.jsonl`
/// filename. Returns `None` for any other shape — never errors so a
/// stray file in the tree cannot poison [`locate`].
fn parse_session_id_from_filename(path: &Path) -> Option<SessionId> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".jsonl")?;
    if stem.len() < 36 {
        return None;
    }
    let uuid_str = &stem[stem.len() - 36..];
    Uuid::parse_str(uuid_str).ok().map(SessionId)
}

#[cfg(test)]
#[path = "query_tests.rs"]
mod tests;
