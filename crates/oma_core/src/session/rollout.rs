//! Append-only JSONL rollout writer.
//!
//! Pattern ported from `claw-code-rust/crates/server/src/persistence.rs::RolloutStore`.
//! Each session owns a single JSONL file under
//! `<data_root>/sessions/<YYYY>/<MM>/<DD>/rollout-<rfc3339-secs>-<id>.jsonl`.
//! Lines are written exactly once and never edited or reordered:
//!
//! 1. The first line is always a [`RolloutLine::SessionMeta`].
//! 2. Subsequent lines are [`RolloutLine::Turn`] / [`RolloutLine::Item`] /
//!    [`RolloutLine::SessionTitleUpdated`] in append order.
//!
//! A sidecar `<rollout>.meta.json` mirrors the most recent [`SessionRecord`]
//! state (mutating fields like `updated_at`, `total_*_tokens`, and `title`)
//! so the query layer (chunk 4) can list sessions cheaply without replaying
//! every JSONL line. The sidecar is overwritten atomically (temp + rename);
//! the JSONL itself is the durable source of truth.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Datelike, SecondsFormat, Utc};
use oma_protocol::{
    ApprovalMode, ItemRecord, RolloutLine, SCHEMA_VERSION, SessionId, SessionMetaLine,
    SessionRecord, SessionTitleUpdatedLine, TurnLine, TurnRecord,
};

use crate::error::Result;

/// Fully-qualified writer for the canonical JSONL rollout journal.
///
/// Construct one per data root; cheap to clone. Internally uses synchronous
/// I/O — every append is a single tiny line, and the agent loop already
/// runs single-threaded with provider futures that are `?Send`, so async
/// I/O would buy nothing here and complicate the error surface.
#[derive(Debug, Clone)]
pub struct RolloutStore {
    data_root: PathBuf,
}

impl RolloutStore {
    /// Builds a store rooted at `data_root`. The `sessions/` subtree is
    /// created lazily when the first rollout is appended.
    pub fn new(data_root: PathBuf) -> Self {
        Self { data_root }
    }

    /// Returns the configured data root.
    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    /// Computes the canonical rollout path for a session created at
    /// `created_at`. Path layout:
    /// `<data_root>/sessions/<YYYY>/<MM>/<DD>/rollout-<rfc3339-secs>-<id>.jsonl`.
    pub fn rollout_path(&self, created_at: DateTime<Utc>, session_id: SessionId) -> PathBuf {
        let partition = self
            .data_root
            .join("sessions")
            .join(format!("{:04}", created_at.year()))
            .join(format!("{:02}", created_at.month()))
            .join(format!("{:02}", created_at.day()));
        let timestamp = created_at
            .to_rfc3339_opts(SecondsFormat::Secs, true)
            .replace(':', "-");
        partition.join(format!("rollout-{timestamp}-{session_id}.jsonl"))
    }

    /// Returns the sidecar metadata path for a given rollout.
    pub fn meta_sidecar_path(&self, rollout_path: &Path) -> PathBuf {
        let mut p = rollout_path.to_path_buf();
        let new_name = match p.file_name().and_then(|n| n.to_str()) {
            Some(name) => format!("{name}.meta.json"),
            None => return p, // pathological — caller passes the rollout path directly.
        };
        p.set_file_name(new_name);
        p
    }

    /// Builds a [`SessionRecord`] with sane defaults wired through. The
    /// returned record's `rollout_path` is what subsequent appends write to.
    pub fn create_session_record(
        &self,
        id: SessionId,
        created_at: DateTime<Utc>,
        cwd: PathBuf,
        model_path: Option<PathBuf>,
        approval_mode: ApprovalMode,
    ) -> SessionRecord {
        SessionRecord {
            id,
            rollout_path: self.rollout_path(created_at, id),
            created_at,
            updated_at: created_at,
            source: "cli".into(),
            model_path,
            model_sha256: None,
            cwd,
            oma_version: env!("CARGO_PKG_VERSION").into(),
            title: None,
            approval_mode,
            total_input_tokens: 0,
            total_output_tokens: 0,
            first_user_message: None,
            schema_version: SCHEMA_VERSION,
        }
    }

    /// Appends the mandatory [`SessionMetaLine`] header. Must be the first
    /// line of every rollout; replay (chunk 3) treats a missing meta as a
    /// fatal error.
    pub fn append_session_meta(&self, record: &SessionRecord) -> Result<()> {
        self.append_line(
            &record.rollout_path,
            &RolloutLine::SessionMeta(Box::new(SessionMetaLine {
                timestamp: Utc::now(),
                session: record.clone(),
            })),
        )
    }

    /// Appends a turn metadata line.
    pub fn append_turn(&self, record: &SessionRecord, turn: TurnRecord) -> Result<()> {
        self.append_line(
            &record.rollout_path,
            &RolloutLine::Turn(TurnLine {
                timestamp: Utc::now(),
                turn,
            }),
        )
    }

    /// Appends an item record line.
    pub fn append_item(&self, record: &SessionRecord, item: ItemRecord) -> Result<()> {
        self.append_line(
            &record.rollout_path,
            &RolloutLine::Item(oma_protocol::ItemLine {
                timestamp: Utc::now(),
                item,
            }),
        )
    }

    /// Appends a title-update line.
    pub fn append_title_update(
        &self,
        record: &SessionRecord,
        title: String,
        previous_title: Option<String>,
    ) -> Result<()> {
        self.append_line(
            &record.rollout_path,
            &RolloutLine::SessionTitleUpdated(SessionTitleUpdatedLine {
                timestamp: Utc::now(),
                session_id: record.id,
                title,
                previous_title,
            }),
        )
    }

    /// Atomically overwrites the sidecar `meta.json` for `record` (write to
    /// `<sidecar>.tmp`, fsync the temp, rename over the final path). The
    /// rename is atomic on the same filesystem; cross-fs writes are not
    /// supported and would surface as an [`std::io::Error`] from `rename`.
    pub fn write_meta_sidecar(&self, record: &SessionRecord) -> Result<()> {
        let final_path = self.meta_sidecar_path(&record.rollout_path);
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = with_extension_suffix(&final_path, "tmp");
        let serialized = serde_json::to_vec_pretty(record)?;
        {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&serialized)?;
            tmp.flush()?;
            tmp.sync_all()?;
        }
        std::fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// Reads the sidecar `meta.json` for the rollout at `rollout_path`,
    /// returning `Ok(None)` when the sidecar does not exist (the rollout
    /// is still authoritative).
    pub fn read_meta_sidecar(&self, rollout_path: &Path) -> Result<Option<SessionRecord>> {
        let path = self.meta_sidecar_path(rollout_path);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let record: SessionRecord = serde_json::from_slice(&bytes)?;
                Ok(Some(record))
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    fn append_line(&self, rollout_path: &Path, line: &RolloutLine) -> Result<()> {
        if let Some(parent) = rollout_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(rollout_path)?;
        serde_json::to_writer(&mut file, line)?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(())
    }
}

pub(super) fn with_extension_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut p = path.to_path_buf();
    let new_name = match p.file_name().and_then(|n| n.to_str()) {
        Some(name) => format!("{name}.{suffix}"),
        None => return p,
    };
    p.set_file_name(new_name);
    p
}

#[cfg(test)]
#[path = "rollout_tests.rs"]
mod tests;
