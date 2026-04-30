//! Shared helpers for integration tests in `crates/oma_cli/tests/`.
//!
//! Placing this module under `tests/common/` (not `tests/common.rs`) tells
//! Cargo to treat it as a submodule and skip building it as its own
//! integration binary. Each test file then `mod common;` to pull these
//! helpers in.
//!
//! `dead_code` is allowed because Cargo recompiles this module separately
//! for every integration-test binary and not every binary uses every
//! helper — the GGUF-gated smoke test only needs `Sandbox`.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

use chrono::{TimeZone, Utc};
use oma_core::RolloutStore;
use oma_protocol::{
    ApprovalMode, ItemId, ItemRecord, SCHEMA_VERSION, SessionId, TextItem, TurnId, TurnItem,
    TurnRecord, TurnStatus, TurnUsage,
};
use tempfile::{TempDir, tempdir};

/// Path to the compiled `oma` binary, set by Cargo for integration tests.
pub const OMA_BIN: &str = env!("CARGO_BIN_EXE_oma");

/// Isolated `OMA_DATA_ROOT` directory paired with a [`RolloutStore`] rooted
/// at the same path. Drops the temp dir on `Drop`.
pub struct Sandbox {
    _temp: TempDir,
    pub data_root: PathBuf,
    pub store: RolloutStore,
}

impl Sandbox {
    pub fn new() -> Self {
        let temp = tempdir().expect("tempdir");
        let data_root = temp.path().to_path_buf();
        let store = RolloutStore::new(data_root.clone());
        Self {
            _temp: temp,
            data_root,
            store,
        }
    }

    /// Builds the canonical [`Command`] rooted at the compiled `oma` binary
    /// with `OMA_DATA_ROOT` pinned to this sandbox and tracing quiet so
    /// stderr stays focused on real errors.
    pub fn cmd(&self) -> Command {
        let mut cmd = Command::new(OMA_BIN);
        cmd.env("OMA_DATA_ROOT", &self.data_root)
            .env("OMA_LOG", "error");
        cmd
    }
}

#[derive(Clone)]
pub struct Seed {
    pub cwd: PathBuf,
    pub user_text: String,
    pub agent_text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Hour-of-day baked into the rollout filename; lets a single test
    /// order multiple seeds deterministically by `last_activity`.
    pub hour_offset: u32,
}

impl Default for Seed {
    fn default() -> Self {
        Self {
            cwd: PathBuf::from("/cwd-test"),
            user_text: "hello".into(),
            agent_text: "hi back".into(),
            input_tokens: 10,
            output_tokens: 5,
            hour_offset: 12,
        }
    }
}

/// Writes a complete rollout (`SessionMeta` → `Turn(Running)` → user item →
/// agent item → `Turn(Completed)` → atomic sidecar) using only the public
/// [`RolloutStore`] API. Mirrors the line order [`oma_core::agent::Agent`]
/// emits during a real turn so query/replay paths see the same shape they
/// would in production.
pub fn seed(store: &RolloutStore, opts: &Seed) -> SessionId {
    let session_id = SessionId::new();
    let created_at = Utc
        .with_ymd_and_hms(2026, 4, 30, opts.hour_offset, 0, 0)
        .single()
        .expect("hour_offset must be a valid hour");

    let mut record = store.create_session_record(
        session_id,
        created_at,
        opts.cwd.clone(),
        Some(PathBuf::from("/models/test.gguf")),
        ApprovalMode::Never,
    );
    record.first_user_message = Some(opts.user_text.clone());

    store
        .append_session_meta(&record)
        .expect("session meta line");

    let turn_id = TurnId::new();
    store
        .append_turn(
            &record,
            TurnRecord {
                id: turn_id,
                session_id,
                sequence: 1,
                started_at: created_at,
                completed_at: None,
                status: TurnStatus::Running,
                model_path: record.model_path.clone(),
                usage: None,
                schema_version: SCHEMA_VERSION,
            },
        )
        .expect("turn running");

    for (idx, item) in [
        TurnItem::UserMessage(TextItem {
            text: opts.user_text.clone(),
        }),
        TurnItem::AgentMessage(TextItem {
            text: opts.agent_text.clone(),
        }),
    ]
    .into_iter()
    .enumerate()
    {
        let item_record = ItemRecord {
            id: ItemId::new(),
            session_id,
            turn_id,
            seq: (idx as u64) + 1,
            timestamp: created_at,
            items: vec![item],
            schema_version: SCHEMA_VERSION,
        };
        store
            .append_item(&record, item_record)
            .expect("append item");
    }

    let usage = TurnUsage {
        input_tokens: opts.input_tokens,
        output_tokens: opts.output_tokens,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
    };
    store
        .append_turn(
            &record,
            TurnRecord {
                id: turn_id,
                session_id,
                sequence: 1,
                started_at: created_at,
                completed_at: Some(created_at),
                status: TurnStatus::Completed,
                model_path: record.model_path.clone(),
                usage: Some(usage),
                schema_version: SCHEMA_VERSION,
            },
        )
        .expect("turn completed");

    record.total_input_tokens = opts.input_tokens;
    record.total_output_tokens = opts.output_tokens;
    record.updated_at = created_at;
    store.write_meta_sidecar(&record).expect("sidecar");

    session_id
}

/// Runs `cmd` to completion and returns `(stdout, stderr, exit_code)`.
pub fn run_oma(cmd: &mut Command) -> (String, String, i32) {
    let out = cmd.output().expect("spawn oma");
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    let code = out.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}
