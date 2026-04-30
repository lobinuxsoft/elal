//! Integration tests for `oma sessions {list,show}` over synthetic
//! rollouts seeded into an isolated `OMA_DATA_ROOT`. Covers argument
//! plumbing, sidecar-first listing, cwd filtering, ordering, transcript
//! rendering, and CLI-level error paths — none of which require a GGUF
//! model on disk.
//!
//! The full agent loop (`oma agent --new`/`--continue`) is exercised by
//! `tests/session_resume_gguf.rs`, which is `#[ignore]`-d behind the
//! `OMA_TEST_MODEL` environment variable.

use std::path::PathBuf;

use oma_protocol::SessionId;

mod common;
use common::{Sandbox, Seed, run_oma, seed};

#[test]
fn sessions_list_empty_root_prints_no_sessions() {
    let sb = Sandbox::new();
    let (stdout, stderr, code) = run_oma(sb.cmd().args(["sessions", "list"]));
    assert_eq!(code, 0, "exit code (stderr: {stderr})");
    assert!(
        stdout.contains("No sessions found"),
        "stdout should report empty: {stdout}"
    );
}

#[test]
fn sessions_list_renders_seeded_session() {
    let sb = Sandbox::new();
    let id = seed(
        &sb.store,
        &Seed {
            cwd: PathBuf::from("/cwd-A"),
            user_text: "first prompt".into(),
            agent_text: "first response".into(),
            input_tokens: 42,
            output_tokens: 7,
            hour_offset: 12,
        },
    );

    let (stdout, stderr, code) = run_oma(sb.cmd().args(["sessions", "list"]));
    assert_eq!(code, 0, "exit code (stderr: {stderr})");
    assert!(stdout.contains(&id.to_string()), "uuid present: {stdout}");
    assert!(stdout.contains("42"), "input tokens present: {stdout}");
    assert!(stdout.contains("7"), "output tokens present: {stdout}");
    assert!(stdout.contains("/cwd-A"), "cwd present: {stdout}");
    assert!(stdout.contains("test"), "model file stem present: {stdout}");
}

#[test]
fn sessions_list_cwd_filter_keeps_match_and_drops_others() {
    let sb = Sandbox::new();
    let kept = seed(
        &sb.store,
        &Seed {
            cwd: PathBuf::from("/cwd-keep"),
            hour_offset: 10,
            ..Seed::default()
        },
    );
    let dropped = seed(
        &sb.store,
        &Seed {
            cwd: PathBuf::from("/cwd-skip"),
            hour_offset: 11,
            ..Seed::default()
        },
    );

    let (stdout, stderr, code) = run_oma(sb.cmd().args(["sessions", "list", "--cwd", "/cwd-keep"]));
    assert_eq!(code, 0, "exit code (stderr: {stderr})");
    assert!(
        stdout.contains(&kept.to_string()),
        "kept session must appear: {stdout}"
    );
    assert!(
        !stdout.contains(&dropped.to_string()),
        "filtered session must NOT appear: {stdout}"
    );
}

#[test]
fn sessions_list_orders_by_last_activity_descending() {
    let sb = Sandbox::new();
    let older = seed(
        &sb.store,
        &Seed {
            hour_offset: 8,
            ..Seed::default()
        },
    );
    let newer = seed(
        &sb.store,
        &Seed {
            hour_offset: 16,
            ..Seed::default()
        },
    );

    let (stdout, stderr, code) = run_oma(sb.cmd().args(["sessions", "list"]));
    assert_eq!(code, 0, "exit code (stderr: {stderr})");
    let pos_newer = stdout
        .find(&newer.to_string())
        .expect("newer session in stdout");
    let pos_older = stdout
        .find(&older.to_string())
        .expect("older session in stdout");
    assert!(
        pos_newer < pos_older,
        "newer session must precede older: {stdout}"
    );
}

#[test]
fn sessions_show_renders_transcript() {
    let sb = Sandbox::new();
    let id = seed(
        &sb.store,
        &Seed {
            user_text: "what is 2 + 2".into(),
            agent_text: "four".into(),
            input_tokens: 5,
            output_tokens: 1,
            hour_offset: 4,
            ..Seed::default()
        },
    );

    let (stdout, stderr, code) = run_oma(sb.cmd().args(["sessions", "show", &id.to_string()]));
    assert_eq!(code, 0, "exit code (stderr: {stderr})");
    assert!(stdout.contains(&id.to_string()), "session id: {stdout}");
    assert!(stdout.contains("> what is 2 + 2"), "user msg: {stdout}");
    assert!(stdout.contains("four"), "agent msg: {stdout}");
    assert!(stdout.contains("--- turn 1"), "turn header: {stdout}");
    assert!(stdout.contains("5 in / 1 out"), "usage line: {stdout}");
}

#[test]
fn sessions_show_rejects_invalid_uuid() {
    let sb = Sandbox::new();
    let (_stdout, stderr, code) = run_oma(sb.cmd().args(["sessions", "show", "not-a-uuid"]));
    assert_ne!(code, 0, "non-zero exit on invalid id");
    assert!(
        stderr.contains("invalid session id"),
        "error explains the cause: {stderr}"
    );
}

#[test]
fn sessions_show_reports_unknown_session() {
    let sb = Sandbox::new();
    let bogus = SessionId::new();
    let (_stdout, stderr, code) = run_oma(sb.cmd().args(["sessions", "show", &bogus.to_string()]));
    assert_ne!(code, 0, "non-zero exit on unknown id");
    assert!(
        stderr.contains("not found"),
        "error mentions missing session: {stderr}"
    );
    assert!(
        stderr.contains(&bogus.to_string()),
        "error names the missing id: {stderr}"
    );
}
