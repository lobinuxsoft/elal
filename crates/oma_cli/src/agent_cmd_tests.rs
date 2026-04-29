//! Unit tests for `agent_cmd.rs`. Cover the pure helpers (`parse_resume_mode`,
//! `resolve_model_path`, `decide_resume_kv_path`); the `repl` and the actual
//! `run` entrypoint live in integration territory (chunk 8).

use super::*;
use oma_core::SessionState;
use oma_protocol::{ApprovalMode, SessionRecord};

fn args_default() -> AgentArgs {
    AgentArgs {
        model: None,
        system: "sys".into(),
        n_gpu_layers: -1,
        resume: None,
        continue_: false,
        new: false,
        save_kv_cache: false,
    }
}

fn make_loaded(model_path: Option<PathBuf>, sha: Option<String>) -> LoadedSession {
    let record = SessionRecord {
        id: SessionId::new(),
        rollout_path: PathBuf::from("/tmp/rollout.jsonl"),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        source: "cli".into(),
        model_path,
        model_sha256: sha,
        cwd: PathBuf::from("/cwd"),
        oma_version: "0.0.0".into(),
        title: None,
        approval_mode: ApprovalMode::Never,
        total_input_tokens: 0,
        total_output_tokens: 0,
        first_user_message: None,
        schema_version: oma_protocol::SCHEMA_VERSION,
    };
    LoadedSession {
        record,
        state: SessionState::new(Default::default(), PathBuf::from("/cwd")),
        last_turn_seq: 0,
        last_item_seq: 0,
    }
}

#[test]
fn parse_resume_mode_defaults_to_new() {
    let mode = parse_resume_mode(&args_default()).unwrap();
    assert!(matches!(mode, ResumeMode::New));
}

#[test]
fn parse_resume_mode_recognises_continue() {
    let mut args = args_default();
    args.continue_ = true;
    let mode = parse_resume_mode(&args).unwrap();
    assert!(matches!(mode, ResumeMode::Continue));
}

#[test]
fn parse_resume_mode_parses_resume_uuid() {
    let id = SessionId::new();
    let mut args = args_default();
    args.resume = Some(id.to_string());
    let mode = parse_resume_mode(&args).unwrap();
    match mode {
        ResumeMode::Resume(parsed) => assert_eq!(parsed, id),
        _ => panic!("expected Resume variant"),
    }
}

#[test]
fn parse_resume_mode_rejects_invalid_uuid() {
    let mut args = args_default();
    args.resume = Some("not-a-uuid".into());
    let err = parse_resume_mode(&args).unwrap_err();
    assert!(format!("{err}").contains("--resume expects a session UUID"));
}

#[test]
fn resolve_model_path_prefers_explicit() {
    let explicit = PathBuf::from("/models/explicit.gguf");
    let resolved = resolve_model_path(Some(&explicit), None).unwrap();
    assert_eq!(resolved, explicit);
}

#[test]
fn resolve_model_path_errors_when_no_inputs() {
    let err = resolve_model_path(None, None).unwrap_err();
    assert!(format!("{err}").contains("required for new sessions"));
}

#[test]
fn resolve_model_path_errors_when_recorded_path_missing() {
    let bogus = PathBuf::from("/nonexistent/model.gguf");
    let loaded = make_loaded(Some(bogus.clone()), None);
    let err = resolve_model_path(None, Some(&loaded)).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("model path no longer exists"));
    assert!(msg.contains(&bogus.display().to_string()));
}

#[test]
fn decide_resume_kv_path_returns_none_when_flag_off() {
    let loaded = make_loaded(None, None);
    assert!(decide_resume_kv_path(false, &loaded, std::path::Path::new("/m.gguf")).is_none());
}

#[test]
fn decide_resume_kv_path_enables_path_when_no_recorded_sha() {
    // Older sessions may not have a SHA — we still allow KV from now on,
    // we just can't validate compatibility for the existing snapshot.
    let loaded = make_loaded(None, None);
    let decision = decide_resume_kv_path(true, &loaded, std::path::Path::new("/m.gguf"));
    assert!(decision.is_some());
}

#[test]
fn decide_resume_kv_path_refuses_on_sha_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    std::fs::write(&model_path, b"current-content").unwrap();
    let bogus_sha = "0".repeat(64);
    let loaded = make_loaded(Some(model_path.clone()), Some(bogus_sha));
    let decision = decide_resume_kv_path(true, &loaded, &model_path);
    assert!(
        decision.is_none(),
        "SHA mismatch must disable kv-cache load"
    );
}

#[test]
fn decide_resume_kv_path_accepts_matching_sha() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    std::fs::write(&model_path, b"deterministic-bytes").unwrap();
    let sha = compute_model_sha256(&model_path).unwrap();
    let loaded = make_loaded(Some(model_path.clone()), Some(sha));
    let decision = decide_resume_kv_path(true, &loaded, &model_path);
    assert!(decision.is_some());
}
