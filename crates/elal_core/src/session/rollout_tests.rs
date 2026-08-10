//! Unit tests for `rollout.rs`. Lives alongside the writer to keep the
//! main file under the project's "no monolithic files" threshold while
//! retaining access to private helpers via `#[path = "..."] mod tests;`.

use super::*;
use chrono::TimeZone;
use elal_protocol::{ItemId, ToolCallId, ToolCallItem, TurnId, TurnItem, TurnStatus, TurnUsage};
use std::io::{BufRead, BufReader};
use tempfile::tempdir;

fn fixed_ts() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 29, 14, 32, 11).unwrap()
}

fn make_record(store: &RolloutStore, cwd: PathBuf) -> SessionRecord {
    store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        cwd,
        Some(PathBuf::from("/models/Qwen3-Coder-30B.gguf")),
        ApprovalMode::Smart,
    )
}

fn read_jsonl_lines(path: &Path) -> Vec<RolloutLine> {
    let file = File::open(path).expect("open rollout");
    BufReader::new(file)
        .lines()
        .map(|l| l.expect("read line"))
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).expect("parse line"))
        .collect()
}

#[test]
fn rollout_path_layout_is_yyyy_mm_dd() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let id = SessionId::new();
    let path = store.rollout_path(fixed_ts(), id);
    let suffix = path.strip_prefix(dir.path().join("sessions")).unwrap();
    let parts: Vec<_> = suffix.iter().collect();
    assert_eq!(parts[0], "2026");
    assert_eq!(parts[1], "04");
    assert_eq!(parts[2], "29");
    let leaf = parts[3].to_string_lossy();
    assert!(leaf.starts_with("rollout-"));
    assert!(leaf.ends_with(".jsonl"));
    assert!(leaf.contains(&id.to_string()));
    // No raw colons in the timestamp segment — Windows-friendly path.
    assert!(!leaf.contains(':'));
}

#[test]
fn create_session_record_populates_defaults() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let id = SessionId::new();
    let cwd = PathBuf::from("/var/mnt/DATA/Repos/elal");
    let record = store.create_session_record(
        id,
        fixed_ts(),
        cwd.clone(),
        Some(PathBuf::from("/m.gguf")),
        ApprovalMode::Never,
    );
    assert_eq!(record.id, id);
    assert_eq!(record.created_at, fixed_ts());
    assert_eq!(record.updated_at, fixed_ts());
    assert_eq!(record.cwd, cwd);
    assert_eq!(record.source, "cli");
    assert_eq!(record.approval_mode, ApprovalMode::Never);
    assert_eq!(record.schema_version, SCHEMA_VERSION);
    assert_eq!(record.total_input_tokens, 0);
    assert_eq!(record.title, None);
    assert!(record.model_path.is_some());
    assert!(record.model_sha256.is_none());
    assert_eq!(record.rollout_path, store.rollout_path(fixed_ts(), id));
}

#[test]
fn append_session_meta_creates_directories_and_first_line() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).expect("append meta");
    assert!(record.rollout_path.exists());
    let lines = read_jsonl_lines(&record.rollout_path);
    assert_eq!(lines.len(), 1);
    assert!(matches!(lines[0], RolloutLine::SessionMeta(_)));
}

#[test]
fn full_session_roundtrip_preserves_order() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).unwrap();

    let turn_id = TurnId::new();
    let turn = TurnRecord {
        id: turn_id,
        session_id: record.id,
        sequence: 1,
        started_at: fixed_ts(),
        completed_at: Some(fixed_ts()),
        status: TurnStatus::Completed,
        model_path: record.model_path.clone(),
        usage: Some(TurnUsage {
            input_tokens: 256,
            output_tokens: 128,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }),
        schema_version: SCHEMA_VERSION,
    };
    store.append_turn(&record, turn.clone()).unwrap();

    let item = ItemRecord {
        id: ItemId::new(),
        session_id: record.id,
        turn_id,
        seq: 1,
        timestamp: fixed_ts(),
        items: vec![TurnItem::ToolCall(ToolCallItem {
            tool_call_id: ToolCallId("call_1".into()),
            tool_name: "list_dir".into(),
            input: serde_json::json!({"path": "."}),
        })],
        schema_version: SCHEMA_VERSION,
    };
    store.append_item(&record, item.clone()).unwrap();

    store
        .append_title_update(&record, "fix grammar bug".into(), None)
        .unwrap();

    let lines = read_jsonl_lines(&record.rollout_path);
    assert_eq!(lines.len(), 4);
    assert!(matches!(lines[0], RolloutLine::SessionMeta(_)));
    match &lines[1] {
        RolloutLine::Turn(line) => assert_eq!(line.turn, turn),
        other => panic!("expected Turn, got {other:?}"),
    }
    match &lines[2] {
        RolloutLine::Item(line) => assert_eq!(line.item, item),
        other => panic!("expected Item, got {other:?}"),
    }
    match &lines[3] {
        RolloutLine::SessionTitleUpdated(line) => {
            assert_eq!(line.title, "fix grammar bug");
            assert_eq!(line.previous_title, None);
            assert_eq!(line.session_id, record.id);
        }
        other => panic!("expected SessionTitleUpdated, got {other:?}"),
    }
}

#[test]
fn append_is_truly_append_only() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).unwrap();
    let after_first = std::fs::metadata(&record.rollout_path).unwrap().len();

    store
        .append_title_update(&record, "second".into(), None)
        .unwrap();
    let after_second = std::fs::metadata(&record.rollout_path).unwrap().len();

    assert!(
        after_second > after_first,
        "second append must extend the file, not rewrite it"
    );
}

#[test]
fn meta_sidecar_path_is_alongside_rollout() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = make_record(&store, PathBuf::from("/tmp"));
    let sidecar = store.meta_sidecar_path(&record.rollout_path);
    assert_eq!(
        sidecar.parent(),
        record.rollout_path.parent(),
        "sidecar must live in the same directory as the rollout"
    );
    let leaf = sidecar.file_name().unwrap().to_string_lossy();
    assert!(leaf.ends_with(".jsonl.meta.json"));
}

#[test]
fn meta_sidecar_atomic_roundtrip() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let mut record = make_record(&store, PathBuf::from("/tmp"));
    record.title = Some("rolling".into());
    record.total_input_tokens = 1234;

    store.write_meta_sidecar(&record).expect("write sidecar");
    let loaded = store
        .read_meta_sidecar(&record.rollout_path)
        .expect("read sidecar")
        .expect("sidecar present");
    assert_eq!(loaded, record);

    // Overwriting must replace, not append.
    let mut bumped = record.clone();
    bumped.total_input_tokens = 9999;
    store.write_meta_sidecar(&bumped).unwrap();
    let reloaded = store
        .read_meta_sidecar(&record.rollout_path)
        .unwrap()
        .unwrap();
    assert_eq!(reloaded.total_input_tokens, 9999);
}

#[test]
fn read_meta_sidecar_missing_returns_none() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = make_record(&store, PathBuf::from("/tmp"));
    let result = store
        .read_meta_sidecar(&record.rollout_path)
        .expect("read should succeed even when missing");
    assert!(result.is_none());
}

#[test]
fn meta_sidecar_tmp_file_does_not_linger_on_success() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.write_meta_sidecar(&record).unwrap();
    let sidecar = store.meta_sidecar_path(&record.rollout_path);
    let tmp = with_extension_suffix(&sidecar, "tmp");
    assert!(sidecar.exists(), "final sidecar must exist");
    assert!(
        !tmp.exists(),
        "tmp file must have been renamed away on success"
    );
}

#[test]
fn append_line_writes_single_line_per_call() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).unwrap();
    store
        .append_title_update(&record, "a".into(), None)
        .unwrap();
    store
        .append_title_update(&record, "b".into(), Some("a".into()))
        .unwrap();

    let raw = std::fs::read_to_string(&record.rollout_path).unwrap();
    let line_count = raw.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(line_count, 3);
    for line in raw.lines() {
        // Each line must be exactly one well-formed JSON object — no
        // multi-line pretty-printing leaked from the writer.
        if line.is_empty() {
            continue;
        }
        let parsed = serde_json::from_str::<RolloutLine>(line);
        assert!(parsed.is_ok(), "line not valid: {line}");
    }
}

#[test]
fn data_root_accessor_returns_input() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    assert_eq!(store.data_root(), dir.path());
}
