//! Unit tests for `replay.rs`. End-to-end roundtrip vs `RolloutStore`,
//! tail-truncation tolerance, header-validation errors, and per-item
//! `TurnItem → Message` mapping.

use super::*;
use crate::session::RolloutStore;
use chrono::{DateTime, TimeZone, Utc};
use elal_protocol::{
    ApprovalMode, ItemId, ItemRecord, ReasoningItem, SCHEMA_VERSION, SessionId, TextItem,
    ToolCallId, ToolCallItem, ToolResultItem, TurnId, TurnItem, TurnRecord, TurnStatus, TurnUsage,
};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use tempfile::tempdir;

fn fixed_ts() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 29, 14, 32, 11).unwrap()
}

fn make_store() -> (tempfile::TempDir, RolloutStore) {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    (dir, store)
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

fn make_turn(record: &SessionRecord, sequence: u32, input: u64, output: u64) -> TurnRecord {
    TurnRecord {
        id: TurnId::new(),
        session_id: record.id,
        sequence,
        started_at: fixed_ts(),
        completed_at: Some(fixed_ts()),
        status: TurnStatus::Completed,
        model_path: record.model_path.clone(),
        usage: Some(TurnUsage {
            input_tokens: input,
            output_tokens: output,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }),
        schema_version: SCHEMA_VERSION,
    }
}

fn make_item(
    record: &SessionRecord,
    turn_id: TurnId,
    seq: u64,
    items: Vec<TurnItem>,
) -> ItemRecord {
    ItemRecord {
        id: ItemId::new(),
        session_id: record.id,
        turn_id,
        seq,
        timestamp: fixed_ts(),
        items,
        schema_version: SCHEMA_VERSION,
    }
}

#[test]
fn roundtrip_preserves_record_and_state() {
    let (_dir, store) = make_store();
    let record = make_record(&store, PathBuf::from("/projects/elal"));
    store.append_session_meta(&record).unwrap();

    let turn = make_turn(&record, 1, 256, 128);
    let turn_id = turn.id;
    store.append_turn(&record, turn).unwrap();

    let user_item = make_item(
        &record,
        turn_id,
        1,
        vec![TurnItem::UserMessage(TextItem {
            text: "list /tmp".into(),
        })],
    );
    store.append_item(&record, user_item).unwrap();

    let assistant_item = make_item(
        &record,
        turn_id,
        2,
        vec![
            TurnItem::Reasoning(ReasoningItem {
                text: "I should use list_dir".into(),
            }),
            TurnItem::ToolCall(ToolCallItem {
                tool_call_id: ToolCallId("call_1".into()),
                tool_name: "list_dir".into(),
                input: serde_json::json!({"path": "/tmp"}),
            }),
        ],
    );
    store.append_item(&record, assistant_item).unwrap();

    let tool_result_item = make_item(
        &record,
        turn_id,
        3,
        vec![TurnItem::ToolResult(ToolResultItem {
            tool_call_id: ToolCallId("call_1".into()),
            output: serde_json::json!(["a.txt", "b.log"]),
            is_error: false,
        })],
    );
    store.append_item(&record, tool_result_item).unwrap();

    let final_item = make_item(
        &record,
        turn_id,
        4,
        vec![TurnItem::AgentMessage(TextItem {
            text: "Found a.txt and b.log".into(),
        })],
    );
    store.append_item(&record, final_item).unwrap();

    let loaded = load_session(&record.rollout_path).expect("replay");
    assert_eq!(loaded.record.id, record.id);
    assert_eq!(loaded.record.cwd, record.cwd);
    assert_eq!(loaded.record.approval_mode, record.approval_mode);

    assert_eq!(loaded.state.turn_count, 1);
    assert_eq!(loaded.state.total_input_tokens, 256);
    assert_eq!(loaded.state.total_output_tokens, 128);
    assert_eq!(loaded.state.last_input_tokens, 256);

    // 5 messages: user, reasoning, tool_call, tool_result, agent text.
    assert_eq!(loaded.state.messages.len(), 5);
    assert_eq!(loaded.state.messages[0].role, Role::User);
    assert_eq!(
        loaded.state.messages[0].content.as_deref(),
        Some("list /tmp")
    );

    assert_eq!(loaded.state.messages[1].role, Role::Assistant);
    assert!(loaded.state.messages[1].content.is_none());
    assert_eq!(
        loaded.state.messages[1].reasoning_content.as_deref(),
        Some("I should use list_dir"),
    );

    assert_eq!(loaded.state.messages[2].role, Role::Assistant);
    let calls = loaded.state.messages[2].tool_calls.as_ref().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "list_dir");
    assert_eq!(calls[0].id, "call_1");
    assert!(calls[0].function.arguments.contains("/tmp"));

    assert_eq!(loaded.state.messages[3].role, Role::Tool);
    assert_eq!(
        loaded.state.messages[3].tool_call_id.as_deref(),
        Some("call_1")
    );
    assert!(
        loaded.state.messages[3]
            .content
            .as_deref()
            .unwrap()
            .contains("a.txt")
    );

    assert_eq!(loaded.state.messages[4].role, Role::Assistant);
    assert_eq!(
        loaded.state.messages[4].content.as_deref(),
        Some("Found a.txt and b.log"),
    );
}

#[test]
fn token_totals_accumulate_across_turns() {
    let (_dir, store) = make_store();
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).unwrap();
    store
        .append_turn(&record, make_turn(&record, 1, 100, 50))
        .unwrap();
    store
        .append_turn(&record, make_turn(&record, 2, 200, 80))
        .unwrap();
    store
        .append_turn(&record, make_turn(&record, 3, 300, 120))
        .unwrap();

    let loaded = load_session(&record.rollout_path).unwrap();
    assert_eq!(loaded.state.turn_count, 3);
    assert_eq!(loaded.state.total_input_tokens, 600);
    assert_eq!(loaded.state.total_output_tokens, 250);
    // last_input_tokens reflects the most recent turn.
    assert_eq!(loaded.state.last_input_tokens, 300);
}

#[test]
fn title_update_is_applied_to_record() {
    let (_dir, store) = make_store();
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).unwrap();
    store
        .append_title_update(&record, "fix grammar bug".into(), None)
        .unwrap();
    store
        .append_title_update(&record, "ship #25".into(), Some("fix grammar bug".into()))
        .unwrap();

    let loaded = load_session(&record.rollout_path).unwrap();
    // Most recent title wins.
    assert_eq!(loaded.record.title.as_deref(), Some("ship #25"));
}

#[test]
fn context_compaction_item_maps_to_marker_message() {
    let (_dir, store) = make_store();
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).unwrap();

    let turn_id = TurnId::new();
    let item = make_item(
        &record,
        turn_id,
        1,
        vec![TurnItem::ContextCompaction(TextItem {
            text: "summary of earlier discussion".into(),
        })],
    );
    store.append_item(&record, item).unwrap();

    let loaded = load_session(&record.rollout_path).unwrap();
    assert_eq!(loaded.state.messages.len(), 1);
    assert_eq!(loaded.state.messages[0].role, Role::System);
    let content = loaded.state.messages[0].content.as_deref().unwrap();
    assert!(content.starts_with("[compacted history]"));
    assert!(content.contains("summary of earlier discussion"));
}

#[test]
fn empty_file_returns_session_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.jsonl");
    std::fs::File::create(&path).unwrap();
    let err = load_session(&path).expect_err("empty file must error");
    assert!(matches!(err, ElalError::Session(_)));
}

#[test]
fn missing_file_returns_io_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("does-not-exist.jsonl");
    let err = load_session(&path).expect_err("missing file must error");
    assert!(matches!(err, ElalError::Io(_)));
}

#[test]
fn first_line_not_meta_returns_session_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.jsonl");
    let mut f = std::fs::File::create(&path).unwrap();
    // Build a valid TurnLine and write it as the first line.
    let session_id = SessionId::new();
    let bogus = elal_protocol::TurnLine {
        timestamp: fixed_ts(),
        turn: TurnRecord {
            id: TurnId::new(),
            session_id,
            sequence: 1,
            started_at: fixed_ts(),
            completed_at: None,
            status: TurnStatus::Pending,
            model_path: None,
            usage: None,
            schema_version: SCHEMA_VERSION,
        },
    };
    let line = serde_json::to_string(&RolloutLine::Turn(bogus)).unwrap();
    writeln!(f, "{line}").unwrap();
    drop(f);

    let err = load_session(&path).expect_err("non-meta first line must error");
    match err {
        ElalError::Session(msg) => assert!(msg.contains("session_meta")),
        other => panic!("expected Session error, got {other:?}"),
    }
}

#[test]
fn truncated_tail_line_is_tolerated() {
    let (_dir, store) = make_store();
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).unwrap();
    store
        .append_turn(&record, make_turn(&record, 1, 100, 50))
        .unwrap();
    // Append a deliberately malformed final line — what a kill-mid-write
    // looks like in practice.
    let mut f = OpenOptions::new()
        .append(true)
        .open(&record.rollout_path)
        .unwrap();
    f.write_all(b"{\"kind\":\"turn\",\"timesta").unwrap();
    drop(f);

    let loaded = load_session(&record.rollout_path).expect("tail truncation must be tolerated");
    assert_eq!(loaded.state.turn_count, 1);
    assert_eq!(loaded.state.total_input_tokens, 100);
}

#[test]
fn malformed_non_tail_line_is_fatal() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad-mid.jsonl");
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        PathBuf::from("/tmp"),
        None,
        ApprovalMode::Smart,
    );
    let meta_line = serde_json::to_string(&RolloutLine::SessionMeta(Box::new(
        elal_protocol::SessionMetaLine {
            timestamp: fixed_ts(),
            session: record.clone(),
        },
    )))
    .unwrap();
    let valid_turn = make_turn(&record, 1, 100, 50);
    let valid_turn_line = serde_json::to_string(&RolloutLine::Turn(elal_protocol::TurnLine {
        timestamp: fixed_ts(),
        turn: valid_turn,
    }))
    .unwrap();

    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "{meta_line}").unwrap();
    writeln!(f, "{{\"kind\":\"turn\",\"oops").unwrap(); // malformed mid-file
    writeln!(f, "{valid_turn_line}").unwrap();
    drop(f);

    let err = load_session(&path).expect_err("non-tail malformed line must error");
    assert!(matches!(err, ElalError::Json(_)));
}

#[test]
fn empty_lines_in_rollout_are_skipped() {
    let (_dir, store) = make_store();
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).unwrap();
    // Inject blank lines between records.
    let mut f = OpenOptions::new()
        .append(true)
        .open(&record.rollout_path)
        .unwrap();
    f.write_all(b"\n\n").unwrap();
    drop(f);
    store
        .append_turn(&record, make_turn(&record, 1, 100, 50))
        .unwrap();

    let loaded = load_session(&record.rollout_path).expect("blank lines tolerated");
    assert_eq!(loaded.state.turn_count, 1);
}

#[test]
fn duplicate_session_meta_is_ignored() {
    let (_dir, store) = make_store();
    let record = make_record(&store, PathBuf::from("/tmp"));
    store.append_session_meta(&record).unwrap();
    store.append_session_meta(&record).unwrap(); // intentional duplicate
    store
        .append_turn(&record, make_turn(&record, 1, 50, 25))
        .unwrap();

    let loaded = load_session(&record.rollout_path).expect("duplicate meta tolerated");
    assert_eq!(loaded.record.id, record.id);
    assert_eq!(loaded.state.turn_count, 1);
}

#[test]
fn last_seq_counters_reflect_highest_emitted_values() {
    let (_dir, store) = make_store();
    let record = make_record(&store, PathBuf::from("/cwd"));
    store.append_session_meta(&record).unwrap();
    store
        .append_turn(&record, make_turn(&record, 1, 10, 5))
        .unwrap();
    let item_record = make_item(
        &record,
        TurnId::new(),
        7,
        vec![TurnItem::UserMessage(TextItem { text: "hi".into() })],
    );
    store.append_item(&record, item_record).unwrap();
    store
        .append_turn(&record, make_turn(&record, 2, 20, 10))
        .unwrap();

    let loaded = load_session(&record.rollout_path).unwrap();
    assert_eq!(loaded.last_turn_seq, 2);
    assert_eq!(loaded.last_item_seq, 7);
}

#[test]
fn empty_rollout_after_meta_yields_zero_seq_counters() {
    let (_dir, store) = make_store();
    let record = make_record(&store, PathBuf::from("/cwd"));
    store.append_session_meta(&record).unwrap();
    let loaded = load_session(&record.rollout_path).unwrap();
    assert_eq!(loaded.last_turn_seq, 0);
    assert_eq!(loaded.last_item_seq, 0);
}
