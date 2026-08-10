//! Unit tests for `persist.rs`. Lifecycle (begin → items → end), seq
//! monotonicity, sidecar update at turn close, and the pure-function
//! `TurnItem` helpers.

use super::*;
use chrono::{TimeZone, Utc};
use elal_protocol::{ApprovalMode, RolloutLine, SessionId, TurnItem, TurnStatus, TurnUsage, Usage};
use std::fs::{File, read_to_string};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn fixed_ts() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 29, 14, 32, 11).unwrap()
}

fn make_persistence() -> (tempfile::TempDir, RolloutStore, Persistence, SessionId) {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let id = SessionId::new();
    let record = store.create_session_record(
        id,
        fixed_ts(),
        PathBuf::from("/cwd"),
        Some(PathBuf::from("/m.gguf")),
        ApprovalMode::Smart,
    );
    let persistence = Persistence::for_new_session(store.clone(), record);
    (dir, store, persistence, id)
}

fn read_lines(path: &Path) -> Vec<RolloutLine> {
    let file = File::open(path).expect("open rollout");
    BufReader::new(file)
        .lines()
        .map(|l| serde_json::from_str::<RolloutLine>(&l.unwrap()).expect("parse"))
        .collect()
}

#[test]
fn for_new_session_has_zero_seq_and_no_meta_written() {
    let (_dir, _store, p, _id) = make_persistence();
    assert_eq!(p.turn_seq, 0);
    assert_eq!(p.item_seq, 0);
    assert!(!p.meta_written);
    assert!(p.current_turn.is_none());
}

#[test]
fn for_resumed_session_preserves_counters_and_skips_meta() {
    let dir = tempdir().unwrap();
    let store = RolloutStore::new(dir.path().to_path_buf());
    let record = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        PathBuf::from("/cwd"),
        None,
        ApprovalMode::Smart,
    );
    let p = Persistence::for_resumed_session(store, record, 7, 42);
    assert_eq!(p.turn_seq, 7);
    assert_eq!(p.item_seq, 42);
    assert!(p.meta_written);
}

#[test]
fn begin_turn_writes_meta_then_running_line() {
    let (_dir, _store, mut p, id) = make_persistence();
    p.begin_turn(id).unwrap();

    let path = p.record.rollout_path.clone();
    let lines = read_lines(&path);
    assert_eq!(lines.len(), 2);
    assert!(matches!(&lines[0], RolloutLine::SessionMeta(_)));
    let RolloutLine::Turn(turn) = &lines[1] else {
        panic!("expected Turn line, got {:?}", lines[1])
    };
    assert_eq!(turn.turn.status, TurnStatus::Running);
    assert_eq!(turn.turn.sequence, 1);
    assert!(turn.turn.completed_at.is_none());
}

#[test]
fn begin_turn_only_writes_meta_once() {
    let (_dir, _store, mut p, id) = make_persistence();
    p.begin_turn(id).unwrap();
    p.end_turn(TurnStatus::Completed, None).unwrap();
    p.begin_turn(id).unwrap();

    let path = p.record.rollout_path.clone();
    let lines = read_lines(&path);
    let meta_count = lines
        .iter()
        .filter(|l| matches!(l, RolloutLine::SessionMeta(_)))
        .count();
    assert_eq!(meta_count, 1, "meta line must be unique per rollout");
}

#[test]
fn append_item_increments_seq_and_writes_item_line() {
    let (_dir, _store, mut p, id) = make_persistence();
    p.begin_turn(id).unwrap();
    p.append_item(user_item("hi")).unwrap();
    p.append_item(agent_item("there")).unwrap();

    let path = p.record.rollout_path.clone();
    let lines = read_lines(&path);
    let items: Vec<_> = lines
        .iter()
        .filter_map(|l| {
            if let RolloutLine::Item(line) = l {
                Some(line)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].item.seq, 1);
    assert_eq!(items[1].item.seq, 2);
    assert!(matches!(items[0].item.items[0], TurnItem::UserMessage(_)));
    assert!(matches!(items[1].item.items[0], TurnItem::AgentMessage(_)));
}

#[test]
fn append_item_outside_turn_is_silent_noop() {
    let (_dir, _store, mut p, _id) = make_persistence();
    // No begin_turn — append should be a no-op without error.
    p.append_item(user_item("orphan")).unwrap();
    // No file should have been written.
    assert!(!p.record.rollout_path.exists());
}

#[test]
fn end_turn_writes_completed_with_usage_and_clears_state() {
    let (_dir, _store, mut p, id) = make_persistence();
    p.begin_turn(id).unwrap();
    let usage = TurnUsage {
        input_tokens: 100,
        output_tokens: 50,
        ..Default::default()
    };
    p.end_turn(TurnStatus::Completed, Some(usage)).unwrap();
    assert!(p.current_turn.is_none(), "turn must be cleared on end");

    let path = p.record.rollout_path.clone();
    let lines = read_lines(&path);
    let turns: Vec<_> = lines
        .iter()
        .filter_map(|l| {
            if let RolloutLine::Turn(line) = l {
                Some(line)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(turns.len(), 2, "Turn(Running) + Turn(Completed)");
    assert_eq!(turns[0].turn.status, TurnStatus::Running);
    assert_eq!(turns[1].turn.status, TurnStatus::Completed);
    assert!(turns[1].turn.completed_at.is_some());
    assert_eq!(turns[1].turn.usage, Some(usage));
}

#[test]
fn end_turn_updates_sidecar_with_running_totals() {
    let (_dir, store, mut p, id) = make_persistence();
    p.begin_turn(id).unwrap();
    let usage = TurnUsage {
        input_tokens: 200,
        output_tokens: 75,
        ..Default::default()
    };
    p.end_turn(TurnStatus::Completed, Some(usage)).unwrap();

    let sidecar = store
        .read_meta_sidecar(&p.record.rollout_path)
        .unwrap()
        .expect("sidecar must exist after end_turn");
    assert_eq!(sidecar.total_input_tokens, 200);
    assert_eq!(sidecar.total_output_tokens, 75);
    assert!(sidecar.updated_at >= sidecar.created_at);
}

#[test]
fn end_turn_accumulates_totals_across_turns() {
    let (_dir, store, mut p, id) = make_persistence();
    let usage_a = TurnUsage {
        input_tokens: 100,
        output_tokens: 30,
        ..Default::default()
    };
    let usage_b = TurnUsage {
        input_tokens: 250,
        output_tokens: 80,
        ..Default::default()
    };
    p.begin_turn(id).unwrap();
    p.end_turn(TurnStatus::Completed, Some(usage_a)).unwrap();
    p.begin_turn(id).unwrap();
    p.end_turn(TurnStatus::Completed, Some(usage_b)).unwrap();

    let sidecar = store
        .read_meta_sidecar(&p.record.rollout_path)
        .unwrap()
        .unwrap();
    assert_eq!(sidecar.total_input_tokens, 350);
    assert_eq!(sidecar.total_output_tokens, 110);
}

#[test]
fn end_turn_outside_turn_is_silent_noop() {
    let (_dir, _store, mut p, _id) = make_persistence();
    // No begin_turn — should be a no-op.
    p.end_turn(TurnStatus::Completed, None).unwrap();
    assert!(!p.record.rollout_path.exists());
}

#[test]
fn record_first_user_message_only_takes_effect_once() {
    let (_dir, _store, mut p, _id) = make_persistence();
    p.record_first_user_message("first");
    p.record_first_user_message("second");
    assert_eq!(p.record.first_user_message.as_deref(), Some("first"));
}

#[test]
fn tool_call_item_parses_valid_json_arguments() {
    let item = tool_call_item("c1", "read", r#"{"path":"/etc/hostname"}"#);
    let TurnItem::ToolCall(call) = item else {
        panic!("expected ToolCall")
    };
    assert_eq!(call.tool_name, "read");
    assert_eq!(call.tool_call_id.0, "c1");
    assert_eq!(call.input["path"], "/etc/hostname");
}

#[test]
fn tool_call_item_falls_back_to_string_for_invalid_json() {
    let item = tool_call_item("c1", "broken", "this is not json");
    let TurnItem::ToolCall(call) = item else {
        panic!("expected ToolCall")
    };
    assert_eq!(
        call.input,
        serde_json::Value::String("this is not json".into())
    );
}

#[test]
fn tool_call_item_empty_args_parses_as_empty_object() {
    let item = tool_call_item("c1", "noop", "   ");
    let TurnItem::ToolCall(call) = item else {
        panic!("expected ToolCall")
    };
    assert_eq!(call.input, serde_json::json!({}));
}

#[test]
fn tool_result_item_parses_json_output_when_possible() {
    let item = tool_result_item("c1", r#"{"ok":true,"lines":3}"#, false);
    let TurnItem::ToolResult(result) = item else {
        panic!("expected ToolResult")
    };
    assert!(!result.is_error);
    assert_eq!(result.output["ok"], true);
}

#[test]
fn tool_result_item_falls_back_to_string_for_plain_text() {
    let item = tool_result_item("c1", "hello world", true);
    let TurnItem::ToolResult(result) = item else {
        panic!("expected ToolResult")
    };
    assert!(result.is_error);
    assert_eq!(
        result.output,
        serde_json::Value::String("hello world".into())
    );
}

#[test]
fn usage_from_round_widens_u32_to_u64() {
    let usage = Usage {
        prompt_tokens: u32::MAX,
        completion_tokens: 1234,
        reasoning_tokens: 0,
        prompt_eval_ms: 0,
        generation_ms: 0,
    };
    let persisted = usage_from_round(&usage);
    assert_eq!(persisted.input_tokens, u64::from(u32::MAX));
    assert_eq!(persisted.output_tokens, 1234);
    assert_eq!(persisted.cache_creation_input_tokens, 0);
}

#[test]
fn helpers_produce_correct_turn_item_kinds() {
    assert!(matches!(user_item("u"), TurnItem::UserMessage(_)));
    assert!(matches!(agent_item("a"), TurnItem::AgentMessage(_)));
    assert!(matches!(reasoning_item("r"), TurnItem::Reasoning(_)));
    assert!(matches!(
        compaction_item("c"),
        TurnItem::ContextCompaction(_)
    ));
}

#[test]
fn full_turn_emits_meta_running_items_completed_in_order() {
    let (_dir, _store, mut p, id) = make_persistence();
    p.begin_turn(id).unwrap();
    p.append_item(user_item("hi")).unwrap();
    p.append_item(reasoning_item("thinking")).unwrap();
    p.append_item(agent_item("hello")).unwrap();
    p.end_turn(
        TurnStatus::Completed,
        Some(TurnUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        }),
    )
    .unwrap();

    let path = p.record.rollout_path.clone();
    let raw = read_to_string(&path).unwrap();
    assert_eq!(raw.lines().count(), 6);
    let lines = read_lines(&path);
    assert!(matches!(&lines[0], RolloutLine::SessionMeta(_)));
    assert!(matches!(&lines[1], RolloutLine::Turn(t) if t.turn.status == TurnStatus::Running));
    assert!(matches!(&lines[2], RolloutLine::Item(_)));
    assert!(matches!(&lines[3], RolloutLine::Item(_)));
    assert!(matches!(&lines[4], RolloutLine::Item(_)));
    assert!(matches!(&lines[5], RolloutLine::Turn(t) if t.turn.status == TurnStatus::Completed));
}
