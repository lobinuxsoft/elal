//! Integration tests for the [`Agent`] ↔ [`RolloutStore`] wiring.
//!
//! Drives `Agent::run_turn` against scripted providers, then verifies the
//! JSONL rollout matches the expected sequence of `SessionMeta` / `Turn` /
//! `Item` lines and that the sidecar reflects running totals.

use oma_protocol::{
    ApprovalMode, RolloutLine, SessionId, StopReason, StreamEvent, TurnItem, TurnStatus,
};
use oma_tools::ToolRegistry;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::sync::mpsc;

use super::test_helpers::{
    CompactionProvider, EchoTool, ScriptedProvider, done, drain, fixed_ts, make_store, read_lines,
};
use super::{Agent, AgentEvent, UserAction};
use crate::session::TokenBudget;

#[tokio::test]
async fn plain_turn_writes_meta_running_user_agent_completed() {
    let dir = tempdir().unwrap();
    let store = make_store(dir.path());
    let record = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        PathBuf::from("/cwd"),
        Some(PathBuf::from("/m.gguf")),
        ApprovalMode::Smart,
    );
    let rollout_path = record.rollout_path.clone();

    let provider = ScriptedProvider::new_last_first(vec![vec![
        StreamEvent::TextDelta("hi there".into()),
        done(StopReason::EndTurn),
    ]]);
    let tools = ToolRegistry::new();
    let mut agent = Agent::new(&provider, &tools, "sys", ApprovalMode::Smart, "/cwd")
        .with_persistence(store.clone(), record);

    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
    agent.run_turn("hello", ev_tx, &mut act_rx).await.unwrap();
    drain(&mut ev_rx).await;

    let lines = read_lines(&rollout_path);
    assert!(matches!(&lines[0], RolloutLine::SessionMeta(_)));
    assert!(matches!(&lines[1], RolloutLine::Turn(t) if t.turn.status == TurnStatus::Running));
    let RolloutLine::Item(user_line) = &lines[2] else {
        panic!("expected user item")
    };
    assert!(matches!(user_line.item.items[0], TurnItem::UserMessage(_)));
    let RolloutLine::Item(agent_line) = &lines[3] else {
        panic!("expected agent item")
    };
    assert!(matches!(
        agent_line.item.items[0],
        TurnItem::AgentMessage(_)
    ));
    assert!(matches!(&lines[4], RolloutLine::Turn(t) if t.turn.status == TurnStatus::Completed));

    let sidecar = store.read_meta_sidecar(&rollout_path).unwrap().unwrap();
    assert_eq!(sidecar.first_user_message.as_deref(), Some("hello"));
    assert!(sidecar.total_input_tokens > 0);
    assert!(sidecar.total_output_tokens > 0);
}

#[tokio::test]
async fn tool_call_round_persists_call_and_result_items() {
    let dir = tempdir().unwrap();
    let store = make_store(dir.path());
    let record = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        PathBuf::from("/cwd"),
        None,
        ApprovalMode::Never,
    );
    let rollout_path = record.rollout_path.clone();

    let provider = ScriptedProvider::new_last_first(vec![
        vec![
            StreamEvent::ToolCallStart {
                id: "c1".into(),
                name: "echo".into(),
            },
            StreamEvent::ToolCallInputDelta("{}".into()),
            StreamEvent::ToolCallEnd,
            done(StopReason::EndTurn),
        ],
        vec![
            StreamEvent::TextDelta("done".into()),
            done(StopReason::EndTurn),
        ],
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool));
    let mut agent = Agent::new(&provider, &tools, "sys", ApprovalMode::Never, "/cwd")
        .with_persistence(store.clone(), record);

    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(128);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
    agent
        .run_turn("use tools", ev_tx, &mut act_rx)
        .await
        .unwrap();
    drain(&mut ev_rx).await;

    let lines = read_lines(&rollout_path);
    let item_kinds: Vec<&'static str> = lines
        .iter()
        .filter_map(|l| {
            let RolloutLine::Item(line) = l else {
                return None;
            };
            Some(match &line.item.items[0] {
                TurnItem::UserMessage(_) => "user",
                TurnItem::AgentMessage(_) => "agent",
                TurnItem::Reasoning(_) => "reasoning",
                TurnItem::ToolCall(_) => "tool_call",
                TurnItem::ToolResult(_) => "tool_result",
                TurnItem::ContextCompaction(_) => "compaction",
            })
        })
        .collect();
    assert!(item_kinds.contains(&"user"));
    assert!(item_kinds.contains(&"tool_call"));
    assert!(item_kinds.contains(&"tool_result"));
    assert!(item_kinds.contains(&"agent"));

    let result_item = lines
        .iter()
        .find_map(|l| {
            let RolloutLine::Item(line) = l else {
                return None;
            };
            if let TurnItem::ToolResult(r) = &line.item.items[0] {
                Some(r.clone())
            } else {
                None
            }
        })
        .expect("must have a tool result");
    assert_eq!(result_item.tool_call_id.0, "c1");
    assert!(!result_item.is_error);
}

#[tokio::test]
async fn compaction_marker_lands_before_user_message_on_next_turn() {
    let dir = tempdir().unwrap();
    let store = make_store(dir.path());
    let record = store.create_session_record(
        SessionId::new(),
        fixed_ts(),
        PathBuf::from("/cwd"),
        None,
        ApprovalMode::Smart,
    );
    let rollout_path = record.rollout_path.clone();

    let provider = CompactionProvider::new();
    let tools = ToolRegistry::new();
    // Tiny input budget (900) so 100_000 prompt tokens trip the 0.9 threshold.
    let mut agent = Agent::new(&provider, &tools, "sys", ApprovalMode::Smart, "/cwd")
        .with_token_budget(TokenBudget::new(1_000, 100))
        .with_persistence(store.clone(), record);

    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
    agent
        .run_turn("turn one", ev_tx, &mut act_rx)
        .await
        .unwrap();
    drain(&mut ev_rx).await;
    assert_eq!(agent.last_input_tokens(), 100_000);

    let history_before = agent.history().len();

    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
    agent
        .run_turn("turn two", ev_tx, &mut act_rx)
        .await
        .unwrap();
    drain(&mut ev_rx).await;

    assert!(
        agent.history().len() < history_before + 2,
        "compaction must drop messages: history before={history_before}, after={}",
        agent.history().len(),
    );

    let lines = read_lines(&rollout_path);
    let second_turn_idx = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| match line {
            RolloutLine::Turn(t) if t.turn.status == TurnStatus::Running => Some(idx),
            _ => None,
        })
        .nth(1)
        .expect("expected two Turn(Running) lines for two turns");

    let after_second_turn = &lines[second_turn_idx + 1..];
    let compaction_pos = after_second_turn.iter().position(|l| {
        let RolloutLine::Item(item) = l else {
            return false;
        };
        matches!(item.item.items[0], TurnItem::ContextCompaction(_))
    });
    let user_pos = after_second_turn.iter().position(|l| {
        let RolloutLine::Item(item) = l else {
            return false;
        };
        matches!(item.item.items[0], TurnItem::UserMessage(_))
    });
    assert!(
        compaction_pos.is_some() && user_pos.is_some() && compaction_pos < user_pos,
        "compaction marker must appear before the user message inside turn 2"
    );
}

#[tokio::test]
async fn no_persistence_means_no_disk_writes() {
    let dir = tempdir().unwrap();
    let provider = ScriptedProvider::new_last_first(vec![vec![
        StreamEvent::TextDelta("ok".into()),
        done(StopReason::EndTurn),
    ]]);
    let tools = ToolRegistry::new();
    let mut agent = Agent::new(&provider, &tools, "sys", ApprovalMode::Never, "/cwd");

    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);
    agent.run_turn("hi", ev_tx, &mut act_rx).await.unwrap();
    drain(&mut ev_rx).await;

    assert!(
        std::fs::read_dir(dir.path()).unwrap().next().is_none(),
        "no rollout file should be created when persistence is None"
    );
}
