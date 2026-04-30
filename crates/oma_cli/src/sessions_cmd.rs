//! `oma sessions {list,show}` subcommand handlers.
//!
//! Both commands resolve the data root via [`crate::data_root::resolve`] and
//! delegate to [`oma_core::session`] for the heavy lifting.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Local;
use oma_core::{RolloutStore, list_sessions, locate};
use oma_protocol::{RolloutLine, SessionId, TurnItem, TurnStatus};

use crate::data_root;

#[derive(clap::Args, Debug)]
pub struct SessionsListArgs {
    /// Filter by absolute working directory. Defaults to no filter.
    #[arg(long)]
    pub cwd: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
pub struct SessionsShowArgs {
    /// Session id to print. Accepts the canonical UUID emitted by
    /// `oma sessions list`.
    pub id: String,
}

pub fn run_list(args: SessionsListArgs) -> Result<()> {
    let root = data_root::resolve()?;
    let store = RolloutStore::new(root);
    let entries = list_sessions(&store, args.cwd.as_deref())
        .context("listing rollouts under the data root failed")?;

    if entries.is_empty() {
        println!("No sessions found.");
        return Ok(());
    }

    println!(
        "{:<36}  {:<19}  {:>10}  {:>10}  {:<32}  CWD",
        "ID", "UPDATED", "TOKENS_IN", "TOKENS_OUT", "MODEL"
    );
    for entry in entries {
        let updated_local = entry.last_activity.with_timezone(&Local);
        let updated = updated_local.format("%Y-%m-%d %H:%M:%S").to_string();
        let model = model_label(&entry.record.model_path);
        println!(
            "{:<36}  {:<19}  {:>10}  {:>10}  {:<32}  {}",
            entry.record.id,
            updated,
            entry.record.total_input_tokens,
            entry.record.total_output_tokens,
            truncate(&model, 32),
            entry.record.cwd.display(),
        );
    }
    Ok(())
}

pub fn run_show(args: SessionsShowArgs) -> Result<()> {
    let id = parse_session_id(&args.id)?;
    let root = data_root::resolve()?;
    let store = RolloutStore::new(root);

    let path = locate(&store, id)
        .context("locating rollout file failed")?
        .with_context(|| format!("session id not found: {id}"))?;

    print_transcript(&path)
}

fn print_transcript(path: &Path) -> Result<()> {
    let file =
        File::open(path).with_context(|| format!("opening rollout {} failed", path.display()))?;
    let mut printed_meta = false;
    for raw in BufReader::new(file).lines() {
        let raw = raw?;
        if raw.trim().is_empty() {
            continue;
        }
        let line: RolloutLine = match serde_json::from_str(&raw) {
            Ok(l) => l,
            Err(err) => {
                // Tail-truncation safety: stop at the first malformed line —
                // matches replay semantics so partial transcripts still
                // render whatever the reader could parse.
                eprintln!("[oma sessions] truncated rollout tail: {err}");
                break;
            }
        };
        match line {
            RolloutLine::SessionMeta(meta) if !printed_meta => {
                let s = &meta.session;
                println!("Session : {}", s.id);
                println!("Created : {}", s.created_at.with_timezone(&Local));
                println!("CWD     : {}", s.cwd.display());
                println!("Model   : {}", model_label(&s.model_path));
                if let Some(t) = &s.title {
                    println!("Title   : {t}");
                }
                println!();
                printed_meta = true;
            }
            RolloutLine::SessionMeta(_) => {}
            RolloutLine::Turn(turn_line) => {
                let turn = turn_line.turn;
                if turn.status == TurnStatus::Completed
                    || turn.status == TurnStatus::Failed
                    || turn.status == TurnStatus::Cancelled
                {
                    let usage = turn.usage.unwrap_or_default();
                    let tag = match turn.status {
                        TurnStatus::Completed => "ok",
                        TurnStatus::Failed => "failed",
                        TurnStatus::Cancelled => "cancelled",
                        _ => "?",
                    };
                    println!(
                        "--- turn {} ({tag}, {} in / {} out) ---",
                        turn.sequence, usage.input_tokens, usage.output_tokens
                    );
                }
            }
            RolloutLine::Item(item_line) => {
                for item in item_line.item.items {
                    print_item(&item);
                }
            }
            RolloutLine::SessionTitleUpdated(t) => {
                println!("[title] {}", t.title);
            }
        }
    }
    Ok(())
}

fn print_item(item: &TurnItem) {
    match item {
        TurnItem::UserMessage(t) => println!("> {}", t.text),
        TurnItem::AgentMessage(t) => println!("{}", t.text),
        TurnItem::Reasoning(r) => println!("[think] {}", r.text),
        TurnItem::ToolCall(c) => println!("[tool ] {}({})", c.tool_name, c.input),
        TurnItem::ToolResult(r) => {
            let tag = if r.is_error { "error" } else { "result" };
            println!("[{tag}] {}", r.output);
        }
        TurnItem::ContextCompaction(c) => println!("[compaction] {}", c.text),
    }
}

fn parse_session_id(raw: &str) -> Result<SessionId> {
    let parsed = uuid::Uuid::parse_str(raw.trim())
        .with_context(|| format!("invalid session id (expected a UUID): {raw}"))?;
    Ok(SessionId(parsed))
}

fn model_label(path: &Option<PathBuf>) -> String {
    path.as_ref()
        .and_then(|p| p.file_stem().and_then(|s| s.to_str()))
        .unwrap_or("-")
        .to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut = s
            .char_indices()
            .nth(max.saturating_sub(1))
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        format!("{}…", &s[..cut])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_session_id_accepts_canonical_uuid() {
        let id = SessionId::new();
        let parsed = parse_session_id(&id.to_string()).expect("must parse own output");
        assert_eq!(parsed, id);
    }

    #[test]
    fn parse_session_id_rejects_non_uuid() {
        let err = parse_session_id("not-a-uuid").unwrap_err();
        assert!(format!("{err}").contains("invalid session id"));
    }

    #[test]
    fn parse_session_id_trims_whitespace() {
        let id = SessionId::new();
        let parsed =
            parse_session_id(&format!("  {id}\n")).expect("trimming surrounding whitespace");
        assert_eq!(parsed, id);
    }

    #[test]
    fn truncate_passes_through_short_strings() {
        assert_eq!(truncate("abc", 10), "abc");
    }

    #[test]
    fn truncate_caps_with_ellipsis() {
        let out = truncate("abcdefghij", 5);
        assert_eq!(out.chars().count(), 5);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn model_label_falls_back_when_path_missing() {
        assert_eq!(model_label(&None), "-");
    }

    #[test]
    fn model_label_uses_file_stem() {
        let p = Some(PathBuf::from("/models/Qwen3-Coder-30B.gguf"));
        assert_eq!(model_label(&p), "Qwen3-Coder-30B");
    }
}
