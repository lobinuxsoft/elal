//! `oma agent` — interactive agent loop driven from stdin.
//!
//! Wires `oma_core::agent::Agent` to a [`oma_provider::EmbeddedProvider`]
//! and persists every turn to a [`oma_core::session::RolloutStore`] under
//! [`crate::data_root::resolve`].
//!
//! Continuity flags (`--resume <id>` / `--continue` / `--new`) follow the
//! Claude-Code semantics:
//! - `--resume <id>` rehydrates the named session regardless of cwd.
//! - `--continue` rehydrates the most recent session whose `cwd` matches
//!   the current working directory.
//! - `--new` (or no flag) opens a fresh session.
//!
//! Approval mode is hard-pinned to [`ApprovalMode::Never`] for chunk 6 —
//! interactive approval prompts will land alongside the TUI in a follow-up.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use chrono::Utc;
use oma_core::{
    Agent, AgentEvent, LoadedSession, RolloutStore, SessionId, UserAction, find_latest,
    load_session, locate,
};
use oma_protocol::ApprovalMode;
use oma_provider::{ComputeBackend, EmbeddedProvider, ModelLoadParams, Provider};
use oma_tools::ToolRegistry;
use tokio::io::{AsyncBufReadExt, BufReader, stdin};
use tokio::sync::mpsc;

use crate::data_root;

#[derive(clap::Args, Debug)]
pub struct AgentArgs {
    /// Path to a GGUF model file. Required for `--new` and `--continue`
    /// without prior history; optional when `--resume`/`--continue` finds
    /// a session whose recorded `model_path` is still available.
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// System prompt to inject as the first message of new sessions.
    #[arg(
        long,
        default_value = "You are a helpful coding assistant. Respond concisely."
    )]
    pub system: String,

    /// GPU layers to offload. Negative means all layers.
    #[arg(long, default_value_t = -1)]
    pub n_gpu_layers: i32,

    /// Resume a specific session by id.
    #[arg(long, value_name = "ID", conflicts_with_all = ["continue_", "new"])]
    pub resume: Option<String>,

    /// Resume the most recent session whose cwd matches the current
    /// working directory.
    #[arg(long = "continue", conflicts_with_all = ["resume", "new"])]
    pub continue_: bool,

    /// Force a fresh session, ignoring any prior history for the cwd.
    #[arg(long, conflicts_with_all = ["resume", "continue_"])]
    pub new: bool,
}

#[derive(Debug)]
enum ResumeMode {
    Resume(SessionId),
    Continue,
    New,
}

pub async fn run(args: AgentArgs) -> Result<()> {
    let mode = parse_resume_mode(&args)?;
    let store = RolloutStore::new(data_root::resolve()?);
    let cwd = std::env::current_dir().context("could not read current working directory")?;

    let loaded = match mode {
        ResumeMode::Resume(id) => {
            let path = locate(&store, id)
                .context("locating rollout failed")?
                .with_context(|| format!("session not found: {id}"))?;
            Some(load_session(&path).context("replaying rollout failed")?)
        }
        ResumeMode::Continue => find_latest(&store, &cwd)
            .context("scanning rollouts failed")?
            .map(|listed| load_session(&listed.record.rollout_path))
            .transpose()
            .context("replaying latest rollout failed")?,
        ResumeMode::New => None,
    };

    let model_path = resolve_model_path(args.model.as_ref(), loaded.as_ref())?;

    eprintln!(
        "[oma agent] loading {} on {} backend…",
        model_path.display(),
        ComputeBackend::compiled()
    );
    let load_start = Instant::now();
    let load_params = ModelLoadParams {
        n_gpu_layers: args.n_gpu_layers,
        ..Default::default()
    };
    let provider =
        EmbeddedProvider::load(&model_path, &load_params).context("failed to load model")?;
    eprintln!(
        "[oma agent] model `{}` loaded in {} ms (context {} tokens)",
        provider.model_name(),
        load_start.elapsed().as_millis(),
        provider.context_length(),
    );

    let tools = ToolRegistry::new();
    let approval_mode = ApprovalMode::Never;
    let mut agent = Agent::new(&provider, &tools, &args.system, approval_mode, &cwd);

    if let Some(loaded) = loaded {
        eprintln!(
            "[oma agent] resumed session {} ({} message(s) in history)",
            loaded.record.id,
            loaded.state.messages.len()
        );
        agent = agent.resume_session(store, loaded);
    } else {
        let record = store.create_session_record(
            SessionId::new(),
            Utc::now(),
            cwd.clone(),
            Some(provider.model_path().to_path_buf()),
            approval_mode,
        );
        eprintln!("[oma agent] new session {}", record.id);
        agent = agent.with_persistence(store, record);
    }

    repl(&mut agent).await
}

fn parse_resume_mode(args: &AgentArgs) -> Result<ResumeMode> {
    if let Some(raw) = &args.resume {
        let parsed =
            uuid::Uuid::parse_str(raw.trim()).context("--resume expects a session UUID")?;
        return Ok(ResumeMode::Resume(SessionId(parsed)));
    }
    if args.continue_ {
        return Ok(ResumeMode::Continue);
    }
    Ok(ResumeMode::New)
}

fn resolve_model_path(
    explicit: Option<&PathBuf>,
    loaded: Option<&LoadedSession>,
) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.clone());
    }
    if let Some(loaded) = loaded
        && let Some(p) = loaded.record.model_path.as_ref()
    {
        if p.exists() {
            return Ok(p.clone());
        }
        bail!(
            "session's recorded model path no longer exists: {} — pass --model to override",
            p.display()
        );
    }
    bail!("--model <path> is required for new sessions")
}

async fn repl<P: Provider>(agent: &mut Agent<'_, P>) -> Result<()> {
    eprintln!("[oma agent] ready. Press Ctrl+D to exit.");
    let stdin_reader = BufReader::new(stdin());
    let mut lines = stdin_reader.lines();

    loop {
        eprint!("\noma> ");
        // Force the prompt to appear before blocking on stdin.
        let _ = tokio::io::AsyncWriteExt::flush(&mut tokio::io::stderr()).await;

        let Some(line) = lines.next_line().await? else {
            eprintln!();
            break;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        run_one_turn(agent, line).await?;
    }
    Ok(())
}

async fn run_one_turn<P: Provider>(agent: &mut Agent<'_, P>, prompt: &str) -> Result<()> {
    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(128);
    let (_act_tx, mut act_rx) = mpsc::channel::<UserAction>(16);

    let printer = tokio::spawn(async move {
        while let Some(event) = ev_rx.recv().await {
            render_event(event);
        }
    });

    let summary = agent
        .run_turn(prompt, ev_tx, &mut act_rx)
        .await
        .context("agent turn failed")?;
    printer.await.context("event printer task panicked")?;

    let toks_per_sec = if summary.usage.generation_ms > 0 {
        (summary.usage.completion_tokens as f64 * 1000.0) / summary.usage.generation_ms as f64
    } else {
        0.0
    };
    eprintln!(
        "\n[turn done: stop={:?} prompt={} completion={} ({:.1} tok/s)]",
        summary.stop_reason,
        summary.usage.prompt_tokens,
        summary.usage.completion_tokens,
        toks_per_sec,
    );
    Ok(())
}

fn render_event(event: AgentEvent) {
    use std::io::Write as _;
    match event {
        AgentEvent::TurnStart => {}
        AgentEvent::TextDelta(s) => {
            let mut stdout = std::io::stdout().lock();
            let _ = stdout.write_all(s.as_bytes());
            let _ = stdout.flush();
        }
        AgentEvent::ReasoningDelta(_) => {
            // Hidden by default — reasoning surfaces only in `oma sessions show`.
        }
        AgentEvent::ToolCallStart { name, .. } => {
            eprintln!("\n[tool] calling {name}…");
        }
        AgentEvent::ToolCallArgs(_) => {}
        AgentEvent::ApprovalRequired(_) => {
            // Approval prompts are out of scope for chunk 6 — the CLI is
            // pinned to ApprovalMode::Never so this branch is unreachable
            // in practice. Print something safe just in case the mode ever
            // gets wired through and we forget to handle it.
            eprintln!("\n[approval requested but interactive prompts are not implemented yet]");
        }
        AgentEvent::ToolResult {
            content, is_error, ..
        } => {
            let tag = if is_error { "error" } else { "result" };
            eprintln!("\n[tool {tag}] {content}");
        }
        AgentEvent::TurnComplete(_) => {
            // Summary printed by run_one_turn after the channel closes.
        }
    }
}

#[cfg(test)]
mod tests {
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
        let record = SessionRecord {
            id: SessionId::new(),
            rollout_path: PathBuf::from("/tmp/rollout.jsonl"),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            source: "cli".into(),
            model_path: Some(bogus.clone()),
            model_sha256: None,
            cwd: PathBuf::from("/cwd"),
            oma_version: "0.0.0".into(),
            title: None,
            approval_mode: ApprovalMode::Never,
            total_input_tokens: 0,
            total_output_tokens: 0,
            first_user_message: None,
            schema_version: oma_protocol::SCHEMA_VERSION,
        };
        let loaded = LoadedSession {
            record,
            state: SessionState::new(Default::default(), PathBuf::from("/cwd")),
            last_turn_seq: 0,
            last_item_seq: 0,
        };
        let err = resolve_model_path(None, Some(&loaded)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("model path no longer exists"));
        assert!(msg.contains(&bogus.display().to_string()));
    }
}
