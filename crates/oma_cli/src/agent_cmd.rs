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
use oma_provider::{
    ComputeBackend, EmbeddedProvider, ModelLoadParams, Provider, compute_model_sha256, kv_path,
    validate_compatible,
};
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

    /// Context window size in tokens. `0` (default) auto-tunes from
    /// available VRAM under the 80%-of-total budget. Positive values
    /// override the auto-tuner up to the model's `n_ctx_train`.
    #[arg(long, default_value_t = 0)]
    pub n_ctx: u32,

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

    /// Persist the KV cache next to the rollout (`<rollout>.kv`). On
    /// `--resume` against the same model file the cache is loaded so the
    /// next turn skips prompt-eval over the prior context. Opt-in because
    /// snapshots can grow to ~1 GiB at full 32K context.
    #[arg(long = "save-kv-cache")]
    pub save_kv_cache: bool,
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
        n_ctx: args.n_ctx,
        ..Default::default()
    };
    let provider =
        EmbeddedProvider::load(&model_path, &load_params).context("failed to load model")?;
    eprintln!(
        "[oma agent] model `{}` loaded in {} ms (context {} tokens — {})",
        provider.model_name(),
        load_start.elapsed().as_millis(),
        provider.context_length(),
        provider.auto_tune().reason,
    );

    let tools = ToolRegistry::new();
    let approval_mode = ApprovalMode::Never;
    let mut agent = Agent::new(&provider, &tools, &args.system, approval_mode, &cwd);

    if let Some(loaded) = loaded {
        let kv_decision = decide_resume_kv_path(args.save_kv_cache, &loaded, provider.model_path());
        eprintln!(
            "[oma agent] resumed session {} ({} message(s) in history)",
            loaded.record.id,
            loaded.state.messages.len()
        );
        agent = agent.resume_session(store, loaded);
        if let Some(path) = kv_decision {
            eprintln!("[oma agent] kv-cache enabled at {}", path.display());
            agent = agent.with_kv_cache_path(path);
        }
    } else {
        let mut record = store.create_session_record(
            SessionId::new(),
            Utc::now(),
            cwd.clone(),
            Some(provider.model_path().to_path_buf()),
            approval_mode,
        );
        let kv_path_opt = if args.save_kv_cache {
            let sha = compute_model_sha256(provider.model_path())
                .context("computing model SHA-256 for kv-cache failed")?;
            record.model_sha256 = Some(sha);
            Some(kv_path(&record.rollout_path))
        } else {
            None
        };
        eprintln!("[oma agent] new session {}", record.id);
        agent = agent.with_persistence(store, record);
        if let Some(path) = kv_path_opt {
            eprintln!("[oma agent] kv-cache enabled at {}", path.display());
            agent = agent.with_kv_cache_path(path);
        }
    }

    repl(&mut agent).await
}

/// Decide whether to enable KV-cache snapshots when resuming a session.
/// Returns `Some(kv_path)` when the snapshot is safe to use, `None` when
/// the user opted out, the recorded model SHA is missing, or the SHA no
/// longer matches the live model.
fn decide_resume_kv_path(
    flag_on: bool,
    loaded: &LoadedSession,
    current_model: &std::path::Path,
) -> Option<PathBuf> {
    if !flag_on {
        return None;
    }
    let Some(expected_sha) = loaded.record.model_sha256.as_deref() else {
        eprintln!(
            "[oma agent] --save-kv-cache: prior session has no recorded model SHA — skipping kv load (snapshots will populate going forward)"
        );
        return Some(kv_path(&loaded.record.rollout_path));
    };
    match validate_compatible(expected_sha, current_model) {
        Ok(()) => Some(kv_path(&loaded.record.rollout_path)),
        Err(err) => {
            eprintln!(
                "[oma agent] --save-kv-cache: refusing to load snapshot — {err} — falling back to full prompt-eval"
            );
            None
        }
    }
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
#[path = "agent_cmd_tests.rs"]
mod tests;
