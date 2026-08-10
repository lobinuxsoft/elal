//! `elal` — elal command-line entry point.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use elal_protocol::{Message, StreamEvent};
use elal_provider::{
    CompletionRequest, ComputeBackend, EmbeddedProvider, ModelLoadParams, Provider,
    SamplingControls, detect_primary_gpu_vram,
};
use tokio::sync::mpsc;

mod agent_cmd;
mod data_root;
mod sessions_cmd;

#[derive(Parser, Debug)]
#[command(
    name = "elal",
    version,
    about = "elal — local-only coding agent powered by embedded llama-cpp-2",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[arg(long, global = true, env = "ELAL_LOG", default_value = "info")]
    log_level: String,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Launch the interactive TUI session (default when no subcommand is given).
    Tui,
    /// Report diagnostics about the runtime environment.
    Doctor,
    /// Single-shot chat smoke test: load a GGUF and stream one response to stdout.
    Chat(ChatArgs),
    /// Run the interactive agent loop with persistent sessions.
    Agent(agent_cmd::AgentArgs),
    /// Inspect persisted sessions on disk.
    #[command(subcommand)]
    Sessions(SessionsCommand),
}

#[derive(Subcommand, Debug)]
enum SessionsCommand {
    /// List all persisted sessions, newest first.
    List(sessions_cmd::SessionsListArgs),
    /// Print the reconstructed transcript for a session.
    Show(sessions_cmd::SessionsShowArgs),
}

#[derive(clap::Args, Debug)]
struct ChatArgs {
    /// Path to a GGUF model file.
    #[arg(long)]
    model: PathBuf,
    /// User message to send.
    #[arg(long)]
    prompt: String,
    /// Optional system prompt.
    #[arg(
        long,
        default_value = "You are a helpful assistant. Respond concisely."
    )]
    system: String,
    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,
    /// GPU layers to offload. Negative means all layers.
    #[arg(long, default_value_t = -1)]
    n_gpu_layers: i32,
    /// Context window size in tokens. `0` (default) auto-tunes from
    /// available VRAM under the 80%-of-total budget.
    #[arg(long, default_value_t = 0)]
    n_ctx: u32,
    /// Sampling temperature.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(&cli.log_level))
        .init();

    match cli.command.unwrap_or(Command::Tui) {
        Command::Tui => {
            tracing::info!("TUI mode not implemented yet — see issue #7");
        }
        Command::Doctor => run_doctor(),
        Command::Chat(args) => run_chat(args).await?,
        Command::Agent(args) => agent_cmd::run(args).await?,
        Command::Sessions(SessionsCommand::List(args)) => sessions_cmd::run_list(args)?,
        Command::Sessions(SessionsCommand::Show(args)) => sessions_cmd::run_show(args)?,
    }

    Ok(())
}

fn run_doctor() {
    let backend = ComputeBackend::compiled();
    println!("elal {}", env!("CARGO_PKG_VERSION"));
    println!("  compute backend : {backend} ({})", backend.describe());
    println!("  os              : {}", std::env::consts::OS);
    println!("  arch            : {}", std::env::consts::ARCH);
    match detect_primary_gpu_vram() {
        Some(v) => {
            let total_gib = v.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let free_gib = v.free_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
            println!("  gpu vram        : {free_gib:.2} GiB free / {total_gib:.2} GiB total");
        }
        None => {
            println!("  gpu vram        : <undetected — non-AMD or DRM sysfs unavailable>");
        }
    }
    println!(
        "  models dir      : {}",
        dirs::data_dir()
            .map(|d| d.join("elal").join("models").display().to_string())
            .unwrap_or_else(|| "<unknown>".into())
    );
}

async fn run_chat(args: ChatArgs) -> Result<()> {
    anyhow::ensure!(
        args.model.exists(),
        "model file not found: {}",
        args.model.display()
    );

    eprintln!(
        "[elal chat] loading {} on {} backend…",
        args.model.display(),
        ComputeBackend::compiled()
    );
    let load_start = Instant::now();
    let load_params = ModelLoadParams {
        n_gpu_layers: args.n_gpu_layers,
        n_ctx: args.n_ctx,
        ..Default::default()
    };
    let provider =
        EmbeddedProvider::load(&args.model, &load_params).context("failed to load model")?;
    eprintln!(
        "[elal chat] model `{}` loaded in {} ms (context {} tokens — {})",
        provider.model_name(),
        load_start.elapsed().as_millis(),
        provider.context_length(),
        provider.auto_tune().reason,
    );

    let sampling = SamplingControls {
        temperature: args.temperature,
        ..SamplingControls::default()
    };

    let request = CompletionRequest {
        messages: vec![Message::system(args.system), Message::user(args.prompt)],
        tools: vec![],
        sampling,
        max_tokens: Some(args.max_tokens),
        kv_cache_path: None,
    };

    let (tx, mut rx) = mpsc::channel::<StreamEvent>(128);

    // Reader task — only touches the channel, so it's Send-safe.
    // Re-acquires the stdout lock per write so the guard never crosses an await.
    let reader = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::TextStart => {}
                StreamEvent::TextDelta(s) => {
                    let mut stdout = std::io::stdout().lock();
                    let _ = stdout.write_all(s.as_bytes());
                    let _ = stdout.flush();
                }
                StreamEvent::TextEnd => {
                    println!();
                }
                _ => {}
            }
        }
    });

    let summary = provider
        .chat_completion_stream(request, tx)
        .await
        .context("chat completion failed")?;

    // Ensure the reader has drained the channel before we print stats.
    reader.await?;

    let toks_per_sec = if summary.usage.generation_ms > 0 {
        (summary.usage.completion_tokens as f64 * 1000.0) / summary.usage.generation_ms as f64
    } else {
        0.0
    };
    eprintln!(
        "[elal chat] stop={:?} prompt={} completion={} prompt_eval={} ms generation={} ms ({:.1} tok/s)",
        summary.stop_reason,
        summary.usage.prompt_tokens,
        summary.usage.completion_tokens,
        summary.usage.prompt_eval_ms,
        summary.usage.generation_ms,
        toks_per_sec,
    );

    Ok(())
}
