//! `oma` — oh-my-agent command-line entry point.

use clap::{Parser, Subcommand};
use oma_provider::ComputeBackend;

#[derive(Parser, Debug)]
#[command(
    name = "oma",
    version,
    about = "oh-my-agent — local-only coding agent powered by embedded llama-cpp-2",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[arg(long, global = true, env = "OMA_LOG", default_value = "info")]
    log_level: String,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Launch the interactive TUI session (default when no subcommand is given).
    Tui,
    /// Report diagnostics about the runtime environment.
    Doctor,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(&cli.log_level))
        .init();

    match cli.command.unwrap_or(Command::Tui) {
        Command::Tui => {
            tracing::info!("TUI mode not implemented yet — see issue #7");
        }
        Command::Doctor => run_doctor(),
    }

    Ok(())
}

fn run_doctor() {
    let backend = ComputeBackend::compiled();
    println!("oma {}", env!("CARGO_PKG_VERSION"));
    println!("  compute backend : {backend} ({})", backend.describe());
    println!("  os              : {}", std::env::consts::OS);
    println!("  arch            : {}", std::env::consts::ARCH);
    println!(
        "  models dir      : {}",
        dirs::data_dir()
            .map(|d| d.join("oh-my-agent").join("models").display().to_string())
            .unwrap_or_else(|| "<unknown>".into())
    );
}
