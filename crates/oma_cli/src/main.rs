//! `oma` — oh-my-agent command-line entry point.

use clap::{Parser, Subcommand};

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

    tracing::info!(backend = oma_provider::BACKEND, "oma starting");

    match cli.command.unwrap_or(Command::Tui) {
        Command::Tui => {
            tracing::info!("TUI mode not implemented yet — see issue #7");
        }
        Command::Doctor => {
            tracing::info!("Doctor subcommand not implemented yet — see issue #9");
        }
    }

    Ok(())
}
