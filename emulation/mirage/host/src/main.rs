//! `mirage-host`: per-session host process.
//!
//! Usage:
//!
//! ```text
//! mirage-host --session <id>
//! ```
//!
//! Reads the session's directory under
//! `$XDG_RUNTIME_DIR/mirage/session/<id>`, publishes health, watches for
//! new exec definitions, and runs them.

use std::sync::Arc;

use clap::Parser;
use mirage_core::session::SessionId;
use mirage_host::{HostConfig, run};
use tokio::sync::Notify;

/// Per-session mirage host.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Session id (must match the directory under
    /// `$XDG_RUNTIME_DIR/mirage/session/<id>`).
    #[arg(long)]
    session: SessionId,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("MIRAGE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let shutdown = Arc::new(Notify::new());
        run(
            HostConfig {
                session: args.session,
            },
            shutdown,
        )
        .await
        .map_err(anyhow::Error::from)
    })
}
