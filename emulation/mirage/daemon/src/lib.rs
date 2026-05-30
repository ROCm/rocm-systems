//! `mirage_daemon`: optional background daemon for cross-session
//! orchestration.
//!
//! Mirage's normal operation is fully daemon-free: each session has a
//! per-session [`mirage_host`] process and the CLI talks to it through
//! the filesystem. The daemon is reserved for future cross-session
//! features (shared metric collection, scheduling, dashboards, ...).
//!
//! Today this crate only exposes a stub `run` entry point invoked by
//! the `mirage daemon` subcommand of the unified `mirage` binary.

use clap::Args;

/// Command-line flags for the `mirage daemon` subcommand.
#[derive(Args, Debug, Default, Clone)]
pub struct DaemonArgs {
    /// Run in the foreground (do not detach).
    #[arg(long)]
    pub foreground: bool,
}

/// Entry point for `mirage daemon`.
pub fn run(_args: DaemonArgs) -> anyhow::Result<()> {
    // The daemon is not yet implemented; print a friendly message so
    // users discovering the subcommand know what to expect.
    eprintln!("mirage daemon: not yet implemented (no-op).");
    Ok(())
}
