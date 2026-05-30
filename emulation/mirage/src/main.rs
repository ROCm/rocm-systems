//! `mirage` — a single executable that bundles the user-facing
//! control plane (`ctl`), the per-session `host`, and the (optional)
//! cross-session `daemon`.
//!
//! Top-level layout:
//!
//! * `mirage <ctl-command>` — every control-plane subcommand
//!   (`profile`, `session`, `exec`, `run`, `attach`, `logs`, `paths`,
//!   `schema`). These are flattened in from [`mirage_ctl::CtlCmd`].
//! * `mirage host --session <id>` — runs the per-session host.
//! * `mirage daemon` — runs the (currently stub) background daemon.

use std::process::ExitCode;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};
use mirage_core::ctl::FileCtl;
use mirage_core::session::SessionId;
use mirage_ctl::CtlCmd;
use mirage_daemon::DaemonArgs;
use mirage_host::{HostConfig, run as host_run};
use tokio::sync::Notify;

/// `mirage` — a UX for the rocjitsu (and other) GPU emulators.
///
/// Mirage stores all its state on disk under your XDG directories:
///
/// * profiles in `$XDG_CONFIG_HOME/mirage/profile/<name>.json`
/// * sessions in `$XDG_RUNTIME_DIR/mirage/session/<id>/`
///
/// Use `mirage <command> --help` for details on every subcommand.
#[derive(Parser, Debug)]
#[command(name = "mirage", version, about, long_about, propagate_version = true)]
struct Cli {
    /// Emit machine-readable JSON output where applicable.
    #[arg(long, global = true)]
    json: bool,

    /// Increase logging verbosity (-v info, -vv debug). Can also set
    /// `MIRAGE_LOG=debug`.
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: TopCmd,
}

#[derive(Subcommand, Debug)]
enum TopCmd {
    /// Run the per-session host process (used internally by `mirage
    /// session start` and `mirage run`; rarely invoked directly).
    Host(HostArgs),

    /// Run the (optional) background daemon.
    Daemon(DaemonArgs),

    /// All control-plane subcommands (profile, session, exec, run,
    /// attach, logs, paths, schema) are flattened in here.
    #[command(flatten)]
    Ctl(CtlCmd),
}

#[derive(Args, Debug)]
struct HostArgs {
    /// Session id (must match the directory under
    /// `$XDG_RUNTIME_DIR/mirage/session/<id>`).
    #[arg(long)]
    session: SessionId,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    mirage_ctl::init_logging(cli.verbose);
    match dispatch(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn dispatch(cli: Cli) -> anyhow::Result<ExitCode> {
    match cli.command {
        TopCmd::Host(args) => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move {
                let shutdown = Arc::new(Notify::new());
                host_run(
                    HostConfig {
                        session: args.session,
                    },
                    shutdown,
                )
                .await
                .map_err(anyhow::Error::from)
            })?;
            Ok(ExitCode::from(0))
        }
        TopCmd::Daemon(args) => {
            mirage_daemon::run(args)?;
            Ok(ExitCode::from(0))
        }
        TopCmd::Ctl(cmd) => {
            let ctl = FileCtl::new();
            let json = cli.json;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move { mirage_ctl::dispatch(cmd, ctl, json).await })
        }
    }
}
