//! `mirage` — a single executable that bundles the user-facing
//! control plane (`ctl`), the per-session `host`, and the (optional)
//! cross-session `daemon`.
//!
//! Top-level layout:
//!
//! * `mirage <ctl-command>` — every control-plane subcommand
//!   (`profile`, `topology`, `agent`, `emulators`, `session`, `exec`,
//!   `state`, `run`, `attach`, `logs`, `paths`). These are flattened in
//!   from [`mirage_ctl::CtlCmd`].
//! * `mirage host --session <id>` — runs the per-session host.
//! * `mirage webui` — runs the web UI server (formerly `mirage
//!   daemon`, still accepted as an alias).

use std::process::ExitCode;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};
use mirage_core::ctl::FileCtl;
use mirage_core::session::SessionId;
use mirage_ctl::CtlCmd;
#[cfg(feature = "daemon")]
use mirage_daemon::WebuiArgs;
use mirage_host::{HostConfig, run as host_run};
use tokio::sync::Notify;

// Link-only dependencies on the emulator backend crates. The binary
// never names them: each crate registers itself into the emulator
// registry via `inventory` (an `inventory::submit!` in its `lib.rs`).
// Referencing them with `extern crate` guarantees the linker keeps the
// crate object - and therefore its registration - even though no symbol
// is used directly. Each is gated on its feature so a backend can be
// dropped from the build entirely.
#[cfg(feature = "hotswap")]
extern crate mirage_hotswap as _;
#[cfg(feature = "noop")]
extern crate mirage_noop as _;
#[cfg(feature = "rocjitsu")]
extern crate mirage_rocjitsu as _;

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

    /// Run the web UI server (formerly `daemon`). Use `mirage webui
    /// install` to register it as a systemd service.
    #[cfg(feature = "daemon")]
    #[command(alias = "daemon")]
    Webui(WebuiArgs),

    /// Show version, copyright, and the third-party crates mirage is
    /// built from (with their licenses).
    About,

    /// All control-plane subcommands (profile, topology, agent,
    /// emulators, session, exec, state, run, attach, logs, paths) are
    /// flattened in here.
    #[command(flatten)]
    Ctl(CtlCmd),
}

#[derive(Args, Debug)]
struct HostArgs {
    /// Session id (must match the directory under
    /// `$XDG_RUNTIME_DIR/mirage/session/<id>`).
    #[arg(long)]
    session: SessionId,

    /// Node rank this host serves. Set internally when a containerised
    /// session launches a per-node host *inside* its container: that
    /// host runs only its own node's execs and never manages
    /// containers. Omitted for the orchestrator host on the real host,
    /// which brings up the containers and (for non-containerised
    /// sessions) runs every node directly.
    #[arg(long)]
    rank: Option<u32>,
}

fn main() -> ExitCode {
    let cli = Cli::parse_from(dropin_argv(std::env::args().collect()));
    mirage_ctl::init_logging(cli.verbose);
    match dispatch(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

/// Top-level subcommands `mirage` understands. Used to decide whether an
/// invocation is a normal `mirage <subcommand> …` call or a bare,
/// `rocjitsu`-style `mirage [--config …] [--daemon] -- <app>` call that
/// should be routed to `run`.
const SUBCOMMANDS: &[&str] = &[
    "host",
    "webui",
    "daemon",
    "profile",
    "topology",
    "agent",
    "emulators",
    "session",
    "exec",
    "state",
    "run",
    "attach",
    "logs",
    "paths",
    "about",
    "help",
];

/// The third-party dependency/license manifest, generated at build time
/// by `build.rs` from `cargo metadata` and embedded into the binary.
const THIRD_PARTY: &str = include_str!(concat!(env!("OUT_DIR"), "/about.txt"));

/// Print version, copyright, and the embedded third-party manifest for
/// `mirage about`.
fn print_about() {
    println!("mirage {}", env!("CARGO_PKG_VERSION"));
    println!("A UX for the rocjitsu (and other) GPU emulators.");
    println!();
    println!("Copyright (c) Advanced Micro Devices, Inc. All rights reserved.");
    println!("Licensed under the terms of mirage's LICENSE.");
    println!();
    print!("{THIRD_PARTY}");
}

/// Make `mirage` a drop-in replacement for the `rocjitsu` CLI by routing
/// bare `mirage [opts] -- <app> [args…]` invocations to `mirage run`.
///
/// The upstream `rocjitsu` CLI is invoked as
/// `rocjitsu --config <cfg> [--daemon|--attach] -- <app>`; there is no
/// subcommand. `mirage` is subcommand-based, so when an invocation has
/// the `rocjitsu` shape — a `--` application separator with no
/// recognised subcommand before it — we splice in `run` and translate
/// `--attach` to `--daemon` (mirage manages the daemon's lifecycle, so
/// "attach to a daemon" and "use a daemon" collapse to the same opt-in).
/// Everything `run` already accepts (`--config`, `--profile`, `--daemon`,
/// `--env`, …) then flows straight through.
///
/// Invocations that name a subcommand (`mirage run …`, `mirage profile
/// …`, `mirage exec start … -- cmd`) and those with no `--` separator
/// (so `--help`/`--version` keep working) are left untouched.
fn dropin_argv(args: Vec<String>) -> Vec<String> {
    // Only rocjitsu-style invocations carry a `--` app separator.
    let Some(sep) = args.iter().position(|a| a == "--") else {
        return args;
    };
    // Find the first token before the separator that isn't a global
    // flag; that's where a subcommand would appear.
    let mut head: Option<&str> = None;
    let mut head_idx = sep;
    for (i, a) in args.iter().enumerate().take(sep).skip(1) {
        if a == "--json" || a == "-v" || a == "--verbose" || a == "-vv" {
            continue;
        }
        head = Some(a.as_str());
        head_idx = i;
        break;
    }
    // A recognised subcommand (or a bare `--help`/`--version`) means this
    // is a normal mirage call; leave it alone.
    if let Some(h) = head
        && (SUBCOMMANDS.contains(&h) || matches!(h, "--help" | "-h" | "--version" | "-V"))
    {
        return args;
    }
    // Drop-in: splice `run` in where the subcommand would go and map the
    // rocjitsu-only `--attach` onto `--daemon`.
    let mut out = Vec::with_capacity(args.len() + 1);
    out.extend(args[..head_idx].iter().cloned());
    out.push("run".to_string());
    for a in &args[head_idx..sep] {
        if a == "--attach" {
            out.push("--daemon".to_string());
        } else {
            out.push(a.clone());
        }
    }
    out.extend(args[sep..].iter().cloned());
    out
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
                        rank: args.rank,
                    },
                    shutdown,
                )
                .await
                .map_err(anyhow::Error::from)
            })?;
            Ok(ExitCode::from(0))
        }
        #[cfg(feature = "daemon")]
        TopCmd::Webui(args) => {
            mirage_daemon::run(args)?;
            Ok(ExitCode::from(0))
        }
        TopCmd::About => {
            print_about();
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

#[cfg(test)]
mod tests {
    use super::dropin_argv;

    fn v(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn bare_dropin_routes_to_run() {
        assert_eq!(
            dropin_argv(v(&["mirage", "--", "./app", "arg"])),
            v(&["mirage", "run", "--", "./app", "arg"])
        );
    }

    #[test]
    fn rocjitsu_config_and_daemon_route_to_run() {
        assert_eq!(
            dropin_argv(v(&[
                "mirage", "--config", "c.json", "--daemon", "--", "./app"
            ])),
            v(&[
                "mirage", "run", "--config", "c.json", "--daemon", "--", "./app"
            ])
        );
    }

    #[test]
    fn attach_maps_to_daemon() {
        assert_eq!(
            dropin_argv(v(&[
                "mirage", "--attach", "--config", "c.json", "--", "./app"
            ])),
            v(&[
                "mirage", "run", "--daemon", "--config", "c.json", "--", "./app"
            ])
        );
    }

    #[test]
    fn global_flags_before_dropin_are_preserved() {
        assert_eq!(
            dropin_argv(v(&[
                "mirage",
                "--json",
                "--profile",
                "mi350x",
                "--",
                "./app"
            ])),
            v(&[
                "mirage",
                "--json",
                "run",
                "--profile",
                "mi350x",
                "--",
                "./app"
            ])
        );
    }

    #[test]
    fn explicit_run_subcommand_is_untouched() {
        let args = v(&["mirage", "run", "--profile", "mi350x", "--", "./app"]);
        assert_eq!(dropin_argv(args.clone()), args);
    }

    #[test]
    fn other_subcommands_are_untouched() {
        let args = v(&["mirage", "exec", "start", "s", "--", "cmd"]);
        assert_eq!(dropin_argv(args.clone()), args);
    }

    #[test]
    fn no_separator_is_untouched() {
        let args = v(&["mirage", "profile", "list"]);
        assert_eq!(dropin_argv(args.clone()), args);
        let help = v(&["mirage", "--help"]);
        assert_eq!(dropin_argv(help.clone()), help);
    }
}
