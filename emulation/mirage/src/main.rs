//! `mirage` — one executable, and no background anything.
//!
//! Every subcommand is flattened in from [`mirage_ctl::CtlCmd`]:
//! `profile`, `topology`, `agent`, `emulators`, `state`, `paths`, `run`
//! and `exec`. There is no daemon to start, so there is no `mirage
//! daemon`; there is no web UI, so there is no `mirage webui`.
//!
//! `mirage run` is the runtime. It brings a session up inside its own
//! process, runs the command, and takes the session with it when it
//! exits.

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use mirage_ctl::CtlCmd;

// Link-only dependencies on the emulator backend crates. The binary
// never names them: each crate registers itself into the emulator
// registry via `inventory` (an `inventory::submit!` in its `lib.rs`).
// Referencing them with `extern crate` guarantees the linker keeps the
// crate object - and therefore its registration - even though no symbol
// is used directly. Each is gated on its feature so a backend can be
// dropped from the build entirely.
#[cfg(feature = "hotswap")]
extern crate mirage_hotswap as _;
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
// 344 bytes, effectively all of it in `Ctl(CtlCmd)`. Boxing to even the
// variants out would be a pessimisation, not a saving: exactly one of
// these is ever built — by `Cli::parse_from` in `main` — and it is moved
// once into `dispatch` and dropped. That would trade a stack move for a
// heap allocation and a pointer chase, to shrink a value with a single
// instance and no container.
#[allow(clippy::large_enum_variant)]
enum TopCmd {
    /// Show version, copyright, and the third-party crates mirage is
    /// built from (with their licenses).
    About,

    /// Every other subcommand (profile, topology, agent, emulators,
    /// state, run, exec, paths) is flattened in here.
    #[command(flatten)]
    Ctl(CtlCmd),
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
    "profile",
    "topology",
    "agent",
    "emulators",
    "exec",
    "state",
    "run",
    "paths",
    "about",
    "help",
];

/// The third-party dependency/license manifest, generated at build time
/// by `build.rs` from `cargo metadata` and embedded into the binary.
const THIRD_PARTY: &str = include_str!(concat!(env!("OUT_DIR"), "/about.txt"));

/// The same manifest as a JSON array of `{name, version, license}`,
/// generated alongside [`THIRD_PARTY`] by the same pass over
/// `cargo metadata`.
const THIRD_PARTY_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/about.json"));

/// One-line summary of what mirage is, shared by `about` and the JSON
/// form so the two cannot drift.
const DESCRIPTION: &str = "A UX for the rocjitsu (and other) GPU emulators.";

const COPYRIGHT: &str = "Copyright (c) Advanced Micro Devices, Inc. All rights reserved.";

const LICENSE: &str = "Licensed under the terms of mirage's LICENSE.";

/// Print version, copyright, and the embedded third-party manifest for
/// `mirage about`.
///
/// `--json` is a global flag, so it is accepted here whether or not this
/// command has anything machine-readable to say. It does, and printing
/// the prose anyway would be worse than rejecting the flag: a script that
/// asked for JSON and got a paragraph has no way to tell that it did.
fn print_about(json: bool) -> anyhow::Result<()> {
    if json {
        let doc = serde_json::json!({
            "name": "mirage",
            "version": env!("CARGO_PKG_VERSION"),
            "description": DESCRIPTION,
            "copyright": COPYRIGHT,
            "license": LICENSE,
            "third_party": serde_json::from_str::<serde_json::Value>(THIRD_PARTY_JSON)?,
        });
        println!("{}", serde_json::to_string_pretty(&doc)?);
        return Ok(());
    }
    println!("mirage {}", env!("CARGO_PKG_VERSION"));
    println!("{DESCRIPTION}");
    println!();
    println!("{COPYRIGHT}");
    println!("{LICENSE}");
    println!();
    print!("{THIRD_PARTY}");
    Ok(())
}

/// Whether `arg` is one of [`Cli`]'s global flags, which may appear
/// before the subcommand and so must be stepped over when looking for it.
///
/// `-v` is an [`ArgAction::Count`][clap::ArgAction::Count], so clap
/// accepts it bundled to any depth — `-v`, `-vv`, `-vvv`, … This function
/// therefore matches the *shape* rather than a list of spellings. It used
/// to be a list, and the list stopped at `-vv`: `mirage -vvv -- ./app`
/// mistook `-vvv` for the subcommand, spliced `run` in front of it, and
/// left the real `run` to be executed as the workload — so the user got
/// `command not found: run` from a flag that only differed by one `v`.
///
/// A global flag added to [`Cli`] must be added here too, which is the
/// kind of coupling that rots quietly. `tests::every_global_flag_is_known`
/// asks clap for the actual list and fails if this function does not
/// recognise one of them.
fn is_global_flag(arg: &str) -> bool {
    match arg {
        "--json" | "--verbose" => true,
        _ => {
            arg.len() >= 2
                && arg.starts_with('-')
                && !arg.starts_with("--")
                && arg[1..].chars().all(|c| c == 'v')
        }
    }
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
/// …`, `mirage exec --session s -- cmd`) and those with no `--` separator
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
        if is_global_flag(a) {
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
        TopCmd::About => {
            print_about(cli.json)?;
            Ok(ExitCode::from(0))
        }
        // Everything else, including `run`, happens right here in this
        // process. There is no routing decision to make: no command
        // reaches a session it does not own, because the only command
        // that owns one is `run`, and the only command that borrows one
        // — `exec` — dials it directly.
        TopCmd::Ctl(cmd) => {
            let json = cli.json;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(mirage_ctl::dispatch(cmd, json))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::{Cli, dropin_argv, is_global_flag};

    fn v_args(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn bare_dropin_routes_to_run() {
        assert_eq!(
            dropin_argv(v_args(&["mirage", "--", "./app", "arg"])),
            v_args(&["mirage", "run", "--", "./app", "arg"])
        );
    }

    #[test]
    fn rocjitsu_config_and_daemon_route_to_run() {
        assert_eq!(
            dropin_argv(v_args(&[
                "mirage", "--config", "c.json", "--daemon", "--", "./app"
            ])),
            v_args(&[
                "mirage", "run", "--config", "c.json", "--daemon", "--", "./app"
            ])
        );
    }

    #[test]
    fn attach_maps_to_daemon() {
        assert_eq!(
            dropin_argv(v_args(&[
                "mirage", "--attach", "--config", "c.json", "--", "./app"
            ])),
            v_args(&[
                "mirage", "run", "--daemon", "--config", "c.json", "--", "./app"
            ])
        );
    }

    #[test]
    fn global_flags_before_dropin_are_preserved() {
        assert_eq!(
            dropin_argv(v_args(&[
                "mirage",
                "--json",
                "--profile",
                "mi350x",
                "--",
                "./app"
            ])),
            v_args(&[
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

    /// `-vvv` used to be mistaken for the subcommand, which spliced `run`
    /// in front of it and turned the real `run` into the workload —
    /// `mirage -vvv -- ./app` died with `command not found: run` while
    /// `-vv` worked. Counted flags have no upper bound, so neither does
    /// this.
    #[test]
    fn bundled_verbosity_of_any_depth_finds_the_subcommand() {
        for v in ["-v", "-vv", "-vvv", "-vvvv", "-vvvvvvvvvv"] {
            let args = v_args(&["mirage", v, "run", "--", "./app"]);
            assert_eq!(
                dropin_argv(args.clone()),
                args,
                "{v} should leave an explicit `run` alone"
            );
            assert_eq!(
                dropin_argv(v_args(&["mirage", v, "--", "./app"])),
                v_args(&["mirage", v, "run", "--", "./app"]),
                "{v} should be stepped over when splicing `run`"
            );
        }
    }

    /// `is_global_flag` duplicates knowledge that lives in `Cli`'s derive.
    /// Ask clap what the global flags actually are, so adding one and
    /// forgetting this function is a test failure rather than a bug
    /// report about a flag that eats the subcommand.
    #[test]
    fn every_global_flag_is_known() {
        use clap::CommandFactory as _;
        for arg in Cli::command().get_arguments() {
            if !arg.is_global_set() {
                continue;
            }
            if let Some(long) = arg.get_long() {
                let spelling = format!("--{long}");
                assert!(
                    is_global_flag(&spelling),
                    "`{spelling}` is global on `Cli` but `is_global_flag` \
                     does not recognise it, so it would be mistaken for a \
                     subcommand in a drop-in invocation"
                );
            }
            if let Some(short) = arg.get_short() {
                let spelling = format!("-{short}");
                assert!(
                    is_global_flag(&spelling),
                    "`{spelling}` is global on `Cli` but `is_global_flag` \
                     does not recognise it"
                );
            }
        }
    }

    #[test]
    fn explicit_run_subcommand_is_untouched() {
        let args = v_args(&["mirage", "run", "--profile", "mi350x", "--", "./app"]);
        assert_eq!(dropin_argv(args.clone()), args);
    }

    #[test]
    fn other_subcommands_are_untouched() {
        let args = v_args(&["mirage", "exec", "--session", "s", "--", "cmd"]);
        assert_eq!(dropin_argv(args.clone()), args);
    }

    #[test]
    fn no_separator_is_untouched() {
        let args = v_args(&["mirage", "profile", "list"]);
        assert_eq!(dropin_argv(args.clone()), args);
        let help = v_args(&["mirage", "--help"]);
        assert_eq!(dropin_argv(help.clone()), help);
    }
}
