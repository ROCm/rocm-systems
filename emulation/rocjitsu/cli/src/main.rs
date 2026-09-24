//! `rocjitsu` — one executable, and no background anything.
//!
//! Every subcommand is flattened in from [`rj_ctl::CtlCmd`]:
//! `profile`, `topology`, `agent`, `emulators`, `state`, `paths`, `run`
//! and `exec`. There is no daemon to start, so there is no `rocjitsu
//! daemon`; there is no web UI, so there is no `rocjitsu webui`.
//!
//! `rocjitsu run` is the runtime. It brings a session up inside its own
//! process, runs the command, and takes the session with it when it
//! exits.

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use rj_ctl::CtlCmd;

// Link-only dependencies on the emulator backend crates. The binary
// never names them: each crate registers itself into the emulator
// registry via `inventory` (an `inventory::submit!` in its `lib.rs`).
// Referencing them with `extern crate` guarantees the linker keeps the
// crate object - and therefore its registration - even though no symbol
// is used directly. Each is gated on its feature so a backend can be
// dropped from the build entirely.
#[cfg(feature = "hotswap")]
extern crate rj_backend_hotswap as _;
#[cfg(feature = "rocjitsu")]
extern crate rj_backend_rocjitsu as _;

/// `rocjitsu` — a UX for the rocjitsu (and other) GPU emulators.
///
/// RocJITsu stores all its state on disk under your XDG directories:
///
/// * profiles in `$XDG_CONFIG_HOME/rocjitsu/profile/<name>.json`
/// * sessions in `$XDG_RUNTIME_DIR/rocjitsu/session/<id>/`
///
/// Use `rocjitsu <command> --help` for details on every subcommand.
#[derive(Parser, Debug)]
#[command(
    name = "rocjitsu",
    version,
    about,
    long_about,
    propagate_version = true,
    after_help = DROPIN_HELP,
    after_long_help = DROPIN_LONG_HELP
)]
struct Cli {
    /// Emit machine-readable JSON output where applicable.
    // `overrides_with` naming the flag itself is clap's way of saying a
    // repeat is not an error. Without it `rocjitsu paths --json --json`
    // exits 2, which a user hits by composing a command line from a
    // variable that already carries the flag. Written with `//` and not
    // `///`: a doc comment here is printed to the user by `--help`, and
    // why the attribute is there is nobody's business but ours.
    #[arg(long, global = true, overrides_with = "json")]
    json: bool,

    /// Increase logging verbosity (-v info, -vv debug). Can also set
    /// `ROCJITSU_CLI_LOG=debug`.
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
    /// Show version, copyright, and the third-party crates rocjitsu is
    /// built from (with their licenses).
    About,

    /// Serve an emulated GPU to a VMM as a PCI device, over vfio-user.
    ///
    /// The machine comes from `--config`, so a different part is a
    /// different config file. Needs a rocjitsu built with
    /// `-DROCJITSU_ENABLE_VFIO=ON`; an ordinary one does not carry the
    /// transport.
    #[cfg(feature = "rocjitsu")]
    #[command(name = VFIO_SERVE)]
    VfioServe(VfioArgs),

    /// Show the execution threads a config would allocate, at its own
    /// budget and at a range of ceilings.
    ///
    /// Reads the config and applies the allocation rule; it builds no
    /// GPU and runs no workload, so it answers what a machine will do
    /// before there is one.
    #[cfg(feature = "rocjitsu")]
    #[command(name = THREAD_BUDGET_TABLE)]
    ThreadBudgetTable(ThreadBudgetArgs),

    /// Report whether this build of the emulator can serve a GPU over
    /// vfio-user, and exit nonzero when it cannot.
    ///
    /// A build option of the library rather than of this CLI, so the
    /// answer comes from asking the library for the entry point the
    /// server lives behind.
    #[cfg(feature = "rocjitsu")]
    #[command(name = CHECK_VFIO_USER)]
    CheckVfioUser,

    /// Every other subcommand (profile, topology, agent, emulators,
    /// state, run, exec, paths) is flattened in here.
    #[command(flatten)]
    Ctl(CtlCmd),
}

/// `rocjitsu vfio-serve`, and the older `--vfio-socket` that routes to it.
#[cfg(feature = "rocjitsu")]
#[derive(clap::Args, Debug)]
struct VfioArgs {
    /// Simulation config describing the GPU to present.
    #[arg(long, value_name = "PATH")]
    config: String,
    /// Filesystem path of the AF_UNIX socket to listen on. Spelled as
    /// the older CLI spelled it, which is how that spelling still works.
    #[arg(long = VFIO_SOCKET_FLAG, value_name = "PATH")]
    vfio_socket: String,
    /// Descriptor to write to once the socket is accepting connections.
    ///
    /// A launcher that supervises the server needs to know when it may
    /// start the guest, and polling for the socket file cannot tell
    /// "listening" from "created but not yet bound". `scripts/run-vfio-guest.py`
    /// passes the write end of a pipe here and waits on the read end.
    #[arg(long = "vfio-ready-fd", value_name = "FD")]
    vfio_ready_fd: Option<i32>,
}

/// `rocjitsu thread-budget-table`.
#[cfg(feature = "rocjitsu")]
#[derive(clap::Args, Debug)]
struct ThreadBudgetArgs {
    /// Simulation config whose allocations to report.
    #[arg(long, value_name = "PATH")]
    config: String,
}

/// The status clap exits with on a usage error, and therefore the one a
/// usage error rocjitsu diagnoses for itself must use too.
const USAGE_EXIT: u8 = 2;

fn main() -> ExitCode {
    let argv = match dropin_argv(std::env::args().collect()) {
        Ok(argv) => argv,
        Err(usage) => {
            eprintln!("{usage}");
            return ExitCode::from(USAGE_EXIT);
        }
    };
    let cli = match Cli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(error) if error.kind() == clap::error::ErrorKind::DisplayVersion => {
            return print_version();
        }
        Err(error) => {
            let exit_code = u8::try_from(error.exit_code()).unwrap_or(1);
            let _ = error.print();
            return ExitCode::from(exit_code);
        }
    };
    rj_ctl::init_logging(cli.verbose);
    match dispatch(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

/// Serve the GPU `a.config` describes on `a.vfio_socket` until signalled.
///
/// Returns the server's own exit status, as the CLI this replaced did:
/// what went wrong belongs to the transport and the config, and the
/// server has already said which on stderr.
#[cfg(feature = "rocjitsu")]
fn serve_vfio(a: &VfioArgs) -> anyhow::Result<ExitCode> {
    // Absolute, and existing, before anything is stood up. The old CLI
    // checked both here too, and the reason is the same: the server
    // reports a missing config as a parse failure several layers down,
    // by which point the message names an internal step rather than the
    // path the user typed.
    let config = std::fs::canonicalize(&a.config).map_err(|e| {
        anyhow::anyhow!(
            "--config {}: {e}. Name an emulator config file that exists.",
            a.config
        )
    })?;
    // Negative means "tell nobody", which is what the C API takes for a
    // caller that is not waiting on readiness.
    let ready_fd = a.vfio_ready_fd.unwrap_or(-1);
    if ready_fd < 0 && a.vfio_ready_fd.is_some() {
        anyhow::bail!("--vfio-ready-fd requires a nonnegative descriptor");
    }
    let status =
        rj_backend_rocjitsu::serve_vfio(&config, std::path::Path::new(&a.vfio_socket), ready_fd)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(ExitCode::from(u8::try_from(status).unwrap_or(1)))
}

/// Say whether this build can serve a GPU over vfio-user.
///
/// Exits zero only when it can, which is what makes it usable as the
/// capability probe a build check wants; the CLI this replaced behaved
/// the same way, and a ctest case depends on it.
#[cfg(feature = "rocjitsu")]
fn check_vfio_user() -> anyhow::Result<ExitCode> {
    if rj_backend_rocjitsu::has_vfio_support().map_err(|e| anyhow::anyhow!("{e}"))? {
        println!("vfio-user support enabled");
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!(
        "rocjitsu: this build has no vfio-user support; reconfigure rocjitsu with \
         -DROCJITSU_ENABLE_VFIO=ON"
    );
    Ok(ExitCode::FAILURE)
}

/// Print what `a.config` would allocate, at its own budget and at each
/// of the ceilings the shipped tables are indexed by.
///
/// The columns, their order and their spelling are the older CLI's: this
/// output is quoted in the configuration and concurrent-dispatch docs,
/// and the tables there were read off it.
#[cfg(feature = "rocjitsu")]
fn thread_budget_table(a: &ThreadBudgetArgs) -> anyhow::Result<ExitCode> {
    use rj_backend_rocjitsu::THREAD_BUDGET_TABLE_ROWS as ROWS;

    // Absolute and existing before the library is loaded, for the reason
    // `serve_vfio` does the same: the loader reports a missing file from
    // several layers down, naming an internal step instead of the path
    // the user typed.
    let config = std::fs::canonicalize(&a.config).map_err(|e| {
        anyhow::anyhow!(
            "--config {}: {e}. Name an emulator config file that exists.",
            a.config
        )
    })?;
    let (_host, rows) = rj_backend_rocjitsu::resolve_thread_budgets(&config, ROWS)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Budget | num_threads | cpu_dispatch_threads per GPU | async_helper_threads | Total");
    for (budget, row) in ROWS.iter().zip(&rows) {
        // Budget zero is the config's own request rather than a ceiling,
        // which is why it is labelled instead of numbered.
        let label = if *budget == 0 {
            "Configured".to_string()
        } else {
            budget.to_string()
        };
        let dispatch = row
            .dispatch
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{label} | {} | {dispatch} | {} | {}",
            row.engines,
            row.helpers,
            row.total()
        );
    }
    Ok(ExitCode::SUCCESS)
}

/// Print this CLI's package version and RocJITsu's shared build identity.
///
/// The two are separate version numbers — the Rust workspace's and the
/// emulator library's — so the CLI's line says `cli` rather than leaving
/// a reader to guess which of two bare `rocjitsu <version>` lines is
/// which.
fn print_version() -> ExitCode {
    println!("rocjitsu cli {}", env!("CARGO_PKG_VERSION"));
    #[cfg(feature = "rocjitsu")]
    match rj_backend_rocjitsu::version_string() {
        Ok(version) => println!("{version}"),
        Err(error) => println!("rocjitsu build identity unavailable: {error}"),
    }
    #[cfg(not(feature = "rocjitsu"))]
    println!("rocjitsu build identity unavailable: backend not built");
    ExitCode::SUCCESS
}

/// The `rocjitsu` command tree, built once.
///
/// Everything on the argv-rewrite path asks clap rather than a list of
/// its own — which is the right answer to "what does rocjitsu accept?" and
/// was three separate answers to "build me the whole tree":
/// [`is_subcommand`], [`takes_a_workload`] and [`check_flags`] each called
/// `Cli::command()`, so one invocation materialised every subcommand,
/// every argument and every help string three times over before clap had
/// parsed a single token. The tree cannot differ between those calls — it
/// is generated from the same types — so it is built once and shared.
fn cli() -> &'static clap::Command {
    use clap::CommandFactory as _;
    static COMMAND: std::sync::OnceLock<clap::Command> = std::sync::OnceLock::new();
    COMMAND.get_or_init(Cli::command)
}

/// Whether `name` is a top-level subcommand `rocjitsu` understands.
///
/// Used to decide whether an invocation is a normal `rocjitsu <subcommand>
/// …` call or a bare, `rocjitsu`-style `rocjitsu [--config …] [--daemon] --
/// <app>` call that should be routed to `run`.
///
/// Asked of clap rather than kept as a list here. It *was* a list, and the
/// list was missing `cleanup`: `rocjitsu cleanup -- echo hi` did not run
/// `cleanup`, it brought up a whole emulated session and tried to execute
/// a program called `cleanup` inside it. Every subcommand added from now
/// on is covered the moment it is declared, because this is the
/// declaration.
///
/// Aliases count. A subcommand reachable under a second name is still a
/// subcommand, and missing one would resurrect exactly the bug above —
/// which is why this asks [`clap::Command::find_subcommand`], whose alias
/// rule is the one clap itself dispatches on, rather than spelling the
/// same comparison out a second time.
fn is_subcommand(name: &str) -> bool {
    cli().find_subcommand(name).is_some()
}

/// The third-party dependency/license manifest, generated at build time
/// by `build.rs` from `cargo metadata` and embedded into the binary.
const THIRD_PARTY: &str = include_str!(concat!(env!("OUT_DIR"), "/about.txt"));

/// The same manifest as a JSON array of `{name, version, license}`,
/// generated alongside [`THIRD_PARTY`] by the same pass over
/// `cargo metadata`.
const THIRD_PARTY_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/about.json"));

/// One-line summary of what rocjitsu is, shared by `about` and the JSON
/// form so the two cannot drift.
const DESCRIPTION: &str = "A UX for the rocjitsu (and other) GPU emulators.";

const COPYRIGHT: &str = "Copyright (c) Advanced Micro Devices, Inc. All rights reserved.";

const LICENSE: &str = "Licensed under the terms of rocjitsu's LICENSE.";

/// The `rocjitsu [options] -- <app>` shape, appended to `rocjitsu -h`.
///
/// Clap can only list subcommands, and drop-in mode is the one
/// invocation that has none — so a user reading the help sees every way
/// of running rocjitsu except the headline one. Spelling it out here is
/// what makes it discoverable at all; see [`dropin_argv`].
const DROPIN_HELP: &str = "\
Drop-in mode:
  rocjitsu [OPTIONS] -- <COMMAND> [ARGS]...   run a workload, no subcommand

Use 'rocjitsu --help' for the full form.";

/// The same, for `rocjitsu --help`, where there is room for the reason and
/// an example of each shape.
const DROPIN_LONG_HELP: &str = "\
Drop-in mode:
  A `--` with no subcommand before it runs a workload on an emulated
  machine, exactly as `rocjitsu run` would, so that scripts written for the
  upstream `rocjitsu` CLI keep working unchanged:

      rocjitsu -- ./my-rocm-app --flag
      rocjitsu --config cfg.json -- ./my-rocm-app
      rocjitsu --profile cdna4 --num-nodes 2 -- python train.py

  Everything `rocjitsu run` accepts may appear before the `--`. Anything
  else there is a mistyped flag rather than part of the workload, and is
  refused. See 'rocjitsu run --help' for the full list.

  `--attach` is taken but no longer honoured: it joined a daemon started
  by something else, and every daemon now belongs to the run that
  started it. Use 'rocjitsu exec --session <id>' to join one.";

/// Print version, copyright, and the embedded third-party manifest for
/// `rocjitsu about`.
///
/// `--json` is a global flag, so it is accepted here whether or not this
/// command has anything machine-readable to say. It does, and printing
/// the prose anyway would be worse than rejecting the flag: a script that
/// asked for JSON and got a paragraph has no way to tell that it did.
fn print_about(json: bool) -> anyhow::Result<()> {
    if json {
        let doc = serde_json::json!({
            "name": "rocjitsu",
            "version": env!("CARGO_PKG_VERSION"),
            "description": DESCRIPTION,
            "copyright": COPYRIGHT,
            "license": LICENSE,
            "third_party": serde_json::from_str::<serde_json::Value>(THIRD_PARTY_JSON)?,
        });
        println!("{}", serde_json::to_string_pretty(&doc)?);
        return Ok(());
    }
    println!("rocjitsu cli {}", env!("CARGO_PKG_VERSION"));
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
/// to be a list, and the list stopped at `-vv`: `rocjitsu -vvv -- ./app`
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

/// The subcommand a drop-in invocation is routed to.
///
/// Also the only one that takes `--attach`, which it declares in order to
/// refuse it: upstream's `--attach` joined a daemon started by something
/// else, and there is none to join now. Declared rather than dropped so
/// the spelling reaches an error that says where the capability went,
/// instead of clap's "unexpected argument". It is not translated to
/// `--daemon` — that starts a new session, which is the opposite of
/// joining one. See `check_run_args`.
const RUN: &str = "run";

/// The one-line shape shown by every usage error [`dropin_argv`] raises.
const DROPIN_USAGE: &str = "Usage: rocjitsu [OPTIONS] -- <COMMAND> [ARGS]...";

/// The subcommand a `--vfio-socket` invocation is routed to.
const VFIO_SERVE: &str = "vfio-serve";

/// The flag that names the socket, and the thing the rewriter keys on.
const VFIO_SOCKET_FLAG: &str = "vfio-socket";

/// The subcommand a `--check-vfio-user` invocation is routed to.
///
/// Spelled as a bare flag by the CLI this replaces, and by the ctest case
/// that probes the build, so the rewriter keeps that spelling working.
const CHECK_VFIO_USER: &str = "check-vfio-user";

/// The subcommand a `--thread-budget-table` invocation is routed to.
///
/// The older CLI spelled this as a flag on a bare invocation, and the
/// shipped documentation still does; the rewriter keeps that spelling
/// working by keying on the flag under the same name as the subcommand.
const THREAD_BUDGET_TABLE: &str = "thread-budget-table";

/// Whether `arg` is plausibly a program the user meant to run rather
/// than a mistyped subcommand.
///
/// Only used to decide whether a dead-end invocation is worth explaining
/// with `--` (see [`missing_separator_error`]). It is deliberately
/// narrow: for anything that could be a misspelt subcommand, clap's own
/// "did you mean" is the better answer, and this must not displace it.
fn looks_like_a_program(arg: &str) -> bool {
    arg.contains(std::path::MAIN_SEPARATOR) || std::path::Path::new(arg).is_file()
}

/// Whether `opts` names something to run, as opposed to only describing a
/// machine to bring up.
///
/// Asked of the daemon-only form, which has no `--` to mark where a
/// workload would start, so the only way to tell is to walk the flags and
/// see what is left over.
///
/// Walking is the point. `looks_like_a_program` applied to every token
/// answers for the *values* of options too, and the value of `--config`
/// is a path — so `rocjitsu --daemon --config ../configs/mi355x.json`
/// was read as naming a workload, and the `run` this form needs was
/// never spliced in. It failed with "unexpected argument '--daemon'",
/// while the same command with a bare `c.json` worked, which is why the
/// case that covered this did not catch it.
///
/// Any operand at all, not one that looks like a path: a command found
/// through `PATH` is a workload and looks like nothing in particular.
/// `rocjitsu --daemon true` started a session and ran `true` in it,
/// because `true` has no separator and names no file here — and a
/// workload without a `--` is meant to be refused, not run. The
/// daemon-only form is flags and their values throughout, so an operand
/// left over from that walk is either a workload or a typo, and neither
/// is something to splice `run` in front of.
fn names_a_workload(opts: &[String]) -> bool {
    let known = rj_ctl::usage::AcceptedFlags::of(cli(), RUN);
    rj_ctl::usage::first_operand(opts, &known).is_some()
}

/// Whether the subcommand `name` names ends in a workload, and so has a
/// `--` and a span of rocjitsu's own flags in front of it.
///
/// Every other subcommand is an ordinary clap parse: it has no
/// `trailing_var_arg` positional, so an unknown flag is reported by clap
/// as an unknown flag and never becomes anything else.
///
/// Asked of clap rather than listed, for the same reason
/// [`is_subcommand`] is. This was a list of two names sitting next to
/// that function, and it controls whether the guard runs at all — so a
/// third subcommand declared with a trailing workload, or an alias of
/// one of these two, would have gone unguarded and recreated exactly the
/// bug the list beside it was written to fix. `trailing_var_arg` on a
/// positional *is* the declaration of "everything from here is the
/// workload's", so it is what the question is asked of; aliases come
/// free, because `find_subcommand` resolves them.
fn takes_a_workload(name: &str) -> bool {
    cli().find_subcommand(name).is_some_and(ends_in_a_workload)
}

/// Whether `sub` declares that everything from some point on is the
/// workload's.
///
/// Both of clap's spellings, because either one produces the parse this
/// guard exists in front of. `#[arg(trailing_var_arg = true)]` on the
/// positional is the current one and what `run` and `exec` use;
/// `#[command(trailing_var_arg = true)]` is the clap 3 spelling, still
/// accepted and still setting the same behaviour, and it leaves every
/// `Arg` unset — so reading only the positionals would answer "no" for a
/// subcommand that parses exactly like `run` and leave it unguarded.
///
/// Shared with the test that derives the population it checks from this
/// same question, so the two cannot disagree about who must be guarded.
fn ends_in_a_workload(sub: &clap::Command) -> bool {
    #[allow(deprecated)]
    let command_level = sub.is_trailing_var_arg_set();
    command_level
        || sub
            .get_positionals()
            .any(clap::Arg::is_trailing_var_arg_set)
}

/// Check the span of `args` that belongs to rocjitsu rather than to the
/// workload, and turn the first mistyped flag in it into a usage error.
///
/// `from` is where rocjitsu's own flags start — after the subcommand, or
/// after the program name for a drop-in — and `sep` is the index of the
/// `--`, when there is one. Which there is decides how far the scan goes;
/// see [`rj_ctl::usage::Span`].
///
/// # Errors
///
/// Returns the message to print for the first flag-shaped token rocjitsu
/// does not accept.
fn check_flags(args: &[String], from: usize, sep: Option<usize>, sub: &str) -> Result<(), String> {
    use rj_ctl::usage::{AcceptedFlags, Span, misplaced_flag, unknown_flag_error};

    let known = AcceptedFlags::of(cli(), sub);
    let (span, stop) = match sep {
        Some(sep) => (&args[from.min(sep)..sep], Span::BeforeSeparator),
        None => (&args[from.min(args.len())..], Span::UntilTheCommand),
    };
    let Some(bad) = misplaced_flag(span, &known, stop) else {
        return Ok(());
    };
    // The drop-in spelling has no subcommand to name, so its usage line
    // and help pointer are the bare ones; everything else about the
    // message is identical, and deliberately so.
    let (usage, help) = if args.get(from.saturating_sub(1)).map(String::as_str) == Some(sub) {
        (
            format!("Usage: rocjitsu {sub} [OPTIONS] -- <COMMAND> [ARGS]..."),
            format!("rocjitsu {sub} --help"),
        )
    } else {
        (DROPIN_USAGE.to_string(), "rocjitsu run --help".to_string())
    };
    Err(unknown_flag_error(bad, &known, &usage, &help))
}

/// Make `rocjitsu` a drop-in replacement for the `rocjitsu` CLI by routing
/// bare `rocjitsu [opts] -- <app> [args…]` invocations to `rocjitsu run`.
///
/// The upstream `rocjitsu` CLI is invoked as
/// `rocjitsu --config <cfg> [--daemon|--attach] -- <app>`; there is no
/// subcommand. `rocjitsu` is subcommand-based, so when an invocation has
/// the `rocjitsu` shape — a `--` application separator with no
/// recognised subcommand before it — we splice in `run`. Everything
/// `run` already accepts (`--config`, `--profile`, `--daemon`, `--env`,
/// …) then flows straight through.
///
/// Two things are restored rather than passed through, because the
/// subcommand's defaults are not the old CLI's and a command line
/// written against the old one should still mean what it meant:
///
/// * The execution mode. Upstream ran the workload in-process and forked
///   a daemon only for `--daemon`; `run` uses a daemon unless told not
///   to. A legacy invocation that did not ask for a daemon is given
///   `--in-process`.
/// * The daemon-only form, `rocjitsu --daemon --config <cfg>` with no
///   workload at all, which has no `--` for the rule above to key on.
///
/// Invocations that name a subcommand (`rocjitsu run …`, `rocjitsu profile
/// …`, `rocjitsu exec --session s -- cmd`) are left untouched, and so is
/// a bare `--help`/`--version`.
///
/// # Errors
///
/// Returns the usage message to print when the invocation has the
/// drop-in shape but cannot be one. The rewriter used to treat *any*
/// unrecognised token before `--` as "no subcommand, therefore a
/// workload", which is right for `--config` and wrong for a typo:
/// `rocjitsu --nodes 2 -- ./app` brought a whole emulated machine up and
/// then failed to execute a program called `--nodes` inside it. A token
/// that is shaped like a flag and is not one rocjitsu takes is a mistake
/// the user wants to hear about before anything boots.
fn dropin_argv(args: Vec<String>) -> Result<Vec<String>, String> {
    // `rocjitsu -v`, and nothing else on the line. Upstream documented and
    // implemented `-v` as version; here it is the first step of verbosity,
    // so on its own it parsed as "verbosity 1, no subcommand" and exited 2
    // — a version probe that fails is a poor greeting for a drop-in.
    //
    // Only in isolation. `-vv`, `-v run …` and `-v` before a workload all
    // mean verbosity, because there the user has said what to do and `-v`
    // is modifying it. A lone `-v` asks for nothing, which is the one
    // reading under which "print the version" is the useful answer.
    if args.len() == 2 && args[1] == "-v" {
        return Ok(vec![args[0].clone(), "--version".to_string()]);
    }
    let sep = args.iter().position(|a| a == "--");
    // Where a subcommand could still appear: everything up to the app
    // separator, or the whole command line when there is none.
    let scan_end = sep.unwrap_or(args.len());
    // Find the first token in that span that isn't a global flag; that's
    // where a subcommand would be.
    let mut head: Option<&str> = None;
    let mut head_idx = scan_end;
    for (i, a) in args.iter().enumerate().take(scan_end).skip(1) {
        if is_global_flag(a) {
            continue;
        }
        head = Some(a.as_str());
        head_idx = i;
        break;
    }
    // A recognised subcommand means this is a normal rocjitsu call, so it
    // is not rewritten. That includes `run --attach`: `run` declares that
    // flag itself, in order to refuse it with an explanation, so there is
    // nothing here to translate — and translating it to `--daemon` would
    // start a session rather than join one, which is not what it asked.
    //
    // It still gets the flag guard, though, and that is the point of
    // doing it here. `rocjitsu run --nodes 2 -- ./app` is the same mistake
    // as `rocjitsu --nodes 2 -- ./app` in the spelling people actually use,
    // and it was checked in a different place, by a different copy of the
    // rules, after clap had already thrown away the position that makes
    // it decidable. Same rules, same message, one implementation.
    if let Some(name) = head.filter(|h| is_subcommand(h)) {
        if takes_a_workload(name) {
            check_flags(&args, head_idx + 1, sep, name)?;
        }
        return Ok(args);
    }
    // A bare `--help`/`--version` is not a drop-in either.
    if matches!(head, Some("--help" | "-h" | "--version" | "-V")) {
        return Ok(args);
    }
    // `rocjitsu --config <cfg> --vfio-socket <path>` — the older spelling
    // for serving a GPU to a VMM. It is not a workload invocation and
    // must not be routed to `run`, whose signal handling this mode
    // cannot share; it goes to `vfio-serve`, which is dispatched before
    // any of that is armed.
    if names_a_vfio_socket(&args[1..scan_end]) {
        // The old CLI refused these combinations outright rather than
        // picking one, and so does this: serving a VMM is a whole mode,
        // not a modifier on a run.
        if asks_for_a_daemon(&args[1..scan_end]) || sep.is_some() {
            return Err(vfio_is_its_own_mode_error());
        }
        check_flags(&args, 1, None, VFIO_SERVE)?;
        let mut out = Vec::with_capacity(args.len() + 1);
        out.push(args[0].clone());
        out.push(VFIO_SERVE.to_string());
        out.extend(args[1..].iter().cloned());
        return Ok(out);
    }
    // `rocjitsu --check-vfio-user` — the older spelling of the build
    // probe. It takes nothing and reports on the library rather than on
    // any machine, so it is routed on its own and everything else on the
    // line is a mistake clap should name.
    if args[1..scan_end].iter().any(|a| a == "--check-vfio-user") {
        let mut out = Vec::with_capacity(args.len());
        out.push(args[0].clone());
        out.push(CHECK_VFIO_USER.to_string());
        out.extend(
            args[1..]
                .iter()
                .filter(|a| *a != "--check-vfio-user")
                .cloned(),
        );
        return Ok(out);
    }
    // `rocjitsu --config <cfg> --thread-budget-table` — the older
    // spelling for reporting what a config would allocate. Like serving
    // a VMM it is not a workload invocation, so it must not reach `run`.
    if names_a_thread_budget_table(&args[1..scan_end]) {
        if asks_for_a_daemon(&args[1..scan_end]) || sep.is_some() {
            return Err(thread_budget_is_its_own_mode_error());
        }
        let mut out = Vec::with_capacity(args.len() + 1);
        out.push(args[0].clone());
        out.push(THREAD_BUDGET_TABLE.to_string());
        // The flag named the mode rather than carrying a value, and the
        // mode is now the subcommand, so it would be an unknown argument
        // to the very command it selected. Dropping it before the check
        // below is also what keeps it from being reported as one.
        out.extend(
            args[1..]
                .iter()
                .filter(|a| *a != "--thread-budget-table")
                .cloned(),
        );
        check_flags(&out, 2, None, THREAD_BUDGET_TABLE)?;
        return Ok(out);
    }
    let Some(sep) = sep else {
        // `rocjitsu --daemon --config <cfg>` — upstream's daemon-only
        // form, which served the emulator with no workload of its own.
        // It has no `--`, so the rewrite below never saw it and clap
        // called it a missing subcommand. `run` with an empty argv is
        // the same thing: it brings the session up and holds it until
        // stopped, which is what that invocation asked for.
        //
        // "No workload" is the whole of the shape, so nothing here may
        // look like one. Without that, `rocjitsu --daemon ./app` — a
        // command line missing its `--` — would be rewritten into a
        // session that holds itself open and never runs `./app`, which
        // is a worse answer than the error it gets today.
        if asks_for_a_daemon(&args[1..]) && !names_a_workload(&args[1..]) {
            check_flags(&args, 1, None, RUN)?;
            let mut out = Vec::with_capacity(args.len() + 1);
            out.push(args[0].clone());
            out.push(RUN.to_string());
            out.extend(args[1..].iter().cloned());
            return Ok(out);
        }
        // Otherwise there is nothing to rewrite. Clap reports the
        // unrecognised subcommand, and does it better than we could —
        // unless what the user typed is obviously a program, in which
        // case the missing piece is the `--`, not the spelling.
        return match head {
            Some(h) if looks_like_a_program(h) => Err(missing_separator_error(
                &args[1..head_idx],
                &args[head_idx..],
            )),
            _ => Ok(args),
        };
    };
    // Everything before the separator is rocjitsu's own; a flag there that
    // rocjitsu does not take is a typo, not a workload.
    check_flags(&args, 1, Some(sep), RUN)?;
    if sep + 1 == args.len() {
        return Err(empty_separator_error());
    }
    // Drop-in: splice `run` in where the subcommand would go.
    let mut out = Vec::with_capacity(args.len() + 2);
    out.extend(args[..head_idx].iter().cloned());
    out.push(RUN.to_string());
    // The upstream CLI ran the workload in-process and forked a daemon
    // only for `--daemon`. `rocjitsu run` is the other way round, which
    // is the right default for a subcommand people are typing today but
    // the wrong one to hand an invocation written against the old CLI:
    // it would quietly put a workload in a mode it had never used. So a
    // legacy invocation keeps the legacy default, and `--daemon` is
    // still how it asks for the other one.
    if !names_an_execution_mode(&args[1..sep]) {
        out.push("--in-process".to_string());
    }
    out.extend(args[head_idx..].iter().cloned());
    Ok(out)
}

/// Whether a drop-in invocation's own arguments ask for a daemon.
///
/// `--attach` counts. It is refused later, with an explanation of why
/// there is no daemon to attach to, and that is a better thing to read
/// than a complaint about the in-process mode this would otherwise have
/// added underneath it.
fn asks_for_a_daemon(opts: &[String]) -> bool {
    opts.iter().any(|a| a == "--daemon" || a == "--attach")
}

/// Whether the invocation has already said which mode it wants, and so
/// needs nothing supplied.
///
/// `--in-process` counts as much as `--daemon` does. Supplying a second
/// one would be harmless to clap and confusing to read in a `ps` line,
/// and the point of the rule is to answer a question the user left open
/// — not to answer one they did not.
fn names_an_execution_mode(opts: &[String]) -> bool {
    asks_for_a_daemon(opts) || opts.iter().any(|a| a == "--in-process")
}

/// Whether the invocation asks to serve a VMM.
///
/// Both spellings clap accepts for the flag, since either is how a user
/// would have written it against the older CLI.
fn names_a_vfio_socket(opts: &[String]) -> bool {
    let long = format!("--{VFIO_SOCKET_FLAG}");
    let joined = format!("{long}=");
    opts.iter().any(|a| *a == long || a.starts_with(&joined))
}

/// Whether the invocation asks for a thread allocation table.
///
/// The older CLI took this as a bare flag, so that is the only spelling
/// to look for; unlike `--vfio-socket` it carries no value of its own.
fn names_a_thread_budget_table(opts: &[String]) -> bool {
    opts.iter().any(|a| a == "--thread-budget-table")
}

/// The message for `--thread-budget-table` asked for alongside something
/// it cannot be combined with.
///
/// The older CLI refused the same combinations, and for the same reason:
/// the table is computed from the config alone, so pairing it with a
/// launch would have to either ignore the launch or ignore the table.
fn thread_budget_is_its_own_mode_error() -> String {
    format!(
        "error: --{THREAD_BUDGET_TABLE} reports what a config would allocate and cannot be \
         combined with --daemon, --attach, or a workload\n\n\
         It builds no GPU and runs nothing: it reads the config and applies the \
         allocation\nrule, which is why it can answer before there is a machine.\n\n\
         Usage: rocjitsu {THREAD_BUDGET_TABLE} --config <PATH>\n\n\
         For more information, try 'rocjitsu {THREAD_BUDGET_TABLE} --help'."
    )
}

/// The message for `--vfio-socket` asked for alongside something it
/// cannot be combined with.
fn vfio_is_its_own_mode_error() -> String {
    format!(
        "error: --{VFIO_SOCKET_FLAG} serves a GPU to a VMM and cannot be combined with \
         --daemon, --attach, or a workload\n\n\
         Serving a VMM is a mode of its own: there is no workload to launch and no \
         session to\nshare, only a PCI device on a socket for something else to attach \
         to.\n\n\
         Usage: rocjitsu {VFIO_SERVE} --config <PATH> --{VFIO_SOCKET_FLAG} <PATH>\n\n\
         For more information, try 'rocjitsu {VFIO_SERVE} --help'."
    )
}

/// The message for a `--` with nothing after it.
///
/// Worth spelling out rather than leaving to clap: the rewriter has
/// already spliced `run` in by then, so clap would report a required
/// argument of a subcommand the user never typed.
fn empty_separator_error() -> String {
    format!(
        "error: `--` was given with no command after it\n\n\
         Everything after `--` is the workload to run on the emulated machine, \
         and there\nhas to be one.\n\n\
         {DROPIN_USAGE}\n\n\
         For example:\n  rocjitsu --profile mi350x -- ./my-rocm-app --flag\n\n\
         For more information, try 'rocjitsu run --help'."
    )
}

/// The message for `rocjitsu ./app` — a workload named with no `--` in
/// front of it, which clap can only report as an unrecognised
/// subcommand.
fn missing_separator_error(opts: &[String], argv: &[String]) -> String {
    let program = argv.first().map_or("<command>", String::as_str);
    // Repeat the flags the user already typed, so the suggested line is
    // the one they wanted rather than a shorter one they have to
    // reassemble.
    let opts = opts
        .iter()
        .map(|o| format!("{o} "))
        .collect::<Vec<_>>()
        .concat();
    format!(
        "error: unrecognized subcommand '{program}'\n\n\
         To run '{program}' on an emulated machine, separate it from rocjitsu's \
         own flags\nwith `--`:\n\n  rocjitsu {opts}-- {}\n\n\
         {DROPIN_USAGE}\n\n\
         For more information, try 'rocjitsu --help'.",
        argv.join(" ")
    )
}

fn dispatch(cli: Cli) -> anyhow::Result<ExitCode> {
    match cli.command {
        TopCmd::About => {
            print_about(cli.json)?;
            Ok(ExitCode::from(0))
        }
        // Deliberately here, beside `about`, and not through the runtime
        // below. The server blocks for its whole life on the thread that
        // calls it, and while it runs it blocks SIGINT, SIGTERM and
        // SIGUSR1 and takes them itself — the first two to stop, the
        // third to raise an interrupt on the emulated device. `run` and
        // `exec` install tokio handlers for all three and mean different
        // things by them; SIGUSR1 in particular they forward to a
        // workload. Whichever armed them first would win, and this is
        // not a mode with a workload to forward anything to. So it never
        // enters a runtime that has them.
        #[cfg(feature = "rocjitsu")]
        TopCmd::VfioServe(a) => serve_vfio(&a),
        #[cfg(feature = "rocjitsu")]
        TopCmd::ThreadBudgetTable(a) => thread_budget_table(&a),
        #[cfg(feature = "rocjitsu")]
        TopCmd::CheckVfioUser => check_vfio_user(),
        // Everything else, including `run`, happens right here in this
        // process. There is no routing decision to make: no command
        // reaches a session it does not own, because the only command
        // that owns one is `run`, and the only command that borrows one
        // — `exec` — dials it directly.
        TopCmd::Ctl(cmd) => {
            let json = cli.json;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(rj_ctl::dispatch(cmd, json))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use rj_ctl::usage::AcceptedFlags;

    #[cfg(feature = "rocjitsu")]
    use super::TopCmd;
    use super::{Cli, cli, dropin_argv, ends_in_a_workload, is_global_flag, is_subcommand};

    fn v_args(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    /// The rewritten command line, for the invocations that have one.
    fn rewrite(args: &[&str]) -> Vec<String> {
        dropin_argv(v_args(args)).unwrap_or_else(|usage| panic!("{args:?} was refused:\n{usage}"))
    }

    /// The usage message, for the invocations that are refused.
    fn refuse(args: &[&str]) -> String {
        match dropin_argv(v_args(args)) {
            Err(usage) => usage,
            Ok(out) => panic!("{args:?} should have been refused, but became {out:?}"),
        }
    }

    /// The two spellings of the same mistake give the same answer.
    ///
    /// This is the property the guard exists for, and it is the one that
    /// was false. `rocjitsu --nodes 2 -- ./app` was refused by the drop-in
    /// rewriter; `rocjitsu run --nodes 2 -- ./app` — the commoner spelling
    /// of the same typo — was checked somewhere else, by a second copy of
    /// the rules, and `rocjitsu run --nodes 2 ./app` was not checked at all
    /// and brought a whole emulated machine up to run a program called
    /// `--nodes`.
    #[test]
    fn both_spellings_of_a_mistyped_flag_are_refused_alike() {
        let dropin = refuse(&["rocjitsu", "--nodes", "2", "--", "./app"]);
        let explicit = refuse(&["rocjitsu", "run", "--nodes", "2", "--", "./app"]);

        for msg in [&dropin, &explicit] {
            assert!(msg.contains("unexpected argument '--nodes'"), "{msg}");
            assert!(
                msg.contains("tip: a similar flag exists: '--num-nodes'"),
                "{msg}"
            );
        }
        // Identical but for the usage line and the help pointer, which
        // name the subcommand the user actually typed.
        assert!(explicit.contains("Usage: rocjitsu run [OPTIONS] -- <COMMAND> [ARGS]..."));
        assert!(explicit.contains("try 'rocjitsu run --help'"));
        assert!(dropin.contains("Usage: rocjitsu [OPTIONS] -- <COMMAND> [ARGS]..."));
        let body = |m: &str| m.split("\nUsage:").next().unwrap_or_default().to_string();
        assert_eq!(
            body(&dropin),
            body(&explicit),
            "the two spellings must not disagree about the same mistake"
        );
    }

    /// The typo with no separator at all, which is the one that used to
    /// start a session.
    #[test]
    fn a_mistyped_flag_is_refused_even_with_no_separator() {
        // Nothing in the parsed `argv` distinguishes this from
        // `rocjitsu run -- --nodes 2 /bin/true`, which is why the check
        // cannot live after the parse.
        let msg = refuse(&["rocjitsu", "run", "--nodes", "2", "/bin/true"]);
        assert!(msg.contains("unexpected argument '--nodes'"), "{msg}");

        // And a valued flag in front of it does not hide it: a scan that
        // did not know `--profile` takes a value would stop at `p` and
        // call it the command.
        let msg = refuse(&[
            "rocjitsu",
            "run",
            "--profile",
            "p",
            "--nodes",
            "2",
            "/bin/true",
        ]);
        assert!(msg.contains("unexpected argument '--nodes'"), "{msg}");
    }

    /// `exec` is judged against `exec`'s flags.
    #[test]
    fn exec_is_checked_against_its_own_flags() {
        let msg = refuse(&["rocjitsu", "exec", "--nodes", "2", "--", "./app"]);
        assert!(msg.contains("Usage: rocjitsu exec [OPTIONS] -- <COMMAND> [ARGS]..."));
        assert!(msg.contains("try 'rocjitsu exec --help'"), "{msg}");

        // `--session` is `exec`'s own and must pass, where the drop-in
        // list (which is `run`'s) has no such flag.
        assert_eq!(
            rewrite(&["rocjitsu", "exec", "--session", "s", "--", "./app"]),
            v_args(&["rocjitsu", "exec", "--session", "s", "--", "./app"]),
        );
    }

    /// What the guard must not touch.
    #[test]
    fn a_workloads_own_flags_and_separators_still_pass_through() {
        for case in [
            // The whole point of `allow_hyphen_values`: everything after
            // `--` is the workload's, flags included.
            vec![
                "rocjitsu",
                "run",
                "--",
                "./app",
                "--verbose",
                "--num-nodes",
                "4",
            ],
            // A program that really is named like a flag, spelled the
            // only way it can be — after rocjitsu's own separator.
            vec!["rocjitsu", "run", "--", "--weird"],
            // `git log -- path`: a separator the workload owns.
            vec!["rocjitsu", "run", "--", "git", "log", "--", "path"],
            // No separator, and the workload has flags of its own. This
            // spelling works today and must keep working.
            vec!["rocjitsu", "run", "./app", "--verbose", "--anything"],
            // A negative value is a value, not a flag.
            vec!["rocjitsu", "run", "--num-nodes", "-1", "--", "./app"],
        ] {
            assert_eq!(
                rewrite(&case),
                v_args(&case),
                "{case:?} is a workload, not a usage error"
            );
        }
    }

    #[test]
    fn bare_dropin_routes_to_run() {
        assert_eq!(
            rewrite(&["rocjitsu", "--", "./app", "arg"]),
            v_args(&["rocjitsu", "run", "--in-process", "--", "./app", "arg"])
        );
    }

    /// A legacy invocation keeps the legacy execution mode.
    ///
    /// Upstream ran the workload in-process unless asked for a daemon.
    /// `run` is the other way round, and passing a drop-in command line
    /// through untouched therefore moved every one of them into a mode
    /// it had never used — one that starts a second process and shares
    /// GPU memory through it.
    #[test]
    fn a_dropin_without_daemon_keeps_the_upstream_in_process_default() {
        assert_eq!(
            rewrite(&["rocjitsu", "--config", "c.json", "--", "./app"]),
            v_args(&[
                "rocjitsu",
                "run",
                "--in-process",
                "--config",
                "c.json",
                "--",
                "./app"
            ])
        );
        // And asking for a daemon is still how you get one. Nothing is
        // added there: `--in-process` would contradict the flag.
        for spelling in ["--daemon", "--attach"] {
            let out = rewrite(&["rocjitsu", spelling, "--config", "c.json", "--", "./app"]);
            assert!(
                !out.iter().any(|a| a == "--in-process"),
                "{spelling} asked for a daemon: {out:?}"
            );
        }
        // `rocjitsu run` itself is untouched: its own default is the
        // daemon, and that is not this rule's to change.
        let explicit = &["rocjitsu", "run", "--config", "c.json", "--", "./app"][..];
        assert_eq!(rewrite(explicit), v_args(explicit));
    }

    /// Upstream's daemon-only form, which has no workload and so no `--`
    /// for the rewriter to key on. It served the emulator and nothing
    /// else; `run` with an empty argv does the same and holds the
    /// session until it is stopped.
    #[test]
    fn daemon_only_without_a_workload_routes_to_run() {
        assert_eq!(
            rewrite(&["rocjitsu", "--daemon", "--config", "c.json"]),
            v_args(&["rocjitsu", "run", "--daemon", "--config", "c.json"])
        );
        // Without `--daemon` there is no such form: a config and no
        // workload is the mistake upstream also refused, and clap's
        // report of it is better than one invented here.
        let bare = &["rocjitsu", "--config", "c.json"][..];
        assert_eq!(rewrite(bare), v_args(bare));
    }

    /// Upstream spelled version `-v`, and a bare one has to keep meaning
    /// that.
    ///
    /// It is the cheapest thing a script does — check what it is talking
    /// to — and here it became "verbosity 1 and no subcommand", which
    /// exits 2. Only in isolation, though: everything else keeps `-v` as
    /// the first step of verbosity.
    #[test]
    fn a_bare_dash_v_still_asks_for_the_version() {
        assert_eq!(
            rewrite(&["rocjitsu", "-v"]),
            v_args(&["rocjitsu", "--version"])
        );

        // Not when it is modifying something. In each of these the user
        // has said what to do, so `-v` is how loudly, not what.
        for argv in [
            &["rocjitsu", "-vv"][..],
            &["rocjitsu", "-v", "emulators"][..],
            &["rocjitsu", "-v", "run", "--", "./app"][..],
        ] {
            assert_ne!(
                rewrite(argv),
                v_args(&["rocjitsu", "--version"]),
                "{argv:?} is verbosity, not a version probe"
            );
        }
    }

    /// And with the config paths people actually type.
    ///
    /// `c.json` above has no separator and does not exist, which is the
    /// one shape that survived reading option *values* as workloads. A
    /// documented invocation — `--config ../configs/gfx950_mi355x.json` —
    /// did not: the value was mistaken for the workload, the `run` splice
    /// was skipped, and the user got "unexpected argument '--daemon'".
    #[test]
    fn a_config_path_is_not_mistaken_for_the_workload() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let existing = dir.path().join("cdna4.json");
        std::fs::write(&existing, "{}").expect("a config file that exists");
        let absolute = existing.to_string_lossy().into_owned();

        for path in [
            // Relative, with a separator. The documented spelling.
            "../configs/gfx950_mi355x_kmd.json",
            // Absolute.
            &absolute,
            // Separator-free, but naming a file that is really there —
            // the other half of the old test, and the reason "does it
            // exist" is no better a question than "does it look like a
            // path".
            existing
                .file_name()
                .expect("a file name")
                .to_str()
                .expect("utf-8"),
        ] {
            let argv = ["rocjitsu", "--daemon", "--config", path];
            let expected = ["rocjitsu", "run", "--daemon", "--config", path];
            assert_eq!(rewrite(&argv), v_args(&expected), "--config {path}");
        }

        // The joined spelling is the same request and carries its value
        // with it, so there is no following token to misread either.
        assert_eq!(
            rewrite(&["rocjitsu", "--daemon", "--config=../configs/x.json"]),
            v_args(&["rocjitsu", "run", "--daemon", "--config=../configs/x.json"])
        );
    }

    /// The guard the fix above must not have removed.
    ///
    /// A path that is the *workload* still stops the splice: rewriting
    /// `rocjitsu --daemon ./app` into a session that holds itself open
    /// and never runs `./app` is a worse answer than refusing it.
    #[test]
    fn a_workload_without_a_separator_still_refuses_the_daemon_only_form() {
        for argv in [
            &["rocjitsu", "--daemon", "./app"][..],
            &["rocjitsu", "--daemon", "--config", "c.json", "./app"][..],
            &["rocjitsu", "--daemon", "/usr/bin/env"][..],
            // Found through `PATH`, so it has no separator and names no
            // file here. It is still a workload, and it used to be run:
            // the session came up and executed it, which is the opposite
            // of refusing a workload that brought no `--`.
            &["rocjitsu", "--daemon", "true"][..],
            &["rocjitsu", "--daemon", "--config", "c.json", "env"][..],
        ] {
            assert_eq!(rewrite(argv), v_args(argv), "{argv:?} must be left alone");
        }
    }

    #[test]
    fn rocjitsu_config_and_daemon_route_to_run() {
        assert_eq!(
            rewrite(&["rocjitsu", "--config", "c.json", "--daemon", "--", "./app"]),
            v_args(&[
                "rocjitsu", "run", "--config", "c.json", "--daemon", "--", "./app"
            ])
        );
    }

    /// `--attach` reaches `run` untouched, where it is refused with an
    /// explanation. The rewriter does not translate it: it used to mean
    /// `--daemon`, which is not what upstream did with it, and turning
    /// one into the other here would bury the explanation under a mode
    /// the user did not ask for.
    #[test]
    fn attach_reaches_run_untranslated() {
        assert_eq!(
            rewrite(&["rocjitsu", "--attach", "--config", "c.json", "--", "./app"]),
            v_args(&[
                "rocjitsu", "run", "--attach", "--config", "c.json", "--", "./app"
            ])
        );
        for argv in [
            &["rocjitsu", "run", "--attach", "--", "./app"][..],
            &["rocjitsu", "run", "--attach", "./app"][..],
        ] {
            assert_eq!(rewrite(argv), v_args(argv), "{argv:?} needs no rewriting");
        }
    }

    /// And the parser really does take it, which is what lets it be
    /// refused with an explanation rather than by clap.
    ///
    /// `--attach` is a breaking change, not an alias: the refusal in
    /// `check_run_args` is the contract, and it can only be reached if
    /// the flag parses. If `run` ever stopped declaring it, the test
    /// above would still pass while the spelling this CLI is a drop-in
    /// for started failing with "unexpected argument" — which tells a
    /// script written against upstream nothing about where the
    /// capability went.
    #[test]
    fn run_accepts_the_attach_spelling() {
        use clap::Parser as _;
        for argv in [
            &["rocjitsu", "run", "--attach", "--", "./app"][..],
            &["rocjitsu", "run", "--daemon", "--", "./app"][..],
        ] {
            Cli::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?} should parse: {e}"));
        }
    }

    /// The rewriter stops at the separator: `--attach` is a rocjitsu flag
    /// in front of it and the workload's own argument behind it. Behind
    /// it, it does not even count as asking for a daemon — which is why
    /// the second case still gets the in-process default.
    #[test]
    fn attach_after_the_separator_belongs_to_the_workload() {
        assert_eq!(
            rewrite(&["rocjitsu", "run", "--", "./app", "--attach"]),
            v_args(&["rocjitsu", "run", "--", "./app", "--attach"])
        );
        assert_eq!(
            rewrite(&["rocjitsu", "--", "./app", "--attach"]),
            v_args(&["rocjitsu", "run", "--in-process", "--", "./app", "--attach"])
        );
    }

    /// Only `run` takes `--attach`, and on any other subcommand it is a
    /// mistyped flag like any other.
    ///
    /// This used to be left to clap, because the guard did not look at
    /// explicit subcommands at all. Now that it does, `exec` is judged
    /// against `exec`'s flags and says so itself — which is the same
    /// answer, arrived at before a session could be started for it.
    #[test]
    fn attach_is_not_an_exec_flag() {
        let msg = refuse(&["rocjitsu", "exec", "--attach", "--", "cmd"]);
        assert!(msg.contains("unexpected argument '--attach'"), "{msg}");
        assert!(msg.contains("try 'rocjitsu exec --help'"), "{msg}");

        // And it is still `run`'s, in both spellings of `run`.
        let args = v_args(&["rocjitsu", "run", "--attach", "--", "cmd"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
    }

    #[test]
    fn global_flags_before_dropin_are_preserved() {
        assert_eq!(
            rewrite(&["rocjitsu", "--json", "--profile", "mi350x", "--", "./app"]),
            v_args(&[
                "rocjitsu",
                "--json",
                "run",
                "--in-process",
                "--profile",
                "mi350x",
                "--",
                "./app"
            ])
        );
    }

    /// The heart of the drop-in rule: a token before `--` that is shaped
    /// like a flag has to be one rocjitsu takes. It used to be enough that
    /// it was *not* a subcommand — so `rocjitsu --nodes 2 -- ./app` decided
    /// this was a bare rocjitsu-style call, brought the whole emulated
    /// machine up, and exited 127 trying to execute `--nodes` in it.
    #[test]
    fn a_mistyped_flag_before_the_separator_is_a_usage_error() {
        let usage = refuse(&["rocjitsu", "--nodes", "2", "--", "./app"]);
        assert!(usage.contains("unexpected argument '--nodes'"), "{usage}");
        // And it names the flag the user almost certainly meant.
        assert!(usage.contains("'--num-nodes'"), "{usage}");
    }

    /// The other half of the rule, which is what makes it a rule rather
    /// than a blocklist: everything `run` declares still flows through,
    /// aliases and short spellings included, because the accepted set is
    /// asked of clap rather than written out here.
    #[test]
    fn every_flag_run_accepts_is_accepted_before_the_separator() {
        use clap::CommandFactory as _;
        let cmd = Cli::command();
        let run = cmd.find_subcommand("run").expect("`run` is a subcommand");
        let known = AcceptedFlags::of(&cmd, "run");
        for arg in run.get_arguments() {
            // `--help`/`--version` are clap's, and are answered before
            // the rewriter ever considers routing to `run`.
            if matches!(arg.get_id().as_str(), "help" | "version") {
                continue;
            }
            for spelling in arg
                .get_long()
                .into_iter()
                .chain(arg.get_all_aliases().unwrap_or_default())
            {
                let flag = format!("--{spelling}");
                assert!(
                    known.accepts(&flag),
                    "`rocjitsu run` accepts {flag} but the drop-in rewriter would \
                     refuse it before `--`"
                );
                // A flag that names the execution mode is answering the
                // question the in-process default exists to answer, so
                // nothing is supplied alongside it.
                let names_the_mode =
                    matches!(flag.as_str(), "--daemon" | "--attach" | "--in-process");
                let mut want = vec!["rocjitsu", "run"];
                if !names_the_mode {
                    want.push("--in-process");
                }
                want.extend([flag.as_str(), "x", "--", "./app"]);
                assert_eq!(
                    rewrite(&["rocjitsu", &flag, "x", "--", "./app"]),
                    v_args(&want),
                    "{flag} should route to `run`, not be refused"
                );
            }
            for short in arg
                .get_short()
                .into_iter()
                .chain(arg.get_all_short_aliases().unwrap_or_default())
            {
                let flag = format!("-{short}");
                assert!(
                    known.accepts(&flag),
                    "`rocjitsu run` accepts {flag} but the drop-in rewriter would \
                     refuse it before `--`"
                );
            }
        }
    }

    /// A value that merely starts with `-` is not a flag. Refusing it
    /// here would replace clap's "invalid value for --num-nodes", which
    /// names the valid range, with a worse message about a flag that
    /// does not exist.
    #[test]
    fn a_negative_value_is_not_mistaken_for_a_flag() {
        assert_eq!(
            rewrite(&["rocjitsu", "--num-nodes", "-1", "--", "./app"]),
            v_args(&[
                "rocjitsu",
                "run",
                "--in-process",
                "--num-nodes",
                "-1",
                "--",
                "./app"
            ])
        );
    }

    /// `--long=value` and `-svalue` are spellings clap accepts, so the
    /// guard has to recognise the flag inside them.
    #[test]
    fn joined_flag_values_are_recognised() {
        assert_eq!(
            rewrite(&["rocjitsu", "--config=c.json", "--", "./app"]),
            v_args(&[
                "rocjitsu",
                "run",
                "--in-process",
                "--config=c.json",
                "--",
                "./app"
            ])
        );
        assert_eq!(
            rewrite(&["rocjitsu", "-okey=value", "--", "./app"]),
            v_args(&[
                "rocjitsu",
                "run",
                "--in-process",
                "-okey=value",
                "--",
                "./app"
            ])
        );
    }

    /// `rocjitsu --` used to be rewritten first and reported second, so
    /// clap complained about a missing argument of `rocjitsu run` — a
    /// subcommand the user never typed.
    #[test]
    fn a_separator_with_nothing_after_it_is_explained_without_naming_run() {
        let usage = refuse(&["rocjitsu", "--"]);
        assert!(usage.contains("`--` was given with no command"), "{usage}");
        assert!(
            !usage.contains("rocjitsu run <"),
            "the usage line should not name a subcommand the user did not type: {usage}"
        );
        // The same for a drop-in that got as far as its flags.
        let usage = refuse(&["rocjitsu", "--config", "c.json", "--"]);
        assert!(usage.contains("`--` was given with no command"), "{usage}");
    }

    /// Naming a program with no `--` in front of it is a dead end clap
    /// can only report as an unrecognised subcommand. Drop-in mode is a
    /// headline feature; the error is the one place the user is
    /// guaranteed to read.
    #[test]
    fn a_program_with_no_separator_is_pointed_at_the_separator() {
        let usage = refuse(&["rocjitsu", "./app", "--flag"]);
        assert!(usage.contains("rocjitsu -- ./app --flag"), "{usage}");
    }

    /// …but only when the token cannot plausibly be a misspelt
    /// subcommand, because clap's own "did you mean" is better than
    /// anything said here.
    #[test]
    fn a_misspelt_subcommand_is_left_to_clap() {
        let args = v_args(&["rocjitsu", "profil", "list"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
    }

    /// `-vvv` used to be mistaken for the subcommand, which spliced `run`
    /// in front of it and turned the real `run` into the workload —
    /// `rocjitsu -vvv -- ./app` died with `command not found: run` while
    /// `-vv` worked. Counted flags have no upper bound, so neither does
    /// this.
    #[test]
    fn bundled_verbosity_of_any_depth_finds_the_subcommand() {
        for v in ["-v", "-vv", "-vvv", "-vvvv", "-vvvvvvvvvv"] {
            let args = v_args(&["rocjitsu", v, "run", "--", "./app"]);
            assert_eq!(
                rewrite(&["rocjitsu", v, "run", "--", "./app"]),
                args,
                "{v} should leave an explicit `run` alone"
            );
            assert_eq!(
                rewrite(&["rocjitsu", v, "--", "./app"]),
                v_args(&["rocjitsu", v, "run", "--in-process", "--", "./app"]),
                "{v} should be stepped over when splicing `run`"
            );
        }
    }

    /// `cleanup` was missing from the hardcoded subcommand list, so
    /// `rocjitsu cleanup -- echo hi` brought up an emulated session and
    /// tried to run a program called `cleanup` in it. Every subcommand
    /// clap knows about must be recognised, so ask clap for all of them
    /// rather than trusting a list — including `run` itself, which must
    /// not be spliced in front of.
    #[test]
    fn every_subcommand_is_recognised() {
        use clap::CommandFactory as _;
        for sub in Cli::command().get_subcommands() {
            let name = sub.get_name().to_string();
            assert!(
                is_subcommand(&name),
                "`{name}` is a subcommand but `is_subcommand` does not know it"
            );
            let args = v_args(&["rocjitsu", &name, "--", "./app"]);
            assert_eq!(
                dropin_argv(args.clone()).unwrap(),
                args,
                "`rocjitsu {name} -- ./app` was rewritten; it names a subcommand \
                 and must be left alone"
            );
        }
    }

    /// The guard has to run for *every* subcommand that ends in a
    /// workload, not for the two that did when it was written.
    ///
    /// Whether `check_flags` runs at all was decided by a hardcoded pair
    /// of names sitting next to `is_subcommand` — the function that
    /// exists because its own hardcoded list was missing `cleanup`. A
    /// third workload subcommand, or an alias of one of these, would have
    /// been parsed with `trailing_var_arg` and no guard in front of it,
    /// which is the original bug: `rocjitsu <sub> --nodes 2 ./app` brings a
    /// session up and tries to execute `--nodes`.
    ///
    /// So the population is taken from clap: a positional declared
    /// `trailing_var_arg` is the declaration of "everything from here is
    /// the workload's", and every subcommand that has one must refuse a
    /// flag rocjitsu does not take — in both spellings, since the one with
    /// no `--` is the one that used to be undecidable.
    #[test]
    fn every_subcommand_that_ends_in_a_workload_is_guarded() {
        let mut guarded = 0;
        for sub in cli().get_subcommands() {
            let name = sub.get_name().to_string();
            // Through the same predicate the guard itself uses, so the
            // population this test checks cannot be narrower than the
            // population that gets guarded. Asking a second, hand-written
            // question here would let a subcommand fall out of both at
            // once and the test still go green.
            if !ends_in_a_workload(sub) {
                // No trailing positional, so an unknown flag is clap's to
                // report and the rewriter must leave the line alone.
                let args = v_args(&["rocjitsu", &name, "--not-a-rocjitsu-flag"]);
                assert_eq!(
                    dropin_argv(args.clone()).unwrap(),
                    args,
                    "`rocjitsu {name}` does not end in a workload, so its flags \
                     are clap's business"
                );
                continue;
            }
            guarded += 1;
            // Both spellings. The one with no `--` is the one the old
            // post-parse check could not see at all.
            for argv in [
                v_args(&["rocjitsu", &name, "--not-a-rocjitsu-flag", "--", "./app"]),
                v_args(&["rocjitsu", &name, "--not-a-rocjitsu-flag", "./app"]),
            ] {
                match dropin_argv(argv.clone()) {
                    Err(usage) => assert!(
                        usage.contains("unexpected argument '--not-a-rocjitsu-flag'"),
                        "`rocjitsu {name}` refused {argv:?} for the wrong reason:\
                         {usage}"
                    ),
                    Ok(out) => panic!(
                        "`rocjitsu {name}` ends in a workload but its flags are \
                         unguarded: {argv:?} became {out:?}"
                    ),
                }
            }
        }
        // Nothing above asserts anything if clap reports no such
        // subcommand, which would be a silent pass.
        assert!(
            guarded >= 2,
            "`run` and `exec` both end in a workload; only {guarded} \
             subcommand(s) were found to guard"
        );
    }

    /// The inverse: something that is *not* a subcommand still routes to
    /// `run`, so fixing the list did not break the drop-in itself.
    #[test]
    fn a_non_subcommand_still_routes_to_run() {
        assert!(!is_subcommand("definitely-not-a-subcommand"));
        assert_eq!(
            rewrite(&["rocjitsu", "--config", "c.json", "--", "./app"]),
            v_args(&[
                "rocjitsu",
                "run",
                "--in-process",
                "--config",
                "c.json",
                "--",
                "./app"
            ])
        );
    }

    /// Repeating a boolean flag is not an error anywhere else, and a
    /// user reaches this by composing a command line from a variable that
    /// already carries `--json`.
    #[test]
    fn a_repeated_json_flag_is_accepted() {
        use clap::Parser as _;
        for argv in [
            &["rocjitsu", "paths", "--json"][..],
            &["rocjitsu", "paths", "--json", "--json"][..],
            &["rocjitsu", "--json", "paths", "--json"][..],
        ] {
            let cli =
                Cli::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?} should parse: {e}"));
            assert!(cli.json, "{argv:?} should set --json");
        }
        assert!(
            !Cli::try_parse_from(["rocjitsu", "paths"]).unwrap().json,
            "--json must still default to off"
        );
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
        let args = v_args(&["rocjitsu", "run", "--profile", "mi350x", "--", "./app"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
    }

    #[test]
    fn other_subcommands_are_untouched() {
        let args = v_args(&["rocjitsu", "exec", "--session", "s", "--", "cmd"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
    }

    #[test]
    fn no_separator_is_untouched() {
        let args = v_args(&["rocjitsu", "profile", "list"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
        let help = v_args(&["rocjitsu", "--help"]);
        assert_eq!(dropin_argv(help.clone()).unwrap(), help);
    }

    /// `--vfio-socket` routes to its own subcommand, not to `run`.
    ///
    /// The C++ CLI dispatched it before it built a machine, and the mode
    /// it starts cannot share `run`'s signal handling: the server takes
    /// SIGINT, SIGTERM and SIGUSR1 for itself, and `run` means something
    /// else by each of them.
    #[cfg(feature = "rocjitsu")]
    #[test]
    fn a_vfio_socket_routes_to_its_own_subcommand() {
        assert_eq!(
            rewrite(&["rocjitsu", "--config", "c.json", "--vfio-socket", "/tmp/s"]),
            v_args(&[
                "rocjitsu",
                "vfio-serve",
                "--config",
                "c.json",
                "--vfio-socket",
                "/tmp/s"
            ])
        );
        // The joined spelling is the same request.
        assert_eq!(
            rewrite(&["rocjitsu", "--config", "c.json", "--vfio-socket=/tmp/s"]),
            v_args(&[
                "rocjitsu",
                "vfio-serve",
                "--config",
                "c.json",
                "--vfio-socket=/tmp/s"
            ])
        );
        // An explicit subcommand is left alone, as every other one is.
        let explicit = &[
            "rocjitsu",
            "vfio-serve",
            "--config",
            "c.json",
            "--vfio-socket",
            "/tmp/s",
        ][..];
        assert_eq!(rewrite(explicit), v_args(explicit));
    }

    /// And it is a whole mode, so it refuses to be half of one.
    #[cfg(feature = "rocjitsu")]
    #[test]
    fn a_vfio_socket_refuses_to_be_combined_with_a_run() {
        for argv in [
            &[
                "rocjitsu",
                "--config",
                "c.json",
                "--vfio-socket",
                "/tmp/s",
                "--",
                "./app",
            ][..],
            &[
                "rocjitsu",
                "--daemon",
                "--config",
                "c.json",
                "--vfio-socket",
                "/tmp/s",
            ][..],
            &[
                "rocjitsu",
                "--attach",
                "--config",
                "c.json",
                "--vfio-socket",
                "/tmp/s",
            ][..],
        ] {
            let msg = refuse(argv);
            assert!(msg.contains("--vfio-socket"), "{argv:?}: {msg}");
            assert!(msg.contains("vfio-serve"), "{argv:?}: {msg}");
        }
    }

    /// The older CLI reported thread allocations from a bare invocation,
    /// and the shipped docs still spell it that way, so the flag has to
    /// keep selecting the mode it always selected.
    #[cfg(feature = "rocjitsu")]
    #[test]
    fn a_thread_budget_table_routes_to_its_own_subcommand() {
        // The flag named the mode; the subcommand now does, so it is not
        // carried through to be rejected by the command it selected.
        assert_eq!(
            rewrite(&["rocjitsu", "--config", "c.json", "--thread-budget-table"]),
            v_args(&["rocjitsu", "thread-budget-table", "--config", "c.json"])
        );
        // Order is the user's to choose, not ours.
        assert_eq!(
            rewrite(&["rocjitsu", "--thread-budget-table", "--config", "c.json"]),
            v_args(&["rocjitsu", "thread-budget-table", "--config", "c.json"])
        );
        // An explicit subcommand is left alone, as every other one is.
        let explicit = &["rocjitsu", "thread-budget-table", "--config", "c.json"][..];
        assert_eq!(rewrite(explicit), v_args(explicit));
    }

    /// It reads a config and runs nothing, so it is not half of a launch.
    #[cfg(feature = "rocjitsu")]
    #[test]
    fn a_thread_budget_table_refuses_to_be_combined_with_a_run() {
        for argv in [
            &[
                "rocjitsu",
                "--config",
                "c.json",
                "--thread-budget-table",
                "--",
                "./app",
            ][..],
            &[
                "rocjitsu",
                "--daemon",
                "--config",
                "c.json",
                "--thread-budget-table",
            ][..],
            &[
                "rocjitsu",
                "--attach",
                "--config",
                "c.json",
                "--thread-budget-table",
            ][..],
        ] {
            let msg = refuse(argv);
            assert!(msg.contains("--thread-budget-table"), "{argv:?}: {msg}");
        }
    }

    /// The parser really does accept what that rewrite produces.
    #[cfg(feature = "rocjitsu")]
    #[test]
    fn thread_budget_table_parses_what_the_rewriter_builds() {
        use clap::Parser as _;
        let argv = rewrite(&["rocjitsu", "--config", "c.json", "--thread-budget-table"]);
        let cli = Cli::try_parse_from(&argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        match cli.command {
            TopCmd::ThreadBudgetTable(a) => assert_eq!(a.config, "c.json"),
            other => panic!("{argv:?} should be thread-budget-table, not {other:?}"),
        }
        // There is nothing to report on without a config.
        Cli::try_parse_from(["rocjitsu", "thread-budget-table"])
            .expect_err("a table needs a config to report on");
    }

    /// Budget zero leads, and stands for the config's own request; the
    /// rest are the ceilings the shipped tables in the docs are indexed
    /// by, so a reader can line the two up row for row.
    #[cfg(feature = "rocjitsu")]
    #[test]
    fn the_reported_budgets_are_the_ones_the_docs_tabulate() {
        assert_eq!(
            rj_backend_rocjitsu::THREAD_BUDGET_TABLE_ROWS,
            // 34, 36 and 40 are the additive helper rows: engine and
            // dispatch cost is capped at 32, and helpers go above it.
            &[0, 1, 2, 4, 8, 12, 16, 24, 32, 34, 36, 40, 48, 64]
        );
    }

    /// The parser really does accept what the rewriter produces.
    #[cfg(feature = "rocjitsu")]
    #[test]
    fn vfio_serve_parses_what_the_rewriter_builds() {
        use clap::Parser as _;
        let argv = rewrite(&["rocjitsu", "--config", "c.json", "--vfio-socket", "/tmp/s"]);
        let cli = Cli::try_parse_from(&argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        match cli.command {
            TopCmd::VfioServe(a) => {
                assert_eq!(a.config, "c.json");
                assert_eq!(a.vfio_socket, "/tmp/s");
            }
            other => panic!("{argv:?} should be vfio-serve, not {other:?}"),
        }
        // Both halves are required: a socket with nothing to serve on it,
        // or a config with nowhere to serve it, is not a usable request.
        for partial in [
            &["rocjitsu", "vfio-serve", "--config", "c.json"][..],
            &["rocjitsu", "vfio-serve", "--vfio-socket", "/tmp/s"][..],
        ] {
            Cli::try_parse_from(partial).expect_err(&format!("{partial:?} is incomplete"));
        }
    }
}
