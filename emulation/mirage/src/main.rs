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
#[command(
    name = "mirage",
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
    // repeat is not an error. Without it `mirage paths --json --json`
    // exits 2, which a user hits by composing a command line from a
    // variable that already carries the flag. Written with `//` and not
    // `///`: a doc comment here is printed to the user by `--help`, and
    // why the attribute is there is nobody's business but ours.
    #[arg(long, global = true, overrides_with = "json")]
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

/// The status clap exits with on a usage error, and therefore the one a
/// usage error mirage diagnoses for itself must use too.
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
    mirage_ctl::init_logging(cli.verbose);
    match dispatch(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

/// Print Mirage's package version and RocJITsu's shared build identity.
fn print_version() -> ExitCode {
    println!("mirage {}", env!("CARGO_PKG_VERSION"));
    #[cfg(feature = "rocjitsu")]
    match mirage_rocjitsu::version_string() {
        Ok(version) => println!("{version}"),
        Err(error) => println!("rocjitsu build identity unavailable: {error}"),
    }
    #[cfg(not(feature = "rocjitsu"))]
    println!("rocjitsu build identity unavailable: backend not built");
    ExitCode::SUCCESS
}

/// The `mirage` command tree, built once.
///
/// Everything on the argv-rewrite path asks clap rather than a list of
/// its own — which is the right answer to "what does mirage accept?" and
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

/// Whether `name` is a top-level subcommand `mirage` understands.
///
/// Used to decide whether an invocation is a normal `mirage <subcommand>
/// …` call or a bare, `rocjitsu`-style `mirage [--config …] [--daemon] --
/// <app>` call that should be routed to `run`.
///
/// Asked of clap rather than kept as a list here. It *was* a list, and the
/// list was missing `cleanup`: `mirage cleanup -- echo hi` did not run
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

/// One-line summary of what mirage is, shared by `about` and the JSON
/// form so the two cannot drift.
const DESCRIPTION: &str = "A UX for the rocjitsu (and other) GPU emulators.";

const COPYRIGHT: &str = "Copyright (c) Advanced Micro Devices, Inc. All rights reserved.";

const LICENSE: &str = "Licensed under the terms of mirage's LICENSE.";

/// The `mirage [options] -- <app>` shape, appended to `mirage -h`.
///
/// Clap can only list subcommands, and drop-in mode is the one
/// invocation that has none — so a user reading the help sees every way
/// of running mirage except the headline one. Spelling it out here is
/// what makes it discoverable at all; see [`dropin_argv`].
const DROPIN_HELP: &str = "\
Drop-in mode:
  mirage [OPTIONS] -- <COMMAND> [ARGS]...   run a workload, no subcommand

Use 'mirage --help' for the full form.";

/// The same, for `mirage --help`, where there is room for the reason and
/// an example of each shape.
const DROPIN_LONG_HELP: &str = "\
Drop-in mode:
  A `--` with no subcommand before it runs a workload on an emulated
  machine, exactly as `mirage run` would, so that scripts written for the
  upstream `rocjitsu` CLI keep working unchanged:

      mirage -- ./my-rocm-app --flag
      mirage --config cfg.json -- ./my-rocm-app
      mirage --profile cdna4 --num-nodes 2 -- python train.py

  Everything `mirage run` accepts may appear before the `--`, plus the
  rocjitsu-only spelling `--attach` (an alias for `--daemon`). Anything
  else there is a mistyped flag rather than part of the workload, and is
  refused. See 'mirage run --help' for the full list.";

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

/// The subcommand a drop-in invocation is routed to.
///
/// Also the only one that accepts `--attach`, the `rocjitsu` spelling of
/// `--daemon`. It used to be translated here, against an `ATTACH`
/// constant this comment outlived; `run` declares it as a clap alias
/// now, so an explicit `mirage run --attach` needs nothing from the
/// rewriter and neither does a drop-in one.
const RUN: &str = "run";

/// The one-line shape shown by every usage error [`dropin_argv`] raises.
const DROPIN_USAGE: &str = "Usage: mirage [OPTIONS] -- <COMMAND> [ARGS]...";

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

/// Whether the subcommand `name` names ends in a workload, and so has a
/// `--` and a span of mirage's own flags in front of it.
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

/// Check the span of `args` that belongs to mirage rather than to the
/// workload, and turn the first mistyped flag in it into a usage error.
///
/// `from` is where mirage's own flags start — after the subcommand, or
/// after the program name for a drop-in — and `sep` is the index of the
/// `--`, when there is one. Which there is decides how far the scan goes;
/// see [`mirage_ctl::usage::Span`].
///
/// # Errors
///
/// Returns the message to print for the first flag-shaped token mirage
/// does not accept.
fn check_flags(args: &[String], from: usize, sep: Option<usize>, sub: &str) -> Result<(), String> {
    use mirage_ctl::usage::{AcceptedFlags, Span, misplaced_flag, unknown_flag_error};

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
            format!("Usage: mirage {sub} [OPTIONS] -- <COMMAND> [ARGS]..."),
            format!("mirage {sub} --help"),
        )
    } else {
        (DROPIN_USAGE.to_string(), "mirage run --help".to_string())
    };
    Err(unknown_flag_error(bad, &known, &usage, &help))
}

/// Make `mirage` a drop-in replacement for the `rocjitsu` CLI by routing
/// bare `mirage [opts] -- <app> [args…]` invocations to `mirage run`.
///
/// The upstream `rocjitsu` CLI is invoked as
/// `rocjitsu --config <cfg> [--daemon|--attach] -- <app>`; there is no
/// subcommand. `mirage` is subcommand-based, so when an invocation has
/// the `rocjitsu` shape — a `--` application separator with no
/// recognised subcommand before it — we splice in `run` and translate
/// `--attach` to `--daemon`. Everything `run` already accepts
/// (`--config`, `--profile`, `--daemon`, `--env`, …) then flows straight
/// through.
///
/// Invocations that name a subcommand (`mirage run …`, `mirage profile
/// …`, `mirage exec --session s -- cmd`) and those with no `--` separator
/// (so `--help`/`--version` keep working) are left untouched. `run`
/// declares `--attach` as an alias of `--daemon` itself, so an explicit
/// `mirage run --attach` needs nothing from here.
///
/// # Errors
///
/// Returns the usage message to print when the invocation has the
/// drop-in shape but cannot be one. The rewriter used to treat *any*
/// unrecognised token before `--` as "no subcommand, therefore a
/// workload", which is right for `--config` and wrong for a typo:
/// `mirage --nodes 2 -- ./app` brought a whole emulated machine up and
/// then failed to execute a program called `--nodes` inside it. A token
/// that is shaped like a flag and is not one mirage takes is a mistake
/// the user wants to hear about before anything boots.
fn dropin_argv(args: Vec<String>) -> Result<Vec<String>, String> {
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
    // A recognised subcommand means this is a normal mirage call, so it
    // is not rewritten. That includes `run --attach`: `--attach` is a
    // clap alias of `--daemon` on `run` itself now, so there is nothing
    // here to translate.
    //
    // It still gets the flag guard, though, and that is the point of
    // doing it here. `mirage run --nodes 2 -- ./app` is the same mistake
    // as `mirage --nodes 2 -- ./app` in the spelling people actually use,
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
    let Some(sep) = sep else {
        // Without a separator there is nothing to rewrite. Clap reports
        // the unrecognised subcommand, and does it better than we could
        // — unless what the user typed is obviously a program, in which
        // case the missing piece is the `--`, not the spelling.
        return match head {
            Some(h) if looks_like_a_program(h) => Err(missing_separator_error(
                &args[1..head_idx],
                &args[head_idx..],
            )),
            _ => Ok(args),
        };
    };
    // Everything before the separator is mirage's own; a flag there that
    // mirage does not take is a typo, not a workload.
    check_flags(&args, 1, Some(sep), RUN)?;
    if sep + 1 == args.len() {
        return Err(empty_separator_error());
    }
    // Drop-in: splice `run` in where the subcommand would go.
    let mut out = Vec::with_capacity(args.len() + 1);
    out.extend(args[..head_idx].iter().cloned());
    out.push(RUN.to_string());
    out.extend(args[head_idx..].iter().cloned());
    Ok(out)
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
         For example:\n  mirage --profile mi350x -- ./my-rocm-app --flag\n\n\
         For more information, try 'mirage run --help'."
    )
}

/// The message for `mirage ./app` — a workload named with no `--` in
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
         To run '{program}' on an emulated machine, separate it from mirage's \
         own flags\nwith `--`:\n\n  mirage {opts}-- {}\n\n\
         {DROPIN_USAGE}\n\n\
         For more information, try 'mirage --help'.",
        argv.join(" ")
    )
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

    use mirage_ctl::usage::AcceptedFlags;

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
    /// was false. `mirage --nodes 2 -- ./app` was refused by the drop-in
    /// rewriter; `mirage run --nodes 2 -- ./app` — the commoner spelling
    /// of the same typo — was checked somewhere else, by a second copy of
    /// the rules, and `mirage run --nodes 2 ./app` was not checked at all
    /// and brought a whole emulated machine up to run a program called
    /// `--nodes`.
    #[test]
    fn both_spellings_of_a_mistyped_flag_are_refused_alike() {
        let dropin = refuse(&["mirage", "--nodes", "2", "--", "./app"]);
        let explicit = refuse(&["mirage", "run", "--nodes", "2", "--", "./app"]);

        for msg in [&dropin, &explicit] {
            assert!(msg.contains("unexpected argument '--nodes'"), "{msg}");
            assert!(
                msg.contains("tip: a similar flag exists: '--num-nodes'"),
                "{msg}"
            );
        }
        // Identical but for the usage line and the help pointer, which
        // name the subcommand the user actually typed.
        assert!(explicit.contains("Usage: mirage run [OPTIONS] -- <COMMAND> [ARGS]..."));
        assert!(explicit.contains("try 'mirage run --help'"));
        assert!(dropin.contains("Usage: mirage [OPTIONS] -- <COMMAND> [ARGS]..."));
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
        // `mirage run -- --nodes 2 /bin/true`, which is why the check
        // cannot live after the parse.
        let msg = refuse(&["mirage", "run", "--nodes", "2", "/bin/true"]);
        assert!(msg.contains("unexpected argument '--nodes'"), "{msg}");

        // And a valued flag in front of it does not hide it: a scan that
        // did not know `--profile` takes a value would stop at `p` and
        // call it the command.
        let msg = refuse(&[
            "mirage",
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
        let msg = refuse(&["mirage", "exec", "--nodes", "2", "--", "./app"]);
        assert!(msg.contains("Usage: mirage exec [OPTIONS] -- <COMMAND> [ARGS]..."));
        assert!(msg.contains("try 'mirage exec --help'"), "{msg}");

        // `--session` is `exec`'s own and must pass, where the drop-in
        // list (which is `run`'s) has no such flag.
        assert_eq!(
            rewrite(&["mirage", "exec", "--session", "s", "--", "./app"]),
            v_args(&["mirage", "exec", "--session", "s", "--", "./app"]),
        );
    }

    /// What the guard must not touch.
    #[test]
    fn a_workloads_own_flags_and_separators_still_pass_through() {
        for case in [
            // The whole point of `allow_hyphen_values`: everything after
            // `--` is the workload's, flags included.
            vec![
                "mirage",
                "run",
                "--",
                "./app",
                "--verbose",
                "--num-nodes",
                "4",
            ],
            // A program that really is named like a flag, spelled the
            // only way it can be — after mirage's own separator.
            vec!["mirage", "run", "--", "--weird"],
            // `git log -- path`: a separator the workload owns.
            vec!["mirage", "run", "--", "git", "log", "--", "path"],
            // No separator, and the workload has flags of its own. This
            // spelling works today and must keep working.
            vec!["mirage", "run", "./app", "--verbose", "--anything"],
            // A negative value is a value, not a flag.
            vec!["mirage", "run", "--num-nodes", "-1", "--", "./app"],
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
            rewrite(&["mirage", "--", "./app", "arg"]),
            v_args(&["mirage", "run", "--", "./app", "arg"])
        );
    }

    #[test]
    fn rocjitsu_config_and_daemon_route_to_run() {
        assert_eq!(
            rewrite(&["mirage", "--config", "c.json", "--daemon", "--", "./app"]),
            v_args(&[
                "mirage", "run", "--config", "c.json", "--daemon", "--", "./app"
            ])
        );
    }

    /// `--attach` is the upstream `rocjitsu` spelling of `--daemon`, and
    /// `run` declares it as a clap alias, so the rewriter passes it
    /// through untouched rather than translating it. It used to do the
    /// translation itself, which meant the alias worked only on the
    /// drop-in path: `mirage run --attach -- app` brought a session up
    /// and exited 127 with `command not found: --attach`. One mechanism
    /// cannot disagree with itself.
    #[test]
    fn attach_reaches_run_untranslated() {
        assert_eq!(
            rewrite(&["mirage", "--attach", "--config", "c.json", "--", "./app"]),
            v_args(&[
                "mirage", "run", "--attach", "--config", "c.json", "--", "./app"
            ])
        );
        for argv in [
            &["mirage", "run", "--attach", "--", "./app"][..],
            &["mirage", "run", "--attach", "./app"][..],
        ] {
            assert_eq!(rewrite(argv), v_args(argv), "{argv:?} needs no rewriting");
        }
    }

    /// And the alias really is accepted by the parser, which is the half
    /// the rewriter now relies on: if `run` ever stopped declaring it,
    /// the test above would still pass while every spelling broke.
    #[test]
    fn run_accepts_the_attach_alias() {
        use clap::Parser as _;
        for argv in [
            &["mirage", "run", "--attach", "--", "./app"][..],
            &["mirage", "run", "--daemon", "--", "./app"][..],
        ] {
            Cli::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?} should parse: {e}"));
        }
    }

    /// The translation stops at the separator: `--attach` is a mirage
    /// flag in front of it and the workload's own argument behind it.
    #[test]
    fn attach_after_the_separator_belongs_to_the_workload() {
        assert_eq!(
            rewrite(&["mirage", "run", "--", "./app", "--attach"]),
            v_args(&["mirage", "run", "--", "./app", "--attach"])
        );
        assert_eq!(
            rewrite(&["mirage", "--", "./app", "--attach"]),
            v_args(&["mirage", "run", "--", "./app", "--attach"])
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
        let msg = refuse(&["mirage", "exec", "--attach", "--", "cmd"]);
        assert!(msg.contains("unexpected argument '--attach'"), "{msg}");
        assert!(msg.contains("try 'mirage exec --help'"), "{msg}");

        // And it is still `run`'s, in both spellings of `run`.
        let args = v_args(&["mirage", "run", "--attach", "--", "cmd"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
    }

    #[test]
    fn global_flags_before_dropin_are_preserved() {
        assert_eq!(
            rewrite(&["mirage", "--json", "--profile", "mi350x", "--", "./app"]),
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

    /// The heart of the drop-in rule: a token before `--` that is shaped
    /// like a flag has to be one mirage takes. It used to be enough that
    /// it was *not* a subcommand — so `mirage --nodes 2 -- ./app` decided
    /// this was a bare rocjitsu-style call, brought the whole emulated
    /// machine up, and exited 127 trying to execute `--nodes` in it.
    #[test]
    fn a_mistyped_flag_before_the_separator_is_a_usage_error() {
        let usage = refuse(&["mirage", "--nodes", "2", "--", "./app"]);
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
                    "`mirage run` accepts {flag} but the drop-in rewriter would \
                     refuse it before `--`"
                );
                assert_eq!(
                    rewrite(&["mirage", &flag, "x", "--", "./app"]),
                    v_args(&["mirage", "run", &flag, "x", "--", "./app"]),
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
                    "`mirage run` accepts {flag} but the drop-in rewriter would \
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
            rewrite(&["mirage", "--num-nodes", "-1", "--", "./app"]),
            v_args(&["mirage", "run", "--num-nodes", "-1", "--", "./app"])
        );
    }

    /// `--long=value` and `-svalue` are spellings clap accepts, so the
    /// guard has to recognise the flag inside them.
    #[test]
    fn joined_flag_values_are_recognised() {
        assert_eq!(
            rewrite(&["mirage", "--config=c.json", "--", "./app"]),
            v_args(&["mirage", "run", "--config=c.json", "--", "./app"])
        );
        assert_eq!(
            rewrite(&["mirage", "-okey=value", "--", "./app"]),
            v_args(&["mirage", "run", "-okey=value", "--", "./app"])
        );
    }

    /// `mirage --` used to be rewritten first and reported second, so
    /// clap complained about a missing argument of `mirage run` — a
    /// subcommand the user never typed.
    #[test]
    fn a_separator_with_nothing_after_it_is_explained_without_naming_run() {
        let usage = refuse(&["mirage", "--"]);
        assert!(usage.contains("`--` was given with no command"), "{usage}");
        assert!(
            !usage.contains("mirage run <"),
            "the usage line should not name a subcommand the user did not type: {usage}"
        );
        // The same for a drop-in that got as far as its flags.
        let usage = refuse(&["mirage", "--config", "c.json", "--"]);
        assert!(usage.contains("`--` was given with no command"), "{usage}");
    }

    /// Naming a program with no `--` in front of it is a dead end clap
    /// can only report as an unrecognised subcommand. Drop-in mode is a
    /// headline feature; the error is the one place the user is
    /// guaranteed to read.
    #[test]
    fn a_program_with_no_separator_is_pointed_at_the_separator() {
        let usage = refuse(&["mirage", "./app", "--flag"]);
        assert!(usage.contains("mirage -- ./app --flag"), "{usage}");
    }

    /// …but only when the token cannot plausibly be a misspelt
    /// subcommand, because clap's own "did you mean" is better than
    /// anything said here.
    #[test]
    fn a_misspelt_subcommand_is_left_to_clap() {
        let args = v_args(&["mirage", "profil", "list"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
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
                rewrite(&["mirage", v, "run", "--", "./app"]),
                args,
                "{v} should leave an explicit `run` alone"
            );
            assert_eq!(
                rewrite(&["mirage", v, "--", "./app"]),
                v_args(&["mirage", v, "run", "--", "./app"]),
                "{v} should be stepped over when splicing `run`"
            );
        }
    }

    /// `cleanup` was missing from the hardcoded subcommand list, so
    /// `mirage cleanup -- echo hi` brought up an emulated session and
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
            let args = v_args(&["mirage", &name, "--", "./app"]);
            assert_eq!(
                dropin_argv(args.clone()).unwrap(),
                args,
                "`mirage {name} -- ./app` was rewritten; it names a subcommand \
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
    /// which is the original bug: `mirage <sub> --nodes 2 ./app` brings a
    /// session up and tries to execute `--nodes`.
    ///
    /// So the population is taken from clap: a positional declared
    /// `trailing_var_arg` is the declaration of "everything from here is
    /// the workload's", and every subcommand that has one must refuse a
    /// flag mirage does not take — in both spellings, since the one with
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
                let args = v_args(&["mirage", &name, "--not-a-mirage-flag"]);
                assert_eq!(
                    dropin_argv(args.clone()).unwrap(),
                    args,
                    "`mirage {name}` does not end in a workload, so its flags \
                     are clap's business"
                );
                continue;
            }
            guarded += 1;
            // Both spellings. The one with no `--` is the one the old
            // post-parse check could not see at all.
            for argv in [
                v_args(&["mirage", &name, "--not-a-mirage-flag", "--", "./app"]),
                v_args(&["mirage", &name, "--not-a-mirage-flag", "./app"]),
            ] {
                match dropin_argv(argv.clone()) {
                    Err(usage) => assert!(
                        usage.contains("unexpected argument '--not-a-mirage-flag'"),
                        "`mirage {name}` refused {argv:?} for the wrong reason:\
                         {usage}"
                    ),
                    Ok(out) => panic!(
                        "`mirage {name}` ends in a workload but its flags are \
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
            rewrite(&["mirage", "--config", "c.json", "--", "./app"]),
            v_args(&["mirage", "run", "--config", "c.json", "--", "./app"])
        );
    }

    /// Repeating a boolean flag is not an error anywhere else, and a
    /// user reaches this by composing a command line from a variable that
    /// already carries `--json`.
    #[test]
    fn a_repeated_json_flag_is_accepted() {
        use clap::Parser as _;
        for argv in [
            &["mirage", "paths", "--json"][..],
            &["mirage", "paths", "--json", "--json"][..],
            &["mirage", "--json", "paths", "--json"][..],
        ] {
            let cli =
                Cli::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?} should parse: {e}"));
            assert!(cli.json, "{argv:?} should set --json");
        }
        assert!(
            !Cli::try_parse_from(["mirage", "paths"]).unwrap().json,
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
        let args = v_args(&["mirage", "run", "--profile", "mi350x", "--", "./app"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
    }

    #[test]
    fn other_subcommands_are_untouched() {
        let args = v_args(&["mirage", "exec", "--session", "s", "--", "cmd"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
    }

    #[test]
    fn no_separator_is_untouched() {
        let args = v_args(&["mirage", "profile", "list"]);
        assert_eq!(dropin_argv(args.clone()).unwrap(), args);
        let help = v_args(&["mirage", "--help"]);
        assert_eq!(dropin_argv(help.clone()).unwrap(), help);
    }
}
