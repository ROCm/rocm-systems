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
    let cli = Cli::parse_from(argv);
    mirage_ctl::init_logging(cli.verbose);
    match dispatch(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
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
/// subcommand, and missing one would resurrect exactly the bug above.
fn is_subcommand(name: &str) -> bool {
    use clap::CommandFactory as _;
    Cli::command()
        .get_subcommands()
        .any(|sub| sub.get_name() == name || sub.get_all_aliases().any(|alias| alias == name))
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

/// The subcommand a drop-in invocation is routed to, and the only one
/// that also honours [`ATTACH`].
const RUN: &str = "run";

/// The rocjitsu-only spelling of `--daemon`. Mirage manages the daemon's
/// lifecycle, so "attach to a daemon" and "use a daemon" collapse to the
/// same opt-in, and the alias is translated away before clap sees it.
const ATTACH: &str = "--attach";

/// The one-line shape shown by every usage error [`dropin_argv`] raises.
const DROPIN_USAGE: &str = "Usage: mirage [OPTIONS] -- <COMMAND> [ARGS]...";

/// Whether `arg` looks like a flag rather than a value.
///
/// Deliberately stricter than "starts with `-`": a negative number is a
/// value (`--num-nodes -1`), and rejecting it here would replace clap's
/// "invalid value" — which says what the range is — with a worse message
/// about an unknown flag. A short flag is a letter, so that is the test.
fn is_flag_token(arg: &str) -> bool {
    if let Some(long) = arg.strip_prefix("--") {
        return !long.is_empty();
    }
    arg.strip_prefix('-')
        .and_then(|rest| rest.chars().next())
        .is_some_and(char::is_alphabetic)
}

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

/// The flag spellings that may appear before the `--` of a drop-in
/// invocation: [`Cli`]'s global flags, everything `mirage run` accepts,
/// and [`ATTACH`].
///
/// Asked of clap rather than listed here, for the same reason
/// [`is_subcommand`] is: a flag added to `RunArgs` is accepted the moment
/// it is declared, and one removed stops being accepted, with no second
/// copy of the list to forget. Aliases count — `run` gives
/// `--nproc-per-node` a second spelling, and rejecting it would be
/// exactly the bug this guard exists to prevent.
#[derive(Debug)]
struct DropinFlags {
    /// Long flag names without their `--`, sorted, so a suggestion is
    /// deterministic when several are equally close.
    longs: Vec<String>,
    /// Short flag letters.
    shorts: Vec<char>,
}

impl DropinFlags {
    fn collect() -> Self {
        use clap::CommandFactory as _;
        let cmd = Cli::command();
        // `help` and `version` are clap's own and are not in the
        // declared argument list of an unbuilt `Command`, but they are
        // accepted everywhere, so a drop-in may carry them too.
        let mut longs = vec![
            ATTACH.trim_start_matches('-').to_string(),
            "help".to_string(),
            "version".to_string(),
        ];
        let mut shorts = vec!['h', 'V'];
        let globals = cmd.get_arguments().filter(|a| a.is_global_set());
        let run = cmd
            .find_subcommand(RUN)
            .into_iter()
            .flat_map(clap::Command::get_arguments);
        for arg in globals.chain(run) {
            longs.extend(arg.get_long().map(str::to_string));
            longs.extend(
                arg.get_all_aliases()
                    .unwrap_or_default()
                    .into_iter()
                    .map(str::to_string),
            );
            shorts.extend(arg.get_short());
            shorts.extend(arg.get_all_short_aliases().unwrap_or_default());
        }
        longs.sort();
        longs.dedup();
        Self { longs, shorts }
    }

    /// Whether `token` is one of these flags, in any of the spellings
    /// clap itself would accept (`--long`, `--long=value`, `-s`,
    /// `-svalue`, and `-v` bundled to any depth).
    fn accepts(&self, token: &str) -> bool {
        if is_global_flag(token) {
            return true;
        }
        if let Some(long) = token.strip_prefix("--") {
            let name = long.split('=').next().unwrap_or(long);
            return self.longs.iter().any(|known| known == name);
        }
        token
            .strip_prefix('-')
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| self.shorts.contains(&c))
    }

    /// The accepted flag closest to `token`, for a "did you mean".
    ///
    /// Containment in either direction, which is what a dropped or
    /// mis-remembered word looks like: `--nodes` for `--num-nodes`,
    /// `--profil` for `--profile`. Short names are excluded because at
    /// two characters almost anything contains almost anything.
    fn nearest(&self, token: &str) -> Option<String> {
        let name = token.trim_start_matches('-');
        let name = name.split('=').next().unwrap_or(name);
        if name.len() < 3 {
            return None;
        }
        self.longs
            .iter()
            .filter(|known| known.contains(name) || name.contains(known.as_str()))
            .min_by_key(|known| known.len().abs_diff(name.len()))
            .map(|known| format!("--{known}"))
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
/// [`ATTACH`] to `--daemon`. Everything `run` already accepts
/// (`--config`, `--profile`, `--daemon`, `--env`, …) then flows straight
/// through.
///
/// Invocations that name a subcommand (`mirage run …`, `mirage profile
/// …`, `mirage exec --session s -- cmd`) and those with no `--` separator
/// (so `--help`/`--version` keep working) are left untouched, except that
/// an explicit `mirage run` gets the same `--attach` translation: the
/// alias was documented as accepted and worked only on the rewriting
/// path, so `mirage run --attach -- app` passed `--attach` to the
/// workload and exited 127.
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
    // A recognised subcommand means this is a normal mirage call. Leave
    // it alone — `run` excepted, which understands `--attach` wherever
    // the drop-in shape does.
    if let Some(h) = head.filter(|h| is_subcommand(h)) {
        return Ok(if h == RUN {
            map_attach(args, head_idx + 1, scan_end)
        } else {
            args
        });
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
    let known = DropinFlags::collect();
    if let Some(bad) = args[1..sep]
        .iter()
        .find(|a| is_flag_token(a) && !known.accepts(a))
    {
        return Err(unknown_flag_error(bad, &known));
    }
    if sep + 1 == args.len() {
        return Err(empty_separator_error());
    }
    // Drop-in: splice `run` in where the subcommand would go.
    let mut out = Vec::with_capacity(args.len() + 1);
    out.extend(args[..head_idx].iter().cloned());
    out.push(RUN.to_string());
    out.extend(map_attach(args, head_idx, sep).into_iter().skip(head_idx));
    Ok(out)
}

/// `args` with every [`ATTACH`] in `args[from..to]` replaced by
/// `--daemon`, leaving the rest — in particular the workload's own
/// arguments after `--` — byte for byte as the user typed them.
fn map_attach(mut args: Vec<String>, from: usize, to: usize) -> Vec<String> {
    let len = args.len();
    for a in &mut args[from.min(len)..to.min(len)] {
        if a == ATTACH {
            *a = "--daemon".to_string();
        }
    }
    args
}

/// The message for a flag-shaped token before `--` that mirage does not
/// accept.
fn unknown_flag_error(flag: &str, known: &DropinFlags) -> String {
    let mut msg = format!(
        "error: unexpected argument '{flag}' found\n\n\
         Everything before `--` is read as mirage's own flags and everything \
         after it as the\ncommand to run, so '{flag}' is a mistyped flag \
         rather than part of the workload.\n"
    );
    if let Some(nearest) = known.nearest(flag) {
        msg.push_str(&format!("\n  tip: a similar flag exists: '{nearest}'\n"));
    }
    msg.push_str(&format!(
        "\n{DROPIN_USAGE}\n\nFor more information, try 'mirage run --help'."
    ));
    msg
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

    use super::{Cli, DropinFlags, dropin_argv, is_global_flag, is_subcommand};

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

    #[test]
    fn attach_maps_to_daemon() {
        assert_eq!(
            rewrite(&["mirage", "--attach", "--config", "c.json", "--", "./app"]),
            v_args(&[
                "mirage", "run", "--daemon", "--config", "c.json", "--", "./app"
            ])
        );
    }

    /// `--attach` was documented as an accepted alias and translated only
    /// on the rewriting path, so naming `run` explicitly turned it back
    /// into a workload argument: `mirage run --attach -- app` brought a
    /// session up and exited 127 with `command not found: --attach`.
    #[test]
    fn attach_maps_to_daemon_on_an_explicit_run_too() {
        assert_eq!(
            rewrite(&["mirage", "run", "--attach", "--", "./app"]),
            v_args(&["mirage", "run", "--daemon", "--", "./app"])
        );
        // Including without a separator, so the two spellings of the same
        // invocation cannot disagree.
        assert_eq!(
            rewrite(&["mirage", "run", "--attach", "./app"]),
            v_args(&["mirage", "run", "--daemon", "./app"])
        );
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

    /// Only `run` takes `--attach`; on any other subcommand it is that
    /// subcommand's business (and, today, its error).
    #[test]
    fn attach_is_left_alone_on_other_subcommands() {
        let args = v_args(&["mirage", "exec", "--attach", "--", "cmd"]);
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
        let known = DropinFlags::collect();
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
