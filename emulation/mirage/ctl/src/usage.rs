//! One place that decides whether a flag-shaped token is a mistake.
//!
//! `mirage run -- ./app` and `mirage -- ./app` are the same invocation in
//! two spellings, and both have the same failure mode: a mirage flag the
//! user misremembered is not a mirage flag, so it becomes the workload.
//! `mirage run --nodes 2 -- ./app` used to bring a whole emulated machine
//! up and then run a program called `--nodes` in it, with the program the
//! user actually named discarded into its arguments.
//!
//! There used to be two implementations of the guard — one in the binary
//! crate for the drop-in spelling, one here for the explicit one — with
//! two copies of the flag vocabulary, two "did you mean", and two
//! definitions of what a flag token even is. They had already drifted:
//! the drop-in copy knew about short flags and bundled `-vvv`, this one
//! did not, and only one of them knew the root command's globals.
//!
//! # Why this cannot be done after the parse
//!
//! It was, and the position is not recoverable there. `run` and `exec`
//! declare `argv` as `trailing_var_arg` + `allow_hyphen_values`, which is
//! what lets a workload keep its own flags — and it is also what lets an
//! unknown mirage flag start the positional. By the time clap hands back
//! an `argv`, these two are identical:
//!
//! ```text
//! mirage run --nodes 2 ./app      # a typo: `--nodes` became the command
//! mirage run -- --nodes 2 ./app   # deliberate: a program called `--nodes`
//! ```
//!
//! Clap consumes the separator in the second and not in the first, so
//! `argv` is `["--nodes", "2", "./app"]` either way. The old check
//! guessed, by only firing when a `--` was still *in* `argv` — which made
//! `mirage run --nodes 2 -- ./app` an error and left `mirage run --nodes
//! 2 ./app`, the same typo with no separator at all, starting a session.
//!
//! Raw argv still has the distinction, because the separator is still in
//! it. So the check belongs here, before clap, and it is the same check
//! for both spellings.
//!
//! # Requiring `--` would not have helped
//!
//! It looks like the structural fix — no token before the separator could
//! ever become the workload — but the separator is not what is missing.
//! `mirage run --nodes 2 -- ./app` *has* one and was still wrong, because
//! the mistake is in front of it. And `mirage run ./app` is a spelling
//! that works today and reads perfectly well; refusing it would cost real
//! users something to fix a bug they do not have.

/// Whether `arg` looks like a flag rather than a value.
///
/// Deliberately stricter than "starts with `-`": a negative number is a
/// value (`--num-nodes -1`), and rejecting it here would replace clap's
/// "invalid value" — which says what the range is — with a worse message
/// about an unknown flag. A lone `-` is the conventional name for
/// standard input. A short flag is a letter, so that is the test.
#[must_use]
pub fn is_flag_token(arg: &str) -> bool {
    if let Some(long) = arg.strip_prefix("--") {
        return !long.is_empty();
    }
    arg.strip_prefix('-')
        .and_then(|rest| rest.chars().next())
        .is_some_and(char::is_alphabetic)
}

/// The flag spellings one invocation may carry before its `--`.
///
/// Asked of clap rather than listed, so a flag added to `RunArgs` or
/// `ExecArgsCli` is accepted and suggestible the moment it is declared,
/// and one removed stops being either, with no second copy to forget.
/// Aliases count: `--nproc_per_node` is a real spelling of a real flag,
/// and rejecting it would be the very bug this guard exists to prevent.
#[derive(Debug)]
pub struct AcceptedFlags {
    /// Long names without their `--`, sorted so a suggestion is
    /// deterministic when several are equally close.
    longs: Vec<String>,
    /// Long names that consume the token after them.
    valued_longs: Vec<String>,
    /// Short letters.
    shorts: Vec<char>,
    /// Short letters that consume the token after them.
    valued_shorts: Vec<char>,
}

/// How a short-option token ends, once it has been read clap's way.
///
/// The distinction that matters to the scan is not "is this a flag" but
/// "where does the next token begin", and only the value-taking letter
/// answers it — see [`AcceptedFlags::cluster`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cluster {
    /// Every letter is one mirage takes and none of them takes a value:
    /// `-v`, `-vvv`.
    AllFlags,
    /// The bundle reached a value-taking letter with its value attached:
    /// `-okey=value`, `-vokey=value`. Self-contained.
    ValueAttached,
    /// The bundle ended on a value-taking letter, so the value is the
    /// next token: `-o key=value`, `-vo key=value`.
    ValueNext,
    /// A letter mirage does not take, before any value-taking one.
    Unknown,
}

impl AcceptedFlags {
    /// Every flag `root`'s globals and its `subcommand` accept.
    ///
    /// Both halves are needed and neither is obvious. `--json` and
    /// `--verbose` are declared on the root and clap accepts them after
    /// the subcommand too, so a check that only knew the subcommand's own
    /// list would reject `mirage run --json -- ./app`.
    #[must_use]
    pub fn of(root: &clap::Command, subcommand: &str) -> Self {
        // `help` and `version` are clap's own: accepted everywhere, and
        // absent from the declared argument list of a `Command` that has
        // not been built yet.
        let mut this = Self {
            longs: vec!["help".to_string(), "version".to_string()],
            valued_longs: Vec::new(),
            shorts: vec!['h', 'V'],
            valued_shorts: Vec::new(),
        };
        let globals = root.get_arguments().filter(|a| a.is_global_set());
        let sub = root
            .find_subcommand(subcommand)
            .into_iter()
            .flat_map(clap::Command::get_arguments);
        for arg in globals.chain(sub) {
            this.add(arg);
        }
        this.longs.sort();
        this.longs.dedup();
        this
    }

    /// Every flag a already-built subcommand spec accepts, with no root
    /// to take globals from.
    #[must_use]
    pub fn of_spec(spec: &clap::Command) -> Self {
        let mut this = Self {
            longs: vec!["help".to_string(), "version".to_string()],
            valued_longs: Vec::new(),
            shorts: vec!['h', 'V'],
            valued_shorts: Vec::new(),
        };
        for arg in spec.get_arguments() {
            this.add(arg);
        }
        this.longs.sort();
        this.longs.dedup();
        this
    }

    fn add(&mut self, arg: &clap::Arg) {
        let takes_value = arg.get_action().takes_values();
        let longs = arg
            .get_long()
            .into_iter()
            .chain(arg.get_all_aliases().unwrap_or_default())
            .map(str::to_string);
        for long in longs {
            if takes_value {
                self.valued_longs.push(long.clone());
            }
            self.longs.push(long);
        }
        let shorts = arg
            .get_short()
            .into_iter()
            .chain(arg.get_all_short_aliases().unwrap_or_default());
        for short in shorts {
            if takes_value {
                self.valued_shorts.push(short);
            }
            self.shorts.push(short);
        }
    }

    /// Whether `token` is one of these, in any spelling clap would accept
    /// (`--long`, `--long=value`, `-s`, `-svalue`, `-v` bundled to any
    /// depth, and a bundle ending in a valued letter — `-vokey=value`).
    #[must_use]
    pub fn accepts(&self, token: &str) -> bool {
        if let Some(long) = token.strip_prefix("--") {
            let name = long.split('=').next().unwrap_or(long);
            return self.longs.iter().any(|known| known == name);
        }
        !matches!(self.cluster(token), Cluster::Unknown)
    }

    /// Whether `token` consumes the argument after it.
    ///
    /// `--profile mi350x` is two tokens for one flag, and a scan that did
    /// not know it would read `mi350x` as the start of the workload —
    /// which is how a mistyped flag *after* a valued one goes unnoticed.
    /// `--profile=mi350x` and `-pmi350x` carry their own value and
    /// consume nothing. `-vo key=value` does, because the bundle ends on
    /// the valued letter with nothing attached to it.
    #[must_use]
    fn consumes_next(&self, token: &str) -> bool {
        if let Some(long) = token.strip_prefix("--") {
            return !long.contains('=') && self.valued_longs.iter().any(|known| known == long);
        }
        matches!(self.cluster(token), Cluster::ValueNext)
    }

    /// Read a short-option token the way clap does, and say how it ends.
    ///
    /// Clap bundles short options — `-vvv` is three `-v`s — and the
    /// bundle runs *until a letter that takes a value*, after which
    /// everything left is that letter's value rather than more letters.
    /// So `-vokey=value` is `-v` plus `-o key=value`, and the scan has to
    /// stop at `o`. Both halves of this were wrong in different
    /// directions: `accepts` kept reading `key=value` as flag letters and
    /// rejected a spelling clap takes, and `consumes_next` looked only at
    /// the first letter, so `-vo key=value` was read as consuming
    /// nothing — the scan then stopped at `key=value`, taking it for the
    /// command, and never reached a typo such as `--nodes` behind it.
    fn cluster(&self, token: &str) -> Cluster {
        let Some(rest) = token.strip_prefix('-') else {
            return Cluster::Unknown;
        };
        if rest.is_empty() || rest.starts_with('-') {
            return Cluster::Unknown;
        }
        let mut chars = rest.chars();
        while let Some(letter) = chars.next() {
            if !self.shorts.contains(&letter) {
                // `-vqv` is not three of anything.
                return Cluster::Unknown;
            }
            if self.valued_shorts.contains(&letter) {
                return if chars.as_str().is_empty() {
                    Cluster::ValueNext
                } else {
                    Cluster::ValueAttached
                };
            }
        }
        Cluster::AllFlags
    }

    /// The accepted flag closest to `token`, for a "did you mean".
    ///
    /// Containment in either direction, which is what a dropped or
    /// mis-remembered word looks like: `--nodes` for `--num-nodes`,
    /// `--profil` for `--profile`. Anything shorter than three characters
    /// is left alone, because at that length almost everything contains
    /// almost everything.
    #[must_use]
    pub fn nearest(&self, token: &str) -> Option<String> {
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

/// Where a scan of mirage's own flags should stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Span {
    /// The invocation has a `--`, so everything given to this scan is
    /// mirage's and a flag token anywhere in it that mirage does not take
    /// is a mistake.
    BeforeSeparator,
    /// The invocation has no `--`, so mirage's flags run until the first
    /// token that is not one — that token is the command, and everything
    /// after it belongs to the workload.
    ///
    /// `mirage run ./app --verbose` passes `--verbose` to `./app`, which
    /// is what the user meant and what mirage has always done. The scan
    /// stops at `./app` so it never sees the flag, which is why valued
    /// flags have to be tracked: without that, `--profile p --nodes 2`
    /// would stop at `p` and never reach the typo.
    UntilTheCommand,
}

/// The first token in `span` that is shaped like a flag and is not one
/// mirage takes.
#[must_use]
pub fn misplaced_flag<'a>(
    span: &'a [String],
    known: &AcceptedFlags,
    stop: Span,
) -> Option<&'a str> {
    let mut i = 0;
    while let Some(token) = span.get(i) {
        if !is_flag_token(token) {
            match stop {
                // The command. Everything from here is the workload's.
                Span::UntilTheCommand => return None,
                // A value, or a positional mirage takes. Either way not
                // a mistyped flag.
                Span::BeforeSeparator => {
                    i += 1;
                    continue;
                }
            }
        }
        if !known.accepts(token) {
            return Some(token.as_str());
        }
        i += if known.consumes_next(token) { 2 } else { 1 };
    }
    None
}

/// The message for a flag-shaped token before `--` that mirage does not
/// accept.
///
/// `usage` and `help` are the only parts that differ between the two
/// spellings — the drop-in one has no subcommand to name — and the body
/// is deliberately word for word the same, because it is the same
/// mistake and the two must not disagree about it.
#[must_use]
pub fn unknown_flag_error(flag: &str, known: &AcceptedFlags, usage: &str, help: &str) -> String {
    let mut msg = format!(
        "error: unexpected argument '{flag}' found\n\n\
         Everything before `--` is read as mirage's own flags and everything \
         after it as the\ncommand to run, so '{flag}' is a mistyped flag \
         rather than part of the workload.\n"
    );
    if let Some(nearest) = known.nearest(flag) {
        msg.push_str(&format!("\n  tip: a similar flag exists: '{nearest}'\n"));
    }
    msg.push_str(&format!("\n{usage}\n\nFor more information, try '{help}'."));
    msg
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use clap::Args as _;

    use super::*;
    use crate::{ExecArgsCli, RunArgs};

    fn run_flags() -> AcceptedFlags {
        AcceptedFlags::of_spec(&RunArgs::augment_args(clap::Command::new("run")))
    }

    fn v(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn a_flag_mirage_does_not_take_is_found_before_a_separator() {
        let flags = run_flags();
        assert_eq!(
            misplaced_flag(&v("--nodes 2"), &flags, Span::BeforeSeparator),
            Some("--nodes")
        );
        // And the suggestion names the flag they meant.
        assert_eq!(flags.nearest("--nodes").as_deref(), Some("--num-nodes"));
    }

    #[test]
    fn a_flag_mirage_does_not_take_is_found_without_a_separator() {
        // The case the post-parse check could not see at all: no `--`
        // anywhere, so `--nodes` became the command and mirage brought a
        // session up to run it.
        let flags = run_flags();
        assert_eq!(
            misplaced_flag(&v("--nodes 2 /bin/true"), &flags, Span::UntilTheCommand),
            Some("--nodes")
        );
    }

    #[test]
    fn a_valued_flag_does_not_hide_the_typo_after_it() {
        // `--profile mi350x` is two tokens for one flag. A scan that did
        // not know that would stop at `mi350x`, take it for the command,
        // and never reach `--nodes`.
        let flags = run_flags();
        assert_eq!(
            misplaced_flag(
                &v("--profile mi350x --nodes 2 /bin/true"),
                &flags,
                Span::UntilTheCommand
            ),
            Some("--nodes")
        );
        // The attached spellings consume nothing extra.
        assert_eq!(
            misplaced_flag(
                &v("--profile=mi350x --nodes 2"),
                &flags,
                Span::UntilTheCommand
            ),
            Some("--nodes")
        );
    }

    #[test]
    fn a_workloads_own_flags_are_not_mirages_business() {
        // `mirage run ./app --verbose` is the shape that works today.
        // Scanning past the command would reject the workload's own
        // flags, which is a worse bug than the one being fixed.
        let flags = run_flags();
        assert_eq!(
            misplaced_flag(
                &v("--profile p ./app --nodes --whatever"),
                &flags,
                Span::UntilTheCommand
            ),
            None
        );
    }

    #[test]
    fn the_spellings_mirage_really_does_accept_are_accepted() {
        let flags = run_flags();
        for good in [
            "--profile",
            "--profile=mi350x",
            "--num-nodes",
            "--nproc_per_node",
            "--help",
            "-o",
        ] {
            assert!(flags.accepts(good), "{good} is a real `run` spelling");
        }
        // A value, not a flag: `--num-nodes -1` must reach clap's own
        // range error rather than being called an unknown flag.
        assert!(!is_flag_token("-1"));
        assert!(!is_flag_token("-"));
    }

    #[test]
    fn bundled_short_flags_and_root_globals_are_accepted() {
        // The root command lives in the binary crate, which this one
        // cannot see, so this stands in for it with the shape that
        // matters: two global flags, one of them a repeatable short.
        //
        // Both halves have been bugs. `-vvv` is three `-v`s and clap
        // takes it, so a guard that only looked at the first letter --
        // or at the whole token as one name -- would refuse a correct
        // command line. And `--json` is declared on the root while being
        // accepted after the subcommand, so a guard built from the
        // subcommand's own list alone would reject `mirage run --json --
        // ./app`.
        let root = clap::Command::new("mirage")
            .arg(
                clap::Arg::new("json")
                    .long("json")
                    .global(true)
                    .action(clap::ArgAction::SetTrue),
            )
            .arg(
                clap::Arg::new("verbose")
                    .short('v')
                    .long("verbose")
                    .global(true)
                    .action(clap::ArgAction::Count),
            )
            .subcommand(RunArgs::augment_args(clap::Command::new("run")));
        let flags = AcceptedFlags::of(&root, "run");
        assert!(flags.accepts("-v"));
        assert!(flags.accepts("-vvv"));
        assert!(
            flags.accepts("--json"),
            "a root global reaches a subcommand"
        );
        assert!(flags.accepts("--profile"), "and so do the subcommand's own");
        assert!(!flags.accepts("-vqv"), "`q` is not a flag mirage has");
        // A short flag that carries its own value: everything after the
        // letter is data, not more flags.
        assert!(flags.accepts("-okey=value"), "`-o` takes a value");
    }

    /// The root command the two clustered-form tests below share.
    ///
    /// The same stand-in as `bundled_short_flags_and_root_globals_are_accepted`
    /// builds: a repeatable short global (`-v`) and, from `run` itself, a
    /// short that takes a value (`-o`).
    fn root_with_a_short_global() -> clap::Command {
        clap::Command::new("mirage")
            .arg(
                clap::Arg::new("verbose")
                    .short('v')
                    .long("verbose")
                    .global(true)
                    .action(clap::ArgAction::Count),
            )
            .subcommand(RunArgs::augment_args(clap::Command::new("run")))
    }

    #[test]
    fn a_bundle_ending_in_a_valued_short_is_accepted_in_both_forms() {
        // Clap's rule is that a short bundle runs until a letter that
        // takes a value, and everything after that letter is its value.
        // So `-vokey=value` is `-v` plus `-o key=value` — a spelling clap
        // accepts and this guard used to reject, because it went on
        // reading `key=value` as more flag letters and found `k`.
        let root = root_with_a_short_global();
        let flags = AcceptedFlags::of(&root, "run");

        assert!(flags.accepts("-vokey=value"), "`-v` then `-o` with a value");
        assert!(
            flags.accepts("-vo"),
            "`-v` then `-o`, value in the next token"
        );
        assert!(
            flags.accepts("-vvokey=value"),
            "and the bundle may be deeper"
        );
        // The letter that takes the value ends the scan, so nothing after
        // it has to be a flag — but everything before it does.
        assert!(
            !flags.accepts("-qokey=value"),
            "`q` is not a flag mirage has"
        );

        // Clap's own verdict on the same tokens, so this cannot drift
        // from the parser it is guarding.
        for argv in [
            vec!["mirage", "run", "-vokey=value", "--", "./app"],
            vec!["mirage", "run", "-vo", "key=value", "--", "./app"],
        ] {
            assert!(
                root.clone().try_get_matches_from(&argv).is_ok(),
                "clap accepts {argv:?}, so the guard must too"
            );
        }
    }

    #[test]
    fn a_valued_short_at_the_end_of_a_bundle_still_hides_the_typo_after_it() {
        // The other half, and the one that let a typo through. `-vo`
        // consumes `key=value`, and a scan that checked only the bundle's
        // *first* letter did not know it: it read `key=value` as the
        // command, stopped there, and never reached `--nodes`.
        let flags = AcceptedFlags::of(&root_with_a_short_global(), "run");
        assert_eq!(
            misplaced_flag(&v("-vo key=value --nodes 2"), &flags, Span::UntilTheCommand),
            Some("--nodes")
        );
        // With the value attached there is no next token to skip, and the
        // typo is still found.
        assert_eq!(
            misplaced_flag(&v("-vokey=value --nodes 2"), &flags, Span::UntilTheCommand),
            Some("--nodes")
        );
        // And a workload really named `key=value` after a bundle that
        // does *not* end on a valued letter is still the command.
        assert_eq!(
            misplaced_flag(&v("-vv key=value --nodes 2"), &flags, Span::UntilTheCommand),
            None
        );
    }

    #[test]
    fn exec_is_judged_against_execs_own_flags() {
        // The two subcommands do not take the same flags, and a guard
        // that checked one against the other's list would reject honest
        // commands. `--session` is `exec`'s and `run` has no such thing.
        let exec = AcceptedFlags::of_spec(&ExecArgsCli::augment_args(clap::Command::new("exec")));
        assert!(exec.accepts("--session"));
        assert!(!run_flags().accepts("--session"));
    }
}
