//! `mirage_ctl`: the mirage command line.
//!
//! This crate is a **library**: it defines the top-level [`CtlCmd`]
//! subcommand enum and an async [`dispatch`] that runs each command. The
//! `mirage` binary is a thin wrapper around it.
//!
//! Commands fall into two groups, and the split is the whole shape of
//! mirage:
//!
//! * **Configuration** — `profile`, `topology`, `agent`, `state`,
//!   `paths`, `emulators`. Pure filesystem work against
//!   [`mirage_core::store`]. No session, no processes, nothing running.
//! * **Execution** — `run` and `exec`, in [`run`]. These own processes.
//!
//! There is no client/server split any more, and therefore no trait to
//! abstract one. `mirage run` *is* the runtime: it holds its session in
//! its own address space. `mirage exec` is the only command that talks
//! over a socket, and it asks exactly one question — see
//! [`mirage_core::proto`].
//!
//! All commands are documented in `docs/cli.md`.

pub mod run;
pub mod usage;

use std::io::IsTerminal;
use std::process::ExitCode;

use clap::{Args, Subcommand, ValueEnum};
use mirage_core::common::{MaybeRef, SimpleMap, SimpleValue};
use mirage_core::emulator::ExecMode;
use mirage_core::profile::{ContainerizedDef, FileMount, Hack, PortMapping, ProfileDef};
use mirage_core::registry::EmulatorInfo;
use mirage_core::session::SessionId;
use mirage_core::store::{DocKind, Stored};

/// The level mirage logs at when nothing asks for more.
const DEFAULT_LOG: &str = "warn";

/// Initialize the global tracing subscriber.
///
/// `-v` / `-vv` wins over `MIRAGE_LOG`: one is a decision about this
/// invocation, typed by someone watching the output, and the other is
/// ambient state inherited from a shell profile or a CI job. A user who
/// adds `-vv` to a command that prints too little has said what they
/// want, and losing to an exported variable they may not know is set
/// makes the flag look broken.
///
/// An unusable `MIRAGE_LOG` is reported and then ignored, rather than
/// obeyed: it used to turn logging off altogether, which is worse than
/// either honouring it or rejecting it, because the symptom — silence —
/// is exactly what a mirage with nothing to say looks like.
pub fn init_logging(verbose: u8) {
    let explicit = match verbose {
        0 => None,
        1 => Some("info"),
        _ => Some("debug"),
    };
    let filter = match (explicit, std::env::var("MIRAGE_LOG")) {
        (Some(level), _) => tracing_subscriber::EnvFilter::new(level),
        (None, Ok(value)) if !value.is_empty() => parse_log_filter(&value).unwrap_or_else(|why| {
            eprintln!(
                "mirage: MIRAGE_LOG={value:?}: {why}. Logging at the default level \
                 ({DEFAULT_LOG}) instead. It takes a level (`info`, `debug`, `off`) or \
                 a comma-separated list of `<target>=<level>` directives \
                 (`warn,mirage_supervisor=debug`); `-v`/`-vv` say the same thing \
                 without it."
            );
            tracing_subscriber::EnvFilter::new(DEFAULT_LOG)
        }),
        _ => tracing_subscriber::EnvFilter::new(DEFAULT_LOG),
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(stderr_wants_colour())
        .with_writer(std::io::stderr)
        .try_init();
}

/// Read a `MIRAGE_LOG` value, rejecting the ones that only look valid.
///
/// [`EnvFilter`] accepts a bare word as a *target* directive at trace
/// level, so `MIRAGE_LOG=not-a-level` parses cleanly and then matches
/// nothing mirage ever logs to — every message, at every level,
/// discarded, with no error anywhere. A bare word here is therefore
/// required to be a level; naming a target still works, spelled the way
/// the filter syntax spells it (`mirage_supervisor=debug`), which is also
/// the form that does not silently silence everything else.
///
/// [`EnvFilter`]: tracing_subscriber::EnvFilter
///
/// # Errors
///
/// Returns the reason the value cannot be used, as a phrase that
/// completes "MIRAGE_LOG=…: {reason}".
fn parse_log_filter(value: &str) -> Result<tracing_subscriber::EnvFilter, String> {
    for directive in value.split(',') {
        let directive = directive.trim();
        if directive.is_empty() || directive.contains('=') {
            continue;
        }
        if directive
            .parse::<tracing_subscriber::filter::LevelFilter>()
            .is_err()
        {
            return Err(format!(
                "`{directive}` is not a log level, and as a bare word it silences \
                 everything else"
            ));
        }
    }
    tracing_subscriber::EnvFilter::try_new(value).map_err(|e| e.to_string())
}

/// Whether log records may be coloured.
///
/// Escape sequences are for a terminal to interpret; written to a file or
/// a pipe they are noise a log-reading tool has to strip, and `grep` does
/// not. [`NO_COLOR`](https://no-color.org/) is honoured on top of that,
/// because a user who set it means it even on a terminal.
fn stderr_wants_colour() -> bool {
    std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty())
}

/// The full emulator registry: every backend crate compiled into this
/// binary registers itself via the `inventory` crate, and
/// [`mirage_core::registry::registry`] probes each one for its identity
/// and live install / support status. No backend is named here, so
/// enabling or disabling a backend's feature simply adds or removes it
/// from this list.
pub fn registry() -> Vec<EmulatorInfo> {
    mirage_core::registry::registry()
}

/// Lookup an emulator by its canonical name in the full [`registry`].
pub fn find_emulator(name: &str) -> Option<EmulatorInfo> {
    registry().into_iter().find(|e| e.name == name)
}

/// The default emulator for new profiles: the first installed backend,
/// falling back to the first compiled in.
///
/// `None` when this build has no emulator backends compiled in.
#[must_use]
pub fn default_emulator() -> Option<EmulatorInfo> {
    let specs = registry();
    mirage_core::registry::default_emulator(&specs).cloned()
}

/// The default emulator's name, or `"-"` when there is none, for display.
#[must_use]
pub fn default_emulator_name() -> String {
    default_emulator().map_or_else(|| "-".to_string(), |e| e.name)
}

/// Render the emulator registry for the `mirage emulators` command:
/// each backend with whether its runtime is installed and whether this
/// host's hardware supports it. With `json` the full descriptions are
/// emitted as-is; otherwise a compact table (or, with `long`, a
/// detailed block including the runtime library's location and the
/// support reason).
///
/// The long form answers "where did mirage look?" because that is the
/// only question left once a backend reports `installed: no`, and it is
/// the question the command exists to answer — the user is running it
/// precisely because nothing will emulate. The `supported: no` line has
/// named its requirement and what was detected all along; `installed`
/// now does the same, naming the resolved library or the paths that were
/// probed for it.
///
/// The JSON carries every fact the text does, the default backend
/// included: the text form marks it with `(default)` and a script reading
/// the JSON had no way to tell, so it had to re-derive the rule
/// ([`mirage_core::registry::default_emulator`]) from the `installed`
/// flags and hope the two agreed. The runtime location is held to the
/// same rule — and, since the text elides the middle of a very long
/// probe list, the JSON is the complete copy of it.
/// Why `name`'s located runtime could not host an emulator daemon, when
/// it could not.
///
/// A backend that hosts a daemon may need more of its runtime than the
/// runtime search can see — rocjitsu's daemon entry points are newer than
/// the rest of its C API, so an older `librocjitsu.so` is found, is
/// usable in-process, and cannot host a daemon. A user reading
/// `mirage emulators -l` on such a host should be told, rather than
/// finding out at the end of a bring-up.
///
/// `None` when the backend can host one, has none to host, or is not
/// compiled in. Only the first line of the reason, because the rest is a
/// paragraph meant for a refused run rather than for a listing.
fn daemon_caveat(name: &str) -> Option<String> {
    let backend = mirage_core::emulator::get_emulator_backend(name)?;
    let reason = backend.daemon_capability().err()?.to_string();
    Some(reason.lines().next().unwrap_or_default().to_string())
}

fn emulators_cmd(long: bool, json: bool) {
    let specs = registry();
    let default_name = default_emulator_name();

    if json {
        let described: Vec<serde_json::Value> = specs
            .iter()
            .map(|spec| {
                let mut value = serde_json::to_value(spec).unwrap_or(serde_json::Value::Null);
                if let Some(object) = value.as_object_mut() {
                    object.insert(
                        "default".to_string(),
                        serde_json::Value::Bool(spec.name == default_name),
                    );
                }
                value
            })
            .collect();
        match serde_json::to_string_pretty(&described) {
            Ok(s) => println!("{s}"),
            Err(e) => eprintln!("failed to serialize emulators: {e}"),
        }
        return;
    }

    if long {
        for spec in &specs {
            let default_marker = if spec.name == default_name {
                " (default)"
            } else {
                ""
            };
            println!("{}{}", spec.name, default_marker);
            println!("  {}", spec.description);
            println!("  installed: {}", if spec.installed { "yes" } else { "no" });
            // Where the runtime is, or — the case a user is in when they
            // run this at all — where mirage looked for it and what they
            // can set to change that. The lines come from the location
            // itself so the search order is described in one place; an
            // empty key continues the value column of the line above.
            for (key, value) in spec.runtime.report() {
                if key.is_empty() {
                    println!("             {value}");
                } else {
                    println!("  {:<10} {value}", format!("{key}:"));
                }
            }
            println!(
                "  supported: {}  ({})",
                if spec.support.supported { "yes" } else { "no" },
                spec.support.reason
            );
            // Whether the located runtime could host a daemon, asked only
            // here.
            //
            // It is not part of `support`: answering it may mean loading
            // the backend's runtime library, and `registry()` — which
            // fills `support` — is on the path of every `mirage run` that
            // carries an override flag. This is the one command whose job
            // is detail, so it is the one that pays. Printed only when
            // the answer is no, and only as its first line: the full
            // message is a paragraph written for somebody who has just
            // been refused a run, and it would break this aligned block
            // apart.
            if let Some(reason) = daemon_caveat(&spec.name) {
                println!("  daemon:    no   ({reason})");
            }
            // The options and plugins this backend will accept, so that a
            // rejected `-o` or `--plugin` can point here for the list
            // rather than only naming it in the error.
            println!("  options:   {}", name_list(option_names(spec)));
            println!("  plugins:   {}", name_list(spec.plugins.clone()));
            println!();
        }
        return;
    }

    println!(
        "{:<13} {:<10} {:<10} DESCRIPTION",
        "NAME", "INSTALLED", "SUPPORTED"
    );
    for spec in &specs {
        let name = if spec.name == default_name {
            format!("{}*", spec.name)
        } else {
            spec.name.clone()
        };
        println!(
            "{:<13} {:<10} {:<10} {}",
            name,
            if spec.installed { "yes" } else { "no" },
            if spec.support.supported { "yes" } else { "no" },
            spec.description
        );
    }
    println!("\n* = default emulator for new profiles");
}

/// Best-effort: materialise all builtin state on disk — agents,
/// topologies, and profiles — writing only what's missing. Not fatal;
/// the user can always force a full rewrite with `mirage state
/// builtins`.
///
/// Shared by the CLI ([`dispatch`]) and the daemon so both surfaces
/// auto-unpack the builtins the first time they run, instead of
/// requiring the user to invoke `mirage state builtins` by hand.
///
/// A failure here is said out loud rather than only `tracing::warn!`ed,
/// because it is never local to the builtins. Everything mirage knows
/// about profiles, agents and topologies is files in one directory, so a
/// directory it cannot write reads as a machine with no configuration at
/// all: `profile list` prints nothing and exits 0, and `run` then blames
/// a missing `mi350x` — which exists, and would have been written here.
/// `config_dir_hint` turns that into the sentence the user needs.
pub fn ensure_builtins_present() {
    let outcome = [
        mirage_builtin::ensure_agents(false).map(|_| ()),
        mirage_builtin::ensure_topologies(false).map(|_| ()),
        mirage_builtin::ensure_profiles(false).map(|_| ()),
    ]
    .into_iter()
    .find_map(Result::err);
    if let Some(e) = outcome {
        // Through `anyhow` for its `{:#}`, which walks the source chain:
        // the store's own message names the path it failed on and leaves
        // the operating system's reason — "Permission denied" — in the
        // cause, which is the half that says what to change.
        eprintln!(
            "mirage: could not write mirage's builtin configuration: {:#}",
            anyhow::Error::new(e)
        );
        eprintln!("mirage: {}", config_dir_hint());
    }
}

/// What to say about the config directory when something that lives in
/// it could not be read or written.
///
/// Names the directory and how it was chosen, because the two ways it
/// moves — `MIRAGE_CONFIG` and `XDG_CONFIG_HOME` — are both inherited
/// from the environment, and a user looking at an empty profile list is
/// usually looking at the wrong directory rather than at an empty one.
fn config_dir_hint() -> String {
    let dir = mirage_core::paths::mirage_config_dir();
    format!(
        "profiles, agents and topologies live in {}; \
         until it is readable and writable mirage will behave as though \
         there are none. Fix its permissions, or point MIRAGE_CONFIG at a \
         directory you can write.",
        dir.display()
    )
}

/// Validate a profile against its target emulator before it is
/// persisted. Returns a human-readable reason when the emulator can't
/// accept the profile (an unknown emulator, an unresolvable
/// agent/topology reference, or a missing runtime asset) so the
/// failure is reported at creation time rather than only when a
/// session is later started.
///
/// Shared by the CLI profile commands and the daemon's profile
/// endpoint so both validate identically.
pub fn validate_profile(def: &ProfileDef) -> Result<(), String> {
    let kind = &def.emulator.emulator;
    match mirage_core::emulator::get_emulator_backend(kind) {
        Some(backend) => backend.validate_profile(def),
        None => Err(unknown_emulator(kind)),
    }
}

// =============================================================================
// Top-level ctl subcommand enum
// =============================================================================

/// All user-facing `mirage` control subcommands. These are flattened
/// into the top-level `mirage` subcommand list by the root binary.
#[derive(Subcommand, Debug)]
pub enum CtlCmd {
    /// Manage profiles (reusable emulator presets).
    #[command(subcommand)]
    Profile(ProfileCmd),

    /// Manage topologies (rack/node/GPU system layouts).
    #[command(subcommand)]
    Topology(TopologyCmd),

    /// Manage agents (hardware GPU definitions).
    #[command(subcommand)]
    Agent(AgentCmd),

    /// List emulator backends and their install / support status.
    Emulators {
        /// Show long form (description, runtime path, support reason,
        /// and the options and plugins the backend accepts).
        #[arg(short = 'l', long)]
        long: bool,
    },

    /// Run a command inside a session an existing `mirage run` owns.
    ///
    /// The process runs in *this* terminal, as a child of this command,
    /// and dies with it.
    Exec(ExecArgsCli),

    /// Manage mirage's on-disk state (builtin documents, purge).
    #[command(subcommand)]
    State(StateCmd),

    /// Reclaim what a `mirage run` that died abruptly left behind.
    ///
    /// A run owns its session and cleans up when it exits, so in normal
    /// use there is nothing here to do. `kill -9`, the OOM killer, and a
    /// machine losing power leave no code of mirage's to run at all —
    /// containers keep running, workloads are reparented to init, and the
    /// session's scratch directory stays on disk. This is the command
    /// that removes them.
    ///
    /// Derived `--hack` images too, which are the one thing here that a
    /// healthy run also leaves behind: an image is built once per base
    /// and hack combination and reused by every session that asks for
    /// the same one, so teardown keeps it on purpose and this is the only
    /// command that ever reclaims one. An image a container still
    /// references is left where it is.
    ///
    /// Safe to run at any time: sessions whose `mirage run` still answers
    /// are left completely alone, as is anything mirage did not create.
    ///
    /// The scope is this runtime directory. Every workload and container
    /// records the `MIRAGE_RUNTIME` it was started under, and anything
    /// recording a different one — a CI job's, a test suite's, another
    /// terminal's — belongs to a mirage whose live sessions this command
    /// cannot see, so it is left for that mirage to reclaim.
    Cleanup {
        /// List what would be removed without removing anything.
        #[arg(long)]
        dry_run: bool,
    },

    /// Bring up a session, run a command in it, and tear it down.
    ///
    /// This process owns the session: it exists while this command runs
    /// and is gone when it exits. Other terminals can start processes in
    /// it with `mirage exec` for as long as it is up.
    Run(RunArgs),

    /// Print where mirage stores its state on this machine.
    Paths,
}

// ----- profile ---------------------------------------------------------------

#[derive(Subcommand, Debug)]
pub enum ProfileCmd {
    /// List available profiles.
    List {
        /// Show long form (description, emulator).
        #[arg(short = 'l', long)]
        long: bool,
    },
    /// Show a profile as JSON.
    Show { name: String },
    /// Create a new profile.
    ///
    /// Any field not given as a flag is prompted for interactively when
    /// stdin is a terminal; otherwise its default is used. This makes
    /// `profile create <name>` an interactive UI while `profile create
    /// <name> --emulator ... --agent ...` stays fully non-interactive
    /// (e.g. in scripts and tests).
    Create(ProfileCreateArgs),
    /// Import a profile from a JSON file.
    Import {
        /// File to import from (use `-` for stdin).
        file: String,
    },
    /// Delete a profile.
    Delete {
        name: String,
        /// Don't prompt for confirmation.
        #[arg(short = 'f', long)]
        force: bool,
    },
}

/// Arguments for `mirage profile create`.
///
/// A named struct rather than an inline variant body because
/// `build_profile_create` consumes the whole payload: six of these
/// fields are `Option<String>`, so passing them positionally meant a
/// transposition — image where the description belongs, say — compiled
/// silently and wrote the wrong profile to disk.
#[derive(Args, Debug)]
pub struct ProfileCreateArgs {
    /// Profile name. Prompted for when omitted on a terminal.
    pub name: Option<String>,
    /// Emulator name (e.g. `rocjitsu`, `hotswap`). Defaults to the
    /// first installed backend; see `mirage emulators`.
    #[arg(long)]
    pub emulator: Option<String>,
    /// Agent name from `<MIRAGE_CONFIG>/agent/` (e.g. `MI300X`,
    /// `MI350X`). Defaults to `MI350X`.
    #[arg(long)]
    pub agent: Option<String>,
    /// Nodes per rack.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub num_nodes: Option<u32>,
    /// GPUs per node.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub gpus_per_node: Option<u32>,
    /// Optional description.
    #[arg(long)]
    pub description: Option<String>,
    /// Containerise the profile: run every node inside a container
    /// built from this image. Enables `--mount`/`--container-provider`.
    #[arg(long)]
    pub image: Option<String>,
    /// Bind mount applied to every node container, as
    /// `HOST[:CONTAINER[:ro|rw]]`. May be repeated. Requires
    /// `--image`.
    #[arg(long = "mount", value_name = "HOST[:CONTAINER[:ro|rw]]")]
    pub mounts: Vec<String>,
    /// Port published from every node container to the host, as
    /// `HOST_PORT[:CONTAINER_PORT][/tcp|/udp]` (like docker `-p`).
    /// May be repeated. Requires `--image`.
    #[arg(long = "port", value_name = "HOST_PORT[:CONTAINER_PORT][/tcp|/udp]")]
    pub ports: Vec<String>,
    /// Container provider to use (`podman`, `docker`, or a path).
    /// Autodetected (podman, then docker) when omitted. Requires
    /// `--image`.
    #[arg(long = "container-provider")]
    pub provider: Option<String>,
    /// Never prompt; use defaults for any unspecified field even on
    /// a terminal.
    #[arg(long)]
    pub no_input: bool,
}

// ----- topology --------------------------------------------------------------

#[derive(Subcommand, Debug)]
pub enum TopologyCmd {
    /// List available topologies.
    List,
    /// Show a topology as JSON.
    Show { name: String },
    /// Create a topology.
    Create {
        name: String,
        /// Agent name referenced by this topology.
        #[arg(long, default_value = "MI350X")]
        agent: String,
        /// Nodes per rack.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        num_nodes: u32,
        /// GPUs per node.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        gpus_per_node: u32,
    },
    /// Import a topology from a JSON file (use `-` for stdin).
    Import { name: String, file: String },
    /// Delete a topology.
    Delete {
        name: String,
        /// Don't prompt for confirmation.
        #[arg(short = 'f', long)]
        force: bool,
    },
}

// ----- agent -----------------------------------------------------------------

#[derive(Subcommand, Debug)]
pub enum AgentCmd {
    /// List available agents.
    List,
    /// Show an agent as JSON.
    Show { name: String },
    /// Import an agent from a JSON file (use `-` for stdin).
    Import { name: String, file: String },
    /// Delete an agent.
    Delete {
        name: String,
        /// Don't prompt for confirmation.
        #[arg(short = 'f', long)]
        force: bool,
    },
}

// ----- session ---------------------------------------------------------------

/// Arguments for `mirage exec`.
#[derive(Args, Debug)]
pub struct ExecArgsCli {
    /// Session to run in.
    ///
    /// Optional, and usually omitted: with exactly one `mirage run`
    /// live — one terminal running the job, another one exec'ing into it
    /// — mirage picks it. Naming one is only needed when several runs
    /// are up at once.
    // A flag rather than a positional because everything after `--`
    // belongs to the command: with both positional, `mirage exec --
    // bash` could equally mean "session bash". Not a doc comment:
    // `--help` prints those, and why the flag is shaped this way is a
    // maintainer's question rather than a user's.
    #[arg(long, short = 's')]
    pub session: Option<SessionId>,

    /// How many of each node's process slots this exec fills. Defaults
    /// to `1`, which is what makes a plain `mirage exec` interactive.
    ///
    /// It is not the job's shape and cannot change it. Ranks are
    /// numbered in the *session's* grid — the one the `mirage run` that
    /// owns it was started with — so a process started here sees the
    /// same `WORLD_SIZE` and the same rendezvous as the ranks already
    /// running, and `mirage exec --node 2` is that node's rank 0 of the
    /// job the run started rather than rank 0 of a job of its own.
    ///
    /// So an exec may not ask for more processes per node than the run
    /// has: the slot after a node's last one is the next node's rank 0,
    /// and an exec process handed that number would be told it is a rank
    /// of a node it is not running on — wrong `LOCAL_RANK`, wrong
    /// `NCCL_HOSTID`, wrong host for every peer that resolves it. Asking
    /// for more is refused, naming the shape the session does have;
    /// start the run with `--nproc-per-node` if that is the job you
    /// wanted.
    ///
    /// What the cap does *not* do is keep an exec's ranks distinct from
    /// the live ones, and it is worth being plain about that because the
    /// bound looks like it should. An exec always duplicates rank numbers
    /// the run is already using: `mirage exec -- cmd` on a live job puts
    /// a process on rank 0 while the run's own rank 0 is still running.
    /// That is the design. An exec is for *tooling that joins the job* —
    /// a shell, `rocm-smi`, a debugger, a script reading a rank's files —
    /// and it is given the job's identity so that what it sees matches
    /// what that rank sees. It is not a way to add a participant to the
    /// collective, and a command that calls `init_process_group` under
    /// one will hang against a rendezvous that is already formed. The cap
    /// bounds where an exec's ranks *land*, not how many processes answer
    /// to a number.
    #[arg(long, visible_alias = "nproc_per_node", value_parser = clap::value_parser!(u32).range(1..))]
    pub nproc_per_node: Option<u32>,

    /// Run on this node only, instead of on every node in the session.
    ///
    /// This is how you get an interactive shell on a multi-node job. A
    /// job spanning several nodes has every rank's output multiplexed
    /// and nobody's stdin connected, because one terminal cannot be
    /// shared between readers. Naming one node leaves a single process,
    /// which does get the terminal:
    ///
    /// ```text
    /// mirage exec --node 2 -- bash
    /// ```
    ///
    /// The process still believes it is that node: same rank variables,
    /// same `WORLD_SIZE`, same rendezvous as its neighbours.
    ///
    /// It composes with `--nproc-per-node`, which decides how many of
    /// that node's slots to fill: `--node 1 --nproc-per-node 3` starts
    /// three processes on node 1, as ranks `1*P`, `1*P+1` and `1*P+2` of
    /// the job's own grid. What `--node` selects is *where*; what
    /// decides the terminal is the process count, and the count wins —
    /// so an exec asking for more than one process is captured and
    /// labelled whether it named a node or not. `--node` alone is
    /// interactive because it leaves one process, not because it named
    /// a node.
    #[arg(long, short = 'n')]
    pub node: Option<u32>,

    /// Start the workload with an almost-empty environment instead of
    /// inheriting this terminal's.
    ///
    /// By default everything you have exported reaches the workload —
    /// an API token, a `PYTHONPATH`, a proxy, a framework tuning
    /// variable — because mirage's parent is your shell and what is in
    /// it you put there. This drops all of it, keeping only what a
    /// process needs to run (`PATH`, `HOME`, `TERM`, …) plus the
    /// emulator's own variables and any `--env`.
    ///
    /// Use it when a result must not depend on ambient state: a
    /// benchmark, a reproduction, a CI job compared against a baseline.
    ///
    /// No effect on a containerised session, which never inherits the
    /// host environment anyway.
    #[arg(long)]
    pub clear_env_vars: bool,

    /// Extra environment variables, in `KEY=VALUE` form. May be repeated.
    ///
    /// These beat the emulator's own variables and anything you
    /// exported. The exceptions are the job's identity — `MIRAGE_*`,
    /// `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`,
    /// `NCCL_HOSTID` — which mirage sets last, because a rank that
    /// disagrees with the grid deadlocks its own collectives; and
    /// `LD_PRELOAD`, which is prepended to rather than replaced by the
    /// emulator's interposer. Passing one of the first group is
    /// accepted, ignored, and warned about.
    #[arg(long = "env", value_name = "KEY=VALUE")]
    pub envs: Vec<String>,

    /// Working directory for the command.
    ///
    /// On a containerised session this names a directory *inside* the
    /// container; otherwise one on this machine, which must exist.
    #[arg(long)]
    pub workdir: Option<String>,

    /// The command and its arguments. Use `--` to separate from mirage
    /// flags.
    #[arg(trailing_var_arg = true, required = true, allow_hyphen_values = true)]
    pub argv: Vec<String>,
}

#[derive(Subcommand, Debug)]
pub enum StateCmd {
    /// (Re)write the builtin agents, topologies and profiles under
    /// `<MIRAGE_CONFIG>`.
    ///
    /// Every command writes back any builtin that has gone missing, so
    /// this one exists for the case that does not cover: refreshing the
    /// ones already on disk after upgrading mirage.
    ///
    /// A builtin you have edited is left alone and named, because
    /// replacing it would destroy the only copy. Delete it first if you
    /// want the shipped version back — that is what a delete of an
    /// edited builtin is for.
    Builtins,
    /// Remove mirage's runtime directory and reclaim orphaned containers.
    ///
    /// Refuses while any `mirage run` is live: a run owns its session and
    /// cleans up when it exits, so stop those first. The config directory
    /// (agents, topologies, profiles) is left alone unless `--all` is
    /// passed.
    Purge {
        /// Don't prompt for confirmation.
        #[arg(short = 'f', long)]
        force: bool,
        /// Also remove the mirage config directory: every agent,
        /// topology and profile, including the ones you wrote. Mirage
        /// writes its own builtin agents, topologies and profiles back
        /// the next time it runs; nothing else comes back.
        #[arg(long)]
        all: bool,
    },
}

// ----- run -------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Profile to use. Defaults to the `mi350x` builtin.
    #[arg(long, default_value = "mi350x")]
    profile: String,
    /// Override the profile's emulator backend (e.g. `rocjitsu`,
    /// `rocjitsu-dbt`, `hotswap`). See `mirage emulators` for
    /// the available backends.
    #[arg(long)]
    emulator: Option<String>,
    /// Override the profile topology's node count for this run.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    num_nodes: Option<u32>,
    /// Override the profile topology's per-node GPU count for this run.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    gpus_per_node: Option<u32>,
    /// Number of workload processes to launch per node (like
    /// `torchrun --nproc-per-node`). Defaults to `1`. Each process gets a
    /// distinct `LOCAL_RANK` (`0..nproc_per_node`) and global `RANK`, and
    /// the job's `WORLD_SIZE` becomes `num_nodes * nproc_per_node`, so
    /// `torch.distributed` runs without a separate launcher. Give each
    /// node at least this many GPUs (`--gpus-per-node`) so every process
    /// can pin its own device.
    #[arg(long, visible_alias = "nproc_per_node", value_parser = clap::value_parser!(u32).range(1..))]
    nproc_per_node: Option<u32>,
    // No `--session` or `--keep-session`. A run *is* its session: it
    // creates one, owns it, and destroys it on the way out, so there is
    // neither an existing session to reuse nor a way to leave one behind.
    // Both flags used to be declared here and silently ignored by
    // `run_cmd`. Use `mirage exec` to join a run that is already up.
    /// Working directory.
    ///
    /// On a containerised session this names a directory *inside* the
    /// container; otherwise one on this machine, which must exist.
    /// Defaults to the directory `mirage run` was started from.
    #[arg(long)]
    workdir: Option<String>,
    /// Extra environment variables to inject into the exec, in
    /// `KEY=VALUE` form. May be repeated.
    ///
    /// These beat the emulator's own variables and anything you
    /// exported. The exceptions are the job's identity — `MIRAGE_*`,
    /// `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`,
    /// `NCCL_HOSTID` — which mirage sets last, because a rank that
    /// disagrees with the grid deadlocks its own collectives; and
    /// `LD_PRELOAD`, which is prepended to rather than replaced by the
    /// emulator's interposer. Passing one of the first group is
    /// accepted, ignored, and warned about.
    #[arg(long = "env", value_name = "KEY=VALUE")]
    envs: Vec<String>,
    /// Override/enable containerisation: run every node inside a
    /// container built from this image.
    #[arg(long)]
    image: Option<String>,
    /// Extra bind mount (`HOST[:CONTAINER[:ro|rw]]`). May be repeated.
    /// Requires a containerised profile or `--image`.
    #[arg(long = "mount", value_name = "HOST[:CONTAINER[:ro|rw]]")]
    mounts: Vec<String>,
    /// Publish a container port on the host
    /// (`HOST_PORT[:CONTAINER_PORT][/tcp|/udp]`, like docker `-p`). May
    /// be repeated. Requires a containerised profile or `--image`.
    #[arg(long = "port", value_name = "HOST_PORT[:CONTAINER_PORT][/tcp|/udp]")]
    ports: Vec<String>,
    /// Container provider (`podman`, `docker`, or a path). Autodetected
    /// when omitted. The `MIRAGE_CONTAINER_PROVIDER` environment variable
    /// has the same effect.
    #[arg(long = "container-provider")]
    container_provider: Option<String>,
    /// Apply an opt-in image hack by building a derivative image from the
    /// base image before launching containers. May be repeated. Requires
    /// a containerised profile or `--image`.
    #[arg(long = "hack", value_name = "HACK")]
    hacks: Vec<HackArg>,
    /// Override the emulator execution mode (`functional` or `clocked`).
    #[arg(long)]
    exec_mode: Option<ExecModeArg>,
    /// Override an emulator option directly (`KEY=VALUE`). May be
    /// repeated.
    #[arg(long = "option", short = 'o', value_name = "KEY=VALUE")]
    options: Vec<String>,
    /// Enable an execution plugin by name (e.g. `race`, `logging`),
    /// applying its schema defaults. May be repeated. Merges with any
    /// plugins the profile already enables.
    #[arg(long = "plugin", value_name = "NAME")]
    plugins: Vec<String>,
    /// Use an explicit emulator config file instead of synthesising one
    /// from the profile (the upstream `rocjitsu --config`).
    ///
    /// The file is handed to the backend verbatim, so the flags that
    /// would have gone into a synthesised config — `--gpus-per-node`,
    /// `--exec-mode`, `-o`/`--option`, `--plugin` — cannot also be
    /// honoured and are refused rather than ignored. Put them in the
    /// config file instead.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["gpus_per_node", "exec_mode", "options", "plugins"]
    )]
    config: Option<String>,
    /// Run the emulator in out-of-process daemon mode. This is the
    /// default; the flag is accepted for explicitness and under the
    /// upstream `rocjitsu` spelling `--attach`, which means the same
    /// thing here: mirage owns the emulator's whole lifecycle, so
    /// "attach to a daemon" and "use a daemon" are one request.
    // Declared here rather than only rewritten on the way in. `mirage
    // run --attach` already worked, but the rewrite happens before clap
    // ever sees the arguments, so `mirage run --help` omitted a spelling
    // the drop-in help offers — and the one page a user checks a `run`
    // flag against was the page that denied it. Not a doc comment:
    // `--help` prints those.
    #[arg(long, visible_alias = "attach", conflicts_with = "in_process")]
    daemon: bool,
    /// Run the emulator in-process (local mode) instead of the default
    /// out-of-process daemon. In-process mode cannot share GPU memory
    /// across processes, so multi-GPU RCCL collectives require the
    /// daemon (the default).
    #[arg(long = "in-process")]
    in_process: bool,
    // No `--capture-all`. Whether output is multiplexed is decided by
    // the shape of the job, not by a flag: one process gets the terminal
    // and its stdin, several get their output labelled and none of them
    // get stdin. A flag could only ever ask for the behaviour that
    // already applies. Use `mirage exec --node N` for a terminal on one
    // node of a multi-node run.
    /// Start the workload with an almost-empty environment instead of
    /// inheriting this terminal's.
    ///
    /// By default everything you have exported reaches the workload —
    /// an API token, a `PYTHONPATH`, a proxy, a framework tuning
    /// variable — because mirage's parent is your shell and what is in
    /// it you put there. This drops all of it, keeping only what a
    /// process needs to run (`PATH`, `HOME`, `TERM`, …) plus the
    /// emulator's own variables and any `--env`.
    ///
    /// Use it when a result must not depend on ambient state: a
    /// benchmark, a reproduction, a CI job compared against a baseline.
    ///
    /// No effect on a containerised session, which never inherits the
    /// host environment anyway.
    #[arg(long)]
    clear_env_vars: bool,
    /// Debug the workload interactively under ROCgdb: the command is
    /// wrapped as `rocgdb --args <command...>`, with kernel breakpoints
    /// made pending so `break <kernel>` works before the GPU code object
    /// loads at launch. Requires a profile whose container ships ROCgdb.
    #[arg(long)]
    gdb: bool,
    /// Extra ROCgdb command to run before the program, injected as
    /// `-ex <CMD>` (repeatable, applied in order). Implies `--gdb`.
    /// Useful for scripted/batch debugging, e.g.
    /// `--gdb-ex 'break add_one' --gdb-ex run --gdb-ex continue`.
    #[arg(long = "gdb-ex", value_name = "CMD")]
    gdb_ex: Vec<String>,
    /// The command and its arguments.
    #[arg(trailing_var_arg = true, required = true, allow_hyphen_values = true)]
    argv: Vec<String>,
}

/// Hand-written rather than derived, so that `RunArgs::default()` is the
/// same set of arguments clap produces for a bare `mirage run -- …`.
///
/// Every field but `profile` already agrees: an `Option` flag defaults to
/// `None`, a repeatable one to an empty `Vec`, and a switch to `false`.
/// `profile` is the exception — it carries `#[arg(default_value =
/// "mi350x")]`, which a derived `Default` would silently turn into the
/// empty string. The `default_matches_clap` test holds the two in step.
impl Default for RunArgs {
    fn default() -> Self {
        Self {
            profile: "mi350x".to_string(),
            emulator: None,
            num_nodes: None,
            gpus_per_node: None,
            nproc_per_node: None,
            workdir: None,
            envs: Vec::new(),
            image: None,
            mounts: Vec::new(),
            ports: Vec::new(),
            container_provider: None,
            hacks: Vec::new(),
            exec_mode: None,
            options: Vec::new(),
            plugins: Vec::new(),
            config: None,
            daemon: false,
            in_process: false,
            clear_env_vars: false,
            gdb: false,
            gdb_ex: Vec::new(),
            argv: Vec::new(),
        }
    }
}

// =============================================================================
// A mistyped mirage flag is not a workload
// =============================================================================

// =============================================================================
// Dispatch
// =============================================================================

/// The exit code a command uses when the user answered "no" at a
/// confirmation prompt.
///
/// Nothing happened, and nothing went wrong — which are two different
/// things, and `0` can only say one of them. A script that deletes a
/// profile has to be able to tell "it is gone" (0) from "you were asked
/// and said no" (this) from "mirage tried and could not" (1), and a
/// declined delete used to be indistinguishable from a completed one.
const DECLINED: u8 = 2;

/// Dispatch a parsed [`CtlCmd`]. Returns the exit code the process
/// should use.
///
/// Under `--json` a failure is rendered as `{"error": …}` on stderr and
/// reported as exit 1, rather than propagating to `main` to be printed as
/// an English sentence. A caller that asked for machine-readable output
/// and got prose on the one path it cannot parse has to fall back to
/// scraping the text, which is exactly what `--json` exists to avoid.
/// stdout is left alone either way: results go there, diagnostics do not,
/// so the single JSON document a successful command promises is never
/// mixed with an error object.
///
/// # Errors
///
/// Returns an error if the command fails and `json` is not set.
pub async fn dispatch(cmd: CtlCmd, json: bool) -> anyhow::Result<ExitCode> {
    match dispatch_inner(cmd, json).await {
        Err(e) if json => {
            // `{:#}` is what `main` prints, and walks the source chain the
            // same way, so the sentence inside the object is the same
            // sentence a text-mode run would have shown.
            let doc = serde_json::json!({ "error": format!("{e:#}") });
            match serde_json::to_string_pretty(&doc) {
                Ok(text) => eprintln!("{text}"),
                // A string in an object cannot fail to serialize; say
                // something rather than swallow the failure it reports.
                Err(_) => eprintln!("error: {e:#}"),
            }
            Ok(ExitCode::from(1))
        }
        other => other,
    }
}

async fn dispatch_inner(cmd: CtlCmd, json: bool) -> anyhow::Result<ExitCode> {
    // Best-effort: write any missing builtin agents/topologies on
    // startup so they are always available under <MIRAGE_CONFIG>/. Errors
    // here are non-fatal; the user can recover via `mirage state
    // builtins`.
    ensure_builtins_present();
    match cmd {
        CtlCmd::Profile(c) => profile_cmd(c, json).await,
        CtlCmd::Topology(c) => topology_cmd(c, json).await,
        CtlCmd::Agent(c) => agent_cmd(c, json).await,
        CtlCmd::Emulators { long } => {
            emulators_cmd(long, json);
            Ok(ExitCode::from(0))
        }
        CtlCmd::Exec(a) => run::exec_cmd(a).await,
        CtlCmd::State(c) => state_cmd(c, json).await,
        CtlCmd::Cleanup { dry_run } => {
            let reclaimed = cleanup(dry_run).await;
            let unfinished = reclaimed.failures.len();
            reclaimed.report(dry_run, json)?;
            // Zero would say the machine is clean. Something mirage set
            // out to remove is still there.
            Ok(ExitCode::from(u8::from(unfinished > 0)))
        }
        CtlCmd::Run(a) => run::run_cmd(a).await,
        CtlCmd::Paths => {
            print_paths(json);
            Ok(ExitCode::from(0))
        }
    }
}

// ----- profile dispatch ------------------------------------------------------

async fn profile_cmd(cmd: ProfileCmd, json: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        ProfileCmd::List { long } => {
            let names = mirage_core::store::profile_list()?;
            if json && long {
                println!("{}", serde_json::to_string_pretty(&long_profiles(&names))?);
            } else if json {
                println!("{}", serde_json::to_string_pretty(&names)?);
            } else if long {
                if names.is_empty() {
                    eprintln!("(no profiles)");
                }
                println!("{:<24} {:<16} DESCRIPTION", "NAME", "EMULATOR");
                for n in names {
                    match mirage_core::store::profile_get(&n) {
                        Ok(p) => println!(
                            "{:<24} {:<16} {}",
                            p.name,
                            p.emulator.emulator,
                            p.description.as_deref().unwrap_or("")
                        ),
                        Err(_) => println!("{n:<24} (unreadable)"),
                    }
                }
            } else {
                for n in names {
                    println!("{n}");
                }
            }
        }
        ProfileCmd::Show { name } => {
            let p = mirage_core::store::profile_get(&name)?;
            println!("{}", serde_json::to_string_pretty(&p)?);
        }
        ProfileCmd::Create(a) => {
            let interactive = !a.no_input && std::io::stdin().is_terminal();
            let p = build_profile_create(a, interactive)?;
            if let Err(e) = validate_profile(&p) {
                anyhow::bail!("cannot create profile {}: {e}", p.name);
            }
            let stored = mirage_core::store::profile_put(&p)?;
            report_replaced_builtin(DocKind::Profile, &p.name, stored);
            if json {
                println!("{}", serde_json::to_string_pretty(&p)?);
            } else {
                println!("created profile {}", p.name);
            }
        }
        ProfileCmd::Import { file } => {
            let (bytes, from) = read_input(&file)?;
            let p: ProfileDef = serde_json::from_slice(&bytes).map_err(|e| json_error(from, e))?;
            let stored = mirage_core::store::profile_put(&p)?;
            report_replaced_builtin(DocKind::Profile, &p.name, stored);
            report_change(json, DocKind::Profile, &p.name, "imported")?;
        }
        ProfileCmd::Delete { name, force } => {
            return delete_document(json, DocKind::Profile, &name, force, |name| {
                mirage_core::store::profile_delete(name)
            });
        }
    }
    Ok(ExitCode::from(0))
}

/// The long form of `profile list`, as JSON.
///
/// `--long` names the fields it exists to show, and under `--json` it
/// used to be accepted and then drop every one of them: a script asking
/// for the emulator and the description got the same bare array of names
/// as one asking for neither, with nothing in the output to say so.
fn long_profiles(names: &[String]) -> Vec<serde_json::Value> {
    names
        .iter()
        .map(|n| match mirage_core::store::profile_get(n) {
            Ok(p) => serde_json::json!({
                "name": p.name,
                "emulator": p.emulator.emulator,
                "description": p.description,
            }),
            // The text form prints "(unreadable)" for one of these. A
            // reader of the JSON needs the same fact, and needs it as a
            // field rather than as a name it has to recognise — with the
            // reason, which the text form has no room for.
            Err(e) => serde_json::json!({
                "name": n,
                "unreadable": e.full_message(),
            }),
        })
        .collect()
}

/// Say when a `create` or `import` consumed a builtin.
///
/// Replacing an untouched builtin is deliberately allowed — it is mirage's
/// own seed, identical to the copy still in the binary, and taking the
/// name over is how a builtin gets customised. Doing it in silence is
/// another matter: `mirage topology create MI350X-2x8` exited 0 with
/// nothing to say while a shipped two-node layout became whatever the
/// flags defaulted to, which is the very thing the overwrite guard exists
/// to prevent, one level down.
///
/// On stderr, so that `--json`'s single document on stdout stays a single
/// document: this is a remark about what happened, not a result.
fn report_replaced_builtin(kind: DocKind, name: &str, stored: Stored) {
    if stored != Stored::ReplacedBuiltin {
        return;
    }
    let kind = kind.as_str();
    eprintln!(
        "mirage: {name} was a builtin {kind} mirage ships, and this replaced it. \
         Nothing you wrote was lost; `mirage {kind} delete {name}` restores the \
         shipped version."
    );
}

// ----- topology dispatch -----------------------------------------------------

async fn topology_cmd(cmd: TopologyCmd, json: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        TopologyCmd::List => {
            let names = mirage_core::store::topology_list()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&names)?);
            } else {
                for n in names {
                    println!("{n}");
                }
            }
        }
        TopologyCmd::Show { name } => {
            let t = mirage_core::store::topology_get(&name)?;
            println!("{}", serde_json::to_string_pretty(&t)?);
        }
        TopologyCmd::Create {
            name,
            agent,
            num_nodes,
            gpus_per_node,
        } => {
            let t = mirage_core::topology::TopologyDef {
                num_nodes,
                gpus_per_node,
                agent: MaybeRef::Ref(DocKind::Agent.canonical(&agent)),
            };
            let stored = mirage_core::store::topology_put(&name, &t)?;
            report_replaced_builtin(DocKind::Topology, &name, stored);
            if json {
                println!("{}", serde_json::to_string_pretty(&t)?);
            } else {
                println!("created topology {name}");
            }
        }
        TopologyCmd::Import { name, file } => {
            let (bytes, from) = read_input(&file)?;
            let t: mirage_core::topology::TopologyDef =
                serde_json::from_slice(&bytes).map_err(|e| json_error(from, e))?;
            let stored = mirage_core::store::topology_put(&name, &t)?;
            report_replaced_builtin(DocKind::Topology, &name, stored);
            report_change(json, DocKind::Topology, &name, "imported")?;
        }
        TopologyCmd::Delete { name, force } => {
            return delete_document(json, DocKind::Topology, &name, force, |name| {
                mirage_core::store::topology_delete(name)
            });
        }
    }
    Ok(ExitCode::from(0))
}

// ----- agent dispatch --------------------------------------------------------

async fn agent_cmd(cmd: AgentCmd, json: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        AgentCmd::List => {
            let names = mirage_core::store::agent_list()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&names)?);
            } else {
                for n in names {
                    println!("{n}");
                }
            }
        }
        AgentCmd::Show { name } => {
            let a = mirage_core::store::agent_get(&name)?;
            println!("{}", serde_json::to_string_pretty(&a)?);
        }
        AgentCmd::Import { name, file } => {
            let (bytes, from) = read_input(&file)?;
            let a: mirage_core::agent::AgentDef =
                serde_json::from_slice(&bytes).map_err(|e| json_error(from, e))?;
            let stored = mirage_core::store::agent_put(&name, &a)?;
            report_replaced_builtin(DocKind::Agent, &name, stored);
            report_change(json, DocKind::Agent, &name, "imported")?;
        }
        AgentCmd::Delete { name, force } => {
            return delete_document(json, DocKind::Agent, &name, force, |name| {
                mirage_core::store::agent_delete(name)
            });
        }
    }
    Ok(ExitCode::from(0))
}

/// Report a stored document that changed — an import.
///
/// Under `--json` stdout must be exactly one JSON document and nothing
/// else, or the caller that asked for JSON cannot parse what it gets.
/// These commands used to print their sentence either way, so
/// `mirage profile import --json f.json` emitted `imported profile x`
/// and a script's `json.load` failed on the word "imported".
fn report_change(json: bool, kind: DocKind, name: &str, verb: &str) -> anyhow::Result<()> {
    let kind = kind.as_str();
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "kind": kind,
                "name": name,
                verb: true,
            }))?
        );
    } else {
        println!("{verb} {kind} {name}");
    }
    Ok(())
}

/// Delete one stored document, having asked first unless told not to.
///
/// One function for all three resource verbs because everything here is
/// the same for all three and was previously written out three times,
/// which is how `--force` came to be documented on one of them and not
/// the others.
///
/// Three things a delete has to say, and used to say none of:
///
/// * A declined prompt is not a completed delete. Both exited 0, and in
///   text mode both printed nothing at all, so `mirage profile delete x`
///   answered with `n` was indistinguishable from one that worked. It
///   now exits [`DECLINED`] and says so.
/// * There is nothing to decline when there was nothing there. The
///   prompt used to come first, so a mistyped name asked "delete profile
///   proflie?" and, on `n`, reported a document that never existed as
///   one deliberately spared — and on `y` reported it as missing, which
///   is the answer the user wanted before they were asked anything.
///   Whether the document is there is settled first now.
/// * Deleting a builtin the user has *edited* really does remove their
///   copy, and the shipped one takes its place. `deleted profile mi350x`
///   followed by `mi350x` still being listed reads as a delete that
///   failed. Saying which one went, and what came back, is the
///   difference — see [`restore_shipped`] for why that claim is made
///   after the shipped version is on disk rather than before.
fn delete_document(
    json: bool,
    kind: DocKind,
    name: &str,
    force: bool,
    delete: impl FnOnce(&str) -> mirage_core::error::Result<()>,
) -> anyhow::Result<ExitCode> {
    let word = kind.as_str();
    // Through the store's own name check first, so a name that could
    // never address a document (`../../etc/passwd`) is refused as the
    // bad name it is rather than probed for on disk.
    mirage_core::store::validate_name(kind, name)?;
    if !kind.path(name).exists() {
        return Err(mirage_core::error::MirageError::not_found(kind, name).into());
    }
    if !force && !confirm(&format!("delete {word} {name}?"))? {
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "kind": word,
                    "name": name,
                    "deleted": false,
                    "declined": true,
                }))?
            );
        } else {
            println!("{word} {name} not deleted: you declined");
        }
        return Ok(ExitCode::from(DECLINED));
    }
    delete(name)?;
    let restored = restore_shipped(kind, name);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "kind": word,
                "name": name,
                "deleted": true,
                "shipped_version_restored": restored,
            }))?
        );
    } else if restored {
        println!(
            "deleted your {word} {name}; mirage ships a builtin of that name, \
             and the shipped version is back in its place"
        );
    } else {
        println!("deleted {word} {name}");
    }
    Ok(ExitCode::from(0))
}

/// Put the shipped version of a just-deleted builtin back, and report
/// whether it is there.
///
/// The claim this answers — "the shipped version is back in its place" —
/// used to be printed, and asserted as `shipped_version_restored` under
/// `--json`, on the strength of mirage *shipping* a document of that
/// name. Nothing had written it: a delete removes a file and stops, and
/// the shipped copy reappears only when the next command's
/// [`ensure_builtins_present`] notices it missing. So `mirage agent
/// delete mi300x` said the builtin was back while the directory it names
/// no longer held it, and a script that believed the field went looking
/// for a file that would not exist until it ran mirage again.
///
/// Writing it here is what makes the sentence true when it is printed,
/// and the same call every command already makes on the way in, so
/// nothing new can appear on disk that would not have appeared anyway.
/// The verdict is then read off the filesystem rather than inferred: a
/// config directory mirage cannot write is exactly the case where the
/// old claim was worst, and it is the case where this one says no.
fn restore_shipped(kind: DocKind, name: &str) -> bool {
    if mirage_core::store::shipped(kind, name).is_none() {
        return false;
    }
    ensure_builtins_present();
    kind.path(name).exists()
}

/// Read one document named on the command line, or stdin for `-`,
/// returning its bytes and the name to blame if they do not parse.
///
/// An `import` failure used to surface the bare `std::io::Error`, so a
/// missing file was `No such file or directory (os error 2)` with the
/// filename nowhere in it — and the user had typed the filename, so it is
/// the one word that identifies which of their commands went wrong. The
/// config *reader* in the same binary already answers `io error on
/// /…/x.json: No such file or directory (os error 2)`, and there is no
/// reason for the writer's side of the same file to speak differently.
fn read_input(file: &str) -> anyhow::Result<(Vec<u8>, std::path::PathBuf)> {
    // Standard input has no path, and inventing one ("-") would be worse
    // than naming what it actually is in a message a person reads.
    let from = std::path::PathBuf::from(if file == "-" { "<stdin>" } else { file });
    let bytes = if file == "-" {
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin().lock(), &mut buf)
            .map_err(|e| mirage_core::error::MirageError::io(&from, e))?;
        buf
    } else {
        std::fs::read(file).map_err(|e| mirage_core::error::MirageError::io(&from, e))?
    };
    Ok((bytes, from))
}

/// Blame the document an `import` could not parse, by name.
///
/// serde's own message is a line and a column of a file it does not
/// identify, which is unreadable in a script importing several — and it
/// leaks the Rust type it was deserializing into, which names nothing the
/// user has. [`MirageError::Json`] is the shape the rest of mirage
/// reports a bad document in.
///
/// [`MirageError::Json`]: mirage_core::error::MirageError::Json
fn json_error(path: std::path::PathBuf, source: serde_json::Error) -> anyhow::Error {
    anyhow::Error::new(mirage_core::error::MirageError::Json { path, source })
}

fn build_containerize(
    image: Option<String>,
    mounts: &[String],
    ports: &[String],
    provider: Option<String>,
) -> anyhow::Result<Option<ContainerizedDef>> {
    match image {
        Some(image) => Ok(Some(ContainerizedDef {
            provider,
            image,
            mounts: parse_mounts(mounts)?,
            ports: parse_ports(ports)?,
            devices: Vec::new(),
            groups: Vec::new(),
            hacks: Vec::new(),
        })),
        None => {
            let given = container_flags_given(mounts, ports, provider.as_ref(), &[]);
            if !given.is_empty() {
                anyhow::bail!(
                    "{} {} --image",
                    and_list(&given),
                    Plural(given.len()).pick("requires", "require")
                );
            }
            Ok(None)
        }
    }
}

/// Which of the container flags the user actually passed, in the order
/// they are documented.
///
/// The refusal used to name all of them — `--mount/--port/
/// --container-provider/--hack require a containerised profile or
/// --image` — which reads as a rule rather than as a report, and leaves
/// the user checking their own command line for three flags they never
/// typed. Naming the one they did is the whole message.
fn container_flags_given<'a>(
    mounts: &[String],
    ports: &[String],
    provider: Option<&'a String>,
    hacks: &[HackArg],
) -> Vec<&'a str> {
    let mut given = Vec::new();
    if !mounts.is_empty() {
        given.push("--mount");
    }
    if !ports.is_empty() {
        given.push("--port");
    }
    if provider.is_some() {
        given.push("--container-provider");
    }
    if !hacks.is_empty() {
        given.push("--hack");
    }
    given
}

/// Build a [`ProfileDef`] for `profile create`.
///
/// Every field passed as a flag is used verbatim. When `interactive`
/// is set, any field left unspecified is prompted for; otherwise the
/// field's default is used. This keeps `profile create <name>` a
/// friendly interactive UI on a terminal while remaining fully
/// non-interactive (defaults) in scripts, pipes and tests.
fn build_profile_create(a: ProfileCreateArgs, interactive: bool) -> anyhow::Result<ProfileDef> {
    use dialoguer::{Confirm, Input, Select};
    let theme = dialoguer::theme::ColorfulTheme::default();

    // ----- name -----
    let name = match a.name {
        Some(n) => n,
        None if interactive => Input::with_theme(&theme)
            .with_prompt("Profile name")
            .validate_with(|s: &String| -> Result<(), &str> {
                if s.trim().is_empty() {
                    Err("name required")
                } else {
                    Ok(())
                }
            })
            .interact_text()?,
        None => anyhow::bail!("a profile name is required"),
    };

    // ----- emulator -----
    let spec = match a.emulator.as_deref() {
        Some(n) => match find_emulator(n) {
            Some(s) => s,
            None => anyhow::bail!(unknown_emulator(n)),
        },
        None if interactive => {
            let specs = registry();
            let default_name = default_emulator_name();
            let default_idx = specs
                .iter()
                .position(|s| s.name == default_name)
                .unwrap_or(0);
            let labels: Vec<String> = specs
                .iter()
                .map(|s| {
                    let installed = if s.installed {
                        "[installed]"
                    } else {
                        "[not installed]"
                    };
                    let supported = if s.support.supported {
                        ""
                    } else {
                        " [unsupported hardware]"
                    };
                    format!("{:<10} {installed}{supported}  {}", s.name, s.description)
                })
                .collect();
            let pick = Select::with_theme(&theme)
                .with_prompt("Emulator")
                .items(&labels)
                .default(default_idx)
                .interact()?;
            specs[pick].clone()
        }
        None => default_emulator().ok_or_else(|| {
            anyhow::anyhow!(
                "this build of mirage has no emulator backends compiled in; \
                 rebuild with at least one (e.g. --features rocjitsu)"
            )
        })?,
    };

    // ----- topology -----
    let num_nodes = resolve_count(a.num_nodes, "Nodes per rack", interactive, &theme)?;
    let gpus_per_node = resolve_count(a.gpus_per_node, "GPUs per node", interactive, &theme)?;

    // ----- agent -----
    //
    // Whichever branch supplies it, the name is stored in the single
    // spelling agents are addressed by. `--agent MI350X` used to be
    // written verbatim while the interactive picker offered the on-disk
    // (lowercase) names, so the same profile came out as `MI350X` from a
    // script and `mi350x` from a terminal — one document described two
    // ways by two paths of one command.
    let agent = match a.agent {
        Some(a) => a,
        None if interactive => {
            let known = mirage_core::agent::store::list().unwrap_or_default();
            if known.is_empty() {
                "MI350X".to_string()
            } else {
                let default_idx = known
                    .iter()
                    .position(|n| n.eq_ignore_ascii_case("MI350X"))
                    .unwrap_or(0);
                let pick = Select::with_theme(&theme)
                    .with_prompt("Agent")
                    .items(&known)
                    .default(default_idx)
                    .interact()?;
                known[pick].clone()
            }
        }
        None => "MI350X".to_string(),
    };

    // ----- description -----
    let description = match a.description {
        Some(d) => Some(d),
        None if interactive => {
            let d: String = Input::with_theme(&theme)
                .with_prompt("Description (optional)")
                .allow_empty(true)
                .interact_text()?;
            if d.is_empty() { None } else { Some(d) }
        }
        None => None,
    };

    // ----- containerisation -----
    let containerize =
        if a.image.is_some() || !a.mounts.is_empty() || !a.ports.is_empty() || a.provider.is_some()
        {
            // Any explicit container flag: build directly (errors if mounts
            // or provider were given without an image).
            build_containerize(a.image, &a.mounts, &a.ports, a.provider)?
        } else if interactive
            && Confirm::with_theme(&theme)
                .with_prompt("Run each node inside a container?")
                .default(false)
                .interact()?
        {
            let img: String = Input::with_theme(&theme)
                .with_prompt("Image")
                .validate_with(|s: &String| -> Result<(), &str> {
                    if s.trim().is_empty() {
                        Err("image required")
                    } else {
                        Ok(())
                    }
                })
                .interact_text()?;
            let prov: String = Input::with_theme(&theme)
                .with_prompt("Provider (blank to auto-detect)")
                .allow_empty(true)
                .interact_text()?;
            let mut specs: Vec<String> = Vec::new();
            while Confirm::with_theme(&theme)
                .with_prompt("Add a bind mount?")
                .default(false)
                .interact()?
            {
                let m: String = Input::with_theme(&theme)
                    .with_prompt("Mount (HOST[:CONTAINER[:ro|rw]])")
                    .interact_text()?;
                if !m.trim().is_empty() {
                    specs.push(m);
                }
            }
            build_containerize(
                Some(img),
                &specs,
                &a.ports,
                if prov.is_empty() { None } else { Some(prov) },
            )?
        } else {
            None
        };

    let topo = mirage_core::topology::TopologyDef {
        num_nodes,
        gpus_per_node,
        agent: MaybeRef::Ref(DocKind::Agent.canonical(&agent)),
    };
    Ok(ProfileDef {
        name,
        description,
        emulator: mirage_core::registry::make_def(&spec, topo),
        containerize,
    })
}

/// Resolve a topology count: explicit value, interactive prompt, or 1.
fn resolve_count(
    value: Option<u32>,
    prompt: &str,
    interactive: bool,
    theme: &dialoguer::theme::ColorfulTheme,
) -> anyhow::Result<u32> {
    match value {
        Some(v) => Ok(v),
        None if interactive => Ok(dialoguer::Input::with_theme(theme)
            .with_prompt(prompt)
            .default(1)
            .interact_text()?),
        None => Ok(1),
    }
}

/// Parse CLI `--mount` specs into [`FileMount`]s.
fn parse_mounts(mounts: &[String]) -> anyhow::Result<Vec<FileMount>> {
    mounts
        .iter()
        .map(|m| FileMount::parse(m).map_err(|e| anyhow::anyhow!(e)))
        .collect()
}

/// Parse CLI `--port` specs into [`PortMapping`]s.
fn parse_ports(ports: &[String]) -> anyhow::Result<Vec<PortMapping>> {
    ports
        .iter()
        .map(|p| PortMapping::parse(p).map_err(|e| anyhow::anyhow!(e)))
        .collect()
}

/// Emulator execution mode, exposed on the CLI as `--exec-mode`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum ExecModeArg {
    /// Functional emulation (default): correct results, no timing model.
    Functional,
    /// Clocked emulation: model device timing.
    Clocked,
}

impl From<ExecModeArg> for ExecMode {
    fn from(m: ExecModeArg) -> Self {
        match m {
            ExecModeArg::Functional => ExecMode::Functional,
            ExecModeArg::Clocked => ExecMode::Clocked,
        }
    }
}

/// Opt-in image hack, exposed on the CLI as `--hack` (repeatable).
/// Mirrors [`mirage_core::profile::Hack`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum HackArg {
    /// Build a derivative image that updates `libstdc++6`/`libgcc-s1`
    /// from the `ubuntu-toolchain-r/test` PPA, fixing `GLIBCXX_*`/`GCC_*`
    /// "version not found" errors from binaries built against a newer
    /// toolchain than the base image ships.
    UpdateGccViaPpa,
}

impl From<HackArg> for Hack {
    fn from(h: HackArg) -> Self {
        match h {
            HackArg::UpdateGccViaPpa => Hack::UpdateGccViaPpa,
        }
    }
}

/// Parse a `KEY=VALUE` emulator option into a typed [`SimpleValue`].
///
/// Values that look like booleans or integers are stored as such so the
/// override matches what a hand-written profile would carry; everything
/// else is kept as a string.
fn parse_option(spec: &str) -> anyhow::Result<(String, SimpleValue)> {
    let (key, value) = spec
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("invalid option {spec:?} (expected KEY=VALUE)"))?;
    if key.is_empty() {
        anyhow::bail!("invalid option {spec:?} (empty key)");
    }
    let parsed = match value {
        "true" => SimpleValue::Boolean(true),
        "false" => SimpleValue::Boolean(false),
        _ => match value.parse::<i64>() {
            Ok(n) => SimpleValue::Number(n),
            Err(_) => SimpleValue::String(value.to_string()),
        },
    };
    Ok((key.to_string(), parsed))
}

/// Parse a `--plugin` spec (a plugin name) into a plugin entry.
///
/// The CLI flag only selects which plugins to enable; each is added with an
/// empty argument object so the rocjitsu plugin loader applies the plugin's
/// schema defaults. Plugins that need explicit arguments are configured
/// through a profile or an explicit `--config` file. The accepted name
/// characters match the loader's own validation (letters, digits, '_', '-'),
/// so a plugin name can never contain a path separator and escape the
/// loader's plugin directory.
fn parse_plugin(spec: &str) -> anyhow::Result<(String, SimpleMap)> {
    let name = spec.trim();
    if name.is_empty() {
        anyhow::bail!("invalid plugin {spec:?} (empty name)");
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        anyhow::bail!("invalid plugin name {name:?} (allowed: letters, digits, '_', '-')");
    }
    Ok((name.to_string(), SimpleMap::new()))
}

/// The one way mirage says it has no such emulator.
///
/// Three sentences said this in one file — a bare `unknown emulator
/// \`x\``, an `unknown emulator: x. Known: …` and an `unknown emulator
/// \`x\`; available backends: …` — so which of them a user saw depended
/// on whether they had mistyped `--emulator` on a `create` or on a
/// `run`, and only two of the three told them what to type instead. The
/// list is the useful half and belongs in all of them.
fn unknown_emulator(name: &str) -> String {
    format!(
        "unknown emulator `{name}`; this build has: {}. See `mirage emulators`.",
        name_list(registry().into_iter().map(|e| e.name).collect())
    )
}

/// The option names a backend accepts, in schema order.
fn option_names(spec: &EmulatorInfo) -> Vec<String> {
    spec.options_schema.iter().map(|o| o.name.clone()).collect()
}

/// `a, b, c`, or `(none)` for an empty list.
fn name_list(names: Vec<String>) -> String {
    if names.is_empty() {
        "(none)".to_string()
    } else {
        names.join(", ")
    }
}

/// `a`, `a and b`, `a, b and c` — a list inside a sentence.
///
/// [`name_list`] is the other one, and the two are not interchangeable:
/// that one enumerates a machine's answer to "what would have worked",
/// where the comma is a delimiter and an empty list has to say so. This
/// one is prose, read aloud, and never empty because its callers only
/// build it from things the user typed.
fn and_list(items: &[&str]) -> String {
    match items {
        [] => String::new(),
        [only] => (*only).to_string(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// The words a sentence about `count` things needs, so that one count
/// cannot inflect two of them differently.
///
/// A five-way `if n == 1` in a single `eprintln!` is where this started;
/// sixty lines below it a second sentence gave up and wrote
/// `process(es)`, which reads as a form nobody finished filling in. Both
/// ask this now, and so does every sentence added since.
#[derive(Clone, Copy, Debug)]
struct Plural(usize);

impl Plural {
    /// `one` when there is exactly one of them, `many` otherwise.
    ///
    /// Both forms are given rather than a suffix appended, because
    /// English does not agree with itself about the suffix: a noun takes
    /// `s` in the plural and a verb takes it in the singular, and
    /// `process` takes `es`.
    fn pick<'a>(self, one: &'a str, many: &'a str) -> &'a str {
        if self.0 == 1 { one } else { many }
    }
}

/// Reject `-o KEY=VALUE` for a key the backend does not know.
///
/// An override that overrides nothing is not an override: the key is
/// carried into the emulator's option map, the backend reads the keys it
/// has a schema entry for, and a misspelling is silently the same as
/// having passed nothing at all. The backend publishes its schema (see
/// `mirage emulators --json`), so the mistake is knowable here — and it
/// is named the same way a bad `--emulator` is, with the list of what
/// would have worked.
///
/// # Errors
///
/// Returns an error naming the first unknown key and every key the
/// backend does accept.
fn check_option_keys(emulator: &str, schema: &[String], keys: &[String]) -> anyhow::Result<()> {
    for key in keys {
        if schema.iter().any(|known| known == key) {
            continue;
        }
        if schema.is_empty() {
            anyhow::bail!(
                "unknown option `{key}` for emulator `{emulator}`, which accepts no options. \
                 What it emulates comes from its agent and topology (`mirage agent list`, \
                 `mirage topology list`), or from a config file passed with `--config`."
            );
        }
        anyhow::bail!(
            "unknown option `{key}` for emulator `{emulator}`; \
             it accepts: {}. See `mirage emulators -l`.",
            schema.join(", ")
        );
    }
    Ok(())
}

/// Reject `--plugin NAME` for a plugin the backend cannot load.
///
/// Running without the instrumentation that was asked for is worse than
/// not running: a `--plugin race` that loads nothing produces a clean
/// report of a racy program. The backend discovers its plugins on this
/// host (see `mirage emulators -l`), so an unloadable name is knowable
/// before the session exists.
///
/// # Errors
///
/// Returns an error naming the plugin and what this host has instead.
fn check_plugin_names(
    emulator: &str,
    available: &[String],
    names: &[String],
) -> anyhow::Result<()> {
    for name in names {
        if available.iter().any(|known| known == name) {
            continue;
        }
        if available.is_empty() {
            anyhow::bail!(
                "no plugin `{name}` for emulator `{emulator}`: this host has none of its \
                 plugins installed. `mirage emulators -l` lists what was found."
            );
        }
        anyhow::bail!(
            "no plugin `{name}` for emulator `{emulator}`; \
             this host has: {}. See `mirage emulators -l`.",
            available.join(", ")
        );
    }
    Ok(())
}

/// Check a `--workdir` that names a directory on *this* machine.
///
/// The operating system checks it too, at `chdir` time inside the
/// spawned child — and by then the only thing left to blame is the
/// program: `chdir` failing with `ENOENT` is indistinguishable, from the
/// spawn's return value, from the command not existing. That is what a
/// missing workdir used to be reported as (`command not found:
/// /bin/true`, with the path that was actually missing never printed),
/// which sends the user looking for a binary that is exactly where they
/// left it.
///
/// Only for a host-side session: a containerised one runs the workload
/// inside the container, where the path means something else entirely
/// and this filesystem has no opinion about it.
///
/// # Errors
///
/// Returns an error naming the path and what is wrong with it.
fn check_host_workdir(path: &str) -> anyhow::Result<()> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("--workdir {path}: {e}. Name a directory that exists."))?;
    if !metadata.is_dir() {
        anyhow::bail!("--workdir {path}: this is a file, not a directory.");
    }
    // A directory can exist and still not be one a process may sit in:
    // entering it needs the execute bit, which is a separate answer from
    // "it is there".
    nix::unistd::access(path, nix::unistd::AccessFlags::X_OK)
        .map_err(|e| anyhow::anyhow!("--workdir {path}: cannot enter this directory ({e})."))?;
    Ok(())
}

/// Reject a process grid mirage will not start, before it has built the
/// session it would have had to tear down again to say so.
///
/// The same bound the supervisor applies ([`MAX_WORLD_SIZE`]), applied
/// where it costs nothing. Bring-up creates containers, a network and an
/// emulator daemon; a multiplication that was always going to be refused
/// should not cost all of that first.
///
/// [`MAX_WORLD_SIZE`]: mirage_supervisor::spec::MAX_WORLD_SIZE
///
/// # Errors
///
/// Returns an error naming the grid and the limit.
fn check_grid(num_nodes: u32, nproc_per_node: u32) -> anyhow::Result<()> {
    let max = mirage_supervisor::spec::MAX_WORLD_SIZE;
    let world = u64::from(num_nodes) * u64::from(nproc_per_node);
    if world > u64::from(max) {
        anyhow::bail!(
            "{num_nodes} nodes x {nproc_per_node} processes per node is {world} processes, \
             more than the {max} mirage will start for one exec"
        );
    }
    Ok(())
}

/// Descriptors mirage needs for itself, beside the workload's.
///
/// The control socket's listener and one connection per borrower, the
/// emulator's socket, this process's own three streams, whatever the
/// provider clients need while they are being spawned, and room to be
/// wrong. A grid of any width measures twelve before its ranks are
/// counted; this is that with room over. Generous rather than derived:
/// getting it slightly high costs nothing, and getting it low
/// reintroduces the failure it exists to prevent.
const DESCRIPTOR_RESERVE: u64 = 64;

/// Descriptors one captured rank costs the process that started it.
///
/// Measured rather than reasoned about, because reasoning about it gave
/// the wrong answer. A captured rank has its stdout and stderr on pipes
/// and mirage holds the read end of each, which suggests two — but
/// counting `/proc/<mirage>/fd` against a live grid gives a slope of
/// exactly three, on a base of twelve:
///
/// ```text
/// ranks=8    fds=36
/// ranks=32   fds=108
/// ranks=128  fds=396
/// ```
///
/// Three is the number to reserve for, and it is the direction that
/// matters: an underestimate lets the check pass a grid that then runs
/// out of descriptors anyway, which is the failure it exists to prevent.
///
/// A one-process job inherits mirage's own streams instead and costs
/// nothing, which is why the check below skips it.
const DESCRIPTORS_PER_RANK: u64 = 3;

/// Make sure this process may open enough files to run a grid of `world`
/// captured ranks, raising its own limit if that is allowed.
///
/// A wide grid runs into `RLIMIT_NOFILE` before it runs into anything
/// mirage bounds. The symptom is bad in a specific way: the ranks that
/// fit start, the rest fail one by one with `Too many open files`, and
/// the run continues — an eight-rank job silently running five. The
/// control socket's own accept loop hits the same wall from the other
/// side, which is why it backs off on `EMFILE` rather than spinning.
///
/// A process may raise its own soft limit as far as the hard one without
/// privilege, so the first answer to "not enough" is to ask for more
/// rather than to refuse. Only the hard limit is somebody else's to
/// grant, and only that is worth failing on — before bring-up, because a
/// job that cannot start every rank should not first create containers,
/// a network and an emulator daemon.
///
/// # Errors
///
/// Returns an error naming the grid, what it needs, and the ceiling.
fn ensure_descriptors_for(world: u64) -> anyhow::Result<()> {
    use nix::sys::resource::{Resource, getrlimit, setrlimit};

    // One process inherits rather than being captured, so it opens
    // nothing. Worth skipping explicitly: it is the overwhelmingly
    // common shape, and it must not be refused on a machine with a
    // miserly `nofile`.
    if world < 2 {
        return Ok(());
    }
    let need = world
        .saturating_mul(DESCRIPTORS_PER_RANK)
        .saturating_add(DESCRIPTOR_RESERVE);
    let (soft, hard) = getrlimit(Resource::RLIMIT_NOFILE)
        .map_err(|e| anyhow::anyhow!("could not read this process's file limit: {e}"))?;
    if soft >= need {
        return Ok(());
    }
    if hard >= need {
        setrlimit(Resource::RLIMIT_NOFILE, need, hard).map_err(|e| {
            anyhow::anyhow!(
                "this job needs {need} open files and this process is limited to {soft}. \
                 Raising it to {need}, which the hard limit of {hard} allows, failed: {e}"
            )
        })?;
        tracing::debug!("raised RLIMIT_NOFILE from {soft} to {need} for {world} ranks");
        return Ok(());
    }
    anyhow::bail!(
        "{world} processes need about {need} open files ({DESCRIPTORS_PER_RANK} per rank for \
         its captured output, plus {DESCRIPTOR_RESERVE} for mirage itself), and this process \
         may not \
         have more than {hard}. That ceiling is the hard `nofile` limit, which only an \
         administrator can raise — check `ulimit -Hn`, or your container's or scheduler's \
         limits. Until then, run a smaller grid."
    )
}

/// How many nodes a resolved profile describes, if that can be answered
/// from the filesystem.
///
/// `None` when the topology is a by-name reference that does not resolve
/// — which is a real error, but one the store reports far better than a
/// grid check would, so it is left to bring-up rather than guessed at
/// here.
fn profile_node_count(profile: &ProfileDef) -> Option<u32> {
    match &profile.emulator.topology {
        MaybeRef::Owned(t) => Some(t.num_nodes),
        MaybeRef::Ref(name) => mirage_core::store::topology_get(name)
            .ok()
            .map(|t| t.num_nodes),
    }
}

/// The sections a document has to have before it is an emulator config
/// rather than merely some JSON.
///
/// `vm` is the machine to emulate and `topology` is how its devices are
/// wired together; a synthesised config always carries both, and the
/// emulator refuses to start without either. Everything else a config
/// can say has a default, so requiring more would refuse files that
/// work.
const CONFIG_SECTIONS: [&str; 2] = ["vm", "topology"];

/// Resolve `--config <path>` to an absolute path, having checked that it
/// is a config file the emulator can actually be given.
///
/// Read here rather than trusted, and read the same way whichever
/// emulation mode the run asked for. That is the point of doing it here:
/// the two modes used to disagree about the same file. `--daemon` starts
/// the emulator daemon during bring-up, which parses the config and
/// refuses — loudly, and with advice to "pass `--in-process` to run
/// without it". `--in-process` parses nothing: the interposer loads the
/// config lazily, in the workload, at its first GPU call, so a workload
/// that never makes one exits 0 having emulated nothing from a config
/// mirage had already been told was unusable. Following mirage's own
/// advice therefore turned a refusal into a silent success, and the
/// advice is only honest while a config bad enough to stop the daemon is
/// stopped before either mode is chosen.
///
/// The check is structural, and deliberately not a second
/// implementation of the emulator's own parser — two of those can only
/// ever agree or be a bug: the file exists, is readable, is a JSON
/// object, and has the sections that make it a description of a machine
/// (see [`CONFIG_SECTIONS`]). What is left to the backend is the
/// contents of those sections, and a config that gets that far and is
/// still refused is refused by the emulator saying so. What mirage
/// itself will not accept no longer depends on which mode was asked
/// for, which is the part that was making the advice dishonest.
///
/// # Errors
///
/// Returns an error naming the path — as the user spelled it, which is
/// the spelling they can compare against what they typed — and what is
/// wrong with it.
fn check_config_file(path: &str) -> anyhow::Result<std::path::PathBuf> {
    let abs = std::fs::canonicalize(path).map_err(|e| {
        anyhow::anyhow!("--config {path}: {e}. Name an emulator config file that exists.")
    })?;
    // `read` covers both halves of "can the emulator open this": a
    // directory fails with `EISDIR`, an unreadable file with `EACCES`.
    let bytes = std::fs::read(&abs).map_err(|e| anyhow::anyhow!("--config {path}: {e}"))?;
    let doc: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        anyhow::anyhow!(
            "--config {path}: this is not a usable emulator config ({e}). \
             It must be a JSON document — the same file the upstream \
             `rocjitsu --config` takes."
        )
    })?;
    let missing: Vec<&str> = CONFIG_SECTIONS
        .iter()
        .copied()
        .filter(|section| doc.get(section).is_none())
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "--config {path}: this is JSON, but not an emulator config — it has no {} \
             {}. A config describes the machine to emulate, and is the same file the \
             upstream `rocjitsu --config` takes; `mirage profile show <name>` is the \
             profile mirage would have built one from.",
            and_list(&missing),
            Plural(missing.len()).pick("section", "sections")
        );
    }
    Ok(abs)
}

/// Apply direct CLI overrides (containerisation + emulator settings) to
/// a profile fetched by name.
///
/// The overrides are read straight off the parsed [`RunArgs`] rather
/// than passed field by field: every one of them is a `run` flag, and
/// naming them positionally only created a long list of same-typed
/// parameters that a caller could silently transpose.
///
/// When no override is supplied the cheap by-name [`MaybeRef::Ref`] is
/// returned so the session keeps tracking the on-disk profile. As soon
/// as any field is overridden the whole (mutated) profile is inlined as
/// [`MaybeRef::Owned`].
fn apply_profile_overrides(
    profile: &mut ProfileDef,
    a: &RunArgs,
) -> anyhow::Result<MaybeRef<ProfileDef>> {
    if a.image.is_none()
        && a.mounts.is_empty()
        && a.ports.is_empty()
        && a.container_provider.is_none()
        && a.emulator.is_none()
        && a.exec_mode.is_none()
        && a.options.is_empty()
        && a.plugins.is_empty()
        && a.config.is_none()
        && a.num_nodes.is_none()
        && a.gpus_per_node.is_none()
        && a.hacks.is_empty()
    {
        // No overrides: keep the cheap by-name reference.
        return Ok(MaybeRef::Ref(a.profile.clone()));
    }

    // Container overrides.
    if a.image.is_some()
        || !a.mounts.is_empty()
        || !a.ports.is_empty()
        || a.container_provider.is_some()
        || !a.hacks.is_empty()
    {
        let parsed = parse_mounts(&a.mounts)?;
        let parsed_ports = parse_ports(&a.ports)?;
        let parsed_hacks: Vec<Hack> = a.hacks.iter().copied().map(Hack::from).collect();
        match &mut profile.containerize {
            Some(c) => {
                if let Some(img) = &a.image {
                    c.image = img.clone();
                }
                if let Some(p) = &a.container_provider {
                    c.provider = Some(p.clone());
                }
                c.mounts.extend(parsed);
                c.ports.extend(parsed_ports);
                for hack in parsed_hacks {
                    if !c.hacks.contains(&hack) {
                        c.hacks.push(hack);
                    }
                }
            }
            None => {
                let image = a.image.clone().ok_or_else(|| {
                    let given = container_flags_given(
                        &a.mounts,
                        &a.ports,
                        a.container_provider.as_ref(),
                        &a.hacks,
                    );
                    anyhow::anyhow!(
                        "{} {} a containerised profile or --image",
                        and_list(&given),
                        Plural(given.len()).pick("requires", "require")
                    )
                })?;
                profile.containerize = Some(ContainerizedDef {
                    provider: a.container_provider.clone(),
                    image,
                    mounts: parsed,
                    ports: parsed_ports,
                    devices: Vec::new(),
                    groups: Vec::new(),
                    hacks: parsed_hacks,
                });
            }
        }
    }

    // Emulator overrides.
    if let Some(name) = &a.emulator {
        if find_emulator(name).is_none() {
            anyhow::bail!(unknown_emulator(name));
        }
        profile.emulator.emulator = name.clone();
    }
    if let Some(mode) = a.exec_mode {
        profile.emulator.exec_mode = mode.into();
    }

    let options: Vec<(String, SimpleValue)> = a
        .options
        .iter()
        .map(|opt| parse_option(opt))
        .collect::<anyhow::Result<_>>()?;
    let plugins: Vec<(String, SimpleMap)> = a
        .plugins
        .iter()
        .map(|spec| parse_plugin(spec))
        .collect::<anyhow::Result<_>>()?;

    // Check both against the backend that will actually run them — which
    // is the one `--emulator` just selected, if it did. A backend this
    // build does not have compiled in is left to bring-up to report:
    // there is no schema here to check against, and refusing on that
    // basis would blame the option for a missing backend.
    if let Some(spec) = find_emulator(&profile.emulator.emulator) {
        let name = &spec.name;
        check_option_keys(
            name,
            &option_names(&spec),
            &options.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
        )?;
        check_plugin_names(
            name,
            &spec.plugins,
            &plugins.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        )?;
    }

    profile.emulator.options.extend(options);
    profile.emulator.plugins.extend(plugins);
    // Drop-in `--config <path>`: an explicit emulator config file
    // (the upstream `rocjitsu --config`). Stored as the `config`
    // emulator option (absolute, so it resolves regardless of the
    // workload's working directory) for the backend to use verbatim.
    if let Some(cfg) = &a.config {
        let abs = check_config_file(cfg)?;
        profile.emulator.options.insert(
            "config".to_string(),
            SimpleValue::String(abs.display().to_string()),
        );
    }

    // Topology overrides: per-run node and per-node GPU counts. Resolve
    // the emulator's topology (following a by-name reference) so the
    // counts can be mutated, then inline the modified topology.
    if a.num_nodes.is_some() || a.gpus_per_node.is_some() {
        let mut topo = match &profile.emulator.topology {
            MaybeRef::Owned(t) => t.clone(),
            // Through the store's front door, so the name is validated
            // before it is joined to the config directory. The raw
            // `topology::store::get` skips that check, and a name is only
            // a filename by convention: a profile whose topology
            // reference is `../../../../tmp/evil` resolved to a document
            // outside the config root, and a missing one reported an io
            // error naming the traversed path instead of
            // `TopologyNotFound`.
            MaybeRef::Ref(name) => mirage_core::store::topology_get(name)?,
        };
        if let Some(n) = a.num_nodes {
            topo.num_nodes = n;
        }
        if let Some(g) = a.gpus_per_node {
            topo.gpus_per_node = g;
        }
        profile.emulator.topology = MaybeRef::Owned(topo);
    }

    Ok(MaybeRef::Owned(profile.clone()))
}

/// Split a trailing `-- <command> <args…>` into its command and
/// arguments.
///
/// An empty `argv` yields an empty command rather than panicking; clap
/// marks the field `required`, so the case is unreachable from the CLI
/// and is not worth an error path.
fn split_argv(argv: &[String]) -> (String, Vec<String>) {
    let mut it = argv.iter().cloned();
    let cmd = it.next().unwrap_or_default();
    (cmd, it.collect())
}

/// The variables mirage sets on every workload process itself, after
/// the emulator's and the user's, and which therefore win over `--env`.
///
/// They are the job's identity — which session a process belongs to,
/// which rank it is, where the rendezvous is — and a workload that
/// exported a stale `RANK` or `WORLD_SIZE`, or a `--env` that disagreed
/// with the grid mirage actually built, would deadlock its own
/// collectives rather than merely misreport. Naming the constants rather
/// than the strings keeps this list in step with
/// [`mirage_supervisor`]'s, which is what actually applies them.
const MIRAGE_OWNED_ENV: [&str; 11] = [
    mirage_core::container::ENV_SESSION,
    mirage_core::container::ENV_RUNTIME,
    mirage_core::container::ENV_RANK,
    mirage_core::container::ENV_TORCH_RANK,
    mirage_core::container::ENV_HEAD_ADDR,
    mirage_core::container::ENV_HEAD_PORT,
    mirage_core::container::ENV_MASTER_ADDR,
    mirage_core::container::ENV_MASTER_PORT,
    mirage_core::container::ENV_WORLD_SIZE,
    mirage_core::container::ENV_LOCAL_RANK,
    mirage_core::container::ENV_NCCL_HOSTID,
];

/// Which of `env`'s keys mirage will overwrite, in the order given.
///
/// The precedence itself is deliberate (see [`MIRAGE_OWNED_ENV`]); what
/// was not deliberate is that the losing side was invisible, so
/// `--env RANK=3` looked accepted and did nothing.
fn mirage_owned_env<'a>(env: impl IntoIterator<Item = &'a String>) -> Vec<&'static str> {
    env.into_iter()
        .filter_map(|key| MIRAGE_OWNED_ENV.iter().find(|owned| *owned == key).copied())
        .collect()
}

/// Wrap a workload argv so it runs under ROCgdb for GPU kernel debugging
/// (`mirage run --gdb`). `["./add_one", "5"]` becomes
/// `rocgdb -ex "set breakpoint pending on" [-ex EX]... --args ./add_one 5`.
///
/// Kernel breakpoints are made pending because GPU code objects load lazily at
/// dispatch: a `break <kernel>` set before launch would otherwise be rejected.
/// Extra `-ex` commands (`--gdb-ex`) are injected in order, before `--args`, so
/// a run can be fully scripted. When any `-ex` command is present the session is
/// non-interactive, so `--batch` is added: ROCgdb runs the commands and exits
/// (a bare `--gdb`, with no `-ex`, stays interactive and reads the terminal).
fn gdb_wrap_argv(argv: &[String], gdb_ex: &[String]) -> Vec<String> {
    let mut wrapped = vec!["rocgdb".to_string()];
    if !gdb_ex.is_empty() {
        wrapped.push("--batch".to_string());
    }
    wrapped.push("-ex".to_string());
    wrapped.push("set breakpoint pending on".to_string());
    for ex in gdb_ex {
        wrapped.push("-ex".to_string());
        wrapped.push(ex.clone());
    }
    wrapped.push("--args".to_string());
    wrapped.extend_from_slice(argv);
    wrapped
}

/// Parse repeated `KEY=VALUE` pairs from the CLI into the env map
/// used by [`ExecArgs`]. Rejects entries without an `=` so a typo
/// surfaces immediately instead of silently being dropped.
fn parse_envs(entries: &[String]) -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    let mut out = std::collections::BTreeMap::new();
    for raw in entries {
        let Some((k, v)) = raw.split_once('=') else {
            anyhow::bail!("--env expects KEY=VALUE, got: {raw}");
        };
        if k.is_empty() {
            anyhow::bail!("--env key is empty in: {raw}");
        }
        out.insert(k.to_string(), v.to_string());
    }
    Ok(out)
}

// ----- cleanup ---------------------------------------------------------------

/// One container, network or image a cleanup pass found.
///
/// Kept whole rather than reduced to a name, because a user acts on the
/// facts together: what kind of thing it is, which engine holds it, and
/// which session it came from. Reported as "container resource <id>", a
/// network and a container were the same sentence and neither said which
/// engine to type the id at.
#[derive(Debug)]
struct Resource {
    /// The id the engine listed it under.
    id: String,
    /// `"container"`, `"network"` or `"image"`.
    kind: &'static str,
    /// The engine holding it (`podman`, `docker`, or a path).
    provider: String,
    /// The session it was created for, or `None` for a derived image.
    ///
    /// An image is keyed by its base plus the hacks applied to it, built
    /// once and reused by every later session that asks for the same
    /// combination, so it carries no session label — see
    /// [`mirage_core::container::image_labels`]. Naming the session that
    /// happened to build it would name an owner gone for days.
    session: Option<String>,
}

impl Resource {
    /// One line of the text report.
    ///
    /// `verb` is already tensed by the caller, because a dry run says
    /// "would remove" and a real one "removed", and that difference is
    /// the whole reason to offer a dry run.
    fn describe(&self, verb: &str) -> String {
        let Self {
            id,
            kind,
            provider,
            session,
        } = self;
        match session {
            Some(session) => format!("{verb} {kind} {id} of session {session} ({provider})"),
            None => format!("{verb} {kind} {id} ({provider})"),
        }
    }
}

/// What one cleanup pass found — and, unless it was a dry run, removed.
#[derive(Debug, Default)]
struct Reclaimed {
    /// Sessions that had something to reclaim.
    sessions: std::collections::BTreeSet<String>,
    /// Containers, networks and derived images, across every engine
    /// consulted.
    resources: Vec<Resource>,
    /// Stranded workload processes.
    processes: Vec<mirage_core::reclaim::Stranded>,
    /// Session scratch directories.
    scratch: Vec<std::path::PathBuf>,
    /// Things that should have been removed and were not, each already a
    /// sentence naming what and why.
    failures: Vec<String>,
}

impl Reclaimed {
    fn is_empty(&self) -> bool {
        self.resources.is_empty()
            && self.processes.is_empty()
            && self.scratch.is_empty()
            && self.failures.is_empty()
    }

    /// Everything this pass found, as one JSON value.
    ///
    /// Built rather than printed so a caller that has more to say — see
    /// [`purge`] — can nest it and still emit a single document.
    fn as_json(&self, dry_run: bool) -> serde_json::Value {
        serde_json::json!({
            "dry_run": dry_run,
            "sessions": self.sessions,
            "resources": self.resources
                .iter()
                .map(|r| serde_json::json!({
                    "id": r.id,
                    "kind": r.kind,
                    "provider": r.provider,
                    "session": r.session,
                }))
                .collect::<Vec<_>>(),
            "processes": self.processes
                .iter()
                .map(|p| serde_json::json!({"pid": p.pid, "session": p.session}))
                .collect::<Vec<_>>(),
            "scratch": self.scratch,
            "failures": self.failures,
        })
    }

    /// Print what happened, in whichever form the caller asked for.
    fn report(&self, dry_run: bool, json: bool) -> anyhow::Result<()> {
        if json {
            println!("{}", serde_json::to_string_pretty(&self.as_json(dry_run))?);
            return Ok(());
        }
        if self.is_empty() {
            println!("nothing to clean up");
            return Ok(());
        }
        // "would" rather than a past tense on a dry run: the difference
        // between the two modes is the whole reason to offer one.
        fn verb<'a>(dry_run: bool, did: &'a str, would: &'a str) -> &'a str {
            if dry_run { would } else { did }
        }
        let verb = |did, would| verb(dry_run, did, would);
        for p in &self.processes {
            println!(
                "{} stranded process {} (session {})",
                verb("killed", "would kill"),
                p.pid,
                p.session
            );
        }
        for r in &self.resources {
            println!("{}", r.describe(verb("removed", "would remove")));
        }
        for s in &self.scratch {
            println!(
                "{} scratch directory {}",
                verb("removed", "would remove"),
                s.display()
            );
        }
        // On stderr, and after the list: these are not results, and a
        // script reading the list must not have to filter them out of it.
        for failure in &self.failures {
            eprintln!("mirage: {failure}");
        }
        Ok(())
    }
}

/// Every container engine a session under this runtime directory could
/// have been created on.
///
/// `MIRAGE_CONTAINER_PROVIDER` names one deliberately, and is then the
/// only one consulted. Otherwise *every* engine installed here is, rather
/// than the first one autodetection finds: detection prefers podman, so
/// on a host with both engines a session created with
/// `--container-provider docker` left containers this command never even
/// asked docker about — and then reported success, which is the one
/// answer that stops a user from looking.
///
/// The candidate list mirrors [`mirage_core::container::detect_provider`],
/// which returns only the first match and so cannot be reused here.
fn cleanup_providers() -> Vec<String> {
    if let Ok(explicit) = std::env::var("MIRAGE_CONTAINER_PROVIDER")
        && !explicit.is_empty()
    {
        return vec![explicit];
    }
    ["podman", "docker"]
        .into_iter()
        .filter(|engine| on_path(engine))
        .map(String::from)
        .collect()
}

/// A derived `--hack` image one runtime directory's mirage built.
#[derive(Debug, PartialEq, Eq)]
struct DerivedImage {
    /// The id the engine listed it under, and what it is removed by. A
    /// tag can be one of several on one image; the id cannot.
    id: String,
    /// `repository:tag` where the engine reports one, for the report — a
    /// user recognises `mirage-hack-…:latest`, not twelve hex digits.
    name: String,
}

/// `<provider> <args…>`, its stdout trimmed.
///
/// `None` covers every way the question can fail to be answered: the
/// engine could not be spawned, or it exited non-zero. Both mean "no
/// answer", and each caller decides what to do about that — which for a
/// removal is always "leave it alone".
fn provider_output(provider: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(provider)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The `mirage.owner` and `mirage.runtime` labels an engine reports for
/// an image, each `None` when it is absent.
///
/// One `inspect` for both, through `.Config.Labels`, which is the one
/// place both engines expose an image's labels as a map: podman also
/// carries a top-level `.Labels` and docker does not, and a template
/// naming a field the engine has never heard of does not render an empty
/// value — it makes the whole call exit non-zero, which would read as
/// "this image is not ours" for every image on a docker host.
fn image_marks(provider: &str, id: &str) -> Option<(Option<String>, Option<String>)> {
    use mirage_core::container::{LABEL_OWNER, LABEL_RUNTIME};
    let template = format!(
        "{{{{index .Config.Labels {LABEL_OWNER:?}}}}}\t{{{{index .Config.Labels {LABEL_RUNTIME:?}}}}}"
    );
    let out = provider_output(provider, &["image", "inspect", "--format", &template, id])?;
    let (owner, runtime) = out.split_once('\t')?;
    // Go renders a missing key as `<no value>`, and an empty label as the
    // empty string; both mean the mark is not there.
    let present = |v: &str| {
        let v = v.trim();
        (!v.is_empty() && v != "<no value>").then(|| v.to_string())
    };
    Some((present(owner), present(runtime)))
}

/// Every derived `--hack` image this runtime directory built and can
/// still remove.
///
/// Images are the third kind of host state a run creates, and the only
/// one deliberately outliving the session that made it: a derived image
/// is built once, keyed by its base plus the hacks applied to it, and
/// reused by every later session asking for the same combination. That
/// is why teardown does not remove it — the next run probably wants it —
/// and it is also why nothing ever did, so a machine that had run
/// `--hack` accumulated a base-image-sized copy per combination with
/// nothing looking for them.
///
/// What makes them reclaimable now is that they are marked like
/// everything else mirage creates. Both marks are required *positively*:
/// `mirage.owner` says a mirage built this, and `mirage.runtime` says
/// which one. An image the engine listed that carries neither is
/// somebody else's — including a `mirage-hack-…` left by a mirage older
/// than the labels, which has to go by hand, exactly as an unattributable
/// container does.
///
/// An image a container still references is the last thing excluded, and
/// `reclaimed` is what makes that judgement right in both modes: the
/// containers this pass has removed, or on a dry run would remove, do not
/// count as users. Asking first — rather than running `rmi` and reading
/// the failure — is what keeps `--dry-run` a preview of the real pass
/// rather than a list of what it would attempt.
///
/// Blocks on the provider binary.
fn reclaimable_images(provider: &str, reclaimed: &[String]) -> Vec<DerivedImage> {
    use mirage_core::container::{LABEL_OWNER, LABEL_OWNER_VALUE};
    let ours = mirage_core::container::owning_runtime();
    let filter = format!("label={LABEL_OWNER}={LABEL_OWNER_VALUE}");
    let Some(listed) = provider_output(
        provider,
        &[
            "images",
            "--filter",
            &filter,
            "--format",
            "{{.ID}}\t{{.Repository}}:{{.Tag}}",
        ],
    ) else {
        return Vec::new();
    };
    // No answer to "what is still in use" is not an empty answer; see
    // [`images_in_use`]. Nothing is reclaimed until the engine says.
    let Some(in_use) = images_in_use(provider, reclaimed) else {
        return Vec::new();
    };

    // One image with several tags is listed once per tag. It has one id,
    // is removed once, and is in use if a container names *any* of them.
    let mut tags: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    let mut order = Vec::new();
    for line in listed.lines() {
        let (id, tag) = line.split_once('\t').unwrap_or((line, ""));
        let id = id.trim();
        if id.is_empty() {
            continue;
        }
        let entry = tags.entry(id.to_string()).or_insert_with(|| {
            order.push(id.to_string());
            Vec::new()
        });
        // An untagged image reads as `<none>:<none>`, which is not a name
        // a user can type and not one a container can reference.
        let tag = tag.trim();
        if !tag.is_empty() && !tag.contains("<none>") {
            entry.push(tag.to_string());
        }
    }

    let mut found = Vec::new();
    for id in order {
        let tags = tags.remove(&id).unwrap_or_default();
        let Some((Some(owner), Some(runtime))) = image_marks(provider, &id) else {
            continue;
        };
        if owner != LABEL_OWNER_VALUE || !mirage_core::container::same_runtime(&runtime, &ours) {
            continue;
        }
        // A container names its image either by a reference or by an id,
        // and an id may be the short one this listing gave or the whole
        // digest — so the id side is a prefix match and the tag side an
        // exact one.
        if in_use
            .iter()
            .any(|used| used.starts_with(&id) || tags.iter().any(|tag| tag == used))
        {
            continue;
        }
        let name = tags.first().cloned().unwrap_or_else(|| id.clone());
        found.push(DerivedImage { id, name });
    }
    found
}

/// What every container on this engine was created from, as the engine
/// names it, discounting the containers this pass is removing.
///
/// One listing read by reference, rather than a `--filter ancestor=`
/// query per image, because the two engines do not agree on what that
/// filter takes: docker matches an image id, podman only a name, so an
/// id-based query silently finds nothing on podman and every image looks
/// unused. What a container says it came from is the question both of
/// them answer the same way.
///
/// `None` is not an empty set. It means the engine could not be asked,
/// and an unanswerable question is not an answer: read as "nothing is in
/// use" it would take an image out from under a session whose run is
/// alive and well.
fn images_in_use(provider: &str, reclaimed: &[String]) -> Option<Vec<String>> {
    let listed = provider_output(
        provider,
        &["ps", "--all", "--format", "{{.ID}}\t{{.Image}}"],
    )?;
    Some(
        listed
            .lines()
            .filter_map(|line| {
                let (id, image) = line.split_once('\t')?;
                let image = image.trim().trim_start_matches("sha256:");
                if image.is_empty() || reclaimed.iter().any(|gone| gone == id.trim()) {
                    return None;
                }
                Some(image.to_string())
            })
            .collect(),
    )
}

/// Remove each derived image, best-effort and never forced.
///
/// Plain `rmi`: `-f` untags an image a container is still using, and the
/// only containers still using one at this point belong to sessions this
/// command exists not to disturb. [`reclaimable_images`] has already
/// excluded those, so forcing could only ever act against that decision.
///
/// Blocks on the provider binary.
fn remove_images(provider: &str, images: &[DerivedImage]) {
    for image in images {
        let _ = provider_output(provider, &["rmi", &image.id]);
    }
}

/// Whether a bare executable name resolves on `PATH`.
fn on_path(name: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join(name).is_file()))
}

/// Reclaim everything belonging to a session no live `mirage run` owns.
///
/// The recovery path for a run that died without tearing its session
/// down. It never touches a live session: `answering_runs` is the same
/// liveness test `ControlSocket::bind` uses — a socket that answers, not
/// a socket file, which outlives a `SIGKILL`ed run and would otherwise
/// make this refuse to clean up in exactly the situation it exists for.
///
/// The order is the order teardown uses, for the same reason: processes
/// stop before the containers they run in, images go after the
/// containers that were built from them, and the scratch directory —
/// which every node container bind-mounts — goes last.
async fn cleanup(dry_run: bool) -> Reclaimed {
    let live = run::answering_runs().await;

    // 1. Workload processes, found by the `MIRAGE_SESSION` each carries.
    //
    // Scanned once and then acted on, rather than scanned again to act:
    // what is reported and what is removed have to be the same set, or a
    // `--dry-run` is not a preview of anything.
    let processes = tokio::task::spawn_blocking({
        let live = live.clone();
        move || {
            let stranded = mirage_core::reclaim::stranded_workloads(&live);
            if !dry_run {
                mirage_core::reclaim::reap(&stranded);
            }
            stranded
        }
    })
    .await
    .unwrap_or_default();

    let mut out = Reclaimed {
        processes,
        ..Reclaimed::default()
    };
    for p in &out.processes {
        out.sessions.insert(p.session.as_str().to_string());
    }

    // 2. Containers and networks, found by the `mirage.owner` label, on
    //    every engine this host has (see [`cleanup_providers`]).
    for provider in cleanup_providers() {
        let found = tokio::task::spawn_blocking({
            let live = live.clone();
            let provider = provider.clone();
            move || {
                let orphans = mirage_core::container::orphans(&provider, &live);
                if !dry_run {
                    // Not `reclaim_orphans`, for the same reason as
                    // above: it re-lists, and the second listing could
                    // differ from the one being reported.
                    mirage_core::container::remove_orphans(&provider, &orphans);
                }
                orphans
            }
        })
        .await
        .unwrap_or_default();
        // The containers this pass removed — or, on a dry run, would
        // have. An image is only reclaimable once nothing references it,
        // and these are exactly the references that are going away.
        let gone: Vec<String> = found
            .iter()
            .filter(|orphan| !orphan.is_network)
            .map(|orphan| orphan.name.clone())
            .collect();
        for orphan in found {
            out.sessions.insert(orphan.session.clone());
            out.resources.push(Resource {
                id: orphan.name,
                kind: if orphan.is_network {
                    "network"
                } else {
                    "container"
                },
                provider: provider.clone(),
                session: Some(orphan.session),
            });
        }

        // 3. Derived `--hack` images, found by the same owner label and
        //    attributed by the same runtime mark. After the containers,
        //    because an engine will not remove an image one still
        //    references — see [`reclaimable_images`].
        let images = tokio::task::spawn_blocking({
            let provider = provider.clone();
            move || {
                let found = reclaimable_images(&provider, &gone);
                if !dry_run {
                    remove_images(&provider, &found);
                }
                found
            }
        })
        .await
        .unwrap_or_default();
        for image in images {
            out.resources.push(Resource {
                id: image.name,
                kind: "image",
                provider: provider.clone(),
                // No session: a derived image outlives the one that
                // built it, deliberately, and says so by carrying no
                // session label.
                session: None,
            });
        }
    }

    // 4. Scratch directories, one per session that was never torn down.
    //
    // Against a *fresh* answer to "what is live". Everything above shells
    // out to a container engine, which takes as long as an engine takes,
    // and a `mirage run` started in that window binds its socket and
    // creates its scratch directory while this pass is still working from
    // a list that predates it. Deleting that directory takes the emulator
    // socket and config out from under a healthy session. Re-probing is
    // one connect per live run, against seconds of listing.
    let live = run::answering_runs().await;
    let live_ids: std::collections::HashSet<&str> = live.iter().map(SessionId::as_str).collect();
    if let Ok(entries) = std::fs::read_dir(mirage_core::paths::session_runtime_root()) {
        for entry in entries.flatten() {
            let Some(id) = entry
                .file_name()
                .to_str()
                .and_then(|n| SessionId::new(n).ok())
            else {
                continue;
            };
            if live_ids.contains(id.as_str()) || !entry.path().is_dir() {
                continue;
            }
            if !dry_run && let Err(e) = std::fs::remove_dir_all(entry.path()) {
                // Recorded, not just logged: a caller that reports
                // success while the directory it named is still there is
                // telling the user their machine is clean when it is not.
                out.failures.push(format!(
                    "could not remove the scratch directory {}: {e}",
                    entry.path().display()
                ));
                continue;
            }
            out.sessions.insert(id.as_str().to_string());
            out.scratch.push(entry.path());
        }
        out.scratch.sort();
    }

    out
}

// ----- state dispatch --------------------------------------------------------

async fn state_cmd(cmd: StateCmd, json: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        StateCmd::Builtins => builtins_cmd(json),
        StateCmd::Purge { force, all } => {
            let prompt = if all {
                "purge ALL mirage state, including agents, topologies and profiles?"
            } else {
                "purge all mirage runtime/state and stop all sessions?"
            };
            // The prompt goes to stderr (see [`confirm`]), so stdout is
            // still free to carry exactly one JSON document — including
            // when the answer is no, which is a result a script has to be
            // able to read rather than infer from an empty stdout.
            if !force && !confirm(prompt)? {
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "purged": false,
                            "all": all,
                            "declined": true,
                        }))?
                    );
                } else {
                    println!("nothing was purged: you declined");
                }
                return Ok(ExitCode::from(DECLINED));
            }
            purge(all, json).await
        }
    }
}

/// `mirage state builtins`: refresh every builtin document on disk.
///
/// All three kinds, always. This used to be three `?`-chained calls, so
/// the first kind holding an edited builtin abandoned the other two — and
/// then said "Every other builtin was refreshed", which was false, and
/// under `--json` threw away the report of everything that *had* been
/// refreshed. Repairing an edited agent, topology and profile therefore
/// took three runs to even discover.
///
/// Exit 0 when nothing failed, edited builtins included. `--help`
/// presents "a builtin you have edited is left alone and named" as the
/// ordinary outcome, and it is: every document that could be refreshed
/// was, nothing was lost, and there is nothing for the user to fix unless
/// they want the shipped version back. A non-zero exit would make an
/// upgrade script fail on a machine where the upgrade fully succeeded. A
/// document that could not be *written* is a different matter and still
/// fails.
fn builtins_cmd(json: bool) -> anyhow::Result<ExitCode> {
    let kinds = [
        (DocKind::Agent, mirage_builtin::ensure_agents(true)?),
        (DocKind::Topology, mirage_builtin::ensure_topologies(true)?),
        (DocKind::Profile, mirage_builtin::ensure_profiles(true)?),
    ];

    if json {
        let entries: Vec<serde_json::Value> = kinds
            .iter()
            .flat_map(|(kind, ensured)| {
                ensured.documents.iter().map(move |(name, written)| {
                    serde_json::json!({
                        "kind": kind.as_str(),
                        "name": name,
                        "path": kind.path(name),
                        "written": written,
                        // Distinguishes "left alone because you changed
                        // it" from a write that simply did not happen.
                        "edited": ensured.edited.iter().any(|(n, _)| n == name),
                    })
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else {
        for (kind, ensured) in &kinds {
            for (name, written) in &ensured.documents {
                let tag = if *written { "wrote" } else { "kept " };
                println!(
                    "{tag} {:<9} {name} -> {}",
                    kind.as_str(),
                    kind.path(name).display()
                );
            }
        }
    }

    // Every kind's edited documents, in one list, after the report of
    // what did land. On stderr: this is a note about files mirage did not
    // touch, and a script reading the list of what it did must not have to
    // filter it out.
    let edited: Vec<(DocKind, &String, &std::path::PathBuf)> = kinds
        .iter()
        .flat_map(|(kind, ensured)| ensured.edited.iter().map(move |(n, p)| (*kind, n, p)))
        .collect();
    if !edited.is_empty() {
        let n = Plural(edited.len());
        eprintln!(
            "mirage: {} builtin document{} differ{} from the one{} mirage ships and \
             {} left alone:",
            edited.len(),
            n.pick("", "s"),
            n.pick("s", ""),
            n.pick("", "s"),
            n.pick("was", "were"),
        );
        for (kind, name, path) in &edited {
            eprintln!("  {:<9} {name} -> {}", kind.as_str(), path.display());
        }
        eprintln!(
            "mirage: rewriting one would discard your edits, and everything else was \
             refreshed. To take the shipped version of one after all, delete it \
             (`mirage <kind> delete <name>`) and run `mirage state builtins` again."
        );
    }
    Ok(ExitCode::from(0))
}

/// Stop every live run and remove mirage's on-disk state.
async fn purge(all: bool, json: bool) -> anyhow::Result<ExitCode> {
    // Ask every live run to stop, by signalling nothing: a run owns its
    // own session and tears it down when it exits, so there is no
    // "destroy session" call to make. What purge can do is remove the
    // leftovers of runs that are already gone, and reclaim the container
    // resources a run that died abruptly could not.
    //
    // A live run is left alone deliberately. Killing someone else's
    // foreground command from a state-cleanup subcommand would be a
    // surprise, and the user can stop it with Ctrl-C in its own terminal.
    // Only runs that *answer* count. `live_runs` lists socket files, and a
    // file outlives a `SIGKILL`ed run — so counting those would make purge
    // refuse to run in exactly the situation it exists for. Corpse sockets
    // are unlinked on the way past.
    let live = run::answering_runs().await;
    if !live.is_empty() {
        let n = Plural(live.len());
        anyhow::bail!(
            "{} `mirage run` process{} {} still running ({}). \
             Stop {} first: each one owns its session and cleans up \
             when it exits.",
            live.len(),
            n.pick("", "es"),
            n.pick("is", "are"),
            live.iter()
                .map(SessionId::as_str)
                .collect::<Vec<_>>()
                .join(", "),
            n.pick("it", "them"),
        );
    }

    // Reclaim whatever runs that are already gone left behind — the
    // containers, networks and workload processes a `SIGKILL`ed run could
    // not remove itself. This is `mirage cleanup`, run for its effect
    // rather than for its own sake: purge is the blunt "start again from
    // nothing" tool and cleanup is the surgical one, but the set of
    // things worth reclaiming is identical and must not drift.
    //
    // The live-run check above has already established that there are
    // none, so every session it finds is orphaned by construction.
    let reclaimed = cleanup(false).await;

    // Ask again, immediately before the destructive step.
    //
    // Everything between the first check and here shells out to a
    // container engine, one listing and one removal at a time, and a
    // `mirage run` started in that window is a healthy session whose
    // socket and scratch directory both live under the directory about to
    // be deleted. The check that mattered is the one taken last.
    let racing = run::answering_runs().await;
    if !racing.is_empty() {
        let n = Plural(racing.len());
        anyhow::bail!(
            "{} `mirage run` started while this purge was working ({}); \
             nothing further was removed. Stop {} and run the purge again.",
            n.pick("a", "several"),
            racing
                .iter()
                .map(SessionId::as_str)
                .collect::<Vec<_>>()
                .join(", "),
            n.pick("it", "them"),
        );
    }

    let mut targets = vec![mirage_core::paths::mirage_runtime_dir()];
    if all {
        targets.push(mirage_core::paths::mirage_config_dir());
    }
    let mut removed = Vec::new();
    let mut failures = Vec::new();
    for t in targets {
        if !t.exists() {
            continue;
        }
        match std::fs::remove_dir_all(&t) {
            Ok(()) => removed.push(t),
            // Reported and counted, not merely logged. `purged` printed
            // over a directory that is still there — the case a `chmod
            // 500` subdirectory produces — is the one outcome a user
            // cannot recover from, because they have been told there is
            // nothing left to do.
            Err(e) => failures.push(format!("could not remove {}: {e}", t.display())),
        }
    }

    // A purge that left a container behind has not started this machine
    // again from nothing either, whatever happened to the directories, so
    // the reclamation's own failures count towards the verdict.
    let unfinished = failures.len() + reclaimed.failures.len();
    let ok = unfinished == 0;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&purge_json(all, &removed, &failures, &reclaimed))?
        );
    } else {
        // Prints the reclamation's own failures, on stderr.
        if !reclaimed.is_empty() {
            reclaimed.report(false, json)?;
        }
        for t in &removed {
            println!("purged {}", t.display());
        }
        for failure in &failures {
            eprintln!("mirage: {failure}");
        }
        if !ok {
            eprintln!(
                "mirage: the purge is incomplete; {unfinished} thing{} could not be removed",
                Plural(unfinished).pick("", "s")
            );
        }
    }
    Ok(ExitCode::from(u8::from(!ok)))
}

/// Everything a purge did, as the single JSON document `--json` promises.
///
/// One document and nothing else: this used to be the reclamation's JSON
/// followed by a bare `purged` line, so `mirage state purge --json | jq`
/// failed on the trailing word — and, worse, `purged` was printed whether
/// or not anything had actually been removed. `purged` is now the
/// verdict, and every failure is named beside it.
fn purge_json(
    all: bool,
    removed: &[std::path::PathBuf],
    failures: &[String],
    reclaimed: &Reclaimed,
) -> serde_json::Value {
    let mut all_failures = reclaimed.failures.clone();
    all_failures.extend_from_slice(failures);
    serde_json::json!({
        "purged": all_failures.is_empty(),
        "all": all,
        "removed": removed,
        "reclaimed": reclaimed.as_json(false),
        "failures": all_failures,
    })
}

// ----- misc helpers ----------------------------------------------------------

/// Ask a yes/no question on stderr and read the answer from stdin.
///
/// Only ever asked of a terminal. A prompt is a question put to a person
/// who is watching, and mirage cannot tell whether anybody is on the far
/// end of a pipe until it has already committed to waiting for them: an
/// open stdin that never sends a line — a `make` recipe, a CI step, a job
/// in the background — leaves `mirage topology delete x` blocked forever,
/// with the prompt itself invisible in whatever collected the output. Not
/// a terminal therefore means not askable, and `--force` is the answer,
/// named in the error so it can be found from the log.
///
/// End-of-file on a terminal is the same refusal for the same reason:
/// Ctrl-D is not a "no". Taking it for one is how `mirage profile delete
/// x < /dev/null` used to delete nothing, exit 0, and tell a script that
/// forgot `--force` that its cleanup had run.
///
/// stderr rather than stdout: a prompt is not a result, and `--json`
/// promises stdout carries exactly one document.
///
/// # Errors
///
/// Returns an error if stdin is not a terminal, cannot be read, or ends
/// without an answer.
fn confirm(prompt: &str) -> anyhow::Result<bool> {
    confirm_answer(prompt, std::io::stdin().is_terminal(), || {
        use std::io::{BufRead, Write};
        eprint!("{prompt} [y/N] ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        let read = std::io::stdin().lock().read_line(&mut line)?;
        Ok((read > 0).then_some(line))
    })
}

/// The decision itself, separated from the terminal it is taken at.
///
/// `ask` prints the prompt and reads one line, returning `None` at
/// end-of-file. It is called only when there is somebody to read from,
/// which is the whole of this function: reaching for the prompt first
/// and discovering afterwards that nobody could answer it is what
/// blocked, and what printed a question into a log where nobody would
/// see it.
///
/// # Errors
///
/// Returns the reason there is no answer: stdin is not a terminal, it
/// could not be read, or it ended without a reply.
fn confirm_answer(
    prompt: &str,
    interactive: bool,
    ask: impl FnOnce() -> std::io::Result<Option<String>>,
) -> anyhow::Result<bool> {
    if !interactive {
        anyhow::bail!(
            "there is nobody to answer \"{prompt}\": stdin is not a terminal, so waiting \
             for a reply would wait forever. Pass --force to skip the question."
        );
    }
    let Some(line) = ask()? else {
        eprintln!();
        anyhow::bail!(
            "there is nobody to answer \"{prompt}\": stdin ended without a reply. \
             Pass --force to skip the question."
        );
    };
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

/// Every directory `mirage paths` reports, as one document.
fn paths_json() -> serde_json::Value {
    serde_json::json!({
        "config": mirage_core::paths::mirage_config_dir(),
        "runtime": mirage_core::paths::mirage_runtime_dir(),
        "profiles": mirage_core::paths::profile_root(),
        "topologies": mirage_core::paths::topology_root(),
        "agents": mirage_core::paths::agent_root(),
        "sessions": mirage_core::paths::session_runtime_root(),
        "runs": mirage_core::paths::run_socket_root(),
    })
}

/// Print where mirage keeps things on this machine.
///
/// All three document directories, not just profiles. They are siblings
/// under the config directory and a user can infer the other two — but
/// this command exists precisely so that nobody has to infer where mirage
/// is reading, and "I edited the agent and nothing changed" is answered
/// by seeing the directory it is actually read from.
fn print_paths(json: bool) {
    let info = paths_json();
    if json {
        // A `serde_json::Value` built from string paths cannot fail to
        // serialize; print something useful rather than panicking if the
        // impossible happens.
        match serde_json::to_string_pretty(&info) {
            Ok(text) => println!("{text}"),
            Err(e) => eprintln!("could not render paths as JSON: {e}"),
        }
    } else {
        for (label, dir) in [
            ("config", mirage_core::paths::mirage_config_dir()),
            ("runtime", mirage_core::paths::mirage_runtime_dir()),
            ("profiles", mirage_core::paths::profile_root()),
            ("topologies", mirage_core::paths::topology_root()),
            ("agents", mirage_core::paths::agent_root()),
            ("sessions", mirage_core::paths::session_runtime_root()),
            ("runs", mirage_core::paths::run_socket_root()),
        ] {
            println!("{:<11} {}", format!("{label}:"), dir.display());
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use mirage_core::common::SimpleMap;
    use mirage_core::emulator::{EmulatorDef, EmulatorKind};
    use mirage_core::topology::TopologyDef;

    fn sample_profile() -> ProfileDef {
        ProfileDef {
            name: "mi450x".to_string(),
            description: None,
            emulator: EmulatorDef {
                emulator: EmulatorKind::from("rocjitsu"),
                plugins: Default::default(),
                exec_mode: ExecMode::Functional,
                options: SimpleMap::default(),
                topology: MaybeRef::Owned(TopologyDef {
                    num_nodes: 1,
                    gpus_per_node: 1,
                    agent: MaybeRef::Ref("MI450X".to_string()),
                }),
            },
            containerize: None,
        }
    }

    #[test]
    fn parse_option_infers_types() {
        assert_eq!(
            parse_option("gpu_model=cdna3").unwrap(),
            ("gpu_model".to_string(), SimpleValue::String("cdna3".into()))
        );
        assert_eq!(
            parse_option("queues=4").unwrap(),
            ("queues".to_string(), SimpleValue::Number(4))
        );
        assert_eq!(
            parse_option("trace=true").unwrap(),
            ("trace".to_string(), SimpleValue::Boolean(true))
        );
    }

    #[test]
    fn parse_option_rejects_malformed() {
        assert!(parse_option("nope").is_err());
        assert!(parse_option("=value").is_err());
    }

    #[test]
    fn gdb_wrap_prepends_rocgdb_with_pending_breakpoints() {
        let got = gdb_wrap_argv(&["./add_one".to_string(), "5".to_string()], &[]);
        assert_eq!(
            got,
            vec![
                "rocgdb",
                "-ex",
                "set breakpoint pending on",
                "--args",
                "./add_one",
                "5",
            ]
        );
    }

    #[test]
    fn gdb_wrap_injects_extra_ex_commands_in_order() {
        let got = gdb_wrap_argv(
            &["./add_one".to_string()],
            &["break add_one".to_string(), "run".to_string()],
        );
        assert_eq!(
            got,
            vec![
                "rocgdb",
                "--batch",
                "-ex",
                "set breakpoint pending on",
                "-ex",
                "break add_one",
                "-ex",
                "run",
                "--args",
                "./add_one",
            ]
        );
    }

    #[test]
    fn no_overrides_keeps_by_name_ref() {
        let mut p = sample_profile();
        let a = RunArgs {
            profile: "mi450x".into(),
            ..Default::default()
        };
        let r = apply_profile_overrides(&mut p, &a).unwrap();
        assert_eq!(r, MaybeRef::Ref("mi450x".to_string()));
    }

    #[test]
    fn exec_mode_and_options_inline_owned_profile() {
        let mut p = sample_profile();
        let a = RunArgs {
            profile: "mi450x".into(),
            exec_mode: Some(ExecModeArg::Clocked),
            options: vec!["gpu_model=cdna4".to_string(), "queues=8".to_string()],
            ..Default::default()
        };
        let r = apply_profile_overrides(&mut p, &a).unwrap();
        match r {
            MaybeRef::Owned(owned) => {
                assert_eq!(owned.emulator.exec_mode, ExecMode::Clocked);
                assert_eq!(
                    owned.emulator.options.get("gpu_model"),
                    Some(&SimpleValue::String("cdna4".into()))
                );
                assert_eq!(
                    owned.emulator.options.get("queues"),
                    Some(&SimpleValue::Number(8))
                );
            }
            MaybeRef::Ref(_) => panic!("expected an inlined (owned) profile"),
        }
    }

    #[test]
    fn topology_counts_override_inline_owned_profile() {
        let mut p = sample_profile();
        let a = RunArgs {
            profile: "mi450x".into(),
            num_nodes: Some(2),
            gpus_per_node: Some(4),
            ..Default::default()
        };
        let r = apply_profile_overrides(&mut p, &a).unwrap();
        match r {
            MaybeRef::Owned(owned) => match owned.emulator.topology {
                MaybeRef::Owned(topo) => {
                    assert_eq!(topo.num_nodes, 2);
                    assert_eq!(topo.gpus_per_node, 4);
                }
                MaybeRef::Ref(_) => panic!("expected an inlined (owned) topology"),
            },
            MaybeRef::Ref(_) => panic!("expected an inlined (owned) profile"),
        }
    }

    #[test]
    fn parse_plugin_accepts_valid_names_and_rejects_bad() {
        assert_eq!(
            parse_plugin("race").unwrap(),
            ("race".to_string(), SimpleMap::new())
        );
        // Surrounding whitespace is trimmed.
        assert_eq!(
            parse_plugin("  logging ").unwrap(),
            ("logging".to_string(), SimpleMap::new())
        );
        assert!(parse_plugin("").is_err());
        assert!(parse_plugin("   ").is_err());
        // A path separator (or any other special char) is rejected, so a
        // plugin name can never escape the loader's plugin directory.
        assert!(parse_plugin("../evil").is_err());
        assert!(parse_plugin("a b").is_err());
    }

    #[test]
    fn plugins_enable_inline_owned_profile() {
        let mut p = sample_profile();
        let a = RunArgs {
            profile: "mi450x".into(),
            plugins: vec!["race".to_string(), "logging".to_string()],
            ..Default::default()
        };
        let r = apply_profile_overrides(&mut p, &a).unwrap();
        match r {
            MaybeRef::Owned(owned) => {
                assert!(owned.emulator.plugins.contains_key("race"));
                assert!(owned.emulator.plugins.contains_key("logging"));
                // Enabled with empty args; the loader applies schema defaults.
                assert_eq!(owned.emulator.plugins["race"], SimpleMap::new());
            }
            MaybeRef::Ref(_) => panic!("expected an inlined (owned) profile"),
        }
    }

    #[test]
    fn plugins_merge_with_profile_defined_plugins() {
        let mut p = sample_profile();
        p.emulator
            .plugins
            .insert("logging".to_string(), SimpleMap::new());
        let a = RunArgs {
            profile: "mi450x".into(),
            plugins: vec!["race".to_string()],
            ..Default::default()
        };
        let r = apply_profile_overrides(&mut p, &a).unwrap();
        match r {
            MaybeRef::Owned(owned) => {
                // The CLI plugin is added alongside the profile's existing one.
                assert!(owned.emulator.plugins.contains_key("race"));
                assert!(owned.emulator.plugins.contains_key("logging"));
            }
            MaybeRef::Ref(_) => panic!("expected an inlined (owned) profile"),
        }
    }

    #[test]
    fn run_args_parse_repeated_plugin_flags() {
        use clap::Parser;
        #[derive(Parser)]
        struct Wrap {
            #[command(flatten)]
            run: RunArgs,
        }
        let w = Wrap::try_parse_from([
            "mirage", "--plugin", "race", "--plugin", "logging", "--", "./app",
        ])
        .expect("`mirage run --plugin race --plugin logging -- ./app` should parse");
        assert_eq!(
            w.run.plugins,
            vec!["race".to_string(), "logging".to_string()]
        );
    }

    /// Parse `mirage run`'s arguments exactly as the real command does.
    fn parse_run(args: &[&str]) -> Result<RunArgs, clap::Error> {
        use clap::Parser;
        #[derive(Parser)]
        struct Wrap {
            #[command(flatten)]
            run: RunArgs,
        }
        let mut argv = vec!["mirage"];
        argv.extend_from_slice(args);
        Wrap::try_parse_from(argv).map(|w| w.run)
    }

    /// `mirage run --help`, as clap renders it.
    fn run_long_help() -> String {
        use clap::CommandFactory as _;
        #[derive(clap::Parser)]
        struct Wrap {
            #[command(flatten)]
            run: RunArgs,
        }
        Wrap::command().render_long_help().to_string()
    }

    #[test]
    fn run_declares_the_rocjitsu_spelling_of_daemon_rather_than_only_accepting_it() {
        // `--attach` reached `run` by a rewrite in the binary, before
        // clap saw the arguments, so `mirage run --help` did not list a
        // flag the drop-in help offers — and the page a user checks a
        // `run` flag against was the page that denied it.
        // As clap's own alias line, not as a word in the prose beside it:
        // a flag that is only *described* is still a flag `--help` does
        // not offer.
        let help = run_long_help();
        assert!(
            help.contains("[aliases: --attach]"),
            "`mirage run --help` must list the spelling it accepts:\n{help}"
        );

        // Declared, so it parses here too and means exactly `--daemon`.
        let a = parse_run(&["--attach", "--", "./app"]).unwrap();
        assert!(a.daemon, "--attach is --daemon");
        assert!(!parse_run(&["--", "./app"]).unwrap().daemon);
        // Including the conflict, which is the whole content of the flag:
        // one emulator cannot be both in-process and out of it.
        parse_run(&["--attach", "--in-process", "--", "./app"])
            .expect_err("`--attach --in-process` contradicts itself");
    }

    #[test]
    fn exec_help_says_how_nproc_per_node_relates_to_the_shape_of_the_job() {
        // The field carried no doc comment at all, so the flag appeared
        // in `mirage exec --help` with nothing beside it — and the rule
        // `build_specs` enforces (ranks are numbered in the session's own
        // grid, and an exec may not ask for more than the run has) was
        // discoverable only by being refused by it.
        use clap::CommandFactory as _;
        #[derive(clap::Parser)]
        struct Wrap {
            #[command(flatten)]
            exec: ExecArgsCli,
        }
        let help = Wrap::command().render_long_help().to_string();
        let (_, after) = help
            .split_once("--nproc-per-node")
            .expect("`mirage exec --help` must list --nproc-per-node");
        // Up to the next option's header, so what is asserted is this
        // flag's own paragraph. The header is the only `--` at the start
        // of an indented line; `--node` appears inside the prose too.
        let described = after
            .split_once("\n      --")
            .map_or(after, |(before, _)| before);
        assert!(
            described.contains("WORLD_SIZE"),
            "the flag must say which grid its ranks are numbered in:\n{described}"
        );
        assert!(
            described.contains("may not ask for more"),
            "the flag must say that the job's shape bounds it:\n{described}"
        );
    }

    #[test]
    fn a_log_filter_that_would_silence_everything_is_refused() {
        // `EnvFilter` reads a bare word as a target at trace level, so
        // this parses and then matches nothing mirage logs — the whole
        // failure being an absence of output.
        let why = parse_log_filter("not-a-level").unwrap_err();
        assert!(why.contains("not-a-level"), "{why}");
        assert!(why.contains("not a log level"), "{why}");

        // The forms that mean something all still work.
        for good in [
            "info",
            "off",
            "warn,mirage_supervisor=debug",
            "mirage_ctl=trace",
        ] {
            parse_log_filter(good).unwrap_or_else(|e| panic!("`{good}` should parse: {e}"));
        }
    }

    #[test]
    fn an_unknown_option_key_is_rejected_and_names_the_ones_that_work() {
        let schema = ["target_isa".to_string(), "source_isa".to_string()];
        let e = check_option_keys("rocjitsu-dbt", &schema, &["targt_isa".to_string()])
            .unwrap_err()
            .to_string();
        assert!(e.contains("targt_isa"), "the typo must be named: {e}");
        assert!(
            e.contains("target_isa") && e.contains("source_isa"),
            "the error must list what would have worked: {e}"
        );
        check_option_keys("rocjitsu-dbt", &schema, &["target_isa".to_string()]).unwrap();
    }

    #[test]
    fn an_option_for_a_backend_that_takes_none_says_that() {
        // rocjitsu publishes an empty schema, so every `-o` against it
        // was accepted, stored, and read by nobody.
        let e = check_option_keys("rocjitsu", &[], &["gpu_model".to_string()])
            .unwrap_err()
            .to_string();
        assert!(e.contains("gpu_model"), "{e}");
        assert!(e.contains("accepts no options"), "{e}");
        check_option_keys("rocjitsu", &[], &[]).unwrap();
    }

    #[test]
    fn a_plugin_this_host_does_not_have_is_rejected() {
        let available = ["logging".to_string(), "race".to_string()];
        let e = check_plugin_names("rocjitsu", &available, &["raec".to_string()])
            .unwrap_err()
            .to_string();
        assert!(e.contains("raec"), "{e}");
        assert!(
            e.contains("logging") && e.contains("race"),
            "the error must say what this host does have: {e}"
        );
        check_plugin_names("rocjitsu", &available, &["race".to_string()]).unwrap();

        let none = check_plugin_names("hotswap", &[], &["race".to_string()])
            .unwrap_err()
            .to_string();
        assert!(none.contains("none of its plugins"), "{none}");
    }

    #[test]
    fn a_missing_workdir_names_the_path_and_not_the_command() {
        // Reported as `command not found: /bin/true` before, with the
        // path that was actually missing never printed at all.
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope");
        let e = check_host_workdir(&missing.to_string_lossy())
            .unwrap_err()
            .to_string();
        assert!(e.contains("--workdir"), "{e}");
        assert!(e.contains(&missing.display().to_string()), "{e}");
    }

    #[test]
    fn a_workdir_that_is_a_file_says_which_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("cfg.json");
        std::fs::write(&file, "{}").unwrap();
        let e = check_host_workdir(&file.to_string_lossy())
            .unwrap_err()
            .to_string();
        assert!(e.contains(&file.display().to_string()), "{e}");
        assert!(e.contains("not a directory"), "{e}");

        check_host_workdir(&dir.path().to_string_lossy()).unwrap();
    }

    #[test]
    fn a_grid_the_supervisor_would_refuse_is_refused_before_bring_up() {
        let max = mirage_supervisor::spec::MAX_WORLD_SIZE;
        check_grid(2, 4).unwrap();
        check_grid(max, 1).unwrap();
        let e = check_grid(max, 2).unwrap_err().to_string();
        assert!(e.contains(&max.to_string()), "{e}");
    }

    #[test]
    fn the_variables_mirage_owns_are_the_ones_reported_as_ignored() {
        let keys = [
            "RANK".to_string(),
            "PYTHONPATH".to_string(),
            "WORLD_SIZE".to_string(),
        ];
        assert_eq!(mirage_owned_env(keys.iter()), vec!["RANK", "WORLD_SIZE"]);
        assert!(mirage_owned_env(["HSA_XNACK".to_string()].iter()).is_empty());
    }

    #[test]
    fn a_config_that_is_not_a_config_is_refused_before_the_session() {
        // An unusable config file made the emulator daemon fail to start,
        // which was downgraded to running in-process — silently, and with
        // an exit code of zero.
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("cfg.json");
        std::fs::write(&bad, "not json at all").unwrap();
        let e = check_config_file(&bad.to_string_lossy())
            .unwrap_err()
            .to_string();
        assert!(e.contains(&bad.display().to_string()), "{e}");

        let missing = dir.path().join("absent.json");
        let e = check_config_file(&missing.to_string_lossy())
            .unwrap_err()
            .to_string();
        assert!(e.contains(&missing.display().to_string()), "{e}");

        let good = dir.path().join("good.json");
        std::fs::write(&good, r#"{"vm": {}, "topology": {}}"#).unwrap();
        assert!(
            check_config_file(&good.to_string_lossy())
                .unwrap()
                .is_absolute()
        );
    }

    /// A config that parses and describes nothing is refused here, which
    /// is the only place both emulation modes pass through.
    ///
    /// `--daemon` used to be the only thing that looked: the daemon
    /// parsed the file during bring-up and refused, advising
    /// `--in-process` — which parses nothing at all, because the
    /// interposer loads the config in the workload at its first GPU
    /// call. Taking mirage's advice therefore turned a config mirage had
    /// rejected into an exit code of zero, having emulated nothing.
    #[test]
    fn a_config_that_describes_no_machine_is_refused_whichever_mode_runs_it() {
        let dir = tempfile::tempdir().unwrap();
        let json_but_not_a_config = dir.path().join("nope.json");
        std::fs::write(&json_but_not_a_config, r#"{"nope": 1}"#).unwrap();
        let e = check_config_file(&json_but_not_a_config.to_string_lossy())
            .unwrap_err()
            .to_string();
        // Both missing sections, named, and the path the user typed.
        for expected in ["vm", "topology", "sections"] {
            assert!(e.contains(expected), "{expected} missing from: {e}");
        }
        assert!(
            e.contains(&json_but_not_a_config.display().to_string()),
            "{e}"
        );

        // One section present is still not a machine, and the message
        // names only the half that is missing.
        let half = dir.path().join("half.json");
        std::fs::write(&half, r#"{"vm": {}}"#).unwrap();
        let e = check_config_file(&half.to_string_lossy())
            .unwrap_err()
            .to_string();
        assert!(e.contains("no topology section"), "{e}");
        assert!(!e.contains("vm and topology"), "{e}");

        // JSON that is not even an object cannot carry a section.
        let array = dir.path().join("array.json");
        std::fs::write(&array, "[]").unwrap();
        assert!(check_config_file(&array.to_string_lossy()).is_err());

        // The path is quoted one way throughout, not three.
        let missing = dir.path().join("absent.json");
        let e = check_config_file(&missing.to_string_lossy())
            .unwrap_err()
            .to_string();
        assert!(
            !e.contains(&format!("{:?}", missing.display().to_string())),
            "one spelling of the path, unquoted, as `--workdir` uses: {e}"
        );
    }

    #[test]
    fn zero_of_anything_is_not_a_job() {
        // Accepted and silently treated as one, which is a different job
        // from the one that was asked for.
        for flag in ["--num-nodes", "--gpus-per-node", "--nproc-per-node"] {
            let e = parse_run(&[flag, "0", "--", "./app"])
                .expect_err(&format!("`{flag} 0` should be refused"))
                .to_string();
            assert!(e.contains(flag), "{e}");
            parse_run(&[flag, "1", "--", "./app"]).unwrap();
        }
    }

    #[test]
    fn config_refuses_the_flags_it_would_have_to_ignore() {
        // `--config` hands the backend a file verbatim, so anything that
        // would have gone into a synthesised config cannot be honoured.
        for extra in [
            vec!["--gpus-per-node", "2"],
            vec!["--exec-mode", "clocked"],
            vec!["-o", "queues=4"],
            vec!["--plugin", "race"],
        ] {
            let mut argv = vec!["--config", "cfg.json"];
            argv.extend_from_slice(&extra);
            argv.extend_from_slice(&["--", "./app"]);
            let e = parse_run(&argv)
                .expect_err(&format!("`--config` with {extra:?} should be refused"))
                .to_string();
            assert!(e.contains(extra[0]), "{e}");
        }
        // `--num-nodes` is not in that set: how many nodes the emulated
        // machine has is mirage's business, not the emulator config's.
        parse_run(&["--config", "cfg.json", "--num-nodes", "2", "--", "./app"]).unwrap();
    }

    #[test]
    fn a_cleanup_report_says_what_each_thing_was() {
        // "container resource <id>" for a network, with no engine named,
        // left a user nothing to type.
        let reclaimed = Reclaimed {
            resources: vec![
                Resource {
                    id: "9f2c1a".to_string(),
                    kind: "container",
                    provider: "docker".to_string(),
                    session: Some("s-1".to_string()),
                },
                Resource {
                    id: "mirage-s-1".to_string(),
                    kind: "network",
                    provider: "docker".to_string(),
                    session: Some("s-1".to_string()),
                },
                Resource {
                    id: "mirage-hack-90d1e055da77723c:latest".to_string(),
                    kind: "image",
                    provider: "docker".to_string(),
                    session: None,
                },
            ],
            failures: vec!["could not remove /run/mirage/s-2: denied".to_string()],
            ..Reclaimed::default()
        };
        let doc = reclaimed.as_json(false);
        let kinds: Vec<&str> = doc["resources"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["kind"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, vec!["container", "network", "image"]);
        assert_eq!(doc["resources"][0]["provider"], "docker");
        assert_eq!(doc["resources"][1]["session"], "s-1");
        assert_eq!(doc["failures"].as_array().unwrap().len(), 1);

        // An image is a different kind of loss from a container, and the
        // text has to say so — a user reading "removed mirage-hack-…"
        // needs to know a *rebuild* is what it costs them, not a restart.
        assert_eq!(
            reclaimed.resources[0].describe("removed"),
            "removed container 9f2c1a of session s-1 (docker)"
        );
        // And a derived image has no session to name: it outlives the one
        // that built it, which is why it carries no session label.
        assert_eq!(
            reclaimed.resources[2].describe("would remove"),
            "would remove image mirage-hack-90d1e055da77723c:latest (docker)"
        );
        assert_eq!(doc["resources"][2]["session"], serde_json::Value::Null);
    }

    /// A stand-in container engine, answering the three questions
    /// [`reclaimable_images`] asks with canned output.
    ///
    /// A script rather than a mock, because what is under test is a
    /// conversation with another program: the flags, the Go templates and
    /// the shape of what comes back are the part that was wrong before
    /// and the part a Rust double would assume away.
    ///
    /// The listing deliberately includes an image the real engine's
    /// `--filter label=…` would never have returned. That is the negative
    /// case: attribution is re-checked here, positively, so a filter that
    /// is broader than expected — or an engine that ignores the filter —
    /// still cannot cost somebody their image.
    fn fake_engine(dir: &std::path::Path, ours: &str, referenced_by: &str) -> String {
        let path = dir.join("fake-engine");
        let script = format!(
            r#"#!/bin/sh
case "$1" in
  images)
    printf 'aaa111\tmirage-hack-ours:latest\naaa111\tmirage-hack-ours:also\nbbb222\tmirage-hack-theirs:latest\nccc333\tsomebody-elses:latest\n'
    ;;
  image)
    # $5 is the id; the template asks for owner and runtime, tab-joined.
    case "$5" in
      aaa111) printf 'mirage\t{ours}\n' ;;
      bbb222) printf 'mirage\t/some/other/runtime\n' ;;
      *)      printf '<no value>\t<no value>\n' ;;
    esac
    ;;
  ps)
    # One `<container id>\t<the image it came from>` line per container.
    printf '{referenced_by}'
    ;;
esac
exit 0
"#
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(
            &path,
            <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700),
        )
        .unwrap();
        path.display().to_string()
    }

    #[test]
    fn cleanup_reclaims_only_the_derived_images_this_runtime_built() {
        // `--hack` builds a derived image and nothing ever removed one:
        // teardown keeps it deliberately (the next run wants it) and
        // cleanup did not look for images at all, so a machine that had
        // run `--hack` grew a base-image-sized copy per combination.
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());
        std::fs::create_dir_all(mirage_core::paths::mirage_runtime_dir()).unwrap();
        let ours = mirage_core::container::owning_runtime();
        let engine = fake_engine(root.path(), &ours, "");

        let found = reclaimable_images(&engine, &[]);
        mirage_core::paths::clear_test_root();

        assert_eq!(
            found,
            vec![DerivedImage {
                id: "aaa111".to_string(),
                name: "mirage-hack-ours:latest".to_string(),
            }],
            "only the image marked as this runtime directory's, and once \
             despite its two tags"
        );
    }

    #[test]
    fn an_image_a_live_session_is_still_running_is_left_alone() {
        // The one case that makes removing an image unsafe. A derived
        // image is shared, so a container of a session cleanup is
        // deliberately not touching can be running from it — and the
        // preview has to agree with that, or `--dry-run` names something
        // a real pass would leave behind.
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());
        std::fs::create_dir_all(mirage_core::paths::mirage_runtime_dir()).unwrap();
        let ours = mirage_core::container::owning_runtime();
        let engine = fake_engine(
            root.path(),
            &ours,
            "live-container\\tmirage-hack-ours:also\\n",
        );

        let held = reclaimable_images(&engine, &[]);
        // ...unless the container holding it is one this very pass is
        // removing, which is the ordinary case: the orphan containers go
        // first and the image they were built from goes after them.
        let released = reclaimable_images(&engine, &["live-container".to_string()]);
        mirage_core::paths::clear_test_root();

        assert!(
            held.is_empty(),
            "an image a container still references must be left: {held:?}"
        );
        assert_eq!(released.len(), 1, "{released:?}");
    }

    #[test]
    fn a_purge_that_left_something_behind_does_not_claim_to_have_purged() {
        let reclaimed = Reclaimed::default();
        let ok = purge_json(
            false,
            &[std::path::PathBuf::from("/run/mirage")],
            &[],
            &reclaimed,
        );
        assert_eq!(ok["purged"], serde_json::json!(true));

        let failed = purge_json(
            false,
            &[],
            &["could not remove /run/mirage: Permission denied".to_string()],
            &reclaimed,
        );
        assert_eq!(failed["purged"], serde_json::json!(false));
        assert_eq!(failed["failures"].as_array().unwrap().len(), 1);

        // A container that could not be removed counts too: the machine
        // has not been started again from nothing either way.
        let stuck = Reclaimed {
            failures: vec!["could not remove the scratch directory /x: denied".to_string()],
            ..Reclaimed::default()
        };
        let partial = purge_json(false, &[], &[], &stuck);
        assert_eq!(partial["purged"], serde_json::json!(false));
    }

    #[test]
    fn purge_fails_when_the_runtime_directory_survives() {
        use std::os::unix::fs::PermissionsExt as _;

        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());
        let runtime = mirage_core::paths::mirage_runtime_dir();
        // A directory whose contents cannot be unlinked: `chmod 500` on
        // the parent of a file is enough, and is what a stuck mount or a
        // root-owned leftover looks like.
        let stuck = runtime.join("stuck");
        std::fs::create_dir_all(&stuck).unwrap();
        std::fs::write(stuck.join("held"), "x").unwrap();
        std::fs::set_permissions(&stuck, std::fs::Permissions::from_mode(0o500)).unwrap();

        // Blocking rather than `#[tokio::test]`: the lock guard above must
        // not be held across an await.
        let code = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(purge(false, false));

        std::fs::set_permissions(&stuck, std::fs::Permissions::from_mode(0o700)).unwrap();
        let survived = runtime.exists();
        mirage_core::paths::clear_test_root();

        assert!(survived, "the test needs a directory purge cannot remove");
        // `ExitCode` has no `PartialEq`; its `Debug` is the only thing to
        // compare, and comparing it against a known value rather than a
        // literal string keeps the test independent of how std renders it.
        assert_eq!(
            format!("{:?}", code.unwrap()),
            format!("{:?}", ExitCode::from(1)),
            "a purge that could not remove the runtime directory must not exit 0"
        );
    }

    #[test]
    fn paths_names_every_directory_a_document_lives_in() {
        // Profiles were the only document directory reported, so the
        // answer to "where does mirage read the agent I edited?" was left
        // to be inferred from a sibling — by the command that exists so
        // nothing has to be inferred.
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());

        let doc = paths_json();
        for (key, dir) in [
            ("profiles", mirage_core::paths::profile_root()),
            ("topologies", mirage_core::paths::topology_root()),
            ("agents", mirage_core::paths::agent_root()),
        ] {
            assert_eq!(
                doc[key].as_str(),
                Some(dir.display().to_string().as_str()),
                "`mirage paths` must report {key}"
            );
        }

        mirage_core::paths::clear_test_root();
    }

    #[test]
    fn the_long_profile_list_carries_its_long_fields_into_json() {
        // `--long --json` was accepted and then emitted the same bare
        // array of names as `--json` alone, so the fields the flag exists
        // to show were dropped with nothing to say they had been.
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());

        // Written around the store: this crate links no emulator backend,
        // so `profile_put`'s validation would refuse a profile naming one.
        let mut p = sample_profile();
        p.name = "described".to_string();
        p.description = Some("six months of tuning".to_string());
        mirage_core::state::write_json(&mirage_core::paths::profile_path(&p.name), &p).unwrap();

        let listed = long_profiles(&["described".to_string()]);
        assert_eq!(listed[0]["name"], "described");
        assert_eq!(listed[0]["emulator"], "rocjitsu");
        assert_eq!(listed[0]["description"], "six months of tuning");

        mirage_core::paths::clear_test_root();
    }

    #[test]
    fn an_import_failure_names_the_file_it_could_not_use() {
        // A missing file used to be reported as a bare `No such file or
        // directory (os error 2)`, with the filename the user typed
        // nowhere in it, and a malformed one as a line and column of a
        // document it did not identify.
        let dir = tempfile::tempdir().unwrap();

        let missing = dir.path().join("nope.json");
        let e = format!("{:#}", read_input(&missing.to_string_lossy()).unwrap_err());
        assert!(e.contains(&missing.display().to_string()), "{e}");
        assert!(e.starts_with("io error on "), "{e}");

        let broken = dir.path().join("broken.json");
        std::fs::write(&broken, "not json at all").unwrap();
        let (bytes, from) = read_input(&broken.to_string_lossy()).unwrap();
        let parsed = serde_json::from_slice::<ProfileDef>(&bytes)
            .map_err(|source| json_error(from, source))
            .unwrap_err();
        let e = format!("{parsed:#}");
        assert!(e.starts_with("json error on "), "{e}");
        assert!(e.contains(&broken.display().to_string()), "{e}");
        // And nothing of serde's internal vocabulary: `ProfileDef` is not
        // a thing the user has.
        assert!(!e.contains("ProfileDef"), "{e}");
    }

    #[test]
    fn a_failure_under_json_is_a_json_object() {
        // Errors were English prose on every command, `--json` or not, so
        // the one output a script cannot parse was the one it most needed
        // to. The exit code is the observable half here; the object itself
        // goes to stderr, which this process cannot capture from inside.
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());

        // Blocking rather than `#[tokio::test]`: the lock guard above must
        // not be held across an await.
        let code = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(dispatch(
                CtlCmd::Profile(ProfileCmd::Show {
                    name: "ghost".to_string(),
                }),
                true,
            ));

        mirage_core::paths::clear_test_root();
        assert_eq!(
            format!("{:?}", code.unwrap()),
            format!("{:?}", ExitCode::from(1)),
            "a command that failed must still exit 1 under --json"
        );
    }

    #[test]
    fn state_builtins_refreshes_every_kind_and_survives_an_edited_one() {
        // The three kinds were `?`-chained, so the first one holding an
        // edited builtin abandoned the other two — and then claimed
        // "Every other builtin was refreshed", which was false. Repairing
        // an edited agent, topology and profile took three runs to even
        // discover.
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());

        ensure_builtins_present();
        // Edit one of each kind, in the order the command visits them, so
        // that stopping at the first would hide the other two.
        let mut agent = mirage_core::store::agent_get("mi300x").unwrap();
        agent.vm.gpu.num_xcds = 1;
        mirage_core::state::write_json(&mirage_core::paths::agent_path("mi300x"), &agent).unwrap();
        let mut topology = mirage_core::store::topology_get("MI350X-2x8").unwrap();
        topology.num_nodes = 3;
        mirage_core::state::write_json(&mirage_core::paths::topology_path("MI350X-2x8"), &topology)
            .unwrap();
        let mut profile = mirage_core::store::profile_get("mi450x").unwrap();
        profile.description = Some("mine".to_string());
        mirage_core::state::write_json(&mirage_core::paths::profile_path("mi450x"), &profile)
            .unwrap();

        let code = builtins_cmd(false).unwrap();

        // All three survived, which is only possible if all three kinds
        // ran, and the run is not a failure: nothing was lost and every
        // other document was refreshed.
        assert_eq!(
            mirage_core::store::agent_get("mi300x")
                .unwrap()
                .vm
                .gpu
                .num_xcds,
            1
        );
        assert_eq!(
            mirage_core::store::topology_get("MI350X-2x8")
                .unwrap()
                .num_nodes,
            3
        );
        assert_eq!(
            mirage_core::store::profile_get("mi450x")
                .unwrap()
                .description,
            Some("mine".to_string())
        );
        mirage_core::paths::clear_test_root();
        assert_eq!(
            format!("{code:?}"),
            format!("{:?}", ExitCode::from(0)),
            "leaving an edited builtin alone is the outcome `--help` calls normal"
        );
    }
    /// Serialises the tests that change this process's resource limits.
    static CHANGING_LIMITS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A one-process job is never refused for want of descriptors.
    ///
    /// It inherits mirage's own streams rather than being captured, so it
    /// opens nothing — and it is the overwhelmingly common shape, so a
    /// machine with a miserly `nofile` must not be told it cannot run
    /// `mirage run -- ./app`.
    #[test]
    fn a_single_process_job_needs_no_headroom() {
        ensure_descriptors_for(0).unwrap();
        ensure_descriptors_for(1).unwrap();
    }

    /// A grid that fits under the current limit is left alone.
    #[test]
    fn a_grid_that_already_fits_changes_nothing() {
        use nix::sys::resource::{Resource, getrlimit};
        let (soft, _) = getrlimit(Resource::RLIMIT_NOFILE).unwrap();
        // Whatever this machine allows, a grid needing well under it must
        // pass without touching the limit.
        let world = (soft.saturating_sub(DESCRIPTOR_RESERVE) / DESCRIPTORS_PER_RANK) / 4;
        assert!(
            world > 1,
            "this machine's `nofile` is too small to test with"
        );
        ensure_descriptors_for(world).unwrap();
        assert_eq!(
            getrlimit(Resource::RLIMIT_NOFILE).unwrap().0,
            soft,
            "a grid that already fits must not move the limit"
        );
    }

    /// A grid the soft limit cannot hold raises the soft limit, rather
    /// than refusing a job the kernel would in fact allow.
    ///
    /// This is the half that keeps the check from being a new way to fail:
    /// a process may raise its own soft limit as far as the hard one
    /// without privilege, so "not enough right now" is a thing to fix and
    /// not a thing to report.
    #[test]
    fn a_soft_limit_in_the_way_is_raised_rather_than_reported() {
        use nix::sys::resource::{Resource, getrlimit, setrlimit};

        // `RLIMIT_NOFILE` belongs to the process, not to a test, so two
        // of these at once would each be measuring the other's edit —
        // and every other test in this binary opens files under whatever
        // they leave behind.
        let _serialised = CHANGING_LIMITS.lock().unwrap_or_else(|e| e.into_inner());

        let (soft, hard) = getrlimit(Resource::RLIMIT_NOFILE).unwrap();

        // The soft limit has to be somewhere below the hard one for
        // there to be a raise to observe, and on a machine where they
        // are equal — this one, at 1048576 — the only way to get there
        // is to lower it first. `PINCHED` is deliberately roomy rather
        // than minimal: it is in force for the few syscalls below, and
        // anything else in this binary that opens a file meanwhile has
        // to keep working. The mutex above keeps it from overlapping
        // another test that changes it.
        const PINCHED: u64 = 512;
        if hard < PINCHED * 4 {
            return;
        }
        setrlimit(Resource::RLIMIT_NOFILE, PINCHED, hard).unwrap();

        // A grid that does not fit under `PINCHED` but fits easily under
        // the hard limit — which is the whole case: the kernel would
        // allow this job, and only this process's own soft limit is in
        // the way.
        let world = (PINCHED - DESCRIPTOR_RESERVE) / DESCRIPTORS_PER_RANK + 16;
        let need = world * DESCRIPTORS_PER_RANK + DESCRIPTOR_RESERVE;
        assert!(need > PINCHED, "the grid has to be one that does not fit");

        let result = ensure_descriptors_for(world);
        let (after, _) = getrlimit(Resource::RLIMIT_NOFILE).unwrap();
        // Put it back before asserting, so a failure here does not leave
        // the rest of the suite running under a pinched limit.
        setrlimit(Resource::RLIMIT_NOFILE, soft, hard).unwrap();

        result.expect("a grid the hard limit allows must be made to fit");
        assert!(
            after >= need,
            "the soft limit was not raised to fit {world} ranks: \
             {PINCHED} -> {after}, needed {need}"
        );
    }

    /// A grid the hard limit cannot hold is refused, saying whose limit
    /// it is.
    #[test]
    fn a_grid_past_the_hard_limit_is_refused_before_anything_is_created() {
        use nix::sys::resource::{Resource, getrlimit};

        let (_, hard) = getrlimit(Resource::RLIMIT_NOFILE).unwrap();
        // Nothing to refuse where the ceiling is effectively absent, and
        // `RLIM_INFINITY` is a real configuration.
        if hard > u64::MAX / 4 {
            return;
        }
        // A grid nothing could hold. `MAX_WORLD_SIZE` bounds what the CLI
        // accepts, so this asks directly.
        let world = hard + 1;
        let err = ensure_descriptors_for(world).expect_err("nothing can open that many files");
        let msg = err.to_string();
        assert!(msg.contains(&world.to_string()), "{msg}");
        assert!(
            msg.contains("hard") && msg.contains("ulimit -Hn"),
            "the refusal must say whose limit this is and where to look: {msg}"
        );
    }

    /// The two `--help` pages must not carry notes to whoever maintains
    /// the flags.
    ///
    /// A doc comment on a clap field is printed to the user. Two of them
    /// explained why a flag is declared the way it is — clap aliases, a
    /// rewrite in the binary crate — which is a maintainer's question
    /// and reads, on the page a user checks a flag against, as noise
    /// between them and the answer.
    #[test]
    fn help_does_not_explain_the_implementation_to_the_user() {
        use clap::CommandFactory as _;
        #[derive(clap::Parser)]
        struct ExecWrap {
            #[command(flatten)]
            exec: ExecArgsCli,
        }
        let exec = ExecWrap::command().render_long_help().to_string();
        assert!(
            !exec.contains("A flag rather than a positional"),
            "why `--session` is a flag is not a user's question:\n{exec}"
        );
        // What the flag is for is still there.
        assert!(exec.contains("Session to run in"), "{exec}");

        let run = run_long_help();
        assert!(
            !run.contains("Declared here rather than only rewritten"),
            "why `--attach` is declared here is not a user's question:\n{run}"
        );
        assert!(run.contains("[aliases: --attach]"), "{run}");
    }

    /// A prompt is put only to a terminal, and to nothing else.
    ///
    /// End-of-file was already refused, so `mirage topology delete x <
    /// /dev/null` said so. An *open* stdin that never sends a line — a
    /// `make` recipe, a CI step, a background job — reached the read and
    /// stayed there, with the prompt itself buried in whatever collected
    /// the output.
    #[test]
    fn a_prompt_is_never_put_to_something_that_cannot_answer() {
        let e = confirm_answer("delete topology t1?", false, || {
            panic!("nothing may be read from a stdin nobody is typing at")
        })
        .expect_err("a non-terminal stdin cannot answer");
        let e = e.to_string();
        assert!(e.contains("delete topology t1?"), "{e}");
        assert!(e.contains("--force"), "the way out has to be named: {e}");

        // A terminal is still asked, and still told when the answer runs
        // out.
        assert!(confirm_answer("go?", true, || Ok(Some("y\n".to_string()))).unwrap());
        assert!(!confirm_answer("go?", true, || Ok(Some("n\n".to_string()))).unwrap());
        let e = confirm_answer("go?", true, || Ok(None))
            .expect_err("end-of-file is not a \"no\"")
            .to_string();
        assert!(e.contains("--force"), "{e}");
    }

    /// Nothing is deleted, and nothing is asked, about a document that is
    /// not there.
    ///
    /// The prompt came first, so `mirage profile delete proflie` asked
    /// about a profile that had never existed and then reported the
    /// answer as though something had been spared.
    #[test]
    fn a_name_that_names_nothing_is_answered_before_anybody_is_asked() {
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());
        ensure_builtins_present();

        let e = delete_document(false, DocKind::Profile, "proflie", false, |_| {
            panic!("nothing may be deleted")
        })
        .expect_err("there is no profile of that name")
        .to_string();
        assert!(e.contains("proflie"), "{e}");
        assert!(
            !e.contains("answer"),
            "the missing document is the answer, not the prompt: {e}"
        );
    }

    /// The shipped version is on disk before mirage says it is back.
    ///
    /// Deleting an edited builtin removes the user's copy and the
    /// shipped one takes its place — but nothing wrote it: the claim was
    /// made from the fact that mirage *ships* one, and the file only
    /// reappeared when a later command noticed it missing.
    #[test]
    fn a_restored_builtin_is_on_disk_before_it_is_claimed() {
        let _lock = mirage_core::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(root.path());
        ensure_builtins_present();

        // Edited, because a pristine builtin is refused a delete
        // outright: removing it would report a change that never
        // happened.
        let mut agent = mirage_core::store::agent_get("mi300x").unwrap();
        agent.vm.gpu.num_xcds = 7;
        mirage_core::state::write_json(&mirage_core::paths::agent_path("mi300x"), &agent).unwrap();

        let code = delete_document(false, DocKind::Agent, "mi300x", true, |name| {
            mirage_core::store::agent_delete(name)
        })
        .unwrap();
        assert_eq!(code, ExitCode::from(0));
        assert!(
            mirage_core::paths::agent_path("mi300x").is_file(),
            "the shipped agent has to be where the message says it is"
        );
        assert_eq!(
            mirage_core::store::agent_get("mi300x")
                .unwrap()
                .vm
                .gpu
                .num_xcds,
            0,
            "and it has to be the shipped one, not the edited copy"
        );
    }

    /// A refusal names the container flag that was passed, not the four
    /// that could have been.
    #[test]
    fn the_container_refusal_names_what_was_typed() {
        let mut p = sample_profile();
        let a = RunArgs {
            profile: "mi450x".into(),
            hacks: vec![HackArg::UpdateGccViaPpa],
            ..Default::default()
        };
        let e = apply_profile_overrides(&mut p, &a)
            .expect_err("a hack needs an image to build from")
            .to_string();
        assert!(e.starts_with("--hack requires"), "{e}");
        assert!(!e.contains("--mount"), "{e}");

        // Two of them read as a sentence rather than as a slash-list.
        let a = RunArgs {
            profile: "mi450x".into(),
            mounts: vec!["/tmp".to_string()],
            ports: vec!["8080".to_string()],
            ..Default::default()
        };
        let e = apply_profile_overrides(&mut sample_profile(), &a)
            .expect_err("a mount needs an image to mount into")
            .to_string();
        assert!(e.starts_with("--mount and --port require"), "{e}");
    }

    /// One sentence for "mirage has no such emulator", wherever it is
    /// said.
    #[test]
    fn every_unknown_emulator_is_reported_the_same_way() {
        let mut profile = sample_profile();
        profile.emulator.emulator = EmulatorKind::from("nope");
        let from_validate = validate_profile(&profile).expect_err("no such backend");

        let mut p = sample_profile();
        let a = RunArgs {
            profile: "mi450x".into(),
            emulator: Some("nope".to_string()),
            ..Default::default()
        };
        let from_run = apply_profile_overrides(&mut p, &a)
            .expect_err("no such backend")
            .to_string();

        let a = ProfileCreateArgs {
            name: Some("p".to_string()),
            emulator: Some("nope".to_string()),
            agent: None,
            num_nodes: None,
            gpus_per_node: None,
            description: None,
            image: None,
            mounts: Vec::new(),
            ports: Vec::new(),
            provider: None,
            no_input: true,
        };
        let from_create = build_profile_create(a, false)
            .expect_err("no such backend")
            .to_string();

        assert_eq!(from_validate, from_run, "run and validation must agree");
        assert_eq!(from_validate, from_create, "create must agree with both");
        assert!(from_validate.contains("`nope`"), "{from_validate}");
        assert!(
            from_validate.contains("mirage emulators"),
            "the list of what would have worked: {from_validate}"
        );
    }

    /// `state purge --all` says what it destroys and what comes back,
    /// and agents are in both lists.
    #[test]
    fn purge_all_admits_that_agents_go_too() {
        use clap::CommandFactory as _;
        #[derive(clap::Parser)]
        struct Wrap {
            #[command(subcommand)]
            state: StateCmd,
        }
        let help = Wrap::command()
            .find_subcommand_mut("purge")
            .expect("state purge is a subcommand")
            .render_long_help()
            .to_string();
        // The flag's own block, not the mention of it in the command's
        // description: `rsplit` because the declaration comes last.
        let (_, all) = help.rsplit_once("--all").expect("--all is documented");
        let all = all.split("\n\n").next().unwrap_or_default().to_string();
        for word in ["agent", "topolog", "profile"] {
            assert!(
                all.matches(word).count() >= 2,
                "`--all` removes {word}s and writes the builtin ones back; both \
                 belong in its help:\n{all}"
            );
        }
    }

    /// One count, one set of inflections.
    #[test]
    fn a_sentence_agrees_with_its_own_count() {
        assert_eq!(Plural(1).pick("process", "processes"), "process");
        assert_eq!(Plural(0).pick("process", "processes"), "processes");
        assert_eq!(Plural(2).pick("is", "are"), "are");
        assert_eq!(and_list(&["--mount"]), "--mount");
        assert_eq!(and_list(&["--mount", "--port"]), "--mount and --port");
        assert_eq!(
            and_list(&["--mount", "--port", "--hack"]),
            "--mount, --port and --hack"
        );
    }

    #[test]
    fn run_args_default_matches_clap() {
        use clap::Parser;
        #[derive(Parser)]
        struct Wrap {
            #[command(flatten)]
            run: RunArgs,
        }
        let w = Wrap::try_parse_from(["mirage", "--", "./app"])
            .expect("`mirage run -- ./app` should parse");
        assert_eq!(w.run.profile, RunArgs::default().profile);
    }
}
