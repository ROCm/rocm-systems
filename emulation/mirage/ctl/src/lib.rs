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

use std::io::IsTerminal;
use std::process::ExitCode;

use clap::{Args, Subcommand, ValueEnum};
use mirage_core::common::{MaybeRef, SimpleMap, SimpleValue};
use mirage_core::emulator::ExecMode;
use mirage_core::profile::{ContainerizedDef, FileMount, Hack, PortMapping, ProfileDef};
use mirage_core::registry::EmulatorInfo;
use mirage_core::session::SessionId;

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
/// binary registers itself via [`inventory`], and
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
/// detailed block including the support reason).
///
/// The JSON carries every fact the text does, the default backend
/// included: the text form marks it with `(default)` and a script reading
/// the JSON had no way to tell, so it had to re-derive the rule
/// ([`mirage_core::registry::default_emulator`]) from the `installed`
/// flags and hope the two agreed.
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
            println!(
                "  supported: {}  ({})",
                if spec.support.supported { "yes" } else { "no" },
                spec.support.reason
            );
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
/// [`config_dir_hint`] turns that into the sentence the user needs.
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
        None => Err(format!("unknown emulator `{kind}`")),
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
        /// Show long form (description, support reason, and the options
        /// and plugins the backend accepts).
        #[arg(short = 'l', long)]
        long: bool,
    },

    /// Run a command inside a session an existing `mirage run` owns.
    ///
    /// The process runs in *this* terminal, as a child of this command,
    /// and dies with it.
    Exec(ExecArgsCli),

    /// Manage mirage's on-disk state (builtin topologies, purge).
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
/// [`build_profile_create`] consumes the whole payload: six of these
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
    /// Create or overwrite a topology.
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
    ///
    /// A flag rather than a positional because everything after `--`
    /// belongs to the command: with both positional, `mirage exec --
    /// bash` could equally mean "session bash".
    #[arg(long, short = 's')]
    pub session: Option<SessionId>,

    /// Number of workload processes to launch per node.
    #[arg(long, visible_alias = "nproc_per_node", value_parser = clap::value_parser!(u32).range(1..))]
    pub nproc_per_node: Option<u32>,

    /// Run on this node only, instead of on every node in the session.
    ///
    /// This is how you get an interactive shell on a multi-node job. A
    /// job spanning several nodes has every rank's output multiplexed
    /// and nobody's stdin connected, because one terminal cannot be
    /// shared between readers. Naming one node makes this a
    /// single-process exec, which does get the terminal:
    ///
    /// ```text
    /// mirage exec --node 2 -- bash
    /// ```
    ///
    /// The process still believes it is that node: same rank variables,
    /// same `WORLD_SIZE`, same rendezvous as its neighbours.
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
    /// (Re)write the builtin topologies to `<MIRAGE_CONFIG>/topology/`.
    ///
    /// On every run mirage writes any missing builtin topologies on
    /// startup; this command additionally **overwrites** existing
    /// ones, useful after upgrading mirage.
    Builtins,
    /// Remove mirage's runtime directory and reclaim orphaned containers.
    ///
    /// Refuses while any `mirage run` is live: a run owns its session and
    /// cleans up when it exits, so stop those first. The config directory
    /// (profiles, topologies) is left alone unless `--all` is passed.
    Purge {
        /// Don't prompt for confirmation.
        #[arg(short = 'f', long)]
        force: bool,
        /// Also remove the mirage config directory (profiles +
        /// topologies). Builtin topologies will be re-written the
        /// next time mirage runs.
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
    /// default; the flag is accepted for explicitness and for the
    /// `rocjitsu --daemon/--attach` drop-in alias.
    #[arg(long, conflicts_with = "in_process")]
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
            argv: Vec::new(),
        }
    }
}

// =============================================================================
// Dispatch
// =============================================================================

/// Dispatch a parsed [`CtlCmd`]. Returns the exit code the process
/// should use.
///
/// # Errors
///
/// Returns an error if the command fails.
pub async fn dispatch(cmd: CtlCmd, json: bool) -> anyhow::Result<ExitCode> {
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
            if json {
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
            mirage_core::store::profile_put(&p)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&p)?);
            } else {
                println!("created profile {}", p.name);
            }
        }
        ProfileCmd::Import { file } => {
            let bytes = if file == "-" {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut std::io::stdin().lock(), &mut buf)?;
                buf
            } else {
                std::fs::read(&file)?
            };
            let p: ProfileDef = serde_json::from_slice(&bytes)?;
            mirage_core::store::profile_put(&p)?;
            report_change(json, "profile", &p.name, "imported", true)?;
        }
        ProfileCmd::Delete { name, force } => {
            if !force && !confirm(&format!("delete profile {name}?"))? {
                report_change(json, "profile", &name, "deleted", false)?;
                return Ok(ExitCode::from(0));
            }
            mirage_core::store::profile_delete(&name)?;
            report_change(json, "profile", &name, "deleted", true)?;
        }
    }
    Ok(ExitCode::from(0))
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
                agent: MaybeRef::Ref(agent),
            };
            mirage_core::store::topology_put(&name, &t)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&t)?);
            } else {
                println!("created topology {name}");
            }
        }
        TopologyCmd::Import { name, file } => {
            let bytes = read_input(&file)?;
            let t: mirage_core::topology::TopologyDef = serde_json::from_slice(&bytes)?;
            mirage_core::store::topology_put(&name, &t)?;
            report_change(json, "topology", &name, "imported", true)?;
        }
        TopologyCmd::Delete { name, force } => {
            if !force && !confirm(&format!("delete topology {name}?"))? {
                report_change(json, "topology", &name, "deleted", false)?;
                return Ok(ExitCode::from(0));
            }
            mirage_core::store::topology_delete(&name)?;
            report_change(json, "topology", &name, "deleted", true)?;
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
            let bytes = read_input(&file)?;
            let a: mirage_core::agent::AgentDef = serde_json::from_slice(&bytes)?;
            mirage_core::store::agent_put(&name, &a)?;
            report_change(json, "agent", &name, "imported", true)?;
        }
        AgentCmd::Delete { name, force } => {
            if !force && !confirm(&format!("delete agent {name}?"))? {
                report_change(json, "agent", &name, "deleted", false)?;
                return Ok(ExitCode::from(0));
            }
            mirage_core::store::agent_delete(&name)?;
            report_change(json, "agent", &name, "deleted", true)?;
        }
    }
    Ok(ExitCode::from(0))
}

/// Report a one-off change to a stored document — an import, a delete.
///
/// Under `--json` stdout must be exactly one JSON document and nothing
/// else, or the caller that asked for JSON cannot parse what it gets.
/// These commands used to print their sentence either way, so
/// `mirage profile import --json f.json` emitted `imported profile x`
/// and a script's `json.load` failed on the word "imported".
///
/// `done` is false only for a delete the user declined at the prompt,
/// which is a result too: a script that asked and was told no should not
/// have to infer it from an empty stdout.
fn report_change(json: bool, kind: &str, name: &str, verb: &str, done: bool) -> anyhow::Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "kind": kind,
                "name": name,
                verb: done,
            }))?
        );
    } else if done {
        println!("{verb} {kind} {name}");
    }
    Ok(())
}

fn read_input(file: &str) -> anyhow::Result<Vec<u8>> {
    if file == "-" {
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin().lock(), &mut buf)?;
        Ok(buf)
    } else {
        Ok(std::fs::read(file)?)
    }
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
            if !mounts.is_empty() || !ports.is_empty() || provider.is_some() {
                anyhow::bail!("--mount/--port/--container-provider require --image");
            }
            Ok(None)
        }
    }
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
            None => anyhow::bail!(
                "unknown emulator: {n}. Known: {}",
                registry()
                    .into_iter()
                    .map(|e| e.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
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
        agent: MaybeRef::Ref(agent),
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

/// Resolve `--config <path>` to an absolute path, having checked that it
/// is a config file the emulator can actually be given.
///
/// Read here rather than trusted, because the backend's own failure is
/// almost silent: an unreadable or malformed config makes the emulator
/// daemon refuse to start, and the session then continues in-process with
/// nothing on stdout, a `tracing::warn!` nobody sees without `-v`, and an
/// exit code of zero. In-process is not a smaller version of the daemon —
/// it cannot share GPU memory between processes, so a multi-GPU RCCL job
/// silently becomes a different experiment.
///
/// # Errors
///
/// Returns an error naming the path and what is wrong with it.
fn check_config_file(path: &str) -> anyhow::Result<std::path::PathBuf> {
    let abs = std::fs::canonicalize(path).map_err(|e| {
        anyhow::anyhow!("--config {path:?}: {e}. Name an emulator config file that exists.")
    })?;
    // `read` covers both halves of "can the emulator open this": a
    // directory fails with `EISDIR`, an unreadable file with `EACCES`.
    let bytes =
        std::fs::read(&abs).map_err(|e| anyhow::anyhow!("--config {}: {e}", abs.display()))?;
    serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| {
        anyhow::anyhow!(
            "--config {}: this is not a usable emulator config ({e}). \
             It must be a JSON document — the same file the upstream \
             `rocjitsu --config` takes.",
            abs.display()
        )
    })?;
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
                    anyhow::anyhow!("--mount/--port/--container-provider/--hack require a containerised profile or --image")
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
            let available = registry()
                .into_iter()
                .map(|e| e.name)
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!("unknown emulator `{name}`; available backends: {available}");
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

/// One container or network a cleanup pass found.
///
/// Kept whole rather than reduced to a name, because a user acts on the
/// three facts together: what kind of thing it is, which engine holds it,
/// and which session it came from. Reported as "container resource
/// <id>", a network and a container were the same sentence and neither
/// said which engine to type the id at.
#[derive(Debug)]
struct Resource {
    /// The id the engine listed it under.
    id: String,
    /// `"container"` or `"network"`.
    kind: &'static str,
    /// The engine holding it (`podman`, `docker`, or a path).
    provider: String,
    /// The session it was created for.
    session: String,
}

/// What one cleanup pass found — and, unless it was a dry run, removed.
#[derive(Debug, Default)]
struct Reclaimed {
    /// Sessions that had something to reclaim.
    sessions: std::collections::BTreeSet<String>,
    /// Containers and networks, across every engine consulted.
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
            println!(
                "{} {} {} of session {} ({})",
                verb("removed", "would remove"),
                r.kind,
                r.id,
                r.session,
                r.provider
            );
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
/// stop before the containers they run in, and the scratch directory —
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
                session: orphan.session,
            });
        }
    }

    // 3. Scratch directories, one per session that was never torn down.
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
        StateCmd::Builtins => {
            let agents = mirage_builtin::ensure_agents(true)?;
            let topologies = mirage_builtin::ensure_topologies(true)?;
            let profiles = mirage_builtin::ensure_profiles(true)?;
            if json {
                let entries: Vec<_> = agents
                    .iter()
                    .map(|(n, w)| {
                        serde_json::json!({
                            "kind": "agent",
                            "name": n,
                            "path": mirage_core::paths::agent_path(n),
                            "written": w,
                        })
                    })
                    .chain(topologies.iter().map(|(n, w)| {
                        serde_json::json!({
                            "kind": "topology",
                            "name": n,
                            "path": mirage_core::paths::topology_path(n),
                            "written": w,
                        })
                    }))
                    .chain(profiles.iter().map(|(n, w)| {
                        serde_json::json!({
                            "kind": "profile",
                            "name": n,
                            "path": mirage_core::paths::profile_path(n),
                            "written": w,
                        })
                    }))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&entries)?);
            } else {
                for (name, w) in &agents {
                    let p = mirage_core::paths::agent_path(name);
                    let tag = if *w { "wrote" } else { "kept" };
                    println!("{tag} agent     {} -> {}", name, p.display());
                }
                for (name, w) in &topologies {
                    let p = mirage_core::paths::topology_path(name);
                    let tag = if *w { "wrote" } else { "kept" };
                    println!("{tag} topology  {} -> {}", name, p.display());
                }
                for (name, w) in &profiles {
                    let p = mirage_core::paths::profile_path(name);
                    let tag = if *w { "wrote" } else { "kept" };
                    println!("{tag} profile   {} -> {}", name, p.display());
                }
            }
        }
        StateCmd::Purge { force, all } => {
            let prompt = if all {
                "purge ALL mirage state, including profiles and topologies?"
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
                }
                return Ok(ExitCode::from(0));
            }
            return purge(all, json).await;
        }
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
        anyhow::bail!(
            "{} `mirage run` process(es) are still running ({}). \
             Stop them first: each one owns its session and cleans up \
             when it exits.",
            live.len(),
            live.iter()
                .map(SessionId::as_str)
                .collect::<Vec<_>>()
                .join(", "),
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
        anyhow::bail!(
            "a `mirage run` started while this purge was working ({}); \
             nothing further was removed. Stop it and run the purge again.",
            racing
                .iter()
                .map(SessionId::as_str)
                .collect::<Vec<_>>()
                .join(", "),
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
                "mirage: the purge is incomplete; {unfinished} thing(s) could not be removed"
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

fn confirm(prompt: &str) -> anyhow::Result<bool> {
    use std::io::{BufRead, Write};
    eprint!("{prompt} [y/N] ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    let stdin = std::io::stdin();
    stdin.lock().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn print_paths(json: bool) {
    let info = serde_json::json!({
        "config": mirage_core::paths::mirage_config_dir(),
        "runtime": mirage_core::paths::mirage_runtime_dir(),
        "profiles": mirage_core::paths::profile_root(),
        "sessions": mirage_core::paths::session_runtime_root(),
        "runs": mirage_core::paths::run_socket_root(),
    });
    if json {
        // A `serde_json::Value` built from string paths cannot fail to
        // serialize; print something useful rather than panicking if the
        // impossible happens.
        match serde_json::to_string_pretty(&info) {
            Ok(text) => println!("{text}"),
            Err(e) => eprintln!("could not render paths as JSON: {e}"),
        }
    } else {
        println!(
            "config:   {}",
            mirage_core::paths::mirage_config_dir().display()
        );
        println!(
            "runtime:  {}",
            mirage_core::paths::mirage_runtime_dir().display()
        );
        println!("profiles: {}", mirage_core::paths::profile_root().display());
        println!(
            "sessions: {}",
            mirage_core::paths::session_runtime_root().display()
        );
        println!(
            "runs:     {}",
            mirage_core::paths::run_socket_root().display()
        );
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
        std::fs::write(&good, r#"{"vm": {}}"#).unwrap();
        assert!(
            check_config_file(&good.to_string_lossy())
                .unwrap()
                .is_absolute()
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
                    session: "s-1".to_string(),
                },
                Resource {
                    id: "mirage-s-1".to_string(),
                    kind: "network",
                    provider: "docker".to_string(),
                    session: "s-1".to_string(),
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
        assert_eq!(kinds, vec!["container", "network"]);
        assert_eq!(doc["resources"][0]["provider"], "docker");
        assert_eq!(doc["resources"][1]["session"], "s-1");
        assert_eq!(doc["failures"].as_array().unwrap().len(), 1);
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

    /// `RunArgs::default()` is hand-written to reproduce clap's own
    /// defaults, so that a test constructing one with `..Default::default()`
    /// is testing the same starting point a user gets from the command
    /// line. `profile` is the only field where the two could disagree —
    /// every other one defaults to `None`, an empty `Vec`, or `false` in
    /// both — so it is the only one worth pinning here.
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
