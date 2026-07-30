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
use std::sync::OnceLock;

use clap::{Args, Subcommand, ValueEnum};
use mirage_core::common::{MaybeRef, SimpleMap, SimpleValue};
use mirage_core::emulator::ExecMode;
use mirage_core::profile::{ContainerizedDef, FileMount, Hack, PortMapping, ProfileDef};
use mirage_core::registry::EmulatorInfo;
use mirage_core::session::SessionId;

/// Log directive an auto-started daemon should inherit, derived from the
/// CLI's `-v`/`-vv`.
///
/// Set once by [`init_logging`] and applied to the spawned daemon's
/// `Command` environment rather than by mutating this process's own — a
/// process-wide `set_var` is `unsafe` in Rust 2024 and would race every
/// other thread reading the environment.
static DAEMON_LOG_DIRECTIVE: OnceLock<String> = OnceLock::new();

/// The log directive an auto-started daemon should inherit, if the user
/// asked for one with `-v`/`-vv`.
#[must_use]
pub fn daemon_log_directive() -> Option<&'static str> {
    DAEMON_LOG_DIRECTIVE.get().map(String::as_str)
}

/// Initialize the global tracing subscriber. Honours `MIRAGE_LOG` if
/// set, otherwise uses the level implied by `-v` / `-vv`.
pub fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    // Record the chosen level so an auto-started daemon inherits it. The
    // daemon is spawned from this binary with no `-v` flag and would
    // otherwise default to `warn`, silently dropping the session and exec
    // events a user passing `-vv` is asking to see. Applied to the
    // child's environment only when the user has not set `MIRAGE_LOG`
    // themselves.
    if verbose > 0 && std::env::var_os("MIRAGE_LOG").is_none() {
        let _ = DAEMON_LOG_DIRECTIVE.set(level.to_string());
    }
    let env = tracing_subscriber::EnvFilter::try_from_env("MIRAGE_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env)
        .with_writer(std::io::stderr)
        .try_init();
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
fn emulators_cmd(long: bool, json: bool) {
    let specs = registry();
    if json {
        match serde_json::to_string_pretty(&specs) {
            Ok(s) => println!("{s}"),
            Err(e) => eprintln!("failed to serialize emulators: {e}"),
        }
        return;
    }

    let default_name = default_emulator_name();

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
/// topologies, and profiles — writing only what's missing. Errors are
/// logged, never fatal; the user can always force a full rewrite with
/// `mirage state builtins`.
///
/// Shared by the CLI ([`dispatch`]) and the daemon so both surfaces
/// auto-unpack the builtins the first time they run, instead of
/// requiring the user to invoke `mirage state builtins` by hand.
pub fn ensure_builtins_present() {
    if let Err(e) = mirage_builtin::ensure_agents(false) {
        tracing::warn!("failed to preload builtin agents: {e:#}");
    }
    if let Err(e) = mirage_builtin::ensure_topologies(false) {
        tracing::warn!("failed to preload builtin topologies: {e:#}");
    }
    if let Err(e) = mirage_builtin::ensure_profiles(false) {
        tracing::warn!("failed to preload builtin profiles: {e:#}");
    }
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
        /// Show long form (description, runtime path, support reason).
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
    Create {
        /// Profile name. Prompted for when omitted on a terminal.
        name: Option<String>,
        /// Emulator name (e.g. `rocjitsu`, `hotswap`). Defaults to the
        /// first installed backend; see `mirage emulators`.
        #[arg(long)]
        emulator: Option<String>,
        /// Agent name from `<MIRAGE_CONFIG>/agent/` (e.g. `MI300X`,
        /// `MI350X`). Defaults to `MI350X`.
        #[arg(long)]
        agent: Option<String>,
        /// Nodes per rack.
        #[arg(long)]
        num_nodes: Option<u32>,
        /// GPUs per node.
        #[arg(long)]
        gpus_per_node: Option<u32>,
        /// Optional description.
        #[arg(long)]
        description: Option<String>,
        /// Containerise the profile: run every node inside a container
        /// built from this image. Enables `--mount`/`--container-provider`.
        #[arg(long)]
        image: Option<String>,
        /// Bind mount applied to every node container, as
        /// `HOST[:CONTAINER[:ro|rw]]`. May be repeated. Requires
        /// `--image`.
        #[arg(long = "mount", value_name = "HOST[:CONTAINER[:ro|rw]]")]
        mounts: Vec<String>,
        /// Port published from every node container to the host, as
        /// `HOST_PORT[:CONTAINER_PORT][/tcp|/udp]` (like docker `-p`).
        /// May be repeated. Requires `--image`.
        #[arg(long = "port", value_name = "HOST_PORT[:CONTAINER_PORT][/tcp|/udp]")]
        ports: Vec<String>,
        /// Container provider to use (`podman`, `docker`, or a path).
        /// Autodetected (podman, then docker) when omitted. Requires
        /// `--image`.
        #[arg(long = "container-provider")]
        provider: Option<String>,
        /// Never prompt; use defaults for any unspecified field even on
        /// a terminal.
        #[arg(long)]
        no_input: bool,
    },
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
        #[arg(long, default_value_t = 1)]
        num_nodes: u32,
        /// GPUs per node.
        #[arg(long, default_value_t = 1)]
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
    #[arg(long, visible_alias = "nproc_per_node")]
    pub nproc_per_node: Option<u32>,

    /// Send every rank's output to this terminal, prefixed with `[rank]`.
    /// Implies no stdin.
    #[arg(long)]
    pub capture_all: bool,

    /// Extra environment variables, in `KEY=VALUE` form. May be repeated.
    #[arg(long = "env", value_name = "KEY=VALUE")]
    pub envs: Vec<String>,

    /// Working directory for the command.
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
    /// Completely stop and purge all mirage processes and state.
    ///
    /// Stops every running session and then removes the mirage
    /// runtime and state directories. The config directory
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
    #[arg(long)]
    num_nodes: Option<u32>,
    /// Override the profile topology's per-node GPU count for this run.
    #[arg(long)]
    gpus_per_node: Option<u32>,
    /// Number of workload processes to launch per node (like
    /// `torchrun --nproc-per-node`). Defaults to `1`. Each process gets a
    /// distinct `LOCAL_RANK` (`0..nproc_per_node`) and global `RANK`, and
    /// the job's `WORLD_SIZE` becomes `num_nodes * nproc_per_node`, so
    /// `torch.distributed` runs without a separate launcher. Give each
    /// node at least this many GPUs (`--gpus-per-node`) so every process
    /// can pin its own device.
    #[arg(long, visible_alias = "nproc_per_node")]
    nproc_per_node: Option<u32>,
    /// Reuse an existing session by id.
    #[arg(long, conflicts_with_all = ["keep_session"])]
    session: Option<SessionId>,
    /// Keep the session running after the exec finishes.
    #[arg(long)]
    keep_session: bool,
    /// Working directory.
    #[arg(long)]
    workdir: Option<String>,
    /// Extra environment variables to inject into the exec, in
    /// `KEY=VALUE` form. May be repeated.
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
    #[arg(long, value_name = "PATH")]
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
    /// Send every rank's output to this terminal, prefixed with the
    /// rank that produced it.
    ///
    /// Without this each process writes to the terminal directly, which
    /// is what keeps an interactive `bash` interactive and output
    /// byte-exact under redirection — but several nodes writing at once
    /// interleave with nothing to say which wrote what. Capturing labels
    /// every line, at the cost of stdin, which is closed for all ranks.
    #[arg(long)]
    capture_all: bool,
    /// The command and its arguments.
    #[arg(trailing_var_arg = true, required = true, allow_hyphen_values = true)]
    argv: Vec<String>,
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
        CtlCmd::Run(a) => run::run_cmd(a).await,
        CtlCmd::Paths => {
            print_paths(json);
            Ok(ExitCode::from(0))
        }
    }
}

// ----- profile dispatch ------------------------------------------------------

async fn profile_cmd(
    cmd: ProfileCmd,
    json: bool,
) -> anyhow::Result<ExitCode> {
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
        ProfileCmd::Create {
            name,
            emulator,
            agent,
            num_nodes,
            gpus_per_node,
            description,
            image,
            mounts,
            ports,
            provider,
            no_input,
        } => {
            let interactive = !no_input && std::io::stdin().is_terminal();
            let p = build_profile_create(
                name,
                emulator,
                agent,
                num_nodes,
                gpus_per_node,
                description,
                image,
                mounts,
                ports,
                provider,
                interactive,
            )?;
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
            println!("imported profile {}", p.name);
        }
        ProfileCmd::Delete { name, force } => {
            if !force && !confirm(&format!("delete profile {name}?"))? {
                return Ok(ExitCode::from(0));
            }
            mirage_core::store::profile_delete(&name)?;
            println!("deleted profile {name}");
        }
    }
    Ok(ExitCode::from(0))
}

// ----- topology dispatch -----------------------------------------------------

async fn topology_cmd(
    cmd: TopologyCmd,
    json: bool,
) -> anyhow::Result<ExitCode> {
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
            println!("imported topology {name}");
        }
        TopologyCmd::Delete { name, force } => {
            if !force && !confirm(&format!("delete topology {name}?"))? {
                return Ok(ExitCode::from(0));
            }
            mirage_core::store::topology_delete(&name)?;
            println!("deleted topology {name}");
        }
    }
    Ok(ExitCode::from(0))
}

// ----- agent dispatch --------------------------------------------------------

async fn agent_cmd(
    cmd: AgentCmd,
    json: bool,
) -> anyhow::Result<ExitCode> {
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
            println!("imported agent {name}");
        }
        AgentCmd::Delete { name, force } => {
            if !force && !confirm(&format!("delete agent {name}?"))? {
                return Ok(ExitCode::from(0));
            }
            mirage_core::store::agent_delete(&name)?;
            println!("deleted agent {name}");
        }
    }
    Ok(ExitCode::from(0))
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
#[allow(clippy::too_many_arguments)]
fn build_profile_create(
    name: Option<String>,
    emulator: Option<String>,
    agent: Option<String>,
    num_nodes: Option<u32>,
    gpus_per_node: Option<u32>,
    description: Option<String>,
    image: Option<String>,
    mounts: Vec<String>,
    ports: Vec<String>,
    provider: Option<String>,
    interactive: bool,
) -> anyhow::Result<ProfileDef> {
    use dialoguer::{Confirm, Input, Select};
    let theme = dialoguer::theme::ColorfulTheme::default();

    // ----- name -----
    let name = match name {
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
    let spec = match emulator.as_deref() {
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
    let num_nodes = resolve_count(num_nodes, "Nodes per rack", interactive, &theme)?;
    let gpus_per_node = resolve_count(gpus_per_node, "GPUs per node", interactive, &theme)?;

    // ----- agent -----
    let agent = match agent {
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
    let description = match description {
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
        if image.is_some() || !mounts.is_empty() || !ports.is_empty() || provider.is_some() {
            // Any explicit container flag: build directly (errors if mounts
            // or provider were given without an image).
            build_containerize(image, &mounts, &ports, provider)?
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
                &ports,
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

/// Apply container override flags to a freshly-loaded profile.
///
/// When no container flags are present and the profile is referenced by
/// name, returns a [`MaybeRef::Ref`] so the host resolves the profile
/// itself (the common, cheap path). When flags are present, they enable
/// or extend the profile's containerisation and the (now modified)
/// profile is returned inline via [`MaybeRef::Owned`].
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

/// Apply direct CLI overrides (containerisation + emulator settings) to
/// a profile fetched by name.
///
/// When no override is supplied the cheap by-name [`MaybeRef::Ref`] is
/// returned so the session keeps tracking the on-disk profile. As soon
/// as any field is overridden the whole (mutated) profile is inlined as
/// [`MaybeRef::Owned`].
#[allow(clippy::too_many_arguments)]
fn apply_profile_overrides(
    profile: &mut ProfileDef,
    image: Option<String>,
    mounts: &[String],
    ports: &[String],
    provider: Option<String>,
    emulator: Option<String>,
    exec_mode: Option<ExecModeArg>,
    options: &[String],
    plugins: &[String],
    config: Option<String>,
    num_nodes: Option<u32>,
    gpus_per_node: Option<u32>,
    hacks: &[HackArg],
    profile_name: &str,
) -> anyhow::Result<MaybeRef<ProfileDef>> {
    if image.is_none()
        && mounts.is_empty()
        && ports.is_empty()
        && provider.is_none()
        && emulator.is_none()
        && exec_mode.is_none()
        && options.is_empty()
        && plugins.is_empty()
        && config.is_none()
        && num_nodes.is_none()
        && gpus_per_node.is_none()
        && hacks.is_empty()
    {
        // No overrides: keep the cheap by-name reference.
        return Ok(MaybeRef::Ref(profile_name.to_string()));
    }

    // Container overrides.
    if image.is_some()
        || !mounts.is_empty()
        || !ports.is_empty()
        || provider.is_some()
        || !hacks.is_empty()
    {
        let parsed = parse_mounts(mounts)?;
        let parsed_ports = parse_ports(ports)?;
        let parsed_hacks: Vec<Hack> = hacks.iter().copied().map(Hack::from).collect();
        match &mut profile.containerize {
            Some(c) => {
                if let Some(img) = image {
                    c.image = img;
                }
                if let Some(p) = provider {
                    c.provider = Some(p);
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
                let image = image.ok_or_else(|| {
                    anyhow::anyhow!("--mount/--port/--container-provider/--hack require a containerised profile or --image")
                })?;
                profile.containerize = Some(ContainerizedDef {
                    provider,
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
    if let Some(name) = emulator {
        if find_emulator(&name).is_none() {
            let available = registry()
                .into_iter()
                .map(|e| e.name)
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!("unknown emulator `{name}`; available backends: {available}");
        }
        profile.emulator.emulator = name;
    }
    if let Some(mode) = exec_mode {
        profile.emulator.exec_mode = mode.into();
    }
    for opt in options {
        let (key, value) = parse_option(opt)?;
        profile.emulator.options.insert(key, value);
    }
    for spec in plugins {
        let (name, args) = parse_plugin(spec)?;
        profile.emulator.plugins.insert(name, args);
    }
    // Drop-in `--config <path>`: an explicit emulator config file
    // (the upstream `rocjitsu --config`). Stored as the `config`
    // emulator option (absolute, so it resolves regardless of the
    // workload's working directory) for the backend to use verbatim.
    if let Some(cfg) = config {
        let abs =
            std::fs::canonicalize(&cfg).map_err(|e| anyhow::anyhow!("--config {cfg:?}: {e}"))?;
        profile.emulator.options.insert(
            "config".to_string(),
            SimpleValue::String(abs.display().to_string()),
        );
    }

    // Topology overrides: per-run node and per-node GPU counts. Resolve
    // the emulator's topology (following a by-name reference) so the
    // counts can be mutated, then inline the modified topology.
    if num_nodes.is_some() || gpus_per_node.is_some() {
        let mut topo = match &profile.emulator.topology {
            MaybeRef::Owned(t) => t.clone(),
            MaybeRef::Ref(name) => mirage_core::topology::store::get(name)?,
        };
        if let Some(n) = num_nodes {
            topo.num_nodes = n;
        }
        if let Some(g) = gpus_per_node {
            topo.gpus_per_node = g;
        }
        profile.emulator.topology = MaybeRef::Owned(topo);
    }

    Ok(MaybeRef::Owned(profile.clone()))
}

/// Wait for a session to become ready, turning a terminal bring-up
/// failure into a hard error.
///
/// `session_wait_ready` resolves as soon as the session is either healthy
/// *or* terminal, so a failed bring-up (a bad image, a node that won't
/// start, a missing emulator asset) returns `Ok` carrying an *unhealthy,
/// terminal* health rather than an error. Callers about to run a workload
/// must treat that as fatal: otherwise they submit an exec into a session
/// that can never run it. This surfaces the detailed health message
/// instead.
fn split_argv(argv: &[String]) -> (String, Vec<String>) {
    let mut it = argv.iter().cloned();
    let cmd = it.next().unwrap_or_default();
    (cmd, it.collect())
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



// ----- state dispatch --------------------------------------------------------

async fn state_cmd(
    cmd: StateCmd,
    json: bool,
) -> anyhow::Result<ExitCode> {
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
            if !force && !confirm(prompt)? {
                return Ok(ExitCode::from(0));
            }
            purge(all).await?;
            println!("purged");
        }
    }
    Ok(ExitCode::from(0))
}

/// Wait, briefly, for the daemon to stop answering.
///
/// Once it has exited its socket is unlinked, so any request fails. This
/// polls rather than watching the process because the control-plane trait
/// deliberately exposes no pid: a client should not need one.
/// Stop every live run and remove mirage's on-disk state.
async fn purge(all: bool) -> anyhow::Result<()> {
    // Ask every live run to stop, by signalling nothing: a run owns its
    // own session and tears it down when it exits, so there is no
    // "destroy session" call to make. What purge can do is remove the
    // leftovers of runs that are already gone, and reclaim the container
    // resources a run that died abruptly could not.
    //
    // A live run is left alone deliberately. Killing someone else's
    // foreground command from a state-cleanup subcommand would be a
    // surprise, and the user can stop it with Ctrl-C in its own terminal.
    let live = run::live_run_count();
    if live > 0 {
        anyhow::bail!(
            "{live} `mirage run` process(es) are still running. \
             Stop them first: each one owns its session and cleans up \
             when it exits."
        );
    }

    // Reclaim containers and networks no live run accounts for.
    //
    // A run's session lives in its own memory, so a run that was
    // `SIGKILL`ed — or an OOM kill, or a machine that lost power — takes
    // its record of every container with it. `--rm` handles the common
    // case, but a container whose engine was itself interrupted can
    // survive; the mirage label on the resource does too, and this is the
    // command a user reaches for when something has gone wrong.
    //
    // Resources without mirage's label are never candidates, so this is
    // safe on a shared engine.
    if let Some(provider) = mirage_core::container::detect_provider() {
        let removed = tokio::task::spawn_blocking(move || {
            mirage_core::container::reclaim_orphans(&provider, &[])
        })
        .await
        .unwrap_or_default();
        if !removed.is_empty() {
            println!(
                "reclaimed {} orphaned container resource(s): {}",
                removed.len(),
                removed.join(", ")
            );
        }
    }

    let mut targets = vec![mirage_core::paths::mirage_runtime_dir()];
    if all {
        targets.push(mirage_core::paths::mirage_config_dir());
    }
    for t in targets {
        if t.exists()
            && let Err(e) = std::fs::remove_dir_all(&t)
        {
            tracing::warn!("failed to remove {}: {e:#}", t.display());
        }
    }
    Ok(())
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
        println!("runs:     {}", mirage_core::paths::run_socket_root().display());
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

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
        let r = apply_profile_overrides(
            &mut p,
            None,
            &[],
            &[],
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            &[],
            "mi450x",
        )
        .unwrap();
        assert_eq!(r, MaybeRef::Ref("mi450x".to_string()));
    }

    #[test]
    fn exec_mode_and_options_inline_owned_profile() {
        let mut p = sample_profile();
        let r = apply_profile_overrides(
            &mut p,
            None,
            &[],
            &[],
            None,
            None,
            Some(ExecModeArg::Clocked),
            &["gpu_model=cdna4".to_string(), "queues=8".to_string()],
            &[],
            None,
            None,
            None,
            &[],
            "mi450x",
        )
        .unwrap();
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
        let r = apply_profile_overrides(
            &mut p,
            None,
            &[],
            &[],
            None,
            None,
            None,
            &[],
            &[],
            None,
            Some(2),
            Some(4),
            &[],
            "mi450x",
        )
        .unwrap();
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
        let r = apply_profile_overrides(
            &mut p,
            None,
            &[],
            &[],
            None,
            None,
            None,
            &[],
            &["race".to_string(), "logging".to_string()],
            None,
            None,
            None,
            &[],
            "mi450x",
        )
        .unwrap();
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
        let r = apply_profile_overrides(
            &mut p,
            None,
            &[],
            &[],
            None,
            None,
            None,
            &[],
            &["race".to_string()],
            None,
            None,
            None,
            &[],
            "mi450x",
        )
        .unwrap();
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
}
