//! `mirage_ctl`: the user-facing control plane (the `ctl` half of
//! `mirage`).
//!
//! This crate is a **library**: it defines the top-level
//! [`CtlCmd`] subcommand enum and an async [`dispatch`] function that
//! drives a [`mirage_core::ctl::MirageCtl`] implementation (by default
//! `FileCtl`). The unified `mirage` binary wires this up alongside the
//! `host` and `daemon` subcommands.
//!
//! All commands are documented in `docs/cli.md`.

use std::io::IsTerminal;
use std::io::Write;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Context as _;
use clap::{Args, Subcommand, ValueEnum};
use mirage_core::common::{MaybeRef, SimpleValue};
use mirage_core::ctl::{CreateSessionRequest, MirageCtl, StdStream, StreamPacket};
use mirage_core::emulator::ExecMode;
use mirage_core::exec::{ExecArgs, ExecDef, ExecId, ExecRef};
use mirage_core::profile::{ContainerizedDef, FileMount, Hack, PortMapping, ProfileDef};
use mirage_core::registry::EmulatorInfo;
use mirage_core::session::SessionId;
use tokio_stream::StreamExt;

/// Log directive that detached child processes (notably the per-session
/// `mirage host`) should inherit, derived from the CLI's `-v`/`-vv`.
///
/// Set once by [`init_logging`] and applied explicitly to spawned hosts
/// by [`spawn_host_for`] via the `MIRAGE_LOG` environment of the child
/// `Command`, rather than mutating this process's own environment.
static HOST_LOG_DIRECTIVE: OnceLock<String> = OnceLock::new();

/// Initialize the global tracing subscriber. Honours `MIRAGE_LOG` if
/// set, otherwise uses the level implied by `-v` / `-vv`.
pub fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    // Record the chosen level so detached child processes can inherit
    // it. Most importantly this reaches the detached per-session `mirage
    // host`, which is re-exec'd from this binary without any `-v` flag
    // and would otherwise default to `warn`, silently dropping all of
    // its info/debug host events. `spawn_host_for` applies this directly
    // to the child `Command`'s environment (only when the user hasn't
    // set `MIRAGE_LOG` themselves), so a `-v`/`-vv` on the CLI is
    // honoured by the host's own logger too, and host events land in the
    // per-session `node/0/host.log` at the requested verbosity.
    if verbose > 0 && std::env::var_os("MIRAGE_LOG").is_none() {
        let _ = HOST_LOG_DIRECTIVE.set(level.to_string());
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

/// The default emulator for new profiles: the first installed,
/// non-noop entry, falling back to `noop`.
pub fn default_emulator() -> EmulatorInfo {
    let specs = registry();
    mirage_core::registry::default_emulator(&specs).clone()
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

    let default_name = default_emulator().name;

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
pub fn validate_profile(def: &ProfileDef) -> std::result::Result<(), String> {
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

    /// Manage sessions.
    #[command(subcommand)]
    Session(SessionCmd),

    /// Manage execs inside a session.
    #[command(subcommand)]
    Exec(ExecCmd),

    /// Manage mirage's on-disk state (builtin topologies, purge).
    #[command(subcommand)]
    State(StateCmd),

    /// Convenience: create session, run a command, attach, clean up.
    Run(RunArgs),

    /// Re-attach to a running exec's streams.
    Attach(AttachArgs),

    /// Show or follow an exec's stdout.
    Logs(LogsArgs),

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
        /// Emulator name (e.g. `rocjitsu`, `noop`). Defaults to the
        /// first installed emulator (rocjitsu if present, otherwise
        /// noop).
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

#[derive(Subcommand, Debug)]
pub enum SessionCmd {
    /// List sessions.
    List,
    /// Show a session's state.
    Show { id: SessionId },
    /// Wait for a session to become healthy.
    Wait {
        id: SessionId,
        /// Seconds to wait.
        #[arg(long, default_value_t = 30)]
        timeout: u64,
    },
    /// Start a new session and its host process.
    Start(StartArgs),
    /// Stop a session and remove its state.
    Stop { id: SessionId },
    /// Show the per-session directory path.
    Dir { id: SessionId },
}

#[derive(Args, Debug)]
pub struct StartArgs {
    /// Profile to use (by name). Prompted for when omitted on a
    /// terminal.
    #[arg(long)]
    pub profile: Option<String>,
    /// Explicit session id; auto-generated if omitted.
    #[arg(long)]
    pub id: Option<SessionId>,
    /// Working directory for execs in the session.
    #[arg(long)]
    pub workdir: Option<String>,
    /// Don't spawn the host (the caller is expected to start it).
    #[arg(long)]
    pub no_host: bool,
    /// How long to wait for the host to report ready (seconds).
    #[arg(long, default_value_t = 10)]
    pub ready_timeout: u64,
    /// Override/enable containerisation: run every node inside a
    /// container built from this image.
    #[arg(long)]
    pub image: Option<String>,
    /// Extra bind mount (`HOST[:CONTAINER[:ro|rw]]`). May be repeated.
    #[arg(long = "mount", value_name = "HOST[:CONTAINER[:ro|rw]]")]
    pub mounts: Vec<String>,
    /// Container provider (`podman`, `docker`, or a path). Autodetected
    /// when omitted.
    #[arg(long = "container-provider")]
    pub provider: Option<String>,
    /// Override the emulator execution mode (`functional` or `clocked`).
    #[arg(long)]
    pub exec_mode: Option<ExecModeArg>,
    /// Override an emulator option directly (`KEY=VALUE`). May be
    /// repeated.
    #[arg(long = "option", short = 'o', value_name = "KEY=VALUE")]
    pub options: Vec<String>,
    /// Use an explicit emulator config file instead of synthesising one
    /// from the profile (the upstream `rocjitsu --config`).
    #[arg(long, value_name = "PATH")]
    pub config: Option<String>,
    /// Run the emulator in out-of-process daemon mode. This is the
    /// default; the flag is accepted for explicitness and for the
    /// `rocjitsu --daemon/--attach` drop-in alias.
    #[arg(long, conflicts_with = "in_process")]
    pub daemon: bool,
    /// Run the emulator in-process (local mode) instead of the default
    /// out-of-process daemon. In-process mode cannot share GPU memory
    /// across processes, so multi-GPU RCCL collectives require the
    /// daemon (the default).
    #[arg(long = "in-process")]
    pub in_process: bool,
    /// Never prompt; require every field on the command line even on a
    /// terminal.
    #[arg(long)]
    pub no_input: bool,
}

// ----- exec ------------------------------------------------------------------

#[derive(Subcommand, Debug)]
pub enum ExecCmd {
    /// List execs in a session.
    List { session: SessionId },
    /// Show an exec's status.
    Show { session: SessionId, exec: ExecId },
    /// Start a new exec in a session and attach to it.
    ///
    /// Everything after `--` is passed to the command verbatim.
    Start(ExecStartArgs),
    /// Send a signal to an exec.
    Signal {
        session: SessionId,
        exec: ExecId,
        /// Signal name (e.g. TERM, KILL, INT) or number.
        #[arg(default_value = "TERM")]
        sig: String,
    },
    /// Remove a finished exec.
    Remove { session: SessionId, exec: ExecId },
}

#[derive(Args, Debug)]
pub struct ExecStartArgs {
    session: SessionId,
    /// Keep the exec on disk after it finishes.
    #[arg(long)]
    keep: bool,
    /// Don't attach to the exec; just submit and return its id.
    #[arg(long)]
    detach: bool,
    /// Extra environment variables to inject into the exec, in
    /// `KEY=VALUE` form. May be repeated.
    #[arg(long = "env", value_name = "KEY=VALUE")]
    envs: Vec<String>,
    /// The command and its arguments. Use `--` to separate from
    /// mirage flags.
    #[arg(trailing_var_arg = true, required = true, allow_hyphen_values = true)]
    argv: Vec<String>,
}

// ----- state -----------------------------------------------------------------

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
    /// `rocjitsu-dbt`, `hotswap`, `noop`). See `mirage emulators` for
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
    /// The command and its arguments.
    #[arg(trailing_var_arg = true, required = true, allow_hyphen_values = true)]
    argv: Vec<String>,
}

// ----- attach / logs ---------------------------------------------------------

#[derive(Args, Debug)]
pub struct AttachArgs {
    session: SessionId,
    exec: ExecId,
}

#[derive(Args, Debug)]
pub struct LogsArgs {
    session: SessionId,
    exec: ExecId,
    /// Follow output as it is appended.
    #[arg(short = 'f', long)]
    follow: bool,
}

// =============================================================================
// Dispatch
// =============================================================================

/// Dispatch a parsed [`CtlCmd`] against an arbitrary [`MirageCtl`]
/// implementation. Returns the exit code the process should use.
pub async fn dispatch<C: MirageCtl + 'static>(
    cmd: CtlCmd,
    ctl: C,
    json: bool,
) -> anyhow::Result<ExitCode> {
    // Best-effort: write any missing builtin agents/topologies on
    // startup so they're always available under <MIRAGE_CONFIG>/. Errors
    // here are non-fatal; the user can recover via `mirage state
    // builtins`.
    ensure_builtins_present();
    let ctl = Arc::new(ctl);
    match cmd {
        CtlCmd::Profile(c) => profile_cmd(&*ctl, c, json),
        CtlCmd::Topology(c) => topology_cmd(&*ctl, c, json),
        CtlCmd::Agent(c) => agent_cmd(&*ctl, c, json),
        CtlCmd::Emulators { long } => {
            emulators_cmd(long, json);
            Ok(ExitCode::from(0))
        }
        CtlCmd::Session(c) => session_cmd(&*ctl, c, json).await,
        CtlCmd::Exec(c) => exec_cmd(ctl.clone(), c, json).await,
        CtlCmd::State(c) => state_cmd(ctl.clone(), c, json).await,
        CtlCmd::Run(a) => run_cmd(ctl.clone(), a).await,
        CtlCmd::Attach(a) => attach_cmd(ctl.clone(), a).await,
        CtlCmd::Logs(a) => logs_cmd(ctl.clone(), a).await,
        CtlCmd::Paths => {
            print_paths(json);
            Ok(ExitCode::from(0))
        }
    }
}

// ----- profile dispatch ------------------------------------------------------

fn profile_cmd(ctl: &dyn MirageCtl, cmd: ProfileCmd, json: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        ProfileCmd::List { long } => {
            let names = ctl.profile_list()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&names)?);
            } else if long {
                if names.is_empty() {
                    eprintln!("(no profiles)");
                }
                println!("{:<24} {:<16} DESCRIPTION", "NAME", "EMULATOR");
                for n in names {
                    match ctl.profile_get(&n) {
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
            let p = ctl.profile_get(&name)?;
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
            ctl.profile_put(&p)?;
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
            ctl.profile_put(&p)?;
            println!("imported profile {}", p.name);
        }
        ProfileCmd::Delete { name, force } => {
            if !force && !confirm(&format!("delete profile {name}?"))? {
                return Ok(ExitCode::from(0));
            }
            ctl.profile_delete(&name)?;
            println!("deleted profile {name}");
        }
    }
    Ok(ExitCode::from(0))
}

// ----- topology dispatch -----------------------------------------------------

fn topology_cmd(ctl: &dyn MirageCtl, cmd: TopologyCmd, json: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        TopologyCmd::List => {
            let names = ctl.topology_list()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&names)?);
            } else {
                for n in names {
                    println!("{n}");
                }
            }
        }
        TopologyCmd::Show { name } => {
            let t = ctl.topology_get(&name)?;
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
            ctl.topology_put(&name, &t)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&t)?);
            } else {
                println!("created topology {name}");
            }
        }
        TopologyCmd::Import { name, file } => {
            let bytes = read_input(&file)?;
            let t: mirage_core::topology::TopologyDef = serde_json::from_slice(&bytes)?;
            ctl.topology_put(&name, &t)?;
            println!("imported topology {name}");
        }
        TopologyCmd::Delete { name, force } => {
            if !force && !confirm(&format!("delete topology {name}?"))? {
                return Ok(ExitCode::from(0));
            }
            ctl.topology_delete(&name)?;
            println!("deleted topology {name}");
        }
    }
    Ok(ExitCode::from(0))
}

// ----- agent dispatch --------------------------------------------------------

fn agent_cmd(ctl: &dyn MirageCtl, cmd: AgentCmd, json: bool) -> anyhow::Result<ExitCode> {
    match cmd {
        AgentCmd::List => {
            let names = ctl.agent_list()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&names)?);
            } else {
                for n in names {
                    println!("{n}");
                }
            }
        }
        AgentCmd::Show { name } => {
            let a = ctl.agent_get(&name)?;
            println!("{}", serde_json::to_string_pretty(&a)?);
        }
        AgentCmd::Import { name, file } => {
            let bytes = read_input(&file)?;
            let a: mirage_core::agent::AgentDef = serde_json::from_slice(&bytes)?;
            ctl.agent_put(&name, &a)?;
            println!("imported agent {name}");
        }
        AgentCmd::Delete { name, force } => {
            if !force && !confirm(&format!("delete agent {name}?"))? {
                return Ok(ExitCode::from(0));
            }
            ctl.agent_delete(&name)?;
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

// ----- session dispatch ------------------------------------------------------

async fn session_cmd<C: MirageCtl>(
    ctl: &C,
    cmd: SessionCmd,
    json: bool,
) -> anyhow::Result<ExitCode> {
    match cmd {
        SessionCmd::List => {
            let ids = ctl.session_list()?;
            if json {
                let states: Vec<_> = ids
                    .iter()
                    .filter_map(|i| ctl.session_state(i).ok())
                    .collect();
                println!("{}", serde_json::to_string_pretty(&states)?);
            } else {
                if ids.is_empty() {
                    eprintln!("(no sessions)");
                }
                println!("{:<32} {:<10} {:<12} CONTAINER", "ID", "HEALTHY", "STATE");
                for id in ids {
                    let state = ctl.session_state(&id).ok();
                    let h = state
                        .as_ref()
                        .map(|s| s.health.clone())
                        .or_else(|| ctl.session_health(&id).ok())
                        .unwrap_or_default();
                    let container = match state.as_ref().and_then(|s| s.container.as_ref()) {
                        Some(c) => format!(
                            "{} {} ({} node{})",
                            c.provider,
                            c.image,
                            c.nodes.len(),
                            if c.nodes.len() == 1 { "" } else { "s" }
                        ),
                        None => "-".to_string(),
                    };
                    println!(
                        "{:<32} {:<10} {:<12} {}",
                        id,
                        h.healthy,
                        h.state.clone().unwrap_or_default(),
                        container
                    );
                    // Surface the detailed status/error message (image
                    // pull progress, network/node bring-up, stall/crash
                    // diagnostics) on an indented continuation line.
                    if let Some(msg) = h.message.as_deref() {
                        println!("{:>32}   {}", "", msg);
                    }
                }
            }
        }
        SessionCmd::Show { id } => {
            let s = ctl.session_state(&id)?;
            println!("{}", serde_json::to_string_pretty(&s)?);
        }
        SessionCmd::Wait { id, timeout } => {
            let h = ctl.session_wait_ready(&id, Duration::from_secs(timeout))?;
            if !h.healthy {
                let state = h.state.clone().unwrap_or_default();
                match h.message.as_deref() {
                    Some(msg) => eprintln!("session is unhealthy ({state}): {msg}"),
                    None => eprintln!("session is unhealthy: {state}"),
                }
                return Ok(ExitCode::from(2));
            }
            println!("{}", serde_json::to_string_pretty(&h)?);
        }
        SessionCmd::Start(args) => return session_start(ctl, args, json).await,
        SessionCmd::Stop { id } => {
            ctl.session_destroy(&id)?;
            println!("stopped {id}");
        }
        SessionCmd::Dir { id } => {
            let p = mirage_core::paths::session_dir(&id);
            println!("{}", p.display());
        }
    }
    Ok(ExitCode::from(0))
}

/// Build a [`ContainerizedDef`] from CLI container flags.
///
/// Returns `None` when no container flags were given. `--mount` and
/// `--container-provider` require `--image` (there is no base image to
/// attach them to otherwise).
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
            let default_name = default_emulator().name;
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
        None => default_emulator(),
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

/// Wait for a freshly-spawned session host to become ready, turning a
/// terminal bring-up failure into a hard error.
///
/// `session_wait_ready` resolves as soon as the session is either
/// healthy *or* terminal, so a failed host/container bring-up (a bad
/// image, a node that won't start, a missing emulator asset, …) returns
/// `Ok` carrying an *unhealthy, terminal* health rather than an error.
/// Callers about to run a workload must treat that as fatal: otherwise
/// they submit an exec that no (now-exited) host will ever process and
/// the client blocks forever. This surfaces the detailed health message
/// (image pull error, node bring-up failure, …) as an error instead.
fn wait_ready_or_bail<C: MirageCtl>(
    ctl: &C,
    id: &SessionId,
    timeout: Duration,
) -> anyhow::Result<()> {
    let h = ctl.session_wait_ready(id, timeout)?;
    if !h.healthy {
        let state = h.state.unwrap_or_else(|| "failed".to_string());
        match h.message {
            Some(msg) => anyhow::bail!("session failed to start ({state}): {msg}"),
            None => anyhow::bail!("session failed to start ({state})"),
        }
    }
    Ok(())
}

async fn session_start<C: MirageCtl>(
    ctl: &C,
    args: StartArgs,
    json: bool,
) -> anyhow::Result<ExitCode> {
    // Any field left off the command line is prompted for when stdin is
    // a terminal (and `--no-input` wasn't given); otherwise its default
    // is used. This makes `session start` an interactive UI while
    // staying fully non-interactive in scripts, pipes and tests.
    let interactive = !args.no_input && std::io::stdin().is_terminal();

    let profile_name = resolve_start_profile(ctl, args.profile, interactive)?;
    let id = resolve_start_id(args.id, interactive)?;
    let workdir = resolve_start_workdir(args.workdir, interactive)?;
    let ready_timeout = resolve_start_ready_timeout(args.ready_timeout, interactive)?;

    // Validate profile exists and resolve it so container overrides can
    // be applied.
    let mut profile = ctl.profile_get(&profile_name)?;
    let profile_ref = apply_profile_overrides(
        &mut profile,
        args.image,
        &args.mounts,
        &[],
        args.provider,
        None,
        args.exec_mode,
        &args.options,
        args.config,
        None,
        None,
        &[],
        &profile_name,
    )?;
    let def = ctl.session_create(CreateSessionRequest {
        id,
        profile: profile_ref,
        workdir,
        daemon: !args.in_process,
    })?;
    if !args.no_host {
        spawn_host_for(&def.id)?;
        wait_ready_or_bail(ctl, &def.id, Duration::from_secs(ready_timeout))?;
    }
    if json {
        let s = ctl.session_state(&def.id)?;
        println!("{}", serde_json::to_string_pretty(&s)?);
    } else {
        println!("{}", def.id);
    }
    Ok(ExitCode::from(0))
}

/// Resolve the profile name for `session start`: the `--profile` flag, an
/// interactive picker over the known profiles, or a hard error when no
/// profile was given and we can't prompt.
fn resolve_start_profile<C: MirageCtl>(
    ctl: &C,
    profile: Option<String>,
    interactive: bool,
) -> anyhow::Result<String> {
    if let Some(p) = profile {
        return Ok(p);
    }
    if !interactive {
        anyhow::bail!("a profile is required (pass --profile NAME)");
    }
    let profiles = ctl.profile_list()?;
    if profiles.is_empty() {
        anyhow::bail!("no profiles found; run `mirage profile create` first");
    }
    let pick = dialoguer::Select::with_theme(&dialoguer::theme::ColorfulTheme::default())
        .with_prompt("Profile")
        .items(&profiles)
        .default(0)
        .interact()?;
    Ok(profiles[pick].clone())
}

/// Resolve the session id: the `--id` flag, an interactive prompt (blank
/// for auto), or `None` (auto-generated).
fn resolve_start_id(id: Option<SessionId>, interactive: bool) -> anyhow::Result<Option<SessionId>> {
    if id.is_some() || !interactive {
        return Ok(id);
    }
    let id_raw: String = dialoguer::Input::with_theme(&dialoguer::theme::ColorfulTheme::default())
        .with_prompt("Session id (blank for auto)")
        .allow_empty(true)
        .interact_text()?;
    if id_raw.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(SessionId::new(id_raw)?))
    }
}

/// Resolve the working directory: the `--workdir` flag, an interactive
/// prompt defaulting to the current directory, or the current directory.
fn resolve_start_workdir(workdir: Option<String>, interactive: bool) -> anyhow::Result<String> {
    let cwd = || {
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "/".to_string())
    };
    match workdir {
        Some(w) => Ok(w),
        None if interactive => Ok(dialoguer::Input::with_theme(
            &dialoguer::theme::ColorfulTheme::default(),
        )
        .with_prompt("Working directory")
        .with_initial_text(cwd())
        .interact_text()?),
        None => Ok(cwd()),
    }
}

/// Resolve the host ready timeout: prompts (defaulting to the current
/// value) when interactive, otherwise uses the value as-is.
fn resolve_start_ready_timeout(ready_timeout: u64, interactive: bool) -> anyhow::Result<u64> {
    if !interactive {
        return Ok(ready_timeout);
    }
    Ok(
        dialoguer::Input::with_theme(&dialoguer::theme::ColorfulTheme::default())
            .with_prompt("Host ready timeout (seconds)")
            .default(ready_timeout)
            .interact_text()?,
    )
}

/// Look up an executable named `name` on `PATH`, returning the first hit.
fn which_on_path(name: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Resolve the `mirage` binary used to spawn a per-session host process.
///
/// Resolution order, skipping any candidate that no longer exists on disk
/// (e.g. `current_exe()` going stale after the binary was rebuilt or
/// reinstalled while a long-running daemon kept executing):
///   1. `MIRAGE_BIN` (tests / explicit override)
///   2. the current executable, if its path still points at a real file
///   3. a `mirage` binary discovered on `PATH`
fn find_host_bin_for_session_spawn() -> anyhow::Result<std::path::PathBuf> {
    if let Some(b) = std::env::var_os("MIRAGE_BIN") {
        let p = std::path::PathBuf::from(b);
        if p.is_file() {
            return Ok(p);
        }
        anyhow::bail!(
            "MIRAGE_BIN points at `{}`, which does not exist",
            p.display()
        );
    }
    if let Ok(exe) = std::env::current_exe()
        && exe.is_file()
    {
        return Ok(exe);
    }
    if let Some(p) = which_on_path("mirage") {
        return Ok(p);
    }
    anyhow::bail!(
        "could not locate the `mirage` binary to launch a session host. \
         The running daemon's own executable path is stale (it was likely \
         rebuilt or reinstalled while running). Restart the daemon, or set \
         MIRAGE_BIN to the path of the `mirage` binary."
    )
}

/// Spawn the per-session `mirage host` process for `id` and detach it.
///
/// This is `pub` so other binaries (notably `mirage_daemon`) can reuse
/// the exact same host-spawning logic the CLI uses.
pub fn spawn_host_for(id: &SessionId) -> anyhow::Result<()> {
    // The unified `mirage` binary is its own host: we re-exec
    // ourselves with the `host` subcommand. Tests may override which
    // binary is used via `MIRAGE_BIN`.
    let bin = find_host_bin_for_session_spawn()?;
    let layout = mirage_core::paths::SessionLayout::for_id(id);
    // The session host is node 0's host: redirect its stderr to
    // `node/0/host.log`. Ensure the node directory exists first.
    let node0 = layout.node(0);
    std::fs::create_dir_all(&node0.root)?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(node0.host_log())?;
    // spawn and detach into its own process group so terminal-generated
    // signals (e.g. Ctrl-C in the foreground shell) are not delivered to
    // the detached host.
    let mut cmd = std::process::Command::new(bin);
    cmd.arg("host")
        .arg("--session")
        .arg(id.as_str())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(log));
    // Propagate the CLI's chosen log level to the detached host so its
    // events are logged at the requested verbosity. Only set when the
    // user didn't already provide `MIRAGE_LOG` (in which case the host
    // inherits it from our environment as usual).
    if let Some(level) = HOST_LOG_DIRECTIVE.get() {
        cmd.env("MIRAGE_LOG", level);
    }
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);
    cmd.spawn().with_context(|| {
        format!(
            "failed to launch session host via `{}`",
            cmd.get_program().to_string_lossy()
        )
    })?;
    Ok(())
}

// ----- exec dispatch ---------------------------------------------------------

async fn exec_cmd<C: MirageCtl + 'static>(
    ctl: Arc<C>,
    cmd: ExecCmd,
    json: bool,
) -> anyhow::Result<ExitCode> {
    match cmd {
        ExecCmd::List { session } => {
            let ids = ctl.exec_list(&session)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&ids)?);
            } else {
                if ids.is_empty() {
                    eprintln!("(no execs)");
                }
                println!("{:<14} {:<8} {:<8} EXIT", "EXEC", "STARTED", "ENDED");
                for id in ids {
                    let r = ExecRef {
                        session: session.clone(),
                        exec: id.clone(),
                    };
                    let s = ctl.exec_status(&r).unwrap_or_default();
                    println!(
                        "{:<14} {:<8} {:<8} {}",
                        id,
                        s.started,
                        s.ended,
                        s.exit_code
                            .map(|c| c.to_string())
                            .unwrap_or_else(|| "-".to_string())
                    );
                }
            }
        }
        ExecCmd::Show { session, exec } => {
            let r = ExecRef { session, exec };
            let s = ctl.exec_status(&r)?;
            println!("{}", serde_json::to_string_pretty(&s)?);
        }
        ExecCmd::Start(a) => return exec_start(ctl, a).await,
        ExecCmd::Signal { session, exec, sig } => {
            let n = parse_signal(&sig)?;
            let r = ExecRef { session, exec };
            ctl.exec_signal(&r, n)?;
        }
        ExecCmd::Remove { session, exec } => {
            let r = ExecRef { session, exec };
            ctl.exec_remove(&r)?;
            println!("removed");
        }
    }
    Ok(ExitCode::from(0))
}

async fn exec_start<C: MirageCtl + 'static>(
    ctl: Arc<C>,
    a: ExecStartArgs,
) -> anyhow::Result<ExitCode> {
    let (cmd, args) = split_argv(&a.argv);
    let env = parse_envs(&a.envs)?;
    let def = ExecDef {
        timestamp: chrono::Utc::now(),
        session: a.session.clone(),
        exec: ExecArgs {
            command: cmd,
            args,
            env,
            workdir: None,
        },
        worker_exec: None,
        // when attaching, we always keep the exec until after attach
        // completes; otherwise the host might remove the dir before we
        // finish tailing.
        nproc_per_node: 1,
        keep: a.keep || !a.detach,
    };
    let r = ctl.session_exec(&def)?;
    if a.detach {
        println!("{}", r.exec);
        return Ok(ExitCode::from(0));
    }
    let code = follow_attach(ctl.clone(), &r).await?;
    if !a.keep {
        let _ = ctl.exec_remove(&r);
    }
    Ok(code)
}

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

fn parse_signal(s: &str) -> anyhow::Result<i32> {
    if let Ok(n) = s.parse::<i32>() {
        return Ok(n);
    }
    let name = s.trim_start_matches("SIG").to_ascii_uppercase();
    Ok(match name.as_str() {
        "TERM" => libc::SIGTERM,
        "KILL" => libc::SIGKILL,
        "INT" => libc::SIGINT,
        "HUP" => libc::SIGHUP,
        "QUIT" => libc::SIGQUIT,
        "USR1" => libc::SIGUSR1,
        "USR2" => libc::SIGUSR2,
        _ => anyhow::bail!("unknown signal: {s}"),
    })
}

// ----- attach/logs/run -------------------------------------------------------

async fn attach_cmd<C: MirageCtl + 'static>(
    ctl: Arc<C>,
    a: AttachArgs,
) -> anyhow::Result<ExitCode> {
    let r = ExecRef {
        session: a.session,
        exec: a.exec,
    };
    follow_attach(ctl, &r).await
}

/// Put the controlling terminal into raw mode for the duration of an
/// interactive attach, restoring the original settings on drop.
///
/// When attached to an interactive workload (e.g. `bash`) the remote
/// PTY owns echo and line editing, so the local terminal must forward
/// every keystroke verbatim instead of cooking/echoing it. Returns
/// `None` (a no-op) when stdin isn't a TTY, e.g. piped or redirected.
struct RawModeGuard {
    fd: i32,
    original: libc::termios,
}

impl RawModeGuard {
    fn enable_if_tty() -> Option<Self> {
        use std::os::fd::AsRawFd as _;
        let fd = std::io::stdin().as_raw_fd();
        // SAFETY: `fd` is the process stdin; the termios calls only read
        // and write a stack-allocated `termios` we own.
        unsafe {
            if libc::isatty(fd) != 1 {
                return None;
            }
            let mut original: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut original) != 0 {
                return None;
            }
            let mut raw = original;
            libc::cfmakeraw(&mut raw);
            if libc::tcsetattr(fd, libc::TCSANOW, &raw) != 0 {
                return None;
            }
            Some(RawModeGuard { fd, original })
        }
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        // SAFETY: restoring the saved termios on the same fd.
        unsafe {
            libc::tcsetattr(self.fd, libc::TCSANOW, &self.original);
        }
    }
}

async fn follow_attach<C: MirageCtl + 'static>(
    ctl: Arc<C>,
    r: &ExecRef,
) -> anyhow::Result<ExitCode> {
    let mut s = ctl.session_attach(r)?;

    // Forward this process's stdin to the workload's stdin (node 0).
    // Without this an interactive workload such as `bash` blocks forever
    // waiting for input that never arrives. The host exposes node 0's
    // stdin as a FIFO that a bridge task pumps into the workload's PTY;
    // `session_stdin` writes there. Reads are blocking, so they run on a
    // dedicated thread that exits on EOF or the first write error (the
    // workload went away). The thread is detached: the attach loop below
    // owns the lifetime and the process exits shortly after it returns.
    let _raw = RawModeGuard::enable_if_tty();
    let stdin_ctl = ctl.clone();
    let stdin_ref = r.clone();
    std::thread::spawn(move || {
        use std::io::Read as _;
        let mut stdin = std::io::stdin().lock();
        let mut buf = [0u8; 4096];
        loop {
            match stdin.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    // The exec's stdin FIFO is created asynchronously by
                    // the host just after the exec is submitted, so the
                    // first write can briefly race ahead of it. Retry
                    // transient failures so opening keystrokes aren't
                    // dropped; give up if it never appears (the workload
                    // is gone) so the thread can't spin forever.
                    let mut attempts = 0;
                    loop {
                        match stdin_ctl.session_stdin(&stdin_ref, &buf[..n]) {
                            Ok(()) => break,
                            Err(_) if attempts < 250 => {
                                attempts += 1;
                                std::thread::sleep(Duration::from_millis(20));
                            }
                            Err(_) => return,
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });

    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();
    let mut exit: i32 = 0;
    while let Some(pkt) = s.next().await {
        match pkt {
            StreamPacket::Output { stream, data, .. } => match stream {
                StdStream::Stdout => {
                    let _ = stdout.write_all(&data);
                    let _ = stdout.flush();
                }
                StdStream::Stderr => {
                    let _ = stderr.write_all(&data);
                    let _ = stderr.flush();
                }
                StdStream::Stdin => {}
            },
            StreamPacket::NodeExit { .. } => {}
            StreamPacket::ExecExit { exit_code } => {
                exit = exit_code;
                break;
            }
        }
    }
    Ok(ExitCode::from((exit & 0xff) as u8))
}

async fn logs_cmd<C: MirageCtl + 'static>(ctl: Arc<C>, a: LogsArgs) -> anyhow::Result<ExitCode> {
    let layout = mirage_core::paths::SessionLayout::for_id(&a.session).exec(&a.exec);
    if !layout.root.exists() {
        anyhow::bail!("exec not found: {}", a.exec);
    }
    let mut nodes = vec![];
    if let Ok(rd) = std::fs::read_dir(layout.node_root()) {
        for e in rd.flatten() {
            if let Some(s) = e.file_name().to_str()
                && let Ok(n) = s.parse::<u32>()
            {
                nodes.push(n);
            }
        }
    }
    nodes.sort();
    if !a.follow {
        for n in &nodes {
            let nl = layout.node(*n);
            if let Ok(b) = std::fs::read(nl.stdout()) {
                let _ = std::io::stdout().write_all(&b);
            }
        }
        return Ok(ExitCode::from(0));
    }
    let r = ExecRef {
        session: a.session,
        exec: a.exec,
    };
    follow_attach(ctl, &r).await?;
    Ok(ExitCode::from(0))
}

async fn run_cmd<C: MirageCtl + 'static>(ctl: Arc<C>, a: RunArgs) -> anyhow::Result<ExitCode> {
    // Find or create the session.
    let (sid, created) = match a.session {
        Some(id) => (id, false),
        None => {
            // create transient session
            let mut profile = ctl.profile_get(&a.profile)?;
            let profile_ref = apply_profile_overrides(
                &mut profile,
                a.image.clone(),
                &a.mounts,
                &a.ports,
                a.container_provider.clone(),
                a.emulator.clone(),
                a.exec_mode,
                &a.options,
                a.config.clone(),
                a.num_nodes,
                a.gpus_per_node,
                &a.hacks,
                &a.profile,
            )?;
            tracing::info!(profile = %a.profile, "creating transient session");
            let def = ctl.session_create(CreateSessionRequest {
                id: None,
                profile: profile_ref,
                workdir: a.workdir.clone().unwrap_or_else(|| {
                    std::env::current_dir()
                        .map(|p| p.display().to_string())
                        .unwrap_or("/".to_string())
                }),
                daemon: !a.in_process,
            })?;
            tracing::info!(session = %def.id, "session created; spawning host");
            spawn_host_for(&def.id)?;
            if let Err(e) = wait_ready_or_bail(ctl.as_ref(), &def.id, Duration::from_secs(10)) {
                // The transient session we just created never came up
                // (e.g. a container image that couldn't be pulled or a
                // node that wouldn't start). Tear it down so we don't
                // leak a dead session, then surface the failure instead
                // of submitting an exec no host will ever run.
                let _ = ctl.session_destroy(&def.id);
                return Err(e);
            }
            tracing::info!(session = %def.id, "session ready");
            (def.id, true)
        }
    };
    let (cmd, args) = split_argv(&a.argv);
    let env = parse_envs(&a.envs)?;
    let def = ExecDef {
        timestamp: chrono::Utc::now(),
        session: sid.clone(),
        exec: ExecArgs {
            command: cmd,
            args,
            env,
            workdir: a.workdir.clone(),
        },
        worker_exec: None,
        nproc_per_node: a.nproc_per_node.unwrap_or(1).max(1),
        // keep until after attach drains; we may still destroy the
        // whole session below.
        keep: true,
    };
    let r = ctl.session_exec(&def)?;
    tracing::info!(session = %sid, exec = %r.exec, "exec submitted; attaching");
    let code = follow_attach(ctl.clone(), &r).await?;
    if created && !a.keep_session {
        let _ = ctl.session_destroy(&sid);
    }
    Ok(code)
}

// ----- state dispatch --------------------------------------------------------

async fn state_cmd<C: MirageCtl + 'static>(
    ctl: Arc<C>,
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
            purge(&*ctl, all)?;
            println!("purged");
        }
    }
    Ok(ExitCode::from(0))
}

/// Stop every session and remove mirage's on-disk state.
fn purge<C: MirageCtl + ?Sized>(ctl: &C, all: bool) -> anyhow::Result<()> {
    // Best effort: stop every known session (this also wipes their
    // per-session directory under `mirage_runtime_dir`).
    if let Ok(ids) = ctl.session_list() {
        for id in ids {
            if let Err(e) = ctl.session_destroy(&id) {
                tracing::warn!("failed to stop session {id}: {e:#}");
            }
        }
    }

    let mut targets = vec![
        mirage_core::paths::mirage_runtime_dir(),
        mirage_core::paths::mirage_state_dir(),
    ];
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
        "state": mirage_core::paths::mirage_state_dir(),
        "profiles": mirage_core::paths::profile_root(),
        "sessions": mirage_core::paths::session_root(),
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&info).unwrap());
    } else {
        println!(
            "config:   {}",
            mirage_core::paths::mirage_config_dir().display()
        );
        println!(
            "runtime:  {}",
            mirage_core::paths::mirage_runtime_dir().display()
        );
        println!(
            "state:    {}",
            mirage_core::paths::mirage_state_dir().display()
        );
        println!("profiles: {}", mirage_core::paths::profile_root().display());
        println!("sessions: {}", mirage_core::paths::session_root().display());
    }
}

#[cfg(test)]
mod tests {
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
}
