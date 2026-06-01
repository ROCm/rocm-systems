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

use std::io::Write;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use clap::{Args, Subcommand};
use mirage_core::common::MaybeRef;
use mirage_core::ctl::{CreateSessionRequest, MirageCtl, StdStream, StreamPacket};
use mirage_core::exec::{ExecArgs, ExecDef, ExecId, ExecRef};
use mirage_core::profile::ProfileDef;
use mirage_core::registry;
use mirage_core::session::SessionId;
use tokio_stream::StreamExt;

/// Initialize the global tracing subscriber. Honours `MIRAGE_LOG` if
/// set, otherwise uses the level implied by `-v` / `-vv`.
pub fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    let env = tracing_subscriber::EnvFilter::try_from_env("MIRAGE_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Best-effort: materialise all builtin state on disk — agents,
/// topologies, and the rocjitsu runtime assets — writing only what's
/// missing. Errors are logged, never fatal; the user can always force
/// a full rewrite with `mirage state builtins`.
///
/// Shared by the CLI ([`dispatch`]) and the daemon so both surfaces
/// auto-unpack the builtins the first time they run, instead of
/// requiring the user to invoke `mirage state builtins` by hand.
pub fn ensure_builtins_present() {
    if let Err(e) = mirage_core::agent::store::ensure_builtins(false) {
        tracing::warn!("failed to preload builtin agents: {e:#}");
    }
    if let Err(e) = mirage_core::topology::store::ensure_builtins(false) {
        tracing::warn!("failed to preload builtin topologies: {e:#}");
    }
    if let Err(e) = mirage_rocjitsu::ensure_assets(false) {
        tracing::warn!("failed to extract rocjitsu assets: {e:#}");
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
    let emulator = def.emulator.emulator.as_str();
    if registry::find(emulator).is_none() {
        let known = registry::builtins()
            .iter()
            .map(|e| e.name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("unknown emulator `{emulator}` (known: {known})"));
    }
    match emulator {
        "rocjitsu" => {
            // Make sure the runtime assets (the flatbuffer schema in
            // particular) are present so this mirrors exactly what
            // session start will do.
            let _ = mirage_rocjitsu::ensure_assets(false);
            // Building the kmd config resolves the topology + agent
            // references and checks the schema is available; any error
            // here is precisely what would otherwise surface at run
            // time, so we report it now with the profile in hand.
            mirage_rocjitsu::kmd_config(&def.emulator)
                .map(|_| ())
                .map_err(|e| format!("rocjitsu cannot use this profile: {e}"))
        }
        _ => Ok(()),
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

    /// Show or follow an exec's stdout/stderr.
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
    Create {
        name: String,
        /// Emulator name (e.g. `rocjitsu`, `noop`). Defaults to the
        /// first installed emulator (rocjitsu if present, otherwise
        /// noop).
        #[arg(long)]
        emulator: Option<String>,
        /// Agent name from `<MIRAGE_CONFIG>/agent/` (e.g. `MI300X`,
        /// `MI350X`). Defaults to `MI350X`.
        #[arg(long, default_value = "MI350X")]
        agent: String,
        /// Number of racks.
        #[arg(long, default_value_t = 1)]
        racks: u32,
        /// Nodes per rack.
        #[arg(long, default_value_t = 1)]
        nodes_per_rack: u32,
        /// GPUs per node.
        #[arg(long, default_value_t = 1)]
        gpus_per_node: u32,
        /// Optional description.
        #[arg(long)]
        description: Option<String>,
    },
    /// Interactive wizard: prompts for every field.
    Wizard {
        /// Name for the new profile.
        name: Option<String>,
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
        /// Number of racks.
        #[arg(long, default_value_t = 1)]
        racks: u32,
        /// Nodes per rack.
        #[arg(long, default_value_t = 1)]
        nodes_per_rack: u32,
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
    /// Interactive wizard for starting a session.
    Wizard,
    /// Stop a session and remove its state.
    Stop {
        id: SessionId,
        /// Don't prompt for confirmation.
        #[arg(short = 'f', long)]
        force: bool,
    },
    /// Show the per-session directory path.
    Dir { id: SessionId },
}

#[derive(Args, Debug)]
pub struct StartArgs {
    /// Profile to use (by name).
    #[arg(long)]
    pub profile: String,
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
    /// runtime, state, and cache directories. The config directory
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
    /// Profile to use.
    #[arg(long)]
    profile: String,
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
    /// Only show stderr.
    #[arg(long)]
    stderr: bool,
    /// Only show stdout.
    #[arg(long)]
    stdout: bool,
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
    // Best-effort: write any missing builtin agents/topologies and
    // extract the rocjitsu runtime assets on startup so they're always
    // available under <MIRAGE_CONFIG>/ and <MIRAGE_CACHE>/. Errors here
    // are non-fatal; the user can recover via `mirage state builtins`.
    ensure_builtins_present();
    let ctl = Arc::new(ctl);
    match cmd {
        CtlCmd::Profile(c) => profile_cmd(&*ctl, c, json),
        CtlCmd::Topology(c) => topology_cmd(&*ctl, c, json),
        CtlCmd::Agent(c) => agent_cmd(&*ctl, c, json),
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
                println!("{:<24} {:<16} {}", "NAME", "EMULATOR", "DESCRIPTION");
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
            racks,
            nodes_per_rack,
            gpus_per_node,
            description,
        } => {
            // Resolve emulator: explicit > registry default.
            let spec = match emulator.as_deref() {
                Some(n) => match registry::find(n) {
                    Some(s) => s,
                    None => anyhow::bail!(
                        "unknown emulator: {n}. Known: {}",
                        registry::builtins()
                            .iter()
                            .map(|e| e.name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                },
                None => registry::default_emulator(),
            };
            let topo = mirage_core::topology::TopologyDef {
                racks,
                nodes_per_rack,
                gpus_per_node,
                agent: MaybeRef::Ref(agent),
            };
            let p = ProfileDef {
                name: name.clone(),
                description,
                emulator: registry::make_def(spec, topo),
            };
            if let Err(e) = validate_profile(&p) {
                anyhow::bail!("cannot create profile {name}: {e}");
            }
            ctl.profile_put(&p)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&p)?);
            } else {
                println!("created profile {name}");
            }
        }
        ProfileCmd::Wizard { name } => {
            return profile_wizard(ctl, name, json);
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
            racks,
            nodes_per_rack,
            gpus_per_node,
        } => {
            let t = mirage_core::topology::TopologyDef {
                racks,
                nodes_per_rack,
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
                println!("{:<32} {:<10} {}", "ID", "HEALTHY", "STATE");
                for id in ids {
                    let h = ctl.session_health(&id).unwrap_or_default();
                    println!(
                        "{:<32} {:<10} {}",
                        id,
                        h.healthy,
                        h.state.unwrap_or_default()
                    );
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
                eprintln!("session is unhealthy: {}", h.state.unwrap_or_default());
                return Ok(ExitCode::from(2));
            }
            println!("{}", serde_json::to_string_pretty(&h)?);
        }
        SessionCmd::Start(args) => return session_start(ctl, args, json).await,
        SessionCmd::Wizard => return session_wizard(ctl, json).await,
        SessionCmd::Stop { id, force } => {
            if !force && !confirm(&format!("stop session {id}?"))? {
                return Ok(ExitCode::from(0));
            }
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

async fn session_start<C: MirageCtl>(
    ctl: &C,
    args: StartArgs,
    json: bool,
) -> anyhow::Result<ExitCode> {
    // Validate profile exists.
    ctl.profile_get(&args.profile)?;
    let def = ctl.session_create(CreateSessionRequest {
        id: args.id,
        profile: MaybeRef::Ref(args.profile.clone()),
        workdir: args.workdir.unwrap_or_else(|| {
            std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or("/".to_string())
        }),
        container: None,
    })?;
    if !args.no_host {
        spawn_host_for(&def.id)?;
        ctl.session_wait_ready(&def.id, Duration::from_secs(args.ready_timeout))?;
    }
    if json {
        let s = ctl.session_state(&def.id)?;
        println!("{}", serde_json::to_string_pretty(&s)?);
    } else {
        println!("{}", def.id);
    }
    Ok(ExitCode::from(0))
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
    if let Ok(exe) = std::env::current_exe() {
        if exe.is_file() {
            return Ok(exe);
        }
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
    // ensure host.log file exists for stderr redirect
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(layout.host_log())?;
    // spawn and detach via setsid()
    let mut cmd = std::process::Command::new(bin);
    cmd.arg("host")
        .arg("--session")
        .arg(id.as_str())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(log));
    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(|| {
            nix::unistd::setsid().ok();
            Ok(())
        });
    }
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
                println!("{:<14} {:<8} {:<8} {}", "EXEC", "STARTED", "ENDED", "EXIT");
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
        keep: a.keep || !a.detach,
    };
    let r = ctl.session_exec(&def)?;
    if a.detach {
        println!("{}", r.exec);
        return Ok(ExitCode::from(0));
    }
    let code = follow_attach(ctl.as_ref(), &r).await?;
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

async fn attach_cmd<C: MirageCtl>(ctl: Arc<C>, a: AttachArgs) -> anyhow::Result<ExitCode> {
    let r = ExecRef {
        session: a.session,
        exec: a.exec,
    };
    follow_attach(ctl.as_ref(), &r).await
}

async fn follow_attach<C: MirageCtl>(ctl: &C, r: &ExecRef) -> anyhow::Result<ExitCode> {
    let mut s = ctl.session_attach(r)?;
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

async fn logs_cmd<C: MirageCtl>(ctl: Arc<C>, a: LogsArgs) -> anyhow::Result<ExitCode> {
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
            if !a.stderr
                && let Ok(b) = std::fs::read(nl.stdout())
            {
                let _ = std::io::stdout().write_all(&b);
            }
            if !a.stdout
                && let Ok(b) = std::fs::read(nl.stderr())
            {
                let _ = std::io::stderr().write_all(&b);
            }
        }
        return Ok(ExitCode::from(0));
    }
    let r = ExecRef {
        session: a.session,
        exec: a.exec,
    };
    follow_attach(ctl.as_ref(), &r).await?;
    Ok(ExitCode::from(0))
}

async fn run_cmd<C: MirageCtl + 'static>(ctl: Arc<C>, a: RunArgs) -> anyhow::Result<ExitCode> {
    // Find or create the session.
    let (sid, created) = match a.session {
        Some(id) => (id, false),
        None => {
            // create transient session
            ctl.profile_get(&a.profile)?;
            let def = ctl.session_create(CreateSessionRequest {
                id: None,
                profile: MaybeRef::Ref(a.profile.clone()),
                workdir: a.workdir.clone().unwrap_or_else(|| {
                    std::env::current_dir()
                        .map(|p| p.display().to_string())
                        .unwrap_or("/".to_string())
                }),
                container: None,
            })?;
            spawn_host_for(&def.id)?;
            ctl.session_wait_ready(&def.id, Duration::from_secs(10))?;
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
        // keep until after attach drains; we may still destroy the
        // whole session below.
        keep: true,
    };
    let r = ctl.session_exec(&def)?;
    let code = follow_attach(ctl.as_ref(), &r).await?;
    if created && !a.keep_session {
        let _ = ctl.session_destroy(&sid);
    }
    Ok(code)
}

// ----- wizards ---------------------------------------------------------------

/// Interactive `profile wizard` command.
///
/// Prompts the user for every field, defaulting to the registry's
/// recommended emulator (rocjitsu if installed, else noop). The
/// resulting profile is persisted via the standard `profile_put`
/// path so it's indistinguishable from a non-wizard creation.
fn profile_wizard(
    ctl: &dyn MirageCtl,
    suggested_name: Option<String>,
    json: bool,
) -> anyhow::Result<ExitCode> {
    use dialoguer::{Confirm, Input, Select};

    let theme = dialoguer::theme::ColorfulTheme::default();

    let name: String = Input::with_theme(&theme)
        .with_prompt("Profile name")
        .with_initial_text(suggested_name.unwrap_or_default())
        .validate_with(|s: &String| -> Result<(), &str> {
            if s.trim().is_empty() {
                Err("name required")
            } else {
                Ok(())
            }
        })
        .interact_text()?;

    let specs = registry::builtins();
    let default_idx = specs
        .iter()
        .position(|s| s.name == registry::default_emulator().name)
        .unwrap_or(0);
    let labels: Vec<String> = specs
        .iter()
        .map(|s| {
            let installed = if (s.installed)() {
                "[installed]"
            } else {
                "[not installed]"
            };
            format!("{:<10} {installed}  {}", s.name, s.description)
        })
        .collect();
    let pick = Select::with_theme(&theme)
        .with_prompt("Emulator")
        .items(&labels)
        .default(default_idx)
        .interact()?;
    let spec = &specs[pick];

    let nodes: u32 = Input::with_theme(&theme)
        .with_prompt("Nodes per rack")
        .default(1)
        .interact_text()?;
    let racks: u32 = Input::with_theme(&theme)
        .with_prompt("Number of racks")
        .default(1)
        .interact_text()?;
    let gpus_per_node: u32 = Input::with_theme(&theme)
        .with_prompt("GPUs per node")
        .default(1)
        .interact_text()?;
    let known_agents = mirage_core::agent::store::list().unwrap_or_default();
    let agent: String = if known_agents.is_empty() {
        "MI350X".to_string()
    } else {
        let default_idx = known_agents
            .iter()
            .position(|n| n == "MI350X")
            .unwrap_or(0);
        let pick = Select::with_theme(&theme)
            .with_prompt("Agent")
            .items(&known_agents)
            .default(default_idx)
            .interact()?;
        known_agents[pick].clone()
    };
    let description: String = Input::with_theme(&theme)
        .with_prompt("Description (optional)")
        .allow_empty(true)
        .interact_text()?;

    let proceed = Confirm::with_theme(&theme)
        .with_prompt(format!(
            "Create profile {name} using {} ({racks} rack(s) x {nodes} node(s) x {gpus_per_node} GPU(s), agent={agent})?",
            spec.name
        ))
        .default(true)
        .interact()?;
    if !proceed {
        eprintln!("aborted");
        return Ok(ExitCode::from(1));
    }

    let topo = mirage_core::topology::TopologyDef {
        racks,
        nodes_per_rack: nodes,
        gpus_per_node,
        agent: MaybeRef::Ref(agent),
    };
    let p = ProfileDef {
        name: name.clone(),
        description: if description.is_empty() {
            None
        } else {
            Some(description)
        },
        emulator: registry::make_def(spec, topo),
    };
    if let Err(e) = validate_profile(&p) {
        anyhow::bail!("cannot create profile {name}: {e}");
    }
    ctl.profile_put(&p)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&p)?);
    } else {
        println!("created profile {name}");
    }
    Ok(ExitCode::from(0))
}

/// Interactive `session wizard` command.
///
/// Prompts for the profile (from `profile_list`), an optional id,
/// the working directory, and a ready-timeout, then delegates to
/// the standard `session_start` path so the host is spawned and the
/// session becomes ready before the wizard returns.
async fn session_wizard<C: MirageCtl>(ctl: &C, json: bool) -> anyhow::Result<ExitCode> {
    use dialoguer::{Confirm, Input, Select};

    let theme = dialoguer::theme::ColorfulTheme::default();

    let profiles = ctl.profile_list()?;
    if profiles.is_empty() {
        anyhow::bail!("no profiles found; run `mirage profile wizard` first");
    }
    let pick = Select::with_theme(&theme)
        .with_prompt("Profile")
        .items(&profiles)
        .default(0)
        .interact()?;
    let profile = profiles[pick].clone();

    let id_raw: String = Input::with_theme(&theme)
        .with_prompt("Session id (blank for auto)")
        .allow_empty(true)
        .interact_text()?;
    let id = if id_raw.trim().is_empty() {
        None
    } else {
        Some(SessionId::new(id_raw)?)
    };

    let workdir: String = Input::with_theme(&theme)
        .with_prompt("Working directory")
        .with_initial_text(
            std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "/".into()),
        )
        .interact_text()?;

    let ready_timeout: u64 = Input::with_theme(&theme)
        .with_prompt("Host ready timeout (seconds)")
        .default(10)
        .interact_text()?;

    let go = Confirm::with_theme(&theme)
        .with_prompt(format!("Start session using profile {profile}?"))
        .default(true)
        .interact()?;
    if !go {
        eprintln!("aborted");
        return Ok(ExitCode::from(1));
    }

    session_start(
        ctl,
        StartArgs {
            profile,
            id,
            workdir: Some(workdir),
            no_host: false,
            ready_timeout,
        },
        json,
    )
    .await
}

// ----- state dispatch --------------------------------------------------------

fn rocjitsu_asset_path(name: &str) -> std::path::PathBuf {
    match name {
        n if n == mirage_rocjitsu::KMD_LIB_NAME => mirage_rocjitsu::kmd_lib_path(),
        n if n == mirage_rocjitsu::LIB_NAME => mirage_rocjitsu::lib_path(),
        _ => mirage_rocjitsu::schema_fbs_path(),
    }
}

async fn state_cmd<C: MirageCtl + 'static>(
    ctl: Arc<C>,
    cmd: StateCmd,
    json: bool,
) -> anyhow::Result<ExitCode> {
    match cmd {
        StateCmd::Builtins => {
            let agents = mirage_core::agent::store::ensure_builtins(true)?;
            let topologies = mirage_core::topology::store::ensure_builtins(true)?;
            let assets = mirage_rocjitsu::ensure_assets(true)?;
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
                    .chain(assets.iter().map(|(n, w)| {
                        let p = rocjitsu_asset_path(n);
                        serde_json::json!({
                            "kind": "asset",
                            "name": n,
                            "path": p,
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
                for (name, w) in &assets {
                    let p = rocjitsu_asset_path(name);
                    let tag = if *w { "wrote" } else { "kept" };
                    println!("{tag} asset     {} -> {}", name, p.display());
                }
            }
        }
        StateCmd::Purge { force, all } => {
            let prompt = if all {
                "purge ALL mirage state, including profiles and topologies?"
            } else {
                "purge all mirage runtime/state/cache and stop all sessions?"
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
        mirage_core::paths::mirage_cache_dir(),
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
        "cache": mirage_core::paths::mirage_cache_dir(),
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
        println!(
            "cache:    {}",
            mirage_core::paths::mirage_cache_dir().display()
        );
        println!("profiles: {}", mirage_core::paths::profile_root().display());
        println!("sessions: {}", mirage_core::paths::session_root().display());
    }
}
