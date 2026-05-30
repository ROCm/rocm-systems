//! `mirage_cli`: the user-facing CLI implementation.
//!
//! The CLI is intentionally thin: it parses arguments with `clap`,
//! delegates all operations to a `mirage_core::ctl::MirageCtl`
//! implementation (by default `FileCtl`), and renders the result for
//! humans (table/text) or machines (`--json`).
//!
//! All commands are documented in `docs/cli.md`. The entry point is
//! [`main`], which is invoked by the `mirage` binary in the workspace
//! root.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, CommandFactory, Parser, Subcommand};
use mirage_core::common::MaybeRef;
use mirage_core::ctl::{CreateSessionRequest, FileCtl, MirageCtl, StdStream, StreamPacket};
use mirage_core::emulator::{EmulatorDef, ExecMode};
use mirage_core::exec::{ExecArgs, ExecDef, ExecId, ExecRef};
use mirage_core::profile::ProfileDef;
use mirage_core::session::SessionId;
use mirage_core::topology::TopologyDef;
use tokio_stream::StreamExt;

/// Re-export so the root binary doesn't have to depend on `clap`.
pub fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    let ctl = FileCtl::new();
    let json = cli.json;
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move { dispatch(cli.command, ctl, json).await })
}

fn init_logging(verbose: u8) {
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

// =============================================================================
// Top-level CLI definition
// =============================================================================

/// `mirage` — a UX for the rocjitsu (and other) GPU emulators.
///
/// Mirage stores all its state on disk under your XDG directories:
///
/// * profiles in `$XDG_CONFIG_HOME/mirage/profile/<name>.json`
/// * sessions in `$XDG_RUNTIME_DIR/mirage/session/<id>/`
///
/// A typical flow is:
///
/// 1. Define a profile: `mirage profile create my-cdna3 --emulator rocjitsu`
/// 2. Start a session:  `mirage session start --profile my-cdna3`
/// 3. Run a command:    `mirage exec <session> -- ./my-app --flag`
/// 4. Or all-in-one:    `mirage run --profile my-cdna3 -- ./my-app --flag`
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
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Manage profiles (reusable emulator presets).
    #[command(subcommand)]
    Profile(ProfileCmd),

    /// Manage sessions.
    #[command(subcommand)]
    Session(SessionCmd),

    /// Manage execs inside a session.
    #[command(subcommand)]
    Exec(ExecCmd),

    /// Convenience: create session, run a command, attach, clean up.
    Run(RunArgs),

    /// Re-attach to a running exec's streams.
    Attach(AttachArgs),

    /// Show or follow an exec's stdout/stderr.
    Logs(LogsArgs),

    /// Print where mirage stores its state on this machine.
    Paths,

    /// Print the JSON schema of a definition (`profile`, `session`,
    /// `exec`, `status`).
    Schema { what: String },
}

// ----- profile ---------------------------------------------------------------

#[derive(Subcommand, Debug)]
enum ProfileCmd {
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
        /// Emulator name (e.g. `rocjitsu`, `noop`).
        #[arg(long, default_value = "noop")]
        emulator: String,
        /// Number of nodes.
        #[arg(long, default_value_t = 1)]
        nodes: u32,
        /// GPUs per node.
        #[arg(long, default_value_t = 1)]
        gpus_per_node: u32,
        /// Optional description.
        #[arg(long)]
        description: Option<String>,
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

// ----- session ---------------------------------------------------------------

#[derive(Subcommand, Debug)]
enum SessionCmd {
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
struct StartArgs {
    /// Profile to use (by name).
    #[arg(long)]
    profile: String,
    /// Explicit session id; auto-generated if omitted.
    #[arg(long)]
    id: Option<SessionId>,
    /// Working directory for execs in the session.
    #[arg(long)]
    workdir: Option<String>,
    /// Don't spawn the host (the caller is expected to start it).
    #[arg(long)]
    no_host: bool,
    /// Override the `mirage-host` binary path.
    #[arg(long, env = "MIRAGE_HOST_BIN")]
    host_bin: Option<PathBuf>,
    /// How long to wait for the host to report ready (seconds).
    #[arg(long, default_value_t = 10)]
    ready_timeout: u64,
}

// ----- exec ------------------------------------------------------------------

#[derive(Subcommand, Debug)]
enum ExecCmd {
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
struct ExecStartArgs {
    session: SessionId,
    /// Keep the exec on disk after it finishes.
    #[arg(long)]
    keep: bool,
    /// Don't attach to the exec; just submit and return its id.
    #[arg(long)]
    detach: bool,
    /// The command and its arguments. Use `--` to separate from
    /// mirage flags.
    #[arg(trailing_var_arg = true, required = true, allow_hyphen_values = true)]
    argv: Vec<String>,
}

// ----- run -------------------------------------------------------------------

#[derive(Args, Debug)]
struct RunArgs {
    /// Profile to use.
    #[arg(long)]
    profile: String,
    /// Reuse an existing session by id.
    #[arg(long, conflicts_with_all = ["keep_session"])]
    session: Option<SessionId>,
    /// Keep the session running after the exec finishes.
    #[arg(long)]
    keep_session: bool,
    /// Override the `mirage-host` binary path.
    #[arg(long, env = "MIRAGE_HOST_BIN")]
    host_bin: Option<PathBuf>,
    /// Working directory.
    #[arg(long)]
    workdir: Option<String>,
    /// The command and its arguments.
    #[arg(trailing_var_arg = true, required = true, allow_hyphen_values = true)]
    argv: Vec<String>,
}

// ----- attach / logs ---------------------------------------------------------

#[derive(Args, Debug)]
struct AttachArgs {
    session: SessionId,
    exec: ExecId,
}

#[derive(Args, Debug)]
struct LogsArgs {
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

async fn dispatch<C: MirageCtl + 'static>(
    cmd: Cmd,
    ctl: C,
    json: bool,
) -> anyhow::Result<ExitCode> {
    let ctl = Arc::new(ctl);
    match cmd {
        Cmd::Profile(c) => profile_cmd(&*ctl, c, json),
        Cmd::Session(c) => session_cmd(&*ctl, c, json).await,
        Cmd::Exec(c) => exec_cmd(ctl.clone(), c, json).await,
        Cmd::Run(a) => run_cmd(ctl.clone(), a).await,
        Cmd::Attach(a) => attach_cmd(ctl.clone(), a).await,
        Cmd::Logs(a) => logs_cmd(ctl.clone(), a).await,
        Cmd::Paths => {
            print_paths(json);
            Ok(ExitCode::from(0))
        }
        Cmd::Schema { what } => {
            print_schema(&what);
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
            nodes,
            gpus_per_node,
            description,
        } => {
            let p = ProfileDef {
                name: name.clone(),
                description,
                emulator: EmulatorDef {
                    emulator,
                    plugins: Default::default(),
                    nodes,
                    gpus_per_node,
                    exec_mode: ExecMode::default(),
                    options: Default::default(),
                    topology: MaybeRef::Owned(TopologyDef::default()),
                },
            };
            ctl.profile_put(&p)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&p)?);
            } else {
                println!("created profile {name}");
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
        spawn_host_for(&def.id, args.host_bin.as_deref())?;
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

fn spawn_host_for(id: &SessionId, override_bin: Option<&std::path::Path>) -> anyhow::Result<()> {
    let bin = match override_bin {
        Some(b) => b.to_path_buf(),
        None => find_host_bin()?,
    };
    let layout = mirage_core::paths::SessionLayout::for_id(id);
    // ensure host.log file exists for stderr redirect
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(layout.host_log())?;
    // double-fork-ish: just spawn and detach.
    let mut cmd = std::process::Command::new(bin);
    cmd.arg("--session")
        .arg(id.as_str())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(log));
    // detach: new session leader via setsid()
    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(|| {
            nix::unistd::setsid().ok();
            Ok(())
        });
    }
    cmd.spawn()?;
    Ok(())
}

fn find_host_bin() -> anyhow::Result<PathBuf> {
    if let Ok(p) = std::env::var("MIRAGE_HOST_BIN") {
        return Ok(PathBuf::from(p));
    }
    // Try alongside this binary.
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let candidate = dir.join("mirage-host");
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    // Fall back to PATH.
    Ok(PathBuf::from("mirage-host"))
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
    let def = ExecDef {
        timestamp: chrono::Utc::now(),
        session: a.session.clone(),
        exec: ExecArgs {
            command: cmd,
            args,
            env: Default::default(),
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
            spawn_host_for(&def.id, a.host_bin.as_deref())?;
            ctl.session_wait_ready(&def.id, Duration::from_secs(10))?;
            (def.id, true)
        }
    };
    let (cmd, args) = split_argv(&a.argv);
    let def = ExecDef {
        timestamp: chrono::Utc::now(),
        session: sid.clone(),
        exec: ExecArgs {
            command: cmd,
            args,
            env: Default::default(),
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
        "config": mirage_core::paths::xdg_config_home(),
        "runtime": mirage_core::paths::xdg_runtime_dir(),
        "state": mirage_core::paths::xdg_state_home(),
        "profiles": mirage_core::paths::profile_root(),
        "sessions": mirage_core::paths::session_root(),
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&info).unwrap());
    } else {
        println!(
            "config:   {}",
            mirage_core::paths::xdg_config_home().display()
        );
        println!(
            "runtime:  {}",
            mirage_core::paths::xdg_runtime_dir().display()
        );
        println!(
            "state:    {}",
            mirage_core::paths::xdg_state_home().display()
        );
        println!("profiles: {}", mirage_core::paths::profile_root().display());
        println!("sessions: {}", mirage_core::paths::session_root().display());
    }
}

fn print_schema(what: &str) {
    // We don't pull in a JSON-schema crate; instead, emit an example.
    let example = match what {
        "profile" => serde_json::to_string_pretty(&ProfileDef {
            name: "example".to_string(),
            description: Some("a single-node noop emulator".to_string()),
            emulator: EmulatorDef {
                emulator: "noop".to_string(),
                plugins: Default::default(),
                nodes: 1,
                gpus_per_node: 1,
                exec_mode: ExecMode::default(),
                options: Default::default(),
                topology: MaybeRef::Owned(TopologyDef::default()),
            },
        })
        .unwrap(),
        "exec" => serde_json::to_string_pretty(&ExecDef {
            timestamp: chrono::Utc::now(),
            session: SessionId::new("example").unwrap(),
            exec: ExecArgs {
                command: "/bin/sh".to_string(),
                args: vec!["-c".into(), "echo hi".into()],
                env: Default::default(),
                workdir: None,
            },
            worker_exec: None,
            keep: false,
        })
        .unwrap(),
        "status" => {
            serde_json::to_string_pretty(&mirage_core::exec::ExecStatus::default()).unwrap()
        }
        "session" => serde_json::to_string_pretty(&serde_json::json!({
            "id": "example",
            "profile": "example",
            "workdir": "/tmp",
            "created_at": chrono::Utc::now(),
        }))
        .unwrap(),
        _ => {
            eprintln!("unknown schema: {what}. try: profile, session, exec, status");
            return;
        }
    };
    println!("{example}");
}

/// Print the long-form help (used by `mirage --help`).
pub fn long_help() -> String {
    let mut s = Vec::new();
    let _ = Cli::command().write_long_help(&mut s);
    String::from_utf8(s).unwrap_or_default()
}
