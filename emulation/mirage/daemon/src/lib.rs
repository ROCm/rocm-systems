//! `mirage_daemon`: the supervisor daemon.
//!
//! The daemon owns every mirage session in memory (see
//! [`mirage_supervisor`]) and exposes them two ways:
//!
//! * a **Unix-socket control plane** ([`rpc`]) that the `mirage` CLI
//!   speaks — always present, and how every command reaches its session;
//! * an optional **HTTP/WebSocket API** (the `http` feature) plus the
//!   bundled dashboard SPA (the `webui` feature), backed by the same
//!   manager.
//!
//! # Why a daemon at all
//!
//! Sessions outlive the command that created them: `mirage session start`
//! in one shell and `mirage exec start` in another must reach the same
//! processes. Something has to own those processes in between.
//!
//! Mirage used to make that owner a detached per-session `mirage host`
//! process, with the filesystem as the channel between it and the CLI.
//! That is the design this replaced, because a detached process nobody is
//! responsible for is exactly a process that leaks: it was reparented to
//! init, its children were signalled open-loop through pid files, and its
//! liveness had to be inferred from the mtime of a heartbeat file.
//!
//! One daemon, holding its sessions as values and its children as awaited
//! tasks, makes each of those a fact instead of an inference.
//!
//! # Lifecycle
//!
//! * **Start.** The CLI auto-starts the daemon on first use. Startup
//!   takes an `flock` before binding the socket, so two racing CLIs
//!   cannot produce two daemons; the loser simply connects to the winner.
//! * **Run.** Sessions come and go over the socket.
//! * **Idle exit.** With no sessions and no clients for
//!   [`DEFAULT_IDLE_TIMEOUT`], the daemon exits. A background process
//!   that outlives its usefulness is the thing users complain about, and
//!   restarting is cheap.
//! * **Shutdown.** `SIGTERM`, `SIGINT`, `mirage daemon stop`, the idle
//!   timeout, and losing the control socket all converge on the same
//!   path: destroy every session — which reaps every workload process —
//!   then exit. Terminating the daemon therefore terminates everything it
//!   started, rather than orphaning it.
//! * **Unreachable.** If the socket file is deleted or replaced (the
//!   runtime directory was removed out from under us), no client can ever
//!   reach this daemon again. It shuts down rather than lingering
//!   invisibly with a process tree nobody can stop.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Args, Subcommand};
use mirage_core::ctl::MirageCtl;
use mirage_supervisor::SessionManager;

pub mod rpc;
mod service;

#[cfg(feature = "http")]
mod api;
#[cfg(feature = "http")]
mod server;
#[cfg(feature = "webui")]
mod spa;
#[cfg(feature = "http")]
mod state;

pub use service::InstallArgs;

#[cfg(feature = "http")]
pub use server::{build_router, serve};
#[cfg(feature = "http")]
pub use state::AppState;

/// How long the daemon stays up with nothing to do before exiting.
///
/// Only counted while it owns no sessions *and* has no connected client,
/// so this never cuts short an idle-but-in-use session.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(600);

/// Command-line flags for `mirage daemon`.
#[derive(Args, Debug, Clone)]
pub struct DaemonArgs {
    /// Path of the control socket. Defaults to
    /// `$XDG_RUNTIME_DIR/mirage/mirage.sock`.
    #[arg(long, env = mirage_core::proto::ENV_SOCKET)]
    pub socket: Option<PathBuf>,

    /// Exit after this many seconds with no sessions and no clients.
    /// `0` disables the idle timeout.
    #[arg(long, default_value_t = DEFAULT_IDLE_TIMEOUT.as_secs())]
    pub idle_timeout: u64,

    /// Also serve the HTTP/WebSocket API on this address.
    #[cfg(feature = "http")]
    #[arg(long, env = "MIRAGE_WEBUI_ADDR")]
    pub addr: Option<std::net::SocketAddr>,

    #[command(subcommand)]
    pub command: Option<DaemonCmd>,
}

impl Default for DaemonArgs {
    fn default() -> Self {
        Self {
            socket: None,
            idle_timeout: DEFAULT_IDLE_TIMEOUT.as_secs(),
            #[cfg(feature = "http")]
            addr: None,
            command: None,
        }
    }
}

/// Subcommands of `mirage daemon`.
#[derive(Subcommand, Debug, Clone)]
pub enum DaemonCmd {
    /// Run the daemon in the foreground (the default with no subcommand).
    Serve,
    /// Ask a running daemon to destroy every session and exit.
    Stop,
    /// Report whether a daemon is running, and what it owns.
    Status,
    /// Install the daemon as a systemd user service.
    Install(InstallArgs),
}

/// Entry point for `mirage daemon`.
///
/// # Errors
///
/// Returns an error if the daemon cannot bind its socket, or if a
/// subcommand fails.
pub fn run(args: DaemonArgs) -> anyhow::Result<()> {
    match &args.command {
        Some(DaemonCmd::Install(install)) => {
            let addr = installed_addr(&args);
            service::install(addr, install)
        }
        Some(DaemonCmd::Stop) => runtime()?.block_on(stop(&args)),
        Some(DaemonCmd::Status) => runtime()?.block_on(status(&args)),
        Some(DaemonCmd::Serve) | None => runtime()?.block_on(serve_forever(args)),
    }
}

#[cfg(feature = "http")]
fn installed_addr(args: &DaemonArgs) -> Option<std::net::SocketAddr> {
    args.addr
}

#[cfg(not(feature = "http"))]
#[allow(clippy::unnecessary_wraps)]
fn installed_addr(_args: &DaemonArgs) -> Option<std::net::SocketAddr> {
    None
}

fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Runtime::new()?)
}

/// The socket path a set of args resolves to.
#[must_use]
pub fn socket_path(args: &DaemonArgs) -> PathBuf {
    args.socket
        .clone()
        .unwrap_or_else(mirage_core::paths::daemon_socket_path)
}

/// Run the daemon until it is asked to stop.
async fn serve_forever(args: DaemonArgs) -> anyhow::Result<()> {
    // Materialise the builtin agents/topologies/profiles so a fresh
    // machine has something to run.
    mirage_ctl::ensure_builtins_present();

    let socket = socket_path(&args);
    let lock = lock_path_for(&socket);
    let control = match rpc::ControlSocket::bind(&socket, &lock) {
        Ok(c) => c,
        Err(rpc::BindError::AlreadyRunning) => {
            // Not an error worth failing on: the caller wanted a daemon
            // to exist, and one does. This is the normal outcome when two
            // CLIs race to auto-start it.
            tracing::info!("a mirage daemon is already running; nothing to do");
            return Ok(());
        }
        Err(e) => return Err(anyhow::anyhow!("cannot bind {}: {e}", socket.display())),
    };

    let manager = Arc::new(SessionManager::default());
    let started = Instant::now();

    // The HTTP API, when compiled in and asked for, shares the same
    // manager: the dashboard and the CLI see one set of sessions.
    #[cfg(feature = "http")]
    let http = args.addr.map(|addr| {
        let state = Arc::new(AppState::new(manager.clone()));
        tokio::spawn(async move {
            if let Err(e) = serve(addr, build_router(state)).await {
                tracing::error!("http server stopped: {e}");
            }
        })
    });

    let idle = (args.idle_timeout > 0).then(|| Duration::from_secs(args.idle_timeout));
    let reason = {
        let manager = manager.clone();
        let shutdown = shutdown_signal(manager.clone(), idle);
        tokio::pin!(shutdown);
        let serve = rpc::serve(control, manager, started, async {
            (&mut shutdown).await;
        });
        serve.await;
        // `shutdown_signal` has already resolved by the time `serve`
        // returns; recover why for the log.
        shutdown_reason()
    };

    tracing::info!(reason, "daemon shutting down");

    // Destroy every session before exiting. This is the whole reason the
    // daemon owns them: stopping the daemon must stop the workloads, not
    // orphan them.
    manager.shutdown_all().await;

    #[cfg(feature = "http")]
    if let Some(http) = http {
        http.abort();
    }

    // Anything that raced its way into the map while `shutdown_all` was
    // running. `shutdown_all` sweeps the sessions it took itself — they
    // are out of the map by the time it finishes, so this call cannot see
    // them — and latches the shutdown flag so the window is a narrow one,
    // but a workload nobody reaped must not outlive the daemon.
    manager.kill_all_now();
    Ok(())
}

/// Records why the daemon is shutting down, for the log line.
static SHUTDOWN_REASON: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();

fn shutdown_reason() -> &'static str {
    SHUTDOWN_REASON.get().copied().unwrap_or("unknown")
}

/// Resolves when the daemon should stop: a signal, an explicit shutdown
/// request, or the idle timeout.
async fn shutdown_signal(manager: Arc<SessionManager>, idle: Option<Duration>) {
    let mut sigterm = match tokio::signal::unix::signal(
        tokio::signal::unix::SignalKind::terminate(),
    ) {
        Ok(s) => s,
        Err(e) => {
            // Without a SIGTERM handler the daemon would die abruptly on
            // shutdown and orphan its workloads. Say so loudly rather
            // than running in a state where cleanup silently cannot work.
            tracing::error!("cannot install a SIGTERM handler: {e}");
            manager.wait_for_shutdown().await;
            let _ = SHUTDOWN_REASON.set("requested");
            return;
        }
    };
    let mut sigint = match tokio::signal::unix::signal(
        tokio::signal::unix::SignalKind::interrupt(),
    ) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("cannot install a SIGINT handler: {e}");
            manager.wait_for_shutdown().await;
            let _ = SHUTDOWN_REASON.set("requested");
            return;
        }
    };

    let idle_deadline = async {
        let Some(idle) = idle else {
            // No idle timeout: wait forever.
            std::future::pending::<()>().await;
            return;
        };
        // Only count time in which the daemon owns nothing. A session
        // that exists but is doing nothing is still a session someone
        // expects to find later.
        loop {
            tokio::time::sleep(idle).await;
            if manager.session_count() == 0 {
                return;
            }
        }
    };

    tokio::select! {
        _ = sigterm.recv() => {
            let _ = SHUTDOWN_REASON.set("SIGTERM");
        }
        _ = sigint.recv() => {
            let _ = SHUTDOWN_REASON.set("SIGINT");
        }
        () = manager.wait_for_shutdown() => {
            let _ = SHUTDOWN_REASON.set("requested");
        }
        () = idle_deadline => {
            let _ = SHUTDOWN_REASON.set("idle timeout");
        }
    }
}

/// The lock file that guards a given socket path.
#[must_use]
pub fn lock_path_for(socket: &std::path::Path) -> PathBuf {
    socket.with_extension("lock")
}

/// `mirage daemon stop`.
async fn stop(args: &DaemonArgs) -> anyhow::Result<()> {
    let socket = socket_path(args);
    match mirage_ctl::rpc_client::RpcClient::connect_existing(&socket).await {
        Ok(client) => {
            client.daemon_shutdown().await?;
            println!("daemon stopping");
            Ok(())
        }
        Err(_) => {
            // Nothing to stop is a success, not a failure: the caller
            // wanted no daemon running and there is none.
            println!("no daemon running");
            Ok(())
        }
    }
}

/// `mirage daemon status`.
async fn status(args: &DaemonArgs) -> anyhow::Result<()> {
    let socket = socket_path(args);
    match mirage_ctl::rpc_client::RpcClient::connect_existing(&socket).await {
        Ok(client) => {
            let status = client.daemon_status().await?;
            println!("running   pid {}", status.pid);
            println!("version   {}", status.version);
            println!("uptime    {}s", status.uptime_secs);
            println!("sessions  {}", status.sessions);
            println!("socket    {}", socket.display());
            Ok(())
        }
        Err(_) => {
            println!("not running (socket {})", socket.display());
            Ok(())
        }
    }
}

/// Build a router without binding, for in-process integration tests.
#[cfg(feature = "http")]
pub fn router(manager: Arc<SessionManager>) -> axum::Router {
    build_router(Arc::new(AppState::new(manager)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn the_lock_sits_beside_the_socket_it_guards() {
        // A custom `--socket` must get its own lock, or two daemons on
        // two sockets would contend for one lock and the second would
        // refuse to start.
        assert_eq!(
            lock_path_for(std::path::Path::new("/run/mirage/mirage.sock")),
            PathBuf::from("/run/mirage/mirage.lock")
        );
        assert_eq!(
            lock_path_for(std::path::Path::new("/tmp/a/custom.sock")),
            PathBuf::from("/tmp/a/custom.lock")
        );
    }

    #[test]
    fn an_explicit_socket_overrides_the_default() {
        let args = DaemonArgs {
            socket: Some(PathBuf::from("/tmp/explicit.sock")),
            ..DaemonArgs::default()
        };
        assert_eq!(socket_path(&args), PathBuf::from("/tmp/explicit.sock"));
    }

    #[test]
    fn the_default_idle_timeout_is_finite() {
        // A background process with no exit condition is the thing users
        // notice and resent; the default must actually expire.
        assert!(DEFAULT_IDLE_TIMEOUT > Duration::ZERO);
        assert_eq!(DaemonArgs::default().idle_timeout, 600);
    }
}
