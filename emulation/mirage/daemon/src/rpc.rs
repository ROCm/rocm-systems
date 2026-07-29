//! The Unix-socket control plane the CLI talks to.
//!
//! Each connection carries one operation. The client sends a
//! [`Request`], the server answers with [`Response`] frames, and the
//! connection closes. [`Request::Attach`] is the exception: it turns the
//! connection into a duplex stream, with the server pushing output frames
//! while the client may keep sending stdin and signals on the same
//! connection.
//!
//! # Ownership of the socket
//!
//! Exactly one daemon may own the control plane at a time, and that is
//! enforced with an `flock` on a lock file rather than by checking
//! whether the socket exists. The difference matters: a socket file left
//! behind by a daemon that crashed or was `SIGKILL`ed looks identical to
//! a live one, so "the file is there" cannot answer "is anyone serving
//! it". The lock can: if it is free, no daemon is running and the socket
//! is stale and safe to unlink. This is what makes an unclean daemon exit
//! recoverable without the user having to delete files by hand.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use futures::{SinkExt, StreamExt};
use mirage_core::ctl::MirageCtl;
use mirage_core::error::MirageError;
use mirage_core::proto::{PROTOCOL_VERSION, Request, Response, codec, millis};
use mirage_supervisor::SessionManager;
use tokio::net::{UnixListener, UnixStream};
use tokio_util::codec::Framed;

/// A bound control socket, together with the lock proving we own it.
#[derive(Debug)]
pub struct ControlSocket {
    listener: UnixListener,
    path: PathBuf,
    /// Inode of the socket we bound, so the watchdog can tell "still
    /// ours" from "deleted, or replaced by someone else's".
    inode: u64,
    /// Held for the lifetime of the daemon. Releasing it (which happens
    /// on drop, and on any process exit including an abnormal one) frees
    /// the `flock` and lets the next daemon take over.
    _lock: nix::fcntl::Flock<std::fs::File>,
}

/// Why binding the control socket failed.
#[derive(Debug)]
pub enum BindError {
    /// Another daemon already holds the lock.
    AlreadyRunning,
    /// The socket or lock file could not be created.
    Io(std::io::Error),
}

impl std::fmt::Display for BindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRunning => write!(f, "another mirage daemon is already running"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BindError {}

impl ControlSocket {
    /// Take the daemon lock and bind the control socket at `path`.
    ///
    /// Must be called from within a tokio runtime: the returned listener
    /// registers with the reactor.
    ///
    /// # Errors
    ///
    /// Returns [`BindError::AlreadyRunning`] if another daemon holds the
    /// lock, or [`BindError::Io`] if the socket cannot be created.
    pub fn bind(path: &Path, lock_path: &Path) -> Result<Self, BindError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(BindError::Io)?;
        }

        let lock = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path)
            .map_err(BindError::Io)?;

        // Non-blocking exclusive lock: held for as long as this file is
        // open, and released by the kernel however the process exits.
        // That last property is the point — a daemon that is `SIGKILL`ed
        // still frees the lock, so recovery needs no cleanup step.
        let lock = match nix::fcntl::Flock::lock(lock, nix::fcntl::FlockArg::LockExclusiveNonblock)
        {
            Ok(locked) => locked,
            Err((_file, nix::errno::Errno::EWOULDBLOCK)) => {
                return Err(BindError::AlreadyRunning);
            }
            Err((_file, e)) => return Err(BindError::Io(std::io::Error::from(e))),
        };

        // We hold the lock, so any socket at this path is a leftover from
        // a daemon that is no longer running. Remove it: `bind` fails
        // with EADDRINUSE against an existing file, and refusing to start
        // because of a file whose owner is provably gone would be a
        // permanent, self-inflicted outage.
        match std::fs::remove_file(path) {
            Ok(()) => tracing::info!(
                path = %path.display(),
                "removed a stale control socket left by a previous daemon"
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(BindError::Io(e)),
        }

        let listener = UnixListener::bind(path).map_err(BindError::Io)?;
        let inode = socket_inode(path).unwrap_or(0);
        tracing::info!(path = %path.display(), "control socket bound");
        Ok(Self {
            listener,
            path: path.to_path_buf(),
            inode,
            _lock: lock,
        })
    }

    /// The path the socket is bound at.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the socket we bound is still the one at our path.
    ///
    /// False once the file has been deleted or replaced. A bound Unix
    /// socket keeps working for already-connected peers when its
    /// directory entry is removed, but nothing can ever *reach* it again
    /// — so a daemon in that state is invisible while still holding
    /// every workload process it started. That is reachable in practice:
    /// deleting the runtime directory (a stale tempdir, an over-eager
    /// cleanup, `rm -rf $XDG_RUNTIME_DIR/mirage`) does exactly this.
    #[must_use]
    pub fn is_reachable(&self) -> bool {
        socket_inode(&self.path).is_some_and(|inode| inode == self.inode)
    }

    /// Resolve once the socket is no longer reachable.
    ///
    /// Polls, because there is no portable way to watch a path for
    /// deletion, and the cadence only has to be fast enough that an
    /// unreachable daemon does not linger.
    pub async fn wait_until_unreachable(&self) {
        const POLL: std::time::Duration = std::time::Duration::from_secs(10);
        loop {
            tokio::time::sleep(POLL).await;
            if !self.is_reachable() {
                return;
            }
        }
    }
}

/// Inode of the file at `path`, if it exists.
fn socket_inode(path: &Path) -> Option<u64> {
    use std::os::unix::fs::MetadataExt as _;
    std::fs::metadata(path).ok().map(|m| m.ino())
}

impl Drop for ControlSocket {
    fn drop(&mut self) {
        // Unlink on the way out so a client does not connect to a socket
        // with nothing behind it. Best-effort: an unclean exit skips this,
        // which is exactly the case the lock exists to handle.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Serve the control plane until `shutdown` resolves.
///
/// Every connection is handled in its own task. On shutdown the listener
/// stops accepting, but in-flight connections are not forcibly cut: the
/// caller tears the sessions down next, which ends every attach stream
/// naturally with an `ExecExit`, so an attached client learns what
/// happened rather than seeing its socket vanish.
pub async fn serve(
    socket: ControlSocket,
    manager: Arc<SessionManager>,
    started: Instant,
    shutdown: impl Future<Output = ()> + Send,
) {
    tokio::pin!(shutdown);
    let unreachable = socket.wait_until_unreachable();
    tokio::pin!(unreachable);
    loop {
        tokio::select! {
            accepted = socket.listener.accept() => {
                match accepted {
                    Ok((stream, _addr)) => {
                        let manager = manager.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle(stream, manager, started).await {
                                tracing::debug!("control connection ended: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        // A per-connection accept failure (EMFILE, a peer
                        // that vanished) must not take the daemon down.
                        tracing::warn!("failed to accept a control connection: {e}");
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                }
            }
            () = &mut shutdown => {
                tracing::info!("control socket shutting down");
                return;
            }
            // Our socket was deleted or replaced. No client can reach us
            // again, so staying alive would only strand the processes we
            // own. Stop, and let the caller tear the sessions down.
            () = &mut unreachable => {
                tracing::warn!(
                    path = %socket.path.display(),
                    "control socket is gone; shutting down rather than \
                     holding sessions nothing can reach"
                );
                return;
            }
        }
    }
}

/// Handle one client connection.
async fn handle(
    stream: UnixStream,
    manager: Arc<SessionManager>,
    started: Instant,
) -> std::io::Result<()> {
    let mut framed = Framed::new(stream, codec());

    while let Some(frame) = framed.next().await {
        let frame = frame?;
        let request: Request = match serde_json::from_slice(&frame) {
            Ok(r) => r,
            Err(e) => {
                send(&mut framed, &Response::from_error(&MirageError::other(
                    format!("malformed request: {e}"),
                )))
                .await?;
                return Ok(());
            }
        };

        // Attach takes over the connection for the rest of its life.
        if let Request::Attach { exec } = request {
            return attach(framed, manager, exec).await;
        }

        let response = dispatch(&manager, request, started).await;
        let shutting_down = matches!(response, Response::Ok if false);
        send(&mut framed, &response).await?;
        if shutting_down {
            return Ok(());
        }
    }
    Ok(())
}

/// Answer a single non-streaming request.
async fn dispatch(manager: &Arc<SessionManager>, request: Request, started: Instant) -> Response {
    /// Map a `Result<T>` onto a response, turning any error into the
    /// structured wire form so the client can rebuild its kind.
    macro_rules! reply {
        ($expr:expr, $ok:expr) => {
            match $expr.await {
                Ok(value) => {
                    let f: fn(_) -> Response = $ok;
                    f(value)
                }
                Err(e) => Response::from_error(&e),
            }
        };
    }

    match request {
        Request::Hello { version } => {
            if version == PROTOCOL_VERSION {
                Response::Hello {
                    version: PROTOCOL_VERSION,
                    daemon_version: env!("CARGO_PKG_VERSION").to_string(),
                }
            } else {
                // Refuse rather than guess. A CLI and a daemon from
                // different builds meet routinely — the daemon is
                // long-lived and auto-started, so upgrading mirage leaves
                // the old one running — and misinterpreting frames would
                // be far worse than a clear error.
                Response::from_error(&MirageError::daemon(format!(
                    "protocol version mismatch: client speaks v{version}, \
                     this daemon speaks v{PROTOCOL_VERSION}. \
                     Run `mirage daemon stop` and retry to restart it."
                )))
            }
        }

        Request::ProfileList => reply!(manager.profile_list(), Response::Names),
        Request::ProfileGet { name } => reply!(manager.profile_get(&name), |p| {
            Response::Profile(Box::new(p))
        }),
        Request::ProfilePut { profile } => {
            reply!(manager.profile_put(&profile), |()| Response::Ok)
        }
        Request::ProfileDelete { name } => {
            reply!(manager.profile_delete(&name), |()| Response::Ok)
        }

        Request::TopologyList => reply!(manager.topology_list(), Response::Names),
        Request::TopologyGet { name } => reply!(manager.topology_get(&name), |t| {
            Response::Topology(Box::new(t))
        }),
        Request::TopologyPut { name, topology } => {
            reply!(manager.topology_put(&name, &topology), |()| Response::Ok)
        }
        Request::TopologyDelete { name } => {
            reply!(manager.topology_delete(&name), |()| Response::Ok)
        }

        Request::AgentList => reply!(manager.agent_list(), Response::Names),
        Request::AgentGet { name } => reply!(manager.agent_get(&name), |a| {
            Response::Agent(Box::new(a))
        }),
        Request::AgentPut { name, agent } => {
            reply!(manager.agent_put(&name, &agent), |()| Response::Ok)
        }
        Request::AgentDelete { name } => {
            reply!(manager.agent_delete(&name), |()| Response::Ok)
        }

        Request::SessionList => reply!(manager.session_list(), Response::SessionIds),
        Request::SessionState { id } => reply!(manager.session_state(&id), |s| {
            Response::SessionState(Box::new(s))
        }),
        Request::SessionHealth { id } => reply!(manager.session_health(&id), |h| {
            Response::SessionHealth(Box::new(h))
        }),
        Request::SessionCreate { req } => reply!(manager.session_create(*req), |d| {
            Response::SessionDef(Box::new(d))
        }),
        Request::SessionWaitReady { id, timeout_ms } => {
            reply!(manager.session_wait_ready(&id, millis(timeout_ms)), |h| {
                Response::SessionHealth(Box::new(h))
            })
        }
        Request::SessionDestroy { id } => {
            reply!(manager.session_destroy(&id), |()| Response::Ok)
        }

        Request::ExecList { session } => reply!(manager.exec_list(&session), Response::ExecIds),
        Request::ExecStatus { exec } => reply!(manager.exec_status(&exec), |s| {
            Response::ExecStatus(Box::new(s))
        }),
        Request::ExecGet { exec } => reply!(manager.exec_get(&exec), |d| {
            Response::ExecDef(Box::new(d))
        }),
        Request::SessionExec { def } => {
            reply!(manager.session_exec(&def), Response::ExecRef)
        }
        Request::Stdin { exec, data } => {
            reply!(manager.session_stdin(&exec, &data), |()| Response::Ok)
        }
        Request::StdinClose { exec } => {
            reply!(manager.session_stdin_close(&exec), |()| Response::Ok)
        }
        Request::Resize { exec, rows, cols } => {
            reply!(manager.exec_resize(&exec, rows, cols), |()| Response::Ok)
        }
        Request::ExecSignal { exec, sig } => {
            reply!(manager.exec_signal(&exec, sig), |()| Response::Ok)
        }
        Request::ExecRemove { exec } => {
            reply!(manager.exec_remove(&exec), |()| Response::Ok)
        }

        Request::DaemonShutdown => {
            // Answered before the daemon actually exits, so the client
            // learns the request was accepted rather than inferring it
            // from a dropped connection.
            reply!(manager.daemon_shutdown(), |()| Response::Ok)
        }
        Request::Ping => Response::Ok,
        Request::DaemonStatus => Response::DaemonStatus {
            pid: std::process::id(),
            uptime_secs: started.elapsed().as_secs(),
            sessions: manager.session_count(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },

        // Handled by the caller before dispatch.
        Request::Attach { .. } => Response::from_error(&MirageError::other(
            "attach must be the first request on a connection",
        )),
    }
}

/// Stream an exec's output over this connection, while accepting stdin
/// and signal frames from the client on the same connection.
async fn attach(
    framed: Framed<UnixStream, tokio_util::codec::LengthDelimitedCodec>,
    manager: Arc<SessionManager>,
    exec: mirage_core::exec::ExecRef,
) -> std::io::Result<()> {
    let (mut sink, mut client) = framed.split();

    let mut stream = match manager.session_attach(&exec).await {
        Ok(s) => s,
        Err(e) => {
            let payload = encode(&Response::from_error(&e));
            sink.send(payload.into()).await?;
            return Ok(());
        }
    };

    loop {
        tokio::select! {
            // Output from the exec towards the client.
            packet = stream.next() => {
                let Some(packet) = packet else {
                    sink.send(encode(&Response::StreamEnd).into()).await?;
                    return Ok(());
                };
                let terminal = matches!(packet, mirage_core::ctl::StreamPacket::ExecExit { .. });
                sink.send(encode(&Response::Stream(packet)).into()).await?;
                if terminal {
                    sink.send(encode(&Response::StreamEnd).into()).await?;
                    return Ok(());
                }
            }
            // Input from the client towards the exec. Keeping this on the
            // same connection is what lets typed input and a Ctrl-C
            // arrive in order relative to the output the user is reacting
            // to, without a second round trip per keystroke.
            frame = client.next() => {
                let Some(frame) = frame else {
                    // The client disconnected. Detaching must not disturb
                    // the exec: other clients may still be attached, and
                    // the workload is not the CLI's to own.
                    return Ok(());
                };
                let frame = frame?;
                match serde_json::from_slice::<Request>(&frame) {
                    Ok(Request::Stdin { exec, data }) => {
                        if let Err(e) = manager.session_stdin(&exec, &data).await {
                            tracing::debug!("stdin on an attached connection failed: {e}");
                        }
                    }
                    Ok(Request::StdinClose { exec }) => {
                        if let Err(e) = manager.session_stdin_close(&exec).await {
                            tracing::debug!("stdin close on an attached connection failed: {e}");
                        }
                    }
                    Ok(Request::Resize { exec, rows, cols }) => {
                        if let Err(e) = manager.exec_resize(&exec, rows, cols).await {
                            tracing::debug!("resize on an attached connection failed: {e}");
                        }
                    }
                    Ok(Request::ExecSignal { exec, sig }) => {
                        if let Err(e) = manager.exec_signal(&exec, sig).await {
                            tracing::debug!("signal on an attached connection failed: {e}");
                        }
                    }
                    Ok(other) => {
                        tracing::debug!(
                            "ignoring {other:?} on an attached connection; \
                             only Stdin, StdinClose, Resize and ExecSignal \
                             are accepted"
                        );
                    }
                    Err(e) => tracing::debug!("malformed frame on an attached connection: {e}"),
                }
            }
        }
    }
}

/// Serialize a response, falling back to a plain error frame if the
/// response itself cannot be encoded.
fn encode(response: &Response) -> Vec<u8> {
    serde_json::to_vec(response).unwrap_or_else(|e| {
        // Cannot happen for these types, but emitting a valid error frame
        // beats panicking inside a connection task and dropping the
        // client's connection with no explanation.
        tracing::error!("failed to encode a response: {e}");
        serde_json::to_vec(&Response::Error {
            kind: mirage_core::proto::ErrorKind::Other,
            message: "the daemon could not encode its response".to_string(),
        })
        .unwrap_or_default()
    })
}

async fn send(
    framed: &mut Framed<UnixStream, tokio_util::codec::LengthDelimitedCodec>,
    response: &Response,
) -> std::io::Result<()> {
    framed.send(encode(response).into()).await
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[tokio::test]
    async fn binding_twice_reports_the_daemon_is_already_running() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("mirage.sock");
        let lock = dir.path().join("mirage.lock");

        let first = ControlSocket::bind(&sock, &lock).expect("first bind");
        let err = ControlSocket::bind(&sock, &lock).expect_err("second bind must fail");
        assert!(matches!(err, BindError::AlreadyRunning), "{err:?}");
        drop(first);

        // Once the first daemon is gone, the lock is free again.
        ControlSocket::bind(&sock, &lock).expect("rebind after the owner exited");
    }

    #[tokio::test]
    async fn a_stale_socket_left_by_a_dead_daemon_does_not_block_startup() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("mirage.sock");
        let lock = dir.path().join("mirage.lock");

        // Simulate a daemon that was SIGKILLed: the socket file survives,
        // but nothing holds the lock. Refusing to start here would be a
        // permanent outage that the user could only fix by deleting files.
        std::fs::write(&sock, b"").unwrap();
        assert!(sock.exists());

        let bound = ControlSocket::bind(&sock, &lock).expect("stale socket must be reclaimed");
        assert_eq!(bound.path(), sock);
    }

    #[tokio::test]
    async fn dropping_the_socket_unlinks_it() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("mirage.sock");
        let lock = dir.path().join("mirage.lock");
        {
            let _bound = ControlSocket::bind(&sock, &lock).unwrap();
            assert!(sock.exists());
        }
        assert!(
            !sock.exists(),
            "a clean exit must not leave a socket for clients to connect to"
        );
    }

    #[tokio::test]
    async fn the_socket_directory_is_created_on_demand() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a/b/c");
        let sock = nested.join("mirage.sock");
        let lock = nested.join("mirage.lock");
        // The runtime directory may not exist on a fresh machine.
        let _bound = ControlSocket::bind(&sock, &lock).expect("must create the parent directory");
        assert!(sock.exists());
    }
}
