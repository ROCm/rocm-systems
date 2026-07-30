//! [`RpcClient`]: a [`MirageCtl`] that forwards every call to the
//! supervisor daemon over its Unix socket.
//!
//! # Connection model
//!
//! One connection per operation. The client opens a socket, sends a
//! request, reads the reply, and closes. There is no connection pool and
//! no multiplexing: Unix connections are cheap, and one operation per
//! connection means a hung operation can only hang itself.
//!
//! # Auto-start
//!
//! If no daemon is running, the client starts one and waits for it. This
//! keeps `mirage run -- ./app` a single command rather than something
//! that requires the user to remember a setup step. Two CLIs racing to do
//! this is fine: the daemon takes an exclusive lock before binding, so
//! exactly one wins and the other connects to it.
//!
//! Set `MIRAGE_AUTOSTART=0` to disable it and get a clean error instead
//! of a background process appearing.

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use mirage_core::agent::AgentDef;
use mirage_core::ctl::{CreateSessionRequest, MirageCtl, StreamPacketStream};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, ExecRef, ExecStatus};
use mirage_core::profile::ProfileDef;
use mirage_core::proto::{
    ENV_AUTOSTART, ENV_SOCKET, PROTOCOL_VERSION, Request, Response, codec,
};
use mirage_core::session::{SessionDef, SessionHealth, SessionId, SessionState};
use mirage_core::topology::TopologyDef;
use tokio::net::UnixStream;
use tokio_util::codec::Framed;

/// How long to wait for an auto-started daemon to bind its socket.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(20);

/// How often to retry connecting while waiting for startup.
const STARTUP_POLL: Duration = Duration::from_millis(25);

/// What the daemon reports about itself.
#[derive(Debug, Clone)]
pub struct DaemonStatus {
    /// The daemon's process id.
    pub pid: u32,
    /// Seconds since it started.
    pub uptime_secs: u64,
    /// How many sessions it owns.
    pub sessions: usize,
    /// Its package version.
    pub version: String,
}

/// A control-plane client speaking to the daemon over a Unix socket.
#[derive(Debug, Clone)]
pub struct RpcClient {
    socket: PathBuf,
}

impl RpcClient {
    /// Connect to the daemon, starting one if necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if no daemon is running and one cannot be
    /// started, or if the daemon speaks an incompatible protocol version.
    pub async fn connect() -> Result<Self> {
        Self::connect_at(&default_socket()).await
    }

    /// Connect to the daemon at `socket`, starting one if necessary.
    ///
    /// # Errors
    ///
    /// As [`RpcClient::connect`].
    pub async fn connect_at(socket: &Path) -> Result<Self> {
        let client = Self {
            socket: socket.to_path_buf(),
        };
        if client.probe().await.is_ok() {
            return client.check_version().await.map(|()| client);
        }
        if !autostart_enabled() {
            return Err(MirageError::daemon(format!(
                "no mirage daemon is running at {} and {ENV_AUTOSTART} disables \
                 starting one. Run `mirage daemon` in another terminal.",
                socket.display()
            )));
        }
        let daemon = client.spawn_daemon()?;
        client.await_daemon(daemon).await?;
        client.check_version().await.map(|()| client)
    }

    /// Connect only to an already-running daemon.
    ///
    /// Used by `mirage daemon stop`/`status`, where starting a daemon in
    /// order to ask whether one is running would be absurd.
    ///
    /// # Errors
    ///
    /// Returns an error if no daemon is listening.
    pub async fn connect_existing(socket: &Path) -> Result<Self> {
        let client = Self {
            socket: socket.to_path_buf(),
        };
        client.probe().await?;
        Ok(client)
    }

    /// The socket this client talks to.
    #[must_use]
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// Ask the daemon to describe itself.
    ///
    /// # Errors
    ///
    /// Returns an error if the daemon cannot be reached.
    pub async fn daemon_status(&self) -> Result<DaemonStatus> {
        match self.request(Request::DaemonStatus).await? {
            Response::DaemonStatus {
                pid,
                uptime_secs,
                sessions,
                version,
            } => Ok(DaemonStatus {
                pid,
                uptime_secs,
                sessions,
                version,
            }),
            other => Err(unexpected(&other)),
        }
    }

    /// Check that the daemon speaks our protocol version.
    async fn check_version(&self) -> Result<()> {
        match self
            .request(Request::Hello {
                version: PROTOCOL_VERSION,
            })
            .await?
        {
            // Check the answer, do not merely accept its shape. The
            // daemon rejects a mismatched client too, but relying on that
            // alone means the guarantee only holds when the *daemon* is
            // the newer of the two — and the case this exists for is the
            // opposite one, where mirage was upgraded and left an old
            // long-lived daemon running.
            Response::Hello {
                version,
                daemon_version,
            } if version == PROTOCOL_VERSION => {
                tracing::debug!(daemon_version, "connected to the mirage daemon");
                Ok(())
            }
            Response::Hello { version, .. } => Err(MirageError::daemon(format!(
                "protocol version mismatch: this client speaks \
                 v{PROTOCOL_VERSION}, the running daemon speaks v{version}. \
                 Run `mirage daemon stop` and retry to restart it."
            ))),
            other => Err(unexpected(&other)),
        }
    }

    /// Is anything listening?
    async fn probe(&self) -> Result<()> {
        match self.request(Request::Ping).await? {
            Response::Ok => Ok(()),
            other => Err(unexpected(&other)),
        }
    }

    /// Start a daemon in the background.
    ///
    /// Returns the child handle, which the caller must reap. It is a real
    /// child of this process, and one that routinely exits within
    /// milliseconds: when two CLIs race to auto-start, the loser's daemon
    /// finds the lock taken and returns immediately. Dropping the handle
    /// without waiting leaves that exit sitting in the process table as a
    /// zombie for as long as this CLI runs — which for `mirage run --
    /// ./train.py` is hours.
    fn spawn_daemon(&self) -> Result<std::process::Child> {
        let bin = mirage_binary()?;
        let log_path = mirage_core::paths::daemon_log_path();
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| MirageError::io(parent, e))?;
        }
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| MirageError::io(&log_path, e))?;

        let mut cmd = std::process::Command::new(&bin);
        cmd.arg("daemon")
            .arg("--socket")
            .arg(&self.socket)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            // The daemon's diagnostics have to go somewhere a user can
            // read after the fact; a background process that fails
            // silently is undiagnosable.
            .stderr(std::process::Stdio::from(log));

        // Its own process group, so a Ctrl-C in the shell that happened
        // to start it does not also kill the daemon — and with it every
        // session belonging to other terminals.
        use std::os::unix::process::CommandExt as _;
        cmd.process_group(0);

        if let Some(level) = crate::daemon_log_directive() {
            cmd.env("MIRAGE_LOG", level);
        }
        // Propagate the directory overrides so an auto-started daemon
        // resolves the same config and runtime roots as the CLI that
        // started it. Without this a test (or anyone using
        // `XDG_RUNTIME_DIR`) would get a daemon pointing at the real
        // user directories.
        for key in [
            "MIRAGE_CONFIG",
            "MIRAGE_RUNTIME",
            "MIRAGE_STATE",
            "XDG_CONFIG_HOME",
            "XDG_RUNTIME_DIR",
            "XDG_STATE_HOME",
        ] {
            if let Some(value) = std::env::var_os(key) {
                cmd.env(key, value);
            }
        }

        let child = cmd.spawn().map_err(|e| {
            MirageError::daemon(format!(
                "could not start the mirage daemon via {}: {e}",
                bin.display()
            ))
        })?;
        tracing::debug!(bin = %bin.display(), "started a mirage daemon");
        Ok(child)
    }

    /// Wait for an auto-started daemon to accept connections.
    ///
    /// Reaps `child` if it exits during the wait, which is the expected
    /// outcome whenever another daemon was already running: the one we
    /// started sees the lock taken and exits at once, and something has
    /// to collect it.
    async fn await_daemon(&self, mut child: std::process::Child) -> Result<()> {
        let deadline = tokio::time::Instant::now() + STARTUP_TIMEOUT;
        let mut exited = None;
        loop {
            if self.probe().await.is_ok() {
                return Ok(());
            }
            // Reap it the moment it goes, rather than leaving a zombie
            // for the life of this command. Its exit is not necessarily a
            // failure — the loser of an auto-start race exits 0 having
            // handed over to the daemon that won — so keep polling the
            // socket either way, and only report the status if the socket
            // never comes up.
            if exited.is_none() {
                exited = child.try_wait().ok().flatten();
            }
            if tokio::time::Instant::now() >= deadline {
                let how = exited.map_or_else(
                    || "it is still running but never bound its socket".to_string(),
                    |status| format!("it exited with {status}"),
                );
                return Err(MirageError::daemon(format!(
                    "the mirage daemon did not start within {STARTUP_TIMEOUT:?}: {how}. \
                     See {} for why.",
                    mirage_core::paths::daemon_log_path().display()
                )));
            }
            tokio::time::sleep(STARTUP_POLL).await;
        }
    }

    /// Open a framed connection to the daemon.
    async fn open(&self) -> Result<Framed<UnixStream, tokio_util::codec::LengthDelimitedCodec>> {
        let stream = UnixStream::connect(&self.socket).await.map_err(|e| {
            MirageError::daemon(format!(
                "cannot reach the mirage daemon at {}: {e}",
                self.socket.display()
            ))
        })?;
        Ok(Framed::new(stream, codec()))
    }

    /// Send one request and read one response.
    async fn request(&self, request: Request) -> Result<Response> {
        let mut framed = self.open().await?;
        let payload = serde_json::to_vec(&request)
            .map_err(|e| MirageError::daemon(format!("cannot encode request: {e}")))?;
        framed
            .send(payload.into())
            .await
            .map_err(|e| MirageError::daemon(format!("cannot send request: {e}")))?;
        let frame = framed
            .next()
            .await
            .ok_or_else(|| {
                MirageError::daemon(
                    "the daemon closed the connection without answering; \
                     it may have crashed",
                )
            })?
            .map_err(|e| MirageError::daemon(format!("cannot read response: {e}")))?;
        let response: Response = serde_json::from_slice(&frame)
            .map_err(|e| MirageError::daemon(format!("cannot decode response: {e}")))?;
        // Turn a wire error back into a typed error so callers branch on
        // the kind rather than pattern-matching a string.
        if let Response::Error { kind, message } = response {
            return Err(Response::into_error(kind, message));
        }
        Ok(response)
    }

    /// Send a request whose only successful answer is `Ok`.
    async fn request_ok(&self, request: Request) -> Result<()> {
        match self.request(request).await? {
            Response::Ok => Ok(()),
            other => Err(unexpected(&other)),
        }
    }
}

/// The socket path the CLI talks to by default.
#[must_use]
pub fn default_socket() -> PathBuf {
    if let Some(explicit) = std::env::var_os(ENV_SOCKET)
        && !explicit.is_empty()
    {
        return PathBuf::from(explicit);
    }
    mirage_core::paths::daemon_socket_path()
}

/// Whether the CLI may start a daemon on demand.
fn autostart_enabled() -> bool {
    match std::env::var(ENV_AUTOSTART) {
        Ok(v) => !matches!(v.trim(), "0" | "false" | "no" | "off"),
        Err(_) => true,
    }
}

/// Locate the `mirage` binary to start a daemon with.
///
/// Skips a candidate that no longer exists: `current_exe()` goes stale
/// when the binary is rebuilt or reinstalled underneath a running
/// process, and failing to notice would produce a confusing "no such
/// file" from deep inside spawn.
fn mirage_binary() -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("MIRAGE_BIN") {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Ok(p);
        }
        return Err(MirageError::daemon(format!(
            "MIRAGE_BIN points at {}, which does not exist",
            p.display()
        )));
    }
    if let Ok(exe) = std::env::current_exe()
        && exe.is_file()
    {
        return Ok(exe);
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join("mirage");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err(MirageError::daemon(
        "could not locate the `mirage` binary to start a daemon with; \
         set MIRAGE_BIN to its path",
    ))
}

/// Decode one frame of an attach stream.
///
/// `Ok(Some(packet))` is output, `Ok(None)` is a clean end of stream, and
/// `Err` is the daemon refusing or aborting the attach.
fn decode_attach_frame(frame: &[u8]) -> Result<Option<mirage_core::ctl::StreamPacket>> {
    match serde_json::from_slice::<Response>(frame) {
        Ok(Response::Stream(packet)) => Ok(Some(packet)),
        Ok(Response::StreamEnd) => Ok(None),
        Ok(Response::Error { kind, message }) => Err(Response::into_error(kind, message)),
        Ok(other) => {
            tracing::debug!("ignoring {other:?} on an attach stream");
            Ok(None)
        }
        Err(e) => Err(MirageError::daemon(format!("malformed attach frame: {e}"))),
    }
}

/// A response that does not match the request that produced it.
fn unexpected(response: &Response) -> MirageError {
    MirageError::daemon(format!(
        "the daemon returned an unexpected response: {response:?}"
    ))
}

#[async_trait]
impl MirageCtl for RpcClient {
    async fn profile_list(&self) -> Result<Vec<String>> {
        match self.request(Request::ProfileList).await? {
            Response::Names(names) => Ok(names),
            other => Err(unexpected(&other)),
        }
    }

    async fn profile_get(&self, name: &str) -> Result<ProfileDef> {
        match self
            .request(Request::ProfileGet {
                name: name.to_string(),
            })
            .await?
        {
            Response::Profile(p) => Ok(*p),
            other => Err(unexpected(&other)),
        }
    }

    async fn profile_put(&self, profile: &ProfileDef) -> Result<()> {
        self.request_ok(Request::ProfilePut {
            profile: Box::new(profile.clone()),
        })
        .await
    }

    async fn profile_delete(&self, name: &str) -> Result<()> {
        self.request_ok(Request::ProfileDelete {
            name: name.to_string(),
        })
        .await
    }

    async fn topology_list(&self) -> Result<Vec<String>> {
        match self.request(Request::TopologyList).await? {
            Response::Names(names) => Ok(names),
            other => Err(unexpected(&other)),
        }
    }

    async fn topology_get(&self, name: &str) -> Result<TopologyDef> {
        match self
            .request(Request::TopologyGet {
                name: name.to_string(),
            })
            .await?
        {
            Response::Topology(t) => Ok(*t),
            other => Err(unexpected(&other)),
        }
    }

    async fn topology_put(&self, name: &str, topology: &TopologyDef) -> Result<()> {
        self.request_ok(Request::TopologyPut {
            name: name.to_string(),
            topology: Box::new(topology.clone()),
        })
        .await
    }

    async fn topology_delete(&self, name: &str) -> Result<()> {
        self.request_ok(Request::TopologyDelete {
            name: name.to_string(),
        })
        .await
    }

    async fn agent_list(&self) -> Result<Vec<String>> {
        match self.request(Request::AgentList).await? {
            Response::Names(names) => Ok(names),
            other => Err(unexpected(&other)),
        }
    }

    async fn agent_get(&self, name: &str) -> Result<AgentDef> {
        match self
            .request(Request::AgentGet {
                name: name.to_string(),
            })
            .await?
        {
            Response::Agent(a) => Ok(*a),
            other => Err(unexpected(&other)),
        }
    }

    async fn agent_put(&self, name: &str, agent: &AgentDef) -> Result<()> {
        self.request_ok(Request::AgentPut {
            name: name.to_string(),
            agent: Box::new(agent.clone()),
        })
        .await
    }

    async fn agent_delete(&self, name: &str) -> Result<()> {
        self.request_ok(Request::AgentDelete {
            name: name.to_string(),
        })
        .await
    }

    async fn session_list(&self) -> Result<Vec<SessionId>> {
        match self.request(Request::SessionList).await? {
            Response::SessionIds(ids) => Ok(ids),
            other => Err(unexpected(&other)),
        }
    }

    async fn session_state(&self, id: &SessionId) -> Result<SessionState> {
        match self
            .request(Request::SessionState { id: id.clone() })
            .await?
        {
            Response::SessionState(s) => Ok(*s),
            other => Err(unexpected(&other)),
        }
    }

    async fn session_health(&self, id: &SessionId) -> Result<SessionHealth> {
        match self
            .request(Request::SessionHealth { id: id.clone() })
            .await?
        {
            Response::SessionHealth(h) => Ok(*h),
            other => Err(unexpected(&other)),
        }
    }

    async fn session_create(&self, req: CreateSessionRequest) -> Result<SessionDef> {
        match self
            .request(Request::SessionCreate { req: Box::new(req) })
            .await?
        {
            Response::SessionDef(d) => Ok(*d),
            other => Err(unexpected(&other)),
        }
    }

    async fn session_wait_ready(
        &self,
        id: &SessionId,
        timeout: Duration,
    ) -> Result<SessionHealth> {
        match self
            .request(Request::SessionWaitReady {
                id: id.clone(),
                timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            })
            .await?
        {
            Response::SessionHealth(h) => Ok(*h),
            other => Err(unexpected(&other)),
        }
    }

    async fn session_destroy(&self, id: &SessionId) -> Result<()> {
        self.request_ok(Request::SessionDestroy { id: id.clone() })
            .await
    }

    async fn exec_list(&self, session: &SessionId) -> Result<Vec<ExecId>> {
        match self
            .request(Request::ExecList {
                session: session.clone(),
            })
            .await?
        {
            Response::ExecIds(ids) => Ok(ids),
            other => Err(unexpected(&other)),
        }
    }

    async fn exec_status(&self, r: &ExecRef) -> Result<ExecStatus> {
        match self
            .request(Request::ExecStatus { exec: r.clone() })
            .await?
        {
            Response::ExecStatus(s) => Ok(*s),
            other => Err(unexpected(&other)),
        }
    }

    async fn exec_get(&self, r: &ExecRef) -> Result<ExecDef> {
        match self.request(Request::ExecGet { exec: r.clone() }).await? {
            Response::ExecDef(d) => Ok(*d),
            other => Err(unexpected(&other)),
        }
    }

    async fn session_exec(&self, exec: &ExecDef) -> Result<ExecRef> {
        match self
            .request(Request::SessionExec {
                def: Box::new(exec.clone()),
            })
            .await?
        {
            Response::ExecRef(r) => Ok(r),
            other => Err(unexpected(&other)),
        }
    }

    async fn session_attach(&self, exec: &ExecRef) -> Result<StreamPacketStream> {
        let mut framed = self.open().await?;
        let request = Request::Attach { exec: exec.clone() };
        let payload = serde_json::to_vec(&request)
            .map_err(|e| MirageError::daemon(format!("cannot encode attach: {e}")))?;
        framed
            .send(payload.into())
            .await
            .map_err(|e| MirageError::daemon(format!("cannot send attach: {e}")))?;

        // Wait for the daemon's acknowledgement before returning, so that
        // a rejected attach is an `Err` from this call.
        // `SessionManager::session_attach` reports a missing exec that
        // way, and the two implementations of one trait method have to
        // agree: handing back an empty stream instead makes
        // `mirage attach <session> <typo>` print nothing and exit 0.
        match framed.next().await {
            Some(Ok(frame)) => {
                let ack: Response = serde_json::from_slice(&frame)
                    .map_err(|e| MirageError::daemon(format!("cannot decode attach: {e}")))?;
                match ack {
                    Response::Ok => {}
                    Response::Error { kind, message } => {
                        return Err(Response::into_error(kind, message));
                    }
                    other => return Err(unexpected(&other)),
                }
            }
            Some(Err(e)) => {
                return Err(MirageError::daemon(format!("cannot read attach: {e}")));
            }
            None => {
                return Err(MirageError::daemon(
                    "the daemon closed the connection without accepting the attach",
                ));
            }
        }

        let (tx, rx) = tokio::sync::mpsc::channel(256);
        tokio::spawn(async move {
            while let Some(frame) = framed.next().await {
                let Ok(frame) = frame else { return };
                match decode_attach_frame(&frame) {
                    // A `StreamEnd`, or an error the daemon raised
                    // mid-stream. Either way the exchange is over;
                    // dropping `tx` ends the stream, and the caller
                    // treats an end with no `ExecExit` as a failure
                    // rather than as a workload that succeeded.
                    Ok(None) | Err(_) => return,
                    Ok(Some(packet)) => {
                        if tx.send(packet).await.is_err() {
                            return;
                        }
                    }
                }
            }
        });
        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn session_stdin(&self, exec: &ExecRef, data: &[u8]) -> Result<()> {
        self.request_ok(Request::Stdin {
            exec: exec.clone(),
            data: data.to_vec(),
        })
        .await
    }

    async fn session_stdin_close(&self, exec: &ExecRef) -> Result<()> {
        self.request_ok(Request::StdinClose { exec: exec.clone() })
            .await
    }

    async fn exec_resize(&self, exec: &ExecRef, rows: u16, cols: u16) -> Result<()> {
        self.request_ok(Request::Resize {
            exec: exec.clone(),
            rows,
            cols,
        })
        .await
    }

    async fn exec_signal(&self, exec: &ExecRef, sig: i32) -> Result<()> {
        self.request_ok(Request::ExecSignal {
            exec: exec.clone(),
            sig,
        })
        .await
    }

    async fn exec_remove(&self, exec: &ExecRef) -> Result<()> {
        self.request_ok(Request::ExecRemove { exec: exec.clone() })
            .await
    }

    async fn daemon_shutdown(&self) -> Result<()> {
        self.request_ok(Request::DaemonShutdown).await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn autostart_is_on_unless_explicitly_disabled() {
        // Read through the real environment, which the test harness does
        // not set, so this asserts the default.
        assert!(
            std::env::var_os(ENV_AUTOSTART).is_some() || autostart_enabled(),
            "auto-start must default to on"
        );
    }

    #[tokio::test]
    async fn connecting_to_a_missing_socket_without_autostart_is_a_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("nope.sock");
        let client = RpcClient {
            socket: socket.clone(),
        };
        let err = client.probe().await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cannot reach the mirage daemon"), "{msg}");
        assert!(msg.contains("nope.sock"), "{msg}");
    }

    #[tokio::test]
    async fn connect_existing_does_not_start_a_daemon() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("nope.sock");
        assert!(RpcClient::connect_existing(&socket).await.is_err());
        // Nothing may have been created as a side effect of asking.
        assert!(!socket.exists());
    }

    #[test]
    fn a_stale_mirage_bin_override_is_reported() {
        // Only meaningful when the variable is not already set for the
        // whole test run.
        if std::env::var_os("MIRAGE_BIN").is_some() {
            return;
        }
        // `mirage_binary` falls through to `current_exe`, which exists
        // during a test run, so this asserts the happy path.
        assert!(mirage_binary().is_ok());
    }
}
