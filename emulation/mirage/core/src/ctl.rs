//! `MirageCtl`: the control-plane API the CLI drives mirage through.
//!
//! The trait is **async** and has two implementations:
//!
//! * `mirage_supervisor::SessionManager` — the real thing, owning every
//!   session in memory inside the daemon process.
//! * `mirage_ctl::RpcClient` — a transport shim that forwards each call
//!   over the daemon's Unix socket.
//!
//! Keeping both behind one trait means the CLI, the HTTP API and the
//! in-process integration tests all exercise the same surface, and a test
//! can drive the manager directly without a daemon or a socket.
//!
//! Every method is async because the honest implementation of most of
//! them involves waiting: on a child process, on a container engine, on a
//! peer across a socket. The predecessor of this trait was synchronous
//! and implemented by polling the filesystem with `thread::sleep`, which
//! is where a good deal of mirage's latency and its stuck-forever
//! failure modes came from.

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use crate::{
    agent::AgentDef,
    error::Result,
    exec::{ExecDef, ExecId, ExecRef, ExecStatus},
    profile::ProfileDef,
    session::{SessionDef, SessionHealth, SessionId, SessionState},
    topology::TopologyDef,
};

/// A stream of [`StreamPacket`]s produced by an attach.
pub type StreamPacketStream = Pin<Box<dyn Stream<Item = StreamPacket> + Send>>;

/// A frame published while attached to an exec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamPacket {
    /// Raw output from one of the streams.
    Output {
        /// Global rank of the process that produced this output.
        node: u32,
        /// Which of the process's streams it came from.
        stream: StdStream,
        /// The bytes, exactly as read.
        data: Vec<u8>,
    },
    /// A process has exited with the given exit code.
    NodeExit {
        /// Global rank of the process that exited.
        node: u32,
        /// Its exit code (`128 + signal` when killed by a signal).
        exit_code: i32,
    },
    /// The exec as a whole has finished.
    ExecExit {
        /// Aggregate exit code across every process in the exec.
        exit_code: i32,
    },
}

/// Which standard stream a [`StreamPacket::Output`] frame came from.
///
/// stdout and stderr are distinct for an exec running on pipes, which is
/// the default. An exec running on a pseudo-terminal has only one stream
/// by construction — that is what a terminal is — so its output is all
/// reported as [`StdStream::Stdout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum StdStream {
    /// The process's standard input.
    Stdin,
    /// The process's standard output.
    Stdout,
    /// The process's standard error.
    Stderr,
}

/// Request used to create a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    /// Pre-validated id; if `None` mirage generates one.
    pub id: Option<SessionId>,
    /// Inline or by-name profile reference. Containerisation (image,
    /// mounts, provider) travels with the profile via
    /// [`crate::profile::ContainerizedDef`].
    pub profile: crate::common::MaybeRef<ProfileDef>,
    /// Working directory used as the default cwd for execs.
    pub workdir: String,
    /// Run the emulator in out-of-process daemon mode for this session
    /// instead of in-process emulation.
    pub daemon: bool,
}

/// The control-plane API the CLI talks to.
///
/// Implementations must be safe to call concurrently from many tasks:
/// the daemon serves every connected client against one shared instance.
/// No method may block the calling thread; anything that waits does so by
/// awaiting.
#[async_trait]
pub trait MirageCtl: Send + Sync {
    // ---- Profiles -------------------------------------------------------

    /// List the names of all profiles.
    async fn profile_list(&self) -> Result<Vec<String>>;
    /// Fetch one profile by name.
    async fn profile_get(&self, name: &str) -> Result<ProfileDef>;
    /// Write a profile. Overwrites any existing profile with the same name.
    async fn profile_put(&self, profile: &ProfileDef) -> Result<()>;
    /// Delete a profile by name.
    async fn profile_delete(&self, name: &str) -> Result<()>;

    // ---- Topologies -----------------------------------------------------

    /// List the names of all topologies.
    async fn topology_list(&self) -> Result<Vec<String>>;
    /// Fetch one topology by name.
    async fn topology_get(&self, name: &str) -> Result<TopologyDef>;
    /// Write a topology under `name`. Overwrites any existing one.
    async fn topology_put(&self, name: &str, topology: &TopologyDef) -> Result<()>;
    /// Delete a topology by name.
    async fn topology_delete(&self, name: &str) -> Result<()>;

    // ---- Agents ---------------------------------------------------------

    /// List the names of all agents.
    async fn agent_list(&self) -> Result<Vec<String>>;
    /// Fetch one agent by name.
    async fn agent_get(&self, name: &str) -> Result<AgentDef>;
    /// Write an agent under `name`. Overwrites any existing one.
    async fn agent_put(&self, name: &str, agent: &AgentDef) -> Result<()>;
    /// Delete an agent by name.
    async fn agent_delete(&self, name: &str) -> Result<()>;

    // ---- Sessions -------------------------------------------------------

    /// Ids of every live session, sorted.
    async fn session_list(&self) -> Result<Vec<SessionId>>;
    /// Full state (definition + health + container record) of one session.
    async fn session_state(&self, id: &SessionId) -> Result<SessionState>;
    /// Just the health snapshot of one session.
    async fn session_health(&self, id: &SessionId) -> Result<SessionHealth>;

    /// Create a session and bring it up.
    ///
    /// Unlike its predecessor this both registers the session *and*
    /// starts it: there is no separate "now go spawn the host" step for a
    /// caller to forget, and therefore no window in which a session
    /// exists but nothing is driving it. Container bring-up and emulator
    /// daemon startup happen in the background; use
    /// [`MirageCtl::session_wait_ready`] to wait for them.
    async fn session_create(&self, req: CreateSessionRequest) -> Result<SessionDef>;

    /// Wait until a session is healthy, terminally failed, or `timeout`
    /// elapses.
    ///
    /// Time spent pulling or building a container image does not count
    /// against the timeout: those are externally bounded and the timeout
    /// exists to catch a session that is stuck, not one that is slow.
    async fn session_wait_ready(
        &self,
        id: &SessionId,
        timeout: Duration,
    ) -> Result<SessionHealth>;

    /// Tear a session down: terminate every exec and its process tree,
    /// stop the emulator daemon, remove any containers and network, and
    /// drop the session. Returns once everything is actually gone.
    async fn session_destroy(&self, id: &SessionId) -> Result<()>;

    // ---- Execs ----------------------------------------------------------

    /// Ids of every exec in a session, sorted.
    async fn exec_list(&self, session: &SessionId) -> Result<Vec<ExecId>>;
    /// Current status of one exec.
    async fn exec_status(&self, r: &ExecRef) -> Result<ExecStatus>;
    /// The definition an exec was started from.
    async fn exec_get(&self, r: &ExecRef) -> Result<ExecDef>;

    /// Start an exec inside a session. Returns as soon as the processes
    /// have been spawned; use `session_attach`/`exec_status` to follow it.
    async fn session_exec(&self, exec: &ExecDef) -> Result<ExecRef>;

    /// Attach to an exec's output.
    ///
    /// The stream replays whatever the exec has produced so far and then
    /// follows it live, ending with [`StreamPacket::ExecExit`]. Attaching
    /// after an exec has already finished is therefore well defined: the
    /// caller sees the full output and the exit code, rather than an
    /// empty stream or a hang.
    async fn session_attach(&self, exec: &ExecRef) -> Result<StreamPacketStream>;

    /// Write to an exec's stdin (rank 0).
    async fn session_stdin(&self, exec: &ExecRef, data: &[u8]) -> Result<()>;

    /// Resize an exec's terminal.
    ///
    /// A program that draws to the screen — a shell's line editor, an
    /// editor, anything using curses — reads its window size from the
    /// terminal and redraws on `SIGWINCH`. Without this the remote
    /// terminal keeps whatever size it was created with and the display
    /// is wrong the moment the user resizes their window.
    ///
    /// A no-op for an exec running on pipes.
    async fn exec_resize(&self, exec: &ExecRef, rows: u16, cols: u16) -> Result<()>;

    /// Close an exec's stdin, signalling EOF to the workload.
    ///
    /// Without this a workload that reads to EOF — `cat`, `sort`, `wc`,
    /// anything in a pipeline — never finishes, because nothing would
    /// ever close the write end of its stdin pipe. `echo hi | mirage run
    /// -- cat` depends on it.
    async fn session_stdin_close(&self, exec: &ExecRef) -> Result<()>;

    /// Send a signal to every process in an exec. `sig` follows libc
    /// numbering (e.g. `SIGINT == 2`).
    async fn exec_signal(&self, exec: &ExecRef, sig: i32) -> Result<()>;

    /// Forget an exec. A still-running exec has its process tree
    /// terminated first, so a remove never leaves orphans behind.
    async fn exec_remove(&self, exec: &ExecRef) -> Result<()>;

    // ---- Daemon ---------------------------------------------------------

    /// Destroy every session and ask the daemon to exit.
    async fn daemon_shutdown(&self) -> Result<()>;
}
