//! The wire protocol between `mirage exec` and the `mirage run` that owns
//! a session.
//!
//! # Shape
//!
//! A `mirage run` owns exactly one session, for exactly as long as that
//! process lives. While it is up it serves a Unix socket in the runtime
//! directory, named after the session. The socket answers one question:
//! *how do I start a process in this session?*
//!
//! A client connects, sends one [`Request`], reads one [`Response`], and
//! closes. Frames are length-delimited (4-byte big-endian prefix) and
//! carry JSON.
//!
//! # The connection is the lease
//!
//! There is one wrinkle on "sends one request, reads one response, and
//! closes", and it is [`Request::Attach`]: the client does not close.
//!
//! `mirage exec` borrows a session it does not own, and the run that does
//! own it must not tear the emulator daemon, the containers and the
//! scratch directory down while a borrower's workload is still using
//! them. Something therefore has to tell the run that a borrower exists,
//! and — much harder — that it has stopped existing, including when it
//! stopped by crashing.
//!
//! An explicit release message cannot do that: a borrower killed between
//! attaching and releasing would hold the session open forever. The open
//! socket can, because the kernel closes it however the client ends. So
//! the lease *is* the connection: holding it is the claim, and dropping
//! it — deliberately, or by dying — is the release. The run counts open
//! `Attach` connections and waits for the count to reach zero before
//! tearing down.
//!
//! It carries the other direction too. When the run tears down anyway —
//! the user pressed Ctrl-C rather than waiting — it closes the
//! connection, and the borrower reads that as "the session is going away"
//! and stops its own workload cleanly, instead of discovering it when the
//! container is removed out from under it.
//!
//! # Why the protocol is this small
//!
//! Because the exec'd process is not the run's child. `mirage exec` runs
//! in a different terminal from the `mirage run` that owns the session,
//! and the whole point of dropping pseudo-terminals is that a child
//! inherits the *real* terminal of whoever started it. A process spawned
//! by the run process would inherit the run's terminal, not the exec
//! client's — so the exec client spawns it itself.
//!
//! That leaves the run process with nothing to do for an exec except
//! describe the session: which containers exist, what environment the
//! emulator needs, where the rendezvous is. Everything else — spawning,
//! signalling, reaping, printing output — belongs to the process that
//! owns the terminal it is happening in.
//!
//! The previous protocol carried attach streams, stdin frames, terminal
//! resizes, exec lifecycle and a daemon handshake, because a long-lived
//! daemon owned every process and clients had to drive them at a
//! distance. None of that survives the change of ownership.

use serde::{Deserialize, Serialize};
use tokio_util::codec::LengthDelimitedCodec;

use crate::session::SessionId;

/// Largest frame the protocol will encode or accept, in bytes.
///
/// A session description is a few kilobytes at most. The cap exists so a
/// malformed length prefix cannot make either end allocate unboundedly.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// Build the length-delimited codec used on both ends of the socket.
#[must_use]
pub fn codec() -> LengthDelimitedCodec {
    LengthDelimitedCodec::builder()
        .length_field_type::<u32>()
        .max_frame_length(MAX_FRAME_BYTES)
        .new_codec()
}

/// A single client-to-run message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    /// Ask for everything needed to start a process in this session, and
    /// then let go of it.
    ///
    /// A one-shot question with a one-shot answer, for anything that only
    /// wants to *look* at a session. A caller that is about to start
    /// processes wants [`Request::Attach`] instead: the answer is
    /// identical, but this one claims nothing, so the run may tear the
    /// session down the instant after replying.
    Describe,

    /// Take a lease on the session and describe it.
    ///
    /// Answered exactly like [`Request::Describe`], but the run then
    /// holds the connection open and counts it as a borrower until the
    /// client goes away. Refused with [`Response::Error`] if the session
    /// is already tearing down. See the module documentation.
    Attach {
        /// The exec id this client will stamp into its workload's
        /// environment, so the run can recognise that workload as a live
        /// borrower's even once it has stopped descending from the
        /// client — a `setsid`, a double fork, anything daemonised.
        ///
        /// The run learns the client's *pid* from the kernel and does not
        /// have to be told; this is the one thing it cannot observe,
        /// because the id is minted client-side and the run never starts
        /// the processes carrying it.
        ///
        /// Optional so that a client with nothing to protect — anything
        /// that attaches only to hold the session open — can say so, and
        /// so an older client still parses.
        #[serde(default)]
        exec: Option<crate::exec::ExecId>,
    },
}

/// The run process's answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    /// The session description.
    Description(Box<SessionDescription>),
    /// The request could not be served. Carries the rendered message
    /// rather than a structured error: the client's only recourse is to
    /// show it, and a wire-stable error taxonomy would be a second
    /// definition of one that already exists in
    /// [`MirageError`](crate::error::MirageError).
    Error(String),
}

/// Everything `mirage exec` needs to start a process inside a session it
/// does not own.
///
/// This is a snapshot taken after the session is healthy, so the
/// container names and emulator environment in it are final. It is
/// deliberately *data*, not a handle: the client uses it to build spawn
/// specs locally, with no further round trips.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDescription {
    /// The session this describes.
    pub session: SessionId,
    /// Number of nodes in the session's topology.
    pub node_count: u32,
    /// Processes per node in the job this session was created to run,
    /// as the owning `mirage run` was asked for with `--nproc-per-node`.
    ///
    /// Part of the description because it is part of the session's
    /// *shape*, not of one command: `RANK`, `LOCAL_RANK` and
    /// `WORLD_SIZE` are numbered in this grid, and a `mirage exec` that
    /// computed them from its own process count instead handed its
    /// workload a different world from the one the run's ranks are in —
    /// while pointing it at the run's rendezvous, which is how a
    /// collective mis-forms rather than fails.
    pub nproc_per_node: u32,
    /// Default working directory for processes on the host. Meaningless
    /// inside a container, where nothing mounts it.
    pub workdir: String,
    /// Per-rank container targets, when the session is containerised.
    pub containers: Option<ContainerTargets>,
    /// The emulator's environment injection, already remapped onto the
    /// in-container mounts when the session is containerised.
    pub env: std::collections::BTreeMap<String, String>,
    /// The emulator's interposer, to be prepended to `LD_PRELOAD`.
    /// Already resolved to its in-container path when containerised.
    pub ld_preload: Option<String>,
    /// Hostname or address of rank 0, for the job's rendezvous.
    pub head_addr: String,
    /// Rendezvous port on [`SessionDescription::head_addr`].
    pub head_port: u16,
}

/// Where each rank's processes really run, for a containerised session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerTargets {
    /// Provider binary (`podman`, `docker`, or a path).
    pub provider: String,
    /// Container name per rank, indexed by rank.
    pub names: Vec<String>,
    /// Host path of the session scratch directory, bind-mounted into
    /// every node container. Wrapper pid files land here.
    pub scratch: std::path::PathBuf,
}

impl ContainerTargets {
    /// The container hosting `rank`, or `None` if there is none.
    ///
    /// No fallback. Falling back to rank 0's container when the list is
    /// shorter than the topology looked defensive and was the opposite:
    /// the caller's `None` branch raises "session has no container for
    /// node N", which is the truth, whereas the fallback silently ran
    /// rank N's workload inside rank 0's container. Two ranks then share
    /// one PID namespace while each believes it is on a different node,
    /// so their `NCCL_HOSTID`s collide, the pid files they write to the
    /// shared scratch name processes in the same namespace, and the
    /// rendezvous is quietly wrong — with nothing logged anywhere.
    #[must_use]
    pub fn name(&self, rank: u32) -> Option<&str> {
        self.names.get(rank as usize).map(String::as_str)
    }
}
