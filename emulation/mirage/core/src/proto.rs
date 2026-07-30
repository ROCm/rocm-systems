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
    /// Ask for everything needed to start a process in this session.
    Describe,
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
    /// The container hosting `rank`, falling back to rank 0's when the
    /// topology and the container list disagree.
    #[must_use]
    pub fn name(&self, rank: u32) -> Option<&str> {
        self.names
            .get(rank as usize)
            .or_else(|| self.names.first())
            .map(String::as_str)
    }
}
