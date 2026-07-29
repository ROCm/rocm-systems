//! The wire protocol between the `mirage` CLI and the supervisor daemon.
//!
//! # Shape
//!
//! One TCP-less, connection-per-operation protocol over a Unix stream
//! socket. Frames are length-delimited (4-byte big-endian length prefix)
//! and carry JSON. A client opens a connection, sends exactly one
//! [`Request`], and reads [`Response`] frames until the exchange is over.
//!
//! Most operations are a single request and a single response. Two are
//! not:
//!
//! * [`Request::Attach`] turns the connection into a duplex channel. The
//!   server streams [`Response::Stream`] frames until the exec exits;
//!   meanwhile the client may keep sending [`Request::Stdin`] and
//!   [`Request::ExecSignal`] frames on the *same* connection, so
//!   forwarded keystrokes and a Ctrl-C reach the workload in order and
//!   without a second round trip.
//! * [`Request::DaemonShutdown`] is answered before the daemon exits, so
//!   the client learns the request was accepted rather than seeing the
//!   connection drop.
//!
//! # Why connection-per-operation
//!
//! There is no request id and no multiplexing. That is deliberate: a
//! multiplexed protocol has to define what happens when one stream stalls
//! and another needs to make progress, and mirage has no operation
//! frequent enough to justify the complexity. Unix socket connections are
//! cheap, and one connection per operation means a hung operation can
//! only ever hang itself.
//!
//! # Framing limits
//!
//! Frames are capped at [`MAX_FRAME_BYTES`]. Output is chunked well below
//! that by the supervisor's read buffer, so the cap only ever rejects a
//! malformed or hostile peer rather than legitimate traffic.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio_util::codec::LengthDelimitedCodec;

use crate::{
    agent::AgentDef,
    ctl::{CreateSessionRequest, StreamPacket},
    error::MirageError,
    exec::{ExecDef, ExecId, ExecRef, ExecStatus},
    profile::ProfileDef,
    session::{SessionDef, SessionHealth, SessionId, SessionState},
    topology::TopologyDef,
};

/// Largest frame the protocol will encode or accept, in bytes.
///
/// Generous enough for any profile, agent or output chunk mirage
/// produces, and small enough that a malformed length prefix cannot make
/// the daemon allocate unboundedly.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Protocol version, bumped on any incompatible change to [`Request`] or
/// [`Response`].
///
/// A CLI and a daemon from different builds can easily meet: the daemon
/// is long-lived and auto-started, so upgrading mirage leaves the old one
/// running. The client sends its version in [`Request::Hello`] and
/// refuses to talk to a mismatched daemon rather than misinterpreting its
/// frames.
pub const PROTOCOL_VERSION: u32 = 1;

/// Build the length-delimited codec used on both ends of the socket.
#[must_use]
pub fn codec() -> LengthDelimitedCodec {
    LengthDelimitedCodec::builder()
        .length_field_type::<u32>()
        .max_frame_length(MAX_FRAME_BYTES)
        .new_codec()
}

/// A single client-to-daemon message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    /// Version handshake. Answered with [`Response::Hello`].
    Hello {
        /// The client's [`PROTOCOL_VERSION`].
        version: u32,
    },

    // ---- Profiles -------------------------------------------------------
    /// List profile names.
    ProfileList,
    /// Fetch one profile.
    ProfileGet {
        /// Profile name.
        name: String,
    },
    /// Create or overwrite a profile.
    ProfilePut {
        /// The profile to store.
        profile: Box<ProfileDef>,
    },
    /// Delete a profile.
    ProfileDelete {
        /// Profile name.
        name: String,
    },

    // ---- Topologies -----------------------------------------------------
    /// List topology names.
    TopologyList,
    /// Fetch one topology.
    TopologyGet {
        /// Topology name.
        name: String,
    },
    /// Create or overwrite a topology.
    TopologyPut {
        /// Topology name.
        name: String,
        /// The topology to store.
        topology: Box<TopologyDef>,
    },
    /// Delete a topology.
    TopologyDelete {
        /// Topology name.
        name: String,
    },

    // ---- Agents ---------------------------------------------------------
    /// List agent names.
    AgentList,
    /// Fetch one agent.
    AgentGet {
        /// Agent name.
        name: String,
    },
    /// Create or overwrite an agent.
    AgentPut {
        /// Agent name.
        name: String,
        /// The agent to store.
        agent: Box<AgentDef>,
    },
    /// Delete an agent.
    AgentDelete {
        /// Agent name.
        name: String,
    },

    // ---- Sessions -------------------------------------------------------
    /// List live session ids.
    SessionList,
    /// Fetch a session's full state.
    SessionState {
        /// Session id.
        id: SessionId,
    },
    /// Fetch a session's health.
    SessionHealth {
        /// Session id.
        id: SessionId,
    },
    /// Create and bring up a session.
    SessionCreate {
        /// The creation request.
        req: Box<CreateSessionRequest>,
    },
    /// Wait for a session to settle or time out.
    SessionWaitReady {
        /// Session id.
        id: SessionId,
        /// How long to wait, in milliseconds.
        timeout_ms: u64,
    },
    /// Destroy a session and everything under it.
    SessionDestroy {
        /// Session id.
        id: SessionId,
    },

    // ---- Execs ----------------------------------------------------------
    /// List exec ids in a session.
    ExecList {
        /// Session id.
        session: SessionId,
    },
    /// Fetch an exec's status.
    ExecStatus {
        /// The exec.
        exec: ExecRef,
    },
    /// Fetch the definition an exec was started from.
    ExecGet {
        /// The exec.
        exec: ExecRef,
    },
    /// Start an exec.
    SessionExec {
        /// The exec definition.
        def: Box<ExecDef>,
    },
    /// Attach to an exec, upgrading this connection to a duplex stream.
    Attach {
        /// The exec.
        exec: ExecRef,
    },
    /// Write to an exec's stdin. Valid standalone or on an attached
    /// connection.
    Stdin {
        /// The exec.
        exec: ExecRef,
        /// Bytes to write.
        data: Vec<u8>,
    },
    /// Close an exec's stdin, signalling EOF to the workload. Valid
    /// standalone or on an attached connection.
    StdinClose {
        /// The exec.
        exec: ExecRef,
    },
    /// Resize an exec's terminal. Valid standalone or on an attached
    /// connection — the client sends it on `SIGWINCH`, while output is
    /// streaming, which is exactly why the attach connection is duplex.
    Resize {
        /// The exec.
        exec: ExecRef,
        /// New height in character cells.
        rows: u16,
        /// New width in character cells.
        cols: u16,
    },
    /// Signal every process in an exec. Valid standalone or on an
    /// attached connection.
    ExecSignal {
        /// The exec.
        exec: ExecRef,
        /// Signal number, in libc numbering.
        sig: i32,
    },
    /// Forget an exec, terminating it first if it is still running.
    ExecRemove {
        /// The exec.
        exec: ExecRef,
    },

    // ---- Daemon ---------------------------------------------------------
    /// Destroy every session and exit.
    DaemonShutdown,
    /// Liveness probe used when the CLI is waiting for an auto-started
    /// daemon to come up.
    Ping,
    /// Report the daemon's pid, uptime and session count.
    DaemonStatus,
}

/// A single daemon-to-client message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    /// Handshake accepted.
    Hello {
        /// The daemon's [`PROTOCOL_VERSION`].
        version: u32,
        /// The daemon's package version string.
        daemon_version: String,
    },
    /// The operation succeeded and returns nothing.
    Ok,
    /// The operation failed. Carries a structured code so the client can
    /// rebuild a typed [`MirageError`] rather than a flat string.
    Error {
        /// Machine-readable error kind.
        kind: ErrorKind,
        /// Human-readable message.
        message: String,
    },
    /// A list of names.
    Names(Vec<String>),
    /// A profile.
    Profile(Box<ProfileDef>),
    /// A topology.
    Topology(Box<TopologyDef>),
    /// An agent.
    Agent(Box<AgentDef>),
    /// A list of session ids.
    SessionIds(Vec<SessionId>),
    /// A session definition.
    SessionDef(Box<SessionDef>),
    /// A session's full state.
    SessionState(Box<SessionState>),
    /// A session's health.
    SessionHealth(Box<SessionHealth>),
    /// A list of exec ids.
    ExecIds(Vec<ExecId>),
    /// An exec's status.
    ExecStatus(Box<ExecStatus>),
    /// An exec definition.
    ExecDef(Box<ExecDef>),
    /// A reference to a newly started exec.
    ExecRef(ExecRef),
    /// One frame of an attached exec's output.
    Stream(StreamPacket),
    /// The attach stream is finished; no more [`Response::Stream`] frames
    /// will arrive on this connection.
    StreamEnd,
    /// Daemon status report.
    DaemonStatus {
        /// The daemon's process id.
        pid: u32,
        /// Seconds since the daemon started.
        uptime_secs: u64,
        /// How many sessions it currently owns.
        sessions: usize,
        /// The daemon's package version string.
        version: String,
    },
}

/// Machine-readable error kinds carried by [`Response::Error`].
///
/// The wire cannot carry a [`MirageError`] directly (it holds
/// `std::io::Error` and `serde_json::Error`, neither of which is
/// serializable), and flattening everything to a string would lose the
/// distinctions the CLI and the HTTP API map onto exit codes and status
/// codes. This enum is the serializable projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorKind {
    /// A requested profile does not exist.
    ProfileNotFound,
    /// A requested session does not exist.
    SessionNotFound,
    /// A session with that id already exists.
    SessionExists,
    /// A requested exec does not exist.
    ExecNotFound,
    /// An id failed validation.
    InvalidId,
    /// A wait timed out.
    Timeout,
    /// Filesystem or serialization failure inside the daemon.
    Io,
    /// Anything else.
    Other,
}

impl From<&MirageError> for ErrorKind {
    fn from(e: &MirageError) -> Self {
        match e {
            MirageError::ProfileNotFound(_) => Self::ProfileNotFound,
            MirageError::SessionNotFound(_) => Self::SessionNotFound,
            MirageError::SessionExists(_) => Self::SessionExists,
            MirageError::ExecNotFound(_) => Self::ExecNotFound,
            MirageError::Id(_) => Self::InvalidId,
            MirageError::Timeout(_) => Self::Timeout,
            MirageError::Io { .. } | MirageError::Json { .. } => Self::Io,
            MirageError::Daemon(_) | MirageError::Other(_) => Self::Other,
        }
    }
}

impl Response {
    /// Build an error response from a [`MirageError`].
    #[must_use]
    pub fn from_error(e: &MirageError) -> Self {
        Self::Error {
            kind: ErrorKind::from(e),
            message: e.to_string(),
        }
    }

    /// Rebuild a typed [`MirageError`] from an error response.
    ///
    /// Round-tripping is lossy by construction — an `Io` error arrives as
    /// a message, not a `std::io::Error` — but the *kind* survives, which
    /// is what callers actually branch on.
    #[must_use]
    pub fn into_error(kind: ErrorKind, message: String) -> MirageError {
        match kind {
            ErrorKind::ProfileNotFound => MirageError::ProfileNotFound(message),
            ErrorKind::SessionNotFound => MirageError::SessionNotFound(message),
            ErrorKind::SessionExists => MirageError::SessionExists(message),
            ErrorKind::ExecNotFound => MirageError::ExecNotFound(message),
            ErrorKind::Timeout => MirageError::Timeout(message),
            ErrorKind::InvalidId | ErrorKind::Io | ErrorKind::Other => MirageError::Other(message),
        }
    }
}

/// Convert a millisecond count from the wire into a [`Duration`].
#[must_use]
pub fn millis(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

/// Environment variable a client may set to point at a non-default
/// daemon socket. Used by the test suite to give every test its own
/// daemon, and by anyone running several isolated mirage instances.
pub const ENV_SOCKET: &str = "MIRAGE_SOCKET";

/// Environment variable that, when set to a falsey value, stops the CLI
/// from auto-starting a daemon. A client that only wants to talk to an
/// already-running daemon sets this and gets a clean error instead of a
/// surprise background process.
pub const ENV_AUTOSTART: &str = "MIRAGE_AUTOSTART";

/// Extra environment a spawned exec always receives, keyed by name.
pub type EnvMap = BTreeMap<String, String>;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::ctl::StdStream;

    fn round_trip_request(req: &Request) -> Request {
        let bytes = serde_json::to_vec(req).unwrap();
        assert!(bytes.len() < MAX_FRAME_BYTES);
        serde_json::from_slice(&bytes).unwrap()
    }

    fn round_trip_response(resp: &Response) -> Response {
        let bytes = serde_json::to_vec(resp).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn requests_round_trip() {
        let exec = ExecRef {
            session: SessionId::new("s1").unwrap(),
            exec: ExecId::from_counter(3),
        };
        let cases = vec![
            Request::Hello {
                version: PROTOCOL_VERSION,
            },
            Request::ProfileList,
            Request::ProfileGet {
                name: "mi350x".into(),
            },
            Request::SessionList,
            Request::Attach { exec: exec.clone() },
            Request::Stdin {
                exec: exec.clone(),
                data: vec![0, 1, 2, 255],
            },
            Request::StdinClose { exec: exec.clone() },
            Request::Resize {
                exec: exec.clone(),
                rows: 40,
                cols: 120,
            },
            Request::ExecSignal {
                exec: exec.clone(),
                sig: 15,
            },
            Request::SessionWaitReady {
                id: SessionId::new("s1").unwrap(),
                timeout_ms: 10_000,
            },
            Request::DaemonShutdown,
            Request::Ping,
        ];
        for case in &cases {
            let back = round_trip_request(case);
            assert_eq!(
                std::mem::discriminant(case),
                std::mem::discriminant(&back),
                "variant changed across a round trip: {case:?}"
            );
        }
    }

    #[test]
    fn stdin_preserves_arbitrary_bytes() {
        let exec = ExecRef {
            session: SessionId::new("s1").unwrap(),
            exec: ExecId::from_counter(0),
        };
        // Not valid UTF-8, and containing a NUL: an encoding that went
        // through a string would corrupt this.
        let data: Vec<u8> = vec![0x00, 0xff, 0xfe, 0x80, b'\n', 0x7f];
        let back = round_trip_request(&Request::Stdin {
            exec,
            data: data.clone(),
        });
        match back {
            Request::Stdin { data: got, .. } => assert_eq!(got, data),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn output_packets_round_trip_binary_data() {
        let pkt = StreamPacket::Output {
            node: 2,
            stream: StdStream::Stderr,
            data: (0u8..=255).collect(),
        };
        match round_trip_response(&Response::Stream(pkt.clone())) {
            Response::Stream(got) => assert_eq!(got, pkt),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn error_kind_survives_a_round_trip() {
        let cases = [
            MirageError::ProfileNotFound("p".into()),
            MirageError::SessionNotFound("s".into()),
            MirageError::SessionExists("s".into()),
            MirageError::ExecNotFound("e".into()),
            MirageError::Timeout("waiting".into()),
            MirageError::Other("boom".into()),
        ];
        for original in &cases {
            let wire = Response::from_error(original);
            let Response::Error { kind, message } = round_trip_response(&wire) else {
                panic!("expected an error response");
            };
            let rebuilt = Response::into_error(kind, message);
            assert_eq!(
                std::mem::discriminant(original),
                std::mem::discriminant(&rebuilt),
                "error kind lost across the wire: {original:?}"
            );
        }
    }

    #[test]
    fn io_errors_degrade_to_other_but_keep_their_message() {
        let err = MirageError::Io {
            path: "/nope".into(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        };
        let wire = Response::from_error(&err);
        let Response::Error { kind, message } = wire else {
            panic!("expected an error response");
        };
        assert_eq!(kind, ErrorKind::Io);
        assert!(message.contains("/nope"), "{message}");
    }

    #[test]
    fn codec_rejects_frames_over_the_cap() {
        let c = codec();
        // The builder is the only place the cap is configured, so assert
        // it is actually what we think it is.
        assert_eq!(c.max_frame_length(), MAX_FRAME_BYTES);
    }
}
