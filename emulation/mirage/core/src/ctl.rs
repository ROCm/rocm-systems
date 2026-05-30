use std::pin::Pin;

use tokio_stream::Stream;

use crate::{
    exec::{ExecDef, ExecId},
    session::SessionState,
};

pub type StreamPacketStream = Pin<Box<dyn Stream<Item = StreamPacket> + Send>>;

/// the socket used to control the mirage
pub trait MirageCtl {
    /// get the current wall time in milliseconds since the linux epoch
    fn timestamp(&self) -> u64;

    /// get the current session state
    fn session_state(&self, name: &str) -> SessionState;

    /// shut down mirage
    fn daemon_shutdown(&self);

    /// start an exec in a session
    /// returns an exec id that can be used to attach to the exec later
    fn session_exec(&self, exec: &ExecDef) -> Result<ExecId, String>;

    /// attach to a session's streams (stdout, stderr, and exit code)
    fn session_attach(&self, exec_id: &ExecId) -> Result<StreamPacketStream, String>;
}

pub enum StdStream {
    Stdin,
    Stdout,
    Stderr,
}

pub struct StreamPacket {
    pub sig: Option<u8>,
    pub stream: Option<(StdStream, Vec<u8>)>,
    pub exit_code: Option<i32>,
}
