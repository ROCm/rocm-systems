use crate::session::SessionState;

/// the socket used to control the mirage
pub trait MirageCtl {
    /// get the current wall time in milliseconds since the linux epoch
    fn timestamp(&self) -> u64;

    /// get the current session state
    fn session_state(&self, name: &str) -> SessionState;

    /// shut down the mirage
    fn shutdown(&self);

    fn attach(&self) -> Result<(), String>;
}
