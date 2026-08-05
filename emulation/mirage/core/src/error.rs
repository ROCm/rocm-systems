//! Crate-wide error type.

use std::path::PathBuf;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, MirageError>;

#[derive(Debug, Error)]
pub enum MirageError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("json error on {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid id: {0}")]
    Id(#[from] crate::session::IdError),

    #[error("profile not found: {0}")]
    ProfileNotFound(String),

    #[error("session not found: {0}")]
    SessionNotFound(String),

    #[error("session already exists: {0}")]
    SessionExists(String),

    #[error("exec not found: {0}")]
    ExecNotFound(String),

    #[error("session host is not running: {0}")]
    HostNotRunning(String),

    #[error("session host did not become ready within {0:?}")]
    HostStartTimeout(std::time::Duration),

    #[error("session host failed: {0}")]
    HostFailed(String),

    #[error("{0}")]
    Other(String),
}

impl MirageError {
    pub fn other(msg: impl Into<String>) -> Self {
        MirageError::Other(msg.into())
    }
}
