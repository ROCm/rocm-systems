//! Crate-wide error type.

use std::path::PathBuf;

use thiserror::Error;

/// Result alias used throughout mirage.
pub type Result<T> = std::result::Result<T, MirageError>;

/// Everything that can go wrong in mirage's control plane.
#[derive(Debug, Error)]
pub enum MirageError {
    /// A filesystem operation failed, naming the path it was on.
    #[error("io error on {path}: {source}")]
    Io {
        /// The path being operated on.
        path: PathBuf,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// A document could not be parsed or serialized.
    #[error("json error on {path}: {source}")]
    Json {
        /// The document's path.
        path: PathBuf,
        /// The underlying serde error.
        #[source]
        source: serde_json::Error,
    },

    /// A session or exec id failed validation.
    #[error("invalid id: {0}")]
    Id(#[from] crate::session::IdError),

    /// No profile with that name exists.
    #[error("profile not found: {0}")]
    ProfileNotFound(String),

    /// No live session with that id exists.
    #[error("session not found: {0}")]
    SessionNotFound(String),

    /// A session with that id is already live.
    #[error("session already exists: {0}")]
    SessionExists(String),

    /// No exec with that id exists in the session.
    #[error("exec not found: {0}")]
    ExecNotFound(String),

    /// An operation with a deadline did not complete in time.
    #[error("timed out: {0}")]
    Timeout(String),

    /// The supervisor daemon could not be reached, started, or spoke a
    /// protocol this build does not understand.
    #[error("daemon: {0}")]
    Daemon(String),

    /// Anything not worth its own variant.
    #[error("{0}")]
    Other(String),
}

impl MirageError {
    /// Build an [`MirageError::Other`] from anything string-like.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    /// Build a [`MirageError::Daemon`] from anything string-like.
    pub fn daemon(msg: impl Into<String>) -> Self {
        Self::Daemon(msg.into())
    }

    /// Build an [`MirageError::Io`] for `path`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Whether this error means "the thing you named does not exist".
    ///
    /// Callers that clean up opportunistically (destroy a session that
    /// may already be gone, remove an exec that already removed itself)
    /// use this to tell an idempotent no-op from a real failure.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        matches!(
            self,
            Self::ProfileNotFound(_) | Self::SessionNotFound(_) | Self::ExecNotFound(_)
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn not_found_classification() {
        assert!(MirageError::SessionNotFound("s".into()).is_not_found());
        assert!(MirageError::ExecNotFound("e".into()).is_not_found());
        assert!(MirageError::ProfileNotFound("p".into()).is_not_found());
        assert!(!MirageError::Other("boom".into()).is_not_found());
        assert!(!MirageError::SessionExists("s".into()).is_not_found());
    }

    #[test]
    fn io_error_names_the_path() {
        let e = MirageError::io(
            "/tmp/x",
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        );
        let msg = e.to_string();
        assert!(msg.contains("/tmp/x"), "{msg}");
        assert!(msg.contains("denied"), "{msg}");
    }
}
