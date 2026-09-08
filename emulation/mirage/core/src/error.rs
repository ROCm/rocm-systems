//! Crate-wide error type.

use std::path::PathBuf;

use thiserror::Error;

/// Result alias used throughout mirage.
pub type Result<T> = std::result::Result<T, MirageError>;

/// Everything that can go wrong in mirage's control plane.
///
/// Variants that wrap another error name it with `#[source]` and leave it
/// out of their own message. `main` prints failures with anyhow's
/// alternate form (`{:#}`), which walks the source chain and joins it with
/// `": "` — so a message that also interpolated `{source}` printed the
/// underlying reason twice (`io error on /x: denied: denied`). Callers
/// that need the whole chain flattened into one string, rather than a
/// formatter that walks it, use [`MirageError::full_message`].
#[derive(Debug, Error)]
pub enum MirageError {
    /// A filesystem operation failed, naming the path it was on.
    #[error("io error on {path}")]
    Io {
        /// The path being operated on.
        path: PathBuf,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// A document could not be parsed or serialized.
    #[error("json error on {path}")]
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
    #[error("profile not found: {name} (mirage looked in {dir})")]
    ProfileNotFound {
        /// The name that was asked for.
        name: String,
        /// The directory profiles live in on this machine.
        dir: PathBuf,
    },

    /// No topology with that name exists.
    #[error("topology not found: {name} (mirage looked in {dir})")]
    TopologyNotFound {
        /// The name that was asked for.
        name: String,
        /// The directory topologies live in on this machine.
        dir: PathBuf,
    },

    /// No agent with that name exists.
    #[error("agent not found: {name} (mirage looked in {dir})")]
    AgentNotFound {
        /// The name that was asked for.
        name: String,
        /// The directory agents live in on this machine.
        dir: PathBuf,
    },

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

    // No `Daemon` variant. There is no supervisor daemon to fail to
    // reach: a `mirage run` holds its session in its own address space,
    // and the one thing that does speak over a socket — `mirage exec` —
    // reports what it could not reach in its own words. The variant was
    // constructed by nobody and described a process that does not exist.
    /// Anything not worth its own variant.
    #[error("{0}")]
    Other(String),
}

impl MirageError {
    /// Build an [`MirageError::Other`] from anything string-like.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    /// Report a document mirage could not find, naming the directory it
    /// looked in.
    ///
    /// "Where did mirage look?" is the next question every one of these
    /// raises, and it is not a question the reader can answer for
    /// themselves: both `MIRAGE_CONFIG` and `XDG_CONFIG_HOME` move the
    /// config directory, and a user staring at a name they know they
    /// created is usually editing a different one. The directory belongs
    /// in the error that knows it rather than in a sentence one call site
    /// remembers to add.
    #[must_use]
    pub fn not_found(kind: crate::store::DocKind, name: impl Into<String>) -> Self {
        let name = name.into();
        let dir = kind.root();
        match kind {
            crate::store::DocKind::Profile => Self::ProfileNotFound { name, dir },
            crate::store::DocKind::Topology => Self::TopologyNotFound { name, dir },
            crate::store::DocKind::Agent => Self::AgentNotFound { name, dir },
        }
    }

    /// Build an [`MirageError::Io`] for `path`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// This error and every error underneath it, joined with `": "`.
    ///
    /// The same text anyhow's `{:#}` produces, for the callers that have
    /// to hand a plain `String` to something else — the supervisor's RPC
    /// replies, say, where the peer gets one string and no chain to walk.
    /// Prefer `{:#}` (or letting the error propagate to `main`) wherever a
    /// formatter will do.
    #[must_use]
    pub fn full_message(&self) -> String {
        let mut out = self.to_string();
        let mut source = std::error::Error::source(self);
        while let Some(e) = source {
            out.push_str(": ");
            out.push_str(&e.to_string());
            source = e.source();
        }
        out
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
            Self::ProfileNotFound { .. }
                | Self::TopologyNotFound { .. }
                | Self::AgentNotFound { .. }
                | Self::SessionNotFound(_)
                | Self::ExecNotFound(_)
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    use crate::store::DocKind;

    #[test]
    fn not_found_classification() {
        assert!(MirageError::SessionNotFound("s".into()).is_not_found());
        assert!(MirageError::ExecNotFound("e".into()).is_not_found());
        assert!(MirageError::not_found(DocKind::Profile, "p").is_not_found());
        assert!(!MirageError::Other("boom".into()).is_not_found());
        assert!(!MirageError::SessionExists("s".into()).is_not_found());
    }

    #[test]
    fn a_missing_document_says_where_mirage_looked() {
        // The next question a "not found" raises is which directory was
        // read, and it is not one the reader can answer: MIRAGE_CONFIG and
        // XDG_CONFIG_HOME both move it. One call site used to append the
        // answer by hand, so every other command left it unsaid.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        for (kind, root) in [
            (DocKind::Profile, crate::paths::profile_root()),
            (DocKind::Topology, crate::paths::topology_root()),
            (DocKind::Agent, crate::paths::agent_root()),
        ] {
            let e = MirageError::not_found(kind, "ghost").to_string();
            assert!(e.contains("ghost"), "{e}");
            assert!(e.contains(kind.as_str()), "{e}");
            assert!(e.contains(&root.display().to_string()), "{e}");
        }

        crate::paths::clear_test_root();
    }

    #[test]
    fn io_error_names_the_path() {
        let e = MirageError::io(
            "/tmp/x",
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        );
        assert!(e.to_string().contains("/tmp/x"), "{e}");
        let full = e.full_message();
        assert!(full.contains("/tmp/x"), "{full}");
        assert!(full.contains("denied"), "{full}");
    }

    #[test]
    fn a_wrapped_error_is_reported_once() {
        // `main` prints with anyhow's `{:#}`, which walks `source()` and
        // joins with ": ". A variant whose own message also interpolated
        // its source printed the reason twice — "io error on /tmp/x:
        // denied: denied". Reproduced here by flattening the chain the
        // same way anyhow does.
        let io = MirageError::io(
            "/tmp/x",
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        );
        assert_eq!(io.full_message(), "io error on /tmp/x: denied");

        let source = serde_json::from_str::<u32>("nope").unwrap_err();
        let reason = source.to_string();
        let json = MirageError::Json {
            path: "/tmp/y".into(),
            source,
        };
        assert_eq!(
            json.full_message(),
            format!("json error on /tmp/y: {reason}")
        );
    }
}
