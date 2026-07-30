//! The socket a `mirage run` serves so other terminals can find it.
//!
//! One socket per run, named after its session, living for exactly as
//! long as the run does. It answers [`Request::Describe`] with a
//! [`SessionDescription`] and nothing else — see [`mirage_core::proto`]
//! for why that is the whole protocol.
//!
//! # Staleness
//!
//! A socket file outlives the process that bound it if that process was
//! `SIGKILL`ed, so the file's existence proves nothing. Rather than
//! guarding it with a lock file, [`ControlSocket::bind`] simply tries to
//! connect to any socket already at the path: if something answers, a run
//! already owns this session id and we refuse; if nothing does, the file
//! is a corpse and is removed. The test is direct, needs no second file,
//! and cannot be fooled by a stale lock.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use mirage_core::proto::{Request, Response, codec};
use tokio::net::{UnixListener, UnixStream};
use tokio_util::codec::Framed;

use crate::run::Run;

/// A bound control socket, unlinked on drop.
#[derive(Debug)]
pub struct ControlSocket {
    listener: UnixListener,
    path: PathBuf,
}

/// Why binding failed.
#[derive(Debug, thiserror::Error)]
pub enum BindError {
    /// Another live run already owns this session id.
    #[error("a mirage run is already serving session `{0}`")]
    AlreadyRunning(String),
    /// The socket could not be created.
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

impl ControlSocket {
    /// Bind the control socket for `path`.
    ///
    /// Must be called from within a tokio runtime: the returned listener
    /// registers with the reactor.
    ///
    /// # Errors
    ///
    /// Returns [`BindError::AlreadyRunning`] if a live run answers on
    /// this path already, or [`BindError::Io`] if the socket cannot be
    /// created.
    pub async fn bind(path: &Path) -> Result<Self, BindError> {
        use std::os::unix::fs::PermissionsExt as _;

        // Anyone who can connect to this socket learns how to start
        // processes in the session. The default runtime directory is
        // `$XDG_RUNTIME_DIR`, already `0700`, but the fallback when that
        // is unset is under `$TMPDIR` — so the mode is stated rather than
        // inherited from the umask.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
            let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
        }

        if path.exists() {
            if UnixStream::connect(path).await.is_ok() {
                return Err(BindError::AlreadyRunning(
                    path.file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                ));
            }
            // Nothing is listening: the file is left over from a run that
            // died without cleaning up. Refusing to start because of it
            // would strand the user behind a file whose owner is provably
            // gone.
            std::fs::remove_file(path)?;
        }

        let listener = UnixListener::bind(path)?;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        Ok(Self {
            listener,
            path: path.to_path_buf(),
        })
    }

    /// Serve `run` until the future is dropped.
    ///
    /// Never returns on its own: the caller races it against the workload
    /// finishing, and dropping this future stops serving. Each connection
    /// is handled on its own task so a slow client cannot block the next.
    pub async fn serve(&self, run: Arc<Run>) {
        loop {
            let (stream, _) = match self.listener.accept().await {
                Ok(accepted) => accepted,
                Err(e) => {
                    tracing::warn!("control socket accept failed: {e}");
                    continue;
                }
            };
            let run = run.clone();
            tokio::spawn(async move { handle(stream, run).await });
        }
    }

    /// The path this socket is bound to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ControlSocket {
    fn drop(&mut self) {
        // Best effort: a leftover file is recoverable (see `bind`), but
        // leaving one behind on a clean exit would be sloppy.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Answer one client's single request.
async fn handle(stream: UnixStream, run: Arc<Run>) {
    let mut framed = Framed::new(stream, codec());

    let Some(Ok(frame)) = framed.next().await else {
        return;
    };
    let response = match serde_json::from_slice::<Request>(&frame) {
        Ok(Request::Describe) => match run.describe() {
            Ok(desc) => Response::Description(Box::new(desc)),
            Err(e) => Response::Error(e.to_string()),
        },
        Err(e) => Response::Error(format!("malformed request: {e}")),
    };

    match serde_json::to_vec(&response) {
        Ok(bytes) => {
            let _ = framed.send(bytes.into()).await;
        }
        Err(e) => tracing::warn!("could not encode response: {e}"),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A socket path unique to one test.
    ///
    /// Unique per call, not just per tempdir: these tests run in parallel
    /// with the rest of the suite, and a name shared between them makes a
    /// failure ambiguous about *whose* socket answered.
    fn socket_path(dir: &Path, name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(0);
        dir.join(format!(
            "{name}-{}-{}.sock",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[tokio::test]
    async fn a_second_run_cannot_take_a_live_socket() {
        // Two runs owning one session id would each believe they own its
        // containers, and the first to exit would tear the other's down.
        let dir = tempfile::tempdir().unwrap();
        let path = socket_path(dir.path(), "live");

        let _first = ControlSocket::bind(&path).await.unwrap();
        let second = ControlSocket::bind(&path).await;
        assert!(
            matches!(second, Err(BindError::AlreadyRunning(_))),
            "expected AlreadyRunning, got {second:?}"
        );
    }

    #[tokio::test]
    async fn a_socket_left_by_a_dead_run_is_reclaimed() {
        // The file outlives a SIGKILLed run. Refusing to start because of
        // a corpse would strand the user behind a file they have to
        // delete by hand.
        let dir = tempfile::tempdir().unwrap();
        let path = socket_path(dir.path(), "stale");

        // Leave a file where the socket was, with nothing serving it —
        // the state a `SIGKILL`ed run leaves behind, since the kernel
        // does not unlink a socket when its owner dies.
        //
        // A plain file rather than a closed listener: what is under test
        // is the branch `bind` takes when nothing answers, and a file
        // reaches it deterministically. Closing a listener *usually* also
        // reaches it, but "usually" is how a suite acquires a flake, and
        // the branch is the same either way.
        std::fs::write(&path, b"").unwrap();
        assert!(
            UnixStream::connect(&path).await.is_err(),
            "the setup must leave nothing answering on {}",
            path.display()
        );

        let second = ControlSocket::bind(&path).await;
        assert!(second.is_ok(), "a corpse socket must be reclaimed: {second:?}");
        assert!(path.exists(), "the reclaimed path must now be a live socket");
    }

    #[tokio::test]
    async fn the_socket_is_removed_when_the_run_ends() {
        let dir = tempfile::tempdir().unwrap();
        let path = socket_path(dir.path(), "ends");
        {
            let _socket = ControlSocket::bind(&path).await.unwrap();
            assert!(path.exists());
        }
        assert!(
            !path.exists(),
            "a finished run must not leave a socket claiming its session is live"
        );
    }
}
