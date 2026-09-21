//! The socket a `mirage run` serves so other terminals can find it.
//!
//! One socket per run, named after its session, living for exactly as
//! long as the run does. It answers [`Request::Describe`] and
//! [`Request::Attach`] with a [`mirage_core::proto::SessionDescription`] and nothing else —
//! see [`mirage_core::proto`] for why that is the whole protocol.
//!
//! The two differ only in what happens next. `Describe` closes; `Attach`
//! keeps the connection, and the connection *is* the borrower's lease on
//! the session. Each is held by the task serving it, so a lease is
//! released exactly when its client goes away — which is the one thing an
//! explicit release message could not promise for a client that crashed.
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
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use mirage_core::proto::{Request, Response, codec};
use tokio::net::{UnixListener, UnixStream};
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use crate::run::Run;

/// The longest path a Unix socket can be bound to.
///
/// `sockaddr_un.sun_path` is a fixed 108-byte array on Linux and the path
/// in it is NUL-terminated, so 107 bytes is the entire budget: runtime
/// directory, `run/`, session id and `.sock` together. Past it `bind`
/// fails with `EINVAL` — "invalid argument", about an argument the caller
/// never passed — which is why the limit is checked here and named in the
/// error rather than left to the kernel to hint at.
const SUN_PATH_MAX: usize = 107;

/// One client's connection: length-delimited frames of JSON.
type Control = Framed<UnixStream, LengthDelimitedCodec>;

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

    /// The socket path is longer than the kernel's `sun_path` limit.
    #[error(
        "the control socket path is {} bytes, over the {SUN_PATH_MAX}-byte limit \
         the kernel puts on a Unix socket path: {}. \
         Point MIRAGE_RUNTIME (or XDG_RUNTIME_DIR) at a shorter directory.",
        .path.as_os_str().len(),
        .path.display(),
    )]
    PathTooLong {
        /// The path that did not fit.
        path: PathBuf,
    },

    /// The directory the socket lives in is reachable by other users.
    #[error(
        "the directory mirage serves its control sockets from is not private: \
         {} ({reason}). Anyone who can reach it can `mirage exec` into this session, \
         which runs commands as you. Point MIRAGE_RUNTIME at a directory only you \
         can read, or make this one private (`chmod 700 {}`) if it is yours to fix.",
        .path.display(),
        .path.display(),
    )]
    Insecure {
        /// The directory that could not be secured.
        path: PathBuf,
        /// What is wrong with it.
        reason: String,
    },

    /// The socket could not be created.
    ///
    /// The underlying error is the [`source`] and is deliberately *not*
    /// interpolated into this message: mirage prints an error together
    /// with its causes (`{:#}`), so a message that repeated its own
    /// source read "invalid argument: invalid argument" and named
    /// neither the path nor the operation.
    ///
    /// [`source`]: std::error::Error::source
    #[error("could not create mirage's control socket at {}", .path.display())]
    Io {
        /// The path being created when it failed.
        path: PathBuf,
        /// What the operating system said.
        #[source]
        source: std::io::Error,
    },
}

impl BindError {
    /// A [`BindError::Io`] naming the path that failed.
    fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
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
    /// this path already, [`BindError::PathTooLong`] if the path does not
    /// fit in a `sockaddr_un`, [`BindError::Insecure`] if the directory
    /// holding it cannot be made private to this user, or
    /// [`BindError::Io`] if the socket cannot be created.
    pub async fn bind(path: &Path) -> Result<Self, BindError> {
        use std::os::unix::fs::PermissionsExt as _;

        // Checked before anything is created, so an unusable path leaves
        // no directories behind on its way to being rejected.
        if path.as_os_str().len() > SUN_PATH_MAX {
            return Err(BindError::PathTooLong {
                path: path.to_path_buf(),
            });
        }

        // Anyone who can connect to this socket learns how to start
        // processes in the session — which is arbitrary code as its
        // owner, so this is a permission boundary and not tidiness. The
        // default runtime directory is `$XDG_RUNTIME_DIR`, already
        // `0700`, but the fallback when that is unset is under `$TMPDIR`,
        // shared with every other user on the machine.
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            create_private_dir(parent)?;
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
            std::fs::remove_file(path).map_err(|e| BindError::io(path, e))?;
        }

        let listener = UnixListener::bind(path).map_err(|e| BindError::io(path, e))?;
        // The socket file is created with the umask's idea of its mode,
        // and only the private directory above it covers the instant
        // between `bind` and this `chmod`. Owning it as a `ControlSocket`
        // first is what unlinks it if the `chmod` fails, rather than
        // leaving a socket other users can connect to for the next run to
        // reclaim.
        let socket = Self {
            listener,
            path: path.to_path_buf(),
        };
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| BindError::io(path, e))?;
        Ok(socket)
    }

    /// Serve `run` until the future is dropped.
    ///
    /// Never returns on its own: the caller races it against the workload
    /// finishing, and dropping this future stops serving. Each connection
    /// is handled on its own task so a slow client cannot block the next.
    pub async fn serve(&self, run: Arc<Run>) {
        // Back off after a failed accept rather than retrying immediately.
        // Some accept errors are transient (`ECONNABORTED`), but others
        // are not: at the process's fd limit — which a wide
        // `--nproc-per-node` grid with captured stdio can reach — `accept`
        // returns `EMFILE` the instant it is called, and a bare `continue`
        // turns that into a task spinning a core flat out, starving the
        // runtime threads that are supervising the workload and emitting
        // an unbounded stream of warnings.
        const MAX_BACKOFF: Duration = Duration::from_secs(1);
        let mut backoff = Duration::from_millis(5);

        loop {
            let (stream, _) = match self.listener.accept().await {
                Ok(accepted) => {
                    backoff = Duration::from_millis(5);
                    accepted
                }
                Err(e) => {
                    tracing::warn!("control socket accept failed, retrying in {backoff:?}: {e}");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(MAX_BACKOFF);
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

/// Create `dir`, and every missing directory above it, private to this
/// user — and refuse to go on if the result is not.
///
/// Not `create_dir_all` followed by a `chmod`. That gets three things
/// wrong at once: the umask decides the mode of every level created, the
/// `chmod` only ever reaches the last one, and between the two the
/// directory sits there with whatever the umask allowed — a window
/// another user only has to win once. `mkdir(2)` takes the mode the
/// directory is *created* with, which closes all three, and the check
/// afterwards covers the case this call did not create: a directory left
/// by an older mirage, or widened since.
fn create_private_dir(dir: &Path) -> Result<(), BindError> {
    use std::os::unix::fs::{DirBuilderExt as _, MetadataExt as _, PermissionsExt as _};

    // The levels that do not exist yet, innermost first. Created in the
    // opposite order below so each one's parent is already there.
    let mut missing: Vec<&Path> = Vec::new();
    let mut level = Some(dir);
    while let Some(p) = level.filter(|p| !p.as_os_str().is_empty() && !p.exists()) {
        missing.push(p);
        level = p.parent();
    }

    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700);
    for level in missing.iter().rev().copied() {
        match builder.create(level) {
            Ok(()) => {}
            // Another run building the same tree at the same instant.
            // Its mode is this one's, and the check below covers both.
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(BindError::io(level, e)),
        }
    }

    // Verified rather than assumed: the umask can only ever clear bits,
    // so a level created above is `0700` or tighter, but a level that
    // already existed is whatever somebody else made it.
    let meta = std::fs::metadata(dir).map_err(|e| BindError::io(dir, e))?;
    let me = nix::unistd::geteuid().as_raw();
    if meta.uid() != me {
        return Err(BindError::Insecure {
            path: dir.to_path_buf(),
            reason: format!(
                "it belongs to uid {} rather than to you, uid {me}",
                meta.uid()
            ),
        });
    }
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 == 0 {
        return Ok(());
    }

    // Ours, but open to others. Narrow it and confirm that it took —
    // reporting success here on the strength of a `chmod` nobody checked
    // is exactly how the socket came to be served out of a world-writable
    // directory in the first place.
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).map_err(|e| {
        BindError::Insecure {
            path: dir.to_path_buf(),
            reason: format!("its mode is {mode:04o} and it could not be narrowed to 0700: {e}"),
        }
    })?;
    let narrowed = std::fs::metadata(dir)
        .map_err(|e| BindError::io(dir, e))?
        .permissions()
        .mode()
        & 0o777;
    if narrowed & 0o077 != 0 {
        return Err(BindError::Insecure {
            path: dir.to_path_buf(),
            reason: format!("its mode is still {narrowed:04o} after narrowing it to 0700"),
        });
    }
    Ok(())
}

impl Drop for ControlSocket {
    fn drop(&mut self) {
        // Best effort: a leftover file is recoverable (see `bind`), but
        // leaving one behind on a clean exit would be sloppy.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// How long a client has to say what it wants.
///
/// A connection that opens and then sends nothing costs a task and a file
/// descriptor for the rest of the run, and it is invisible while it does:
/// the run is serving normally, one descriptor poorer each time. Generous,
/// because this bounds abandoned connections rather than slow ones — a
/// real client sends its single frame the instant it connects.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Read the one request a client sends, or give up on it.
///
/// `None` covers all three ways there is nothing to answer: the client
/// disconnected without asking (which is also how liveness probes test
/// this socket), the frame was unreadable, or it never arrived.
async fn first_request(framed: &mut Control) -> Option<tokio_util::bytes::BytesMut> {
    match tokio::time::timeout(REQUEST_TIMEOUT, framed.next()).await {
        Ok(Some(Ok(frame))) => Some(frame),
        Ok(Some(Err(e))) => {
            tracing::debug!("control socket client sent an unreadable frame: {e}");
            None
        }
        Ok(None) => None,
        Err(_elapsed) => {
            tracing::debug!(
                "dropping a control socket client that sent no request within {REQUEST_TIMEOUT:?}"
            );
            None
        }
    }
}

/// Why a borrower's lease ended.
#[derive(Debug)]
enum LeaseEnd {
    /// The borrower's connection ended, however it ended.
    ClientGone,
    /// Teardown asked the borrower to let go.
    SessionClosing,
}

/// Hold a borrower's connection open until one side lets go.
///
/// The reads here are not looking for requests — the protocol has none
/// after the description — they are draining the stream to its *end*,
/// which is how a disconnect is observed. Waiting for a single frame is
/// not the same test and is not equivalent: a client that sent anything
/// at all, a zero-length frame included, would have dropped its lease
/// while it was still very much using the session.
///
/// The other arm is teardown deciding not to wait any longer. Dropping
/// the connection is what tells the borrower to stop, and it must not be
/// left to the client to notice by other means.
async fn hold_lease(framed: &mut Control, closing: impl Future<Output = ()>) -> LeaseEnd {
    tokio::select! {
        () = async { while framed.next().await.is_some() {} } => LeaseEnd::ClientGone,
        () = closing => LeaseEnd::SessionClosing,
    }
}

/// Answer one client's request, and hold its lease if it took one.
async fn handle(stream: UnixStream, run: Arc<Run>) {
    // Who is on the other end, asked of the kernel rather than of the
    // client. A borrower cannot misreport it, and it is the only thing
    // that distinguishes one borrower's live processes from another
    // borrower's leftovers once a lease has ended — see
    // [`Session::reap_departed_borrowers`](mirage_supervisor::session::Session::reap_departed_borrowers).
    //
    // A pid of zero is not a pid. `SO_PEERCRED` does not fail for a peer
    // in a namespace this process cannot resolve — the kernel reports `0`
    // — so taking the number at face value records the placeholder the
    // `Borrower` doc says is never stored, and a meaningless `0` then
    // rides along in every ancestry check. Dropped here, at the one place
    // that knows the number came from the kernel, so the borrower falls
    // back to its exec mark exactly as one whose credentials were refused
    // outright does.
    let borrower = stream
        .peer_cred()
        .ok()
        .and_then(|cred| cred.pid())
        .and_then(|pid| u32::try_from(pid).ok())
        .filter(|pid| *pid > 0);
    let mut framed = Framed::new(stream, codec());

    let Some(frame) = first_request(&mut framed).await else {
        return;
    };

    // The lease, for an `Attach`. It lives in this task and nowhere else,
    // so the claim is released by this function returning — which happens
    // when the client disconnects, however it disconnects. That is the
    // whole mechanism: there is no release message to be lost, and a
    // borrower that segfaults releases its lease as reliably as one that
    // exits cleanly.
    let mut lease = None;

    let response = match serde_json::from_slice::<Request>(&frame) {
        Ok(Request::Describe) => match run.describe() {
            Ok(desc) => Response::Description(Box::new(desc)),
            Err(e) => Response::Error(e.to_string()),
        },
        Ok(Request::Attach { exec }) => match run.attach(borrower, exec) {
            // The lease is taken *before* the description is built, not
            // after. Between the two the session could begin tearing
            // down, and a borrower handed a description of containers
            // that are being removed would start its workload into them.
            Some(claim) => match run.describe() {
                Ok(desc) => {
                    lease = Some(claim);
                    Response::Description(Box::new(desc))
                }
                Err(e) => Response::Error(e.to_string()),
            },
            None => Response::Error(format!(
                "session {} is shutting down and cannot be attached to",
                run.id()
            )),
        },
        Err(e) => Response::Error(format!("malformed request: {e}")),
    };

    match serde_json::to_vec(&response) {
        Ok(bytes) => {
            if framed.send(bytes.into()).await.is_err() {
                return;
            }
        }
        Err(e) => {
            tracing::warn!("could not encode response: {e}");
            return;
        }
    }

    let Some(lease) = lease else {
        return;
    };

    // Hold the connection open for as long as the borrower wants it.
    let ended = hold_lease(&mut framed, run.wait_closing()).await;
    // Released before the sweep below, so the departing borrower is not
    // counted among the live ones it must not touch.
    drop(lease);

    match ended {
        LeaseEnd::SessionClosing => {
            tracing::debug!(
                session = %run.id(),
                "closing a borrower's lease: session is tearing down"
            );
        }
        LeaseEnd::ClientGone => {
            // The borrower is gone, and this is the instant the run finds
            // out. Anything it started and did not stop is running in a
            // session that is still very much alive, so nothing else will
            // notice it until teardown — which is how `mirage exec`'s
            // promise that a workload "dies with it" came to hold only
            // for the borrowers that outlived their run.
            run.reap_departed_borrowers().await;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use std::os::unix::fs::PermissionsExt as _;

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
        assert!(
            second.is_ok(),
            "a corpse socket must be reclaimed: {second:?}"
        );
        assert!(
            path.exists(),
            "the reclaimed path must now be a live socket"
        );
    }

    #[tokio::test]
    async fn every_directory_created_for_a_socket_is_private() {
        // The socket grants `exec` into the session, which is arbitrary
        // code as its owner, so the directory holding it may not be
        // reachable by anyone else. `create_dir_all` gave every level but
        // the last whatever the umask allowed — 0775 under the default
        // 002 — and the last one a `chmod` whose result nobody looked at.
        let dir = tempfile::tempdir().unwrap();
        // A world-writable root: what the fallback runtime directory
        // under /tmp looks like when `$XDG_RUNTIME_DIR` is unset.
        let shared = dir.path().join("shared");
        std::fs::create_dir(&shared).unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o777)).unwrap();

        let path = shared.join("mirage/run/private.sock");
        let _socket = ControlSocket::bind(&path).await.unwrap();

        for level in [shared.join("mirage"), shared.join("mirage/run")] {
            let mode = mode_of(&level);
            assert_eq!(
                mode & 0o077,
                0,
                "{} is reachable by other users (mode {mode:04o})",
                level.display()
            );
        }
        let mode = mode_of(&path);
        assert_eq!(
            mode & 0o077,
            0,
            "the socket itself is connectable by other users (mode {mode:04o})"
        );
    }

    #[tokio::test]
    async fn a_directory_that_was_already_there_is_narrowed_before_it_is_used() {
        // The other half, and the half the bug was actually about. Every
        // level mirage *creates* is private by construction —
        // `mkdir(2)` takes the mode — so a test that only exercises
        // those proves nothing about the forty lines that narrow a
        // directory somebody else made. Replace them with `return
        // Ok(())` and the test above still passes.
        //
        // This is the real shape: `$XDG_RUNTIME_DIR` unset, so the
        // runtime root falls back under a shared `$TMPDIR`, and the
        // directory mirage is about to serve its socket out of already
        // exists and is reachable by everyone on the machine.
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("run");
        std::fs::create_dir(&existing).unwrap();
        std::fs::set_permissions(&existing, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert_eq!(mode_of(&existing) & 0o077, 0o077, "the setup must be open");

        let path = socket_path(&existing, "preexisting");
        let _socket = ControlSocket::bind(&path).await.unwrap();

        let mode = mode_of(&existing);
        assert_eq!(
            mode & 0o077,
            0,
            "a directory that already existed was served out of at mode {mode:04o}; \
             anyone who can reach it can `mirage exec` into this session"
        );
    }

    /// The permission bits of `path`.
    fn mode_of(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[tokio::test]
    async fn a_path_too_long_for_the_kernel_says_so_and_creates_nothing() {
        // The limit a deep `$TMPDIR` reaches in practice, and which the
        // e2e harness already has to keep its runtime root shallow to
        // avoid. Unnamed, it arrives as a bare "invalid argument".
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("run");
        let path = parent.join(format!("{}.sock", "s".repeat(SUN_PATH_MAX)));

        let err = ControlSocket::bind(&path).await.unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains(&SUN_PATH_MAX.to_string()),
            "the error must name the limit: {text}"
        );
        assert!(
            text.contains(&path.display().to_string()),
            "the error must name the path that did not fit: {text}"
        );
        assert!(
            !parent.exists(),
            "a rejected path must not leave directories behind"
        );
    }

    #[tokio::test]
    async fn a_failure_to_create_the_socket_names_the_path_once() {
        // `error: {e:#}` prints an error *and its causes*, so a message
        // that was nothing but its own source printed that source twice
        // and named neither the path nor what mirage was doing with it.
        use std::error::Error as _;

        let dir = tempfile::tempdir().unwrap();
        let not_a_dir = dir.path().join("occupied");
        std::fs::write(&not_a_dir, b"").unwrap();

        let err = ControlSocket::bind(&not_a_dir.join("run/s.sock"))
            .await
            .unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("occupied"),
            "the error must name the path it failed on: {text}"
        );
        let cause = err
            .source()
            .expect("an io failure keeps its cause")
            .to_string();
        assert!(
            !text.contains(&cause),
            "the cause is printed again by `{{:#}}`; it must not be in the message too: \
             {text} / {cause}"
        );
    }

    /// A connected pair of framed endpoints over a real socket.
    async fn connected(dir: &Path, name: &str) -> (Control, Control) {
        let path = socket_path(dir, name);
        let listener = UnixListener::bind(&path).unwrap();
        let client = UnixStream::connect(&path).await.unwrap();
        let (server, _) = listener.accept().await.unwrap();
        (Framed::new(server, codec()), Framed::new(client, codec()))
    }

    #[tokio::test(start_paused = true)]
    async fn a_client_that_asks_for_nothing_is_dropped() {
        // Held open, such a connection costs a task and a descriptor for
        // the whole run — and a run that has quietly run out of
        // descriptors stops being able to accept the client that matters.
        let dir = tempfile::tempdir().unwrap();
        let (mut server, _client) = connected(dir.path(), "silent").await;

        let start = tokio::time::Instant::now();
        assert!(
            first_request(&mut server).await.is_none(),
            "a client that never sends a request must not be served forever"
        );
        assert!(
            start.elapsed() >= REQUEST_TIMEOUT,
            "the connection was dropped before its deadline"
        );
    }

    #[tokio::test]
    async fn a_lease_lasts_as_long_as_the_connection_not_until_its_first_frame() {
        // The lease *is* the connection: that is what makes a borrower
        // that segfaults release it as reliably as one that exits. Ending
        // it on the first frame received meant anything the client said
        // — even nothing, in a zero-length frame — tore the session out
        // from under a workload that was still running.
        let dir = tempfile::tempdir().unwrap();
        let (mut server, mut client) = connected(dir.path(), "lease").await;

        client.send(tokio_util::bytes::Bytes::new()).await.unwrap();
        let still_held = tokio::time::timeout(
            Duration::from_millis(200),
            hold_lease(&mut server, std::future::pending()),
        )
        .await;
        assert!(
            still_held.is_err(),
            "a frame from the borrower released its lease: {still_held:?}"
        );

        // And it does end when the borrower goes away.
        drop(client);
        assert!(
            matches!(
                hold_lease(&mut server, std::future::pending()).await,
                LeaseEnd::ClientGone
            ),
            "a disconnected borrower must release its lease"
        );
    }

    #[tokio::test]
    async fn a_lease_ends_when_the_session_starts_closing() {
        let dir = tempfile::tempdir().unwrap();
        let (mut server, _client) = connected(dir.path(), "closing").await;
        assert!(matches!(
            hold_lease(&mut server, std::future::ready(())).await,
            LeaseEnd::SessionClosing
        ));
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
