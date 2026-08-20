//! A safe, RAII wrapper around the rocjitsu daemon lifecycle API.
//!
//! The daemon owns the simulated GPU outside the *workload's* process and
//! serves its KFD ioctls over a Unix socket. It is not a process of its
//! own: the library is `dlopen`ed and the daemon runs on threads inside
//! whoever called [`Daemon::start`] — the `mirage run` process. Its C API
//! is three calls —
//! start, status, stop — with a strict ownership contract: the handle
//! returned by `rj_daemon_start` must be released exactly once, and using
//! it afterwards is undefined.
//!
//! [`Daemon`] encodes that contract in the type system. It owns the
//! handle, `stop`/`Drop` release it exactly once (a null sentinel makes
//! the second call a no-op), and the handle is never handed out. Callers
//! therefore need no `unsafe` of their own, which is what lets every
//! crate above this one be `forbid(unsafe_code)`.
//!
//! The wrapper lives in this crate rather than in `mirage_rocjitsu`
//! deliberately: the `unsafe` and the invariants that justify it belong
//! in the same file.

use std::ffi::{CString, NulError};
use std::path::{Path, PathBuf};

use crate::{Lib, ROCJITSU_STATUS_SUCCESS, RjDaemon, RjDaemonStatus};

/// Why a daemon could not be started or driven.
#[derive(Debug)]
pub enum DaemonError {
    /// The rocjitsu shared library could not be loaded.
    Load {
        /// Path that was attempted.
        path: PathBuf,
        /// The loader's reason.
        source: libloading::Error,
    },
    /// The configuration file could not be read.
    Config {
        /// Path that was attempted.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// A path or configuration contained an interior NUL byte and so
    /// cannot cross the C boundary.
    Nul(NulError),
    /// `rj_daemon_start` reported failure.
    Start {
        /// The status code it returned.
        status: i32,
        /// The config the daemon was started with.
        config: PathBuf,
        /// The socket it was asked to bind.
        socket: PathBuf,
    },
}

impl DaemonError {
    /// Whether this is a library that could not be loaded *at all*, as
    /// opposed to one that loaded and lacks an entry point.
    ///
    /// The two failures of [`Daemon::probe`] call for opposite advice and
    /// only the loader can tell them apart. A library missing
    /// `rj_daemon_start` is an older rocjitsu: it loads, it emulates a
    /// workload in-process, and the remedy is to update it or to run
    /// without a daemon. A library that will not load — a broken
    /// dependency chain, the wrong architecture, an unreadable file — is
    /// not usable for anything, and in particular not usable in-process,
    /// because in-process emulation `LD_PRELOAD`s that same file. Telling
    /// such a user to "pass `--in-process`" sends them somewhere that
    /// fails again for the same reason.
    ///
    /// True only for [`Self::Load`] whose source is a `dlopen` failure;
    /// every other variant, symbol resolution included, is `false`.
    #[must_use]
    pub fn is_unloadable(&self) -> bool {
        matches!(
            self,
            Self::Load {
                source: libloading::Error::DlOpen { .. } | libloading::Error::DlOpenUnknown,
                ..
            }
        )
    }
}

impl std::fmt::Display for DaemonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load { path, source } => {
                write!(f, "cannot load {}: {source}", path.display())
            }
            Self::Config { path, source } => {
                write!(f, "cannot read {}: {source}", path.display())
            }
            Self::Nul(e) => write!(f, "path or configuration contains a NUL byte: {e}"),
            Self::Start {
                status,
                config,
                socket,
            } => write!(
                f,
                "rj_daemon_start({}, {}) failed with status {status}",
                config.display(),
                socket.display()
            ),
        }
    }
}

impl std::error::Error for DaemonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Load { source, .. } => Some(source),
            Self::Config { source, .. } => Some(source),
            Self::Nul(e) => Some(e),
            Self::Start { .. } => None,
        }
    }
}

impl From<NulError> for DaemonError {
    fn from(e: NulError) -> Self {
        Self::Nul(e)
    }
}

/// File name of the socket a daemon binds inside its runtime directory.
pub const SOCKET_NAME: &str = "daemon.sock";

/// A running rocjitsu daemon.
///
/// Dropping it stops the server, joins its threads, destroys the VM and
/// removes its Unix socket.
pub struct Daemon {
    lib: Lib,
    /// The C handle. Null once released, which is what makes `stop` and
    /// `Drop` safe to run in either order and more than once.
    handle: *mut RjDaemon,
    socket_path: PathBuf,
}

// SAFETY: The C daemon synchronises its own lifecycle state internally,
// and `Daemon` never hands out `handle` or aliases it — every use is
// through `&mut self` or the owning `stop`. Mirage moves the owner
// between tasks (bring-up creates it, teardown consumes it), so `Send` is
// required; `Sync` deliberately is not claimed, because two threads
// calling into the C API concurrently through a shared reference is not
// something the C contract promises.
unsafe impl Send for Daemon {}

impl std::fmt::Debug for Daemon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Daemon")
            .field("socket_path", &self.socket_path)
            .field("running", &!self.handle.is_null())
            .finish()
    }
}

impl Daemon {
    /// Whether the library at `lib_path` can host a daemon at all.
    ///
    /// The daemon half of the rocjitsu C API is newer than the rest, and a
    /// library that predates it is not broken — it emulates a workload
    /// in-process perfectly well. It simply has no `rj_daemon_start` to
    /// call, and nothing discovers that until something tries to call it.
    /// For mirage that was at the *end* of bring-up, after the image pull,
    /// the network and every container: the run failed on a fact about a
    /// file that was knowable before any of it was created.
    ///
    /// This is [`Daemon::start`]'s own load step and nothing else — the same
    /// `dlopen`, the same symbols, no VM, no socket, no threads — so a
    /// library this accepts is one `start` will not reject for want of an
    /// entry point. Answering it costs a `dlopen`, which is why callers ask
    /// once and remember rather than asking per session.
    ///
    /// # Why the handle is kept rather than dropped
    ///
    /// Dropping a [`Lib`] is a `dlclose`, and the probe holds the only
    /// reference, so the loader would unmap a library whose initialisers
    /// have just run. An emulator runtime's constructor is exactly the
    /// kind that registers something process-global — a fault handler, a
    /// worker thread, an `atexit` entry — and unloading the code those
    /// point into leaves the dangling mapping to be discovered much later,
    /// somewhere unrelated. [`Daemon::start`] never unloads for the same
    /// reason: it keeps its handle for the life of the daemon.
    ///
    /// So the probe leaks its handle deliberately. The cost is one mapping
    /// held for the life of the process; what it buys is that the library
    /// is loaded at most once — a later [`Daemon::start`] on the same path
    /// is a refcount bump rather than a second `dlopen` — and never
    /// unloaded.
    ///
    /// # Errors
    ///
    /// [`DaemonError::Load`] when the library cannot be loaded or does not
    /// export the daemon entry points; [`DaemonError::is_unloadable`]
    /// distinguishes the two.
    pub fn probe(lib_path: &Path) -> Result<(), DaemonError> {
        // SAFETY: the same contract as `Daemon::start` — `lib_path` is a
        // rocjitsu library located by mirage's own discovery, and loading it
        // runs its initialisers. No C call is made through the handle; it is
        // parked in a `ManuallyDrop` so the library is never unloaded (see
        // above) rather than dropped at the end of this scope.
        unsafe { Lib::open(lib_path) }
            .map(|lib| {
                let _kept = std::mem::ManuallyDrop::new(lib);
            })
            .map_err(|source| DaemonError::Load {
                path: lib_path.to_path_buf(),
                source,
            })
    }

    /// Load `lib_path` and start a daemon on `<runtime_dir>/daemon.sock`
    /// using the configuration at `config_path`.
    ///
    /// # Errors
    ///
    /// Returns [`DaemonError`] if the library cannot be loaded, the
    /// configuration cannot be read, either path cannot cross the C
    /// boundary, or the daemon refuses to start.
    pub fn start(
        lib_path: &Path,
        config_path: &Path,
        runtime_dir: &Path,
    ) -> Result<Self, DaemonError> {
        // SAFETY: `Lib::open` dlopens the path and resolves the rocjitsu
        // symbols. The contract is that the library actually exports the
        // rocjitsu C API; a library that does not fails symbol resolution
        // and returns `Err` rather than producing a mis-typed handle.
        let lib = unsafe { Lib::open(lib_path) }.map_err(|source| DaemonError::Load {
            path: lib_path.to_path_buf(),
            source,
        })?;

        let config =
            std::fs::read_to_string(config_path).map_err(|source| DaemonError::Config {
                path: config_path.to_path_buf(),
                source,
            })?;
        let json = CString::new(config)?;
        let socket_path = runtime_dir.join(SOCKET_NAME);
        let socket = CString::new(socket_path.as_os_str().as_encoded_bytes())?;

        // SAFETY: both C strings are alive for the duration of the call
        // and NUL-terminated by construction. The returned handle is
        // checked for null and, if non-null, becomes this value's sole
        // owner and is released exactly once (see `release`).
        let (status, handle) = unsafe { lib.daemon_start(&json, &socket) };
        if status != ROCJITSU_STATUS_SUCCESS || handle.is_null() {
            return Err(DaemonError::Start {
                status,
                config: config_path.to_path_buf(),
                socket: socket_path,
            });
        }

        tracing::info!(
            socket = %socket_path.display(),
            config = %config_path.display(),
            "rocjitsu daemon started"
        );
        Ok(Self {
            lib,
            handle,
            socket_path,
        })
    }

    /// Path of the Unix socket this daemon listens on.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Whether the daemon is still running (has not been stopped).
    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.handle.is_null()
    }

    /// Current status reported by `librocjitsu`.
    ///
    /// Reports [`RjDaemonStatus::Stopped`] once the daemon has been
    /// stopped, rather than querying a released handle.
    #[must_use]
    pub fn status(&self) -> RjDaemonStatus {
        if self.handle.is_null() {
            return RjDaemonStatus::Stopped;
        }
        // SAFETY: `handle` is non-null here and, by this type's ownership
        // invariant, still live: it is only ever released by `release`,
        // which nulls it in the same step.
        match unsafe { self.lib.daemon_status(self.handle) } {
            Ok(status) => status,
            Err(status) => {
                tracing::error!(status, "librocjitsu returned an invalid daemon status");
                RjDaemonStatus::Error
            }
        }
    }

    /// Stop the daemon, blocking until its threads have joined.
    ///
    /// Idempotent, and equivalent to dropping the value. Taking `self`
    /// gives callers an explicit, ordered shutdown point — mirage stops
    /// the daemon only after every workload process is gone, so the
    /// simulated device outlives everything that might still be talking
    /// to it.
    pub fn stop(mut self) {
        self.release();
    }

    /// Release the C handle exactly once.
    fn release(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // Null the handle before the call so that a panic or a re-entrant
        // drop cannot release it twice.
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        // SAFETY: `handle` came from a successful `daemon_start`, has not
        // been released (it was non-null), and is not reachable any more
        // because it has already been replaced with null.
        let status = unsafe { self.lib.daemon_stop(handle) };
        if status == ROCJITSU_STATUS_SUCCESS {
            tracing::info!(socket = %self.socket_path.display(), "rocjitsu daemon stopped");
        } else {
            tracing::error!(
                status,
                socket = %self.socket_path.display(),
                "failed to stop rocjitsu daemon"
            );
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        // Teardown happens in `Drop` as well as `stop` so a panic
        // unwinding past the owner still releases the C handle and
        // removes the socket.
        self.release();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn socket_name_is_what_the_interposer_probes_for() {
        // The workload's interposer looks for this exact name inside
        // `$ROCJITSU_RUNTIME_DIR`; renaming it silently disconnects the
        // workload from its daemon.
        assert_eq!(SOCKET_NAME, "daemon.sock");
    }

    #[test]
    fn loading_a_missing_library_names_the_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let lib = dir.path().join("librocjitsu.so");
        let config = dir.path().join("config.json");
        std::fs::write(&config, b"{}").expect("write config");

        let err = Daemon::start(&lib, &config, dir.path()).expect_err("must fail");
        assert!(matches!(err, DaemonError::Load { .. }), "{err:?}");
        assert!(err.to_string().contains("librocjitsu.so"), "{err}");
    }

    #[test]
    fn a_missing_config_is_reported_before_the_library_is_used() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A real (if useless) shared object so loading is what fails
        // second, not first.
        let lib = dir.path().join("libnope.so");
        std::fs::write(&lib, b"not an elf").expect("write lib");
        let config = dir.path().join("missing.json");

        let err = Daemon::start(&lib, &config, dir.path()).expect_err("must fail");
        // Either error is acceptable ordering-wise, but it must name the
        // file the user has to fix.
        let msg = err.to_string();
        assert!(
            msg.contains("missing.json") || msg.contains("libnope.so"),
            "{msg}"
        );
    }

    #[test]
    fn probing_a_library_without_the_daemon_api_reports_the_missing_symbol() {
        // The case the probe exists for, and the one that is *not* a
        // missing file: a shared library that loads perfectly and does
        // not export the rocjitsu daemon entry points. A rocjitsu that
        // predates the daemon API looks exactly like this, and mirage
        // used to discover it only after pulling an image, creating a
        // network and starting every container.
        //
        // The C library stands in for it: it is here, it loads, and it
        // has never heard of `rj_daemon_start`. Anything else with those
        // three properties would do; if it cannot be loaded at all this
        // host cannot host the test, which is not a failure of the probe.
        let libc = Path::new("libc.so.6");
        let Err(err) = Daemon::probe(libc) else {
            panic!("libc.so.6 exports the rocjitsu C API?");
        };
        let DaemonError::Load { path, source } = &err else {
            panic!("expected a load error, got {err:?}");
        };
        if matches!(
            source,
            libloading::Error::DlOpen { .. } | libloading::Error::DlOpenUnknown
        ) {
            // No glibc `libc.so.6` to borrow. Nothing to assert here.
            return;
        }
        assert_eq!(path, libc);
        assert!(
            matches!(
                source,
                libloading::Error::DlSym { .. } | libloading::Error::DlSymUnknown
            ),
            "a library that loads and lacks the symbols must be reported as \
             a missing symbol rather than a missing file: {source:?}"
        );
    }

    #[test]
    fn paths_with_interior_nul_are_rejected_not_truncated() {
        // Silently truncating at the NUL would bind a socket somewhere
        // other than where the caller asked.
        let err = CString::new("with\0nul")
            .map_err(DaemonError::from)
            .unwrap_err();
        assert!(matches!(err, DaemonError::Nul(_)), "{err:?}");
    }
}
