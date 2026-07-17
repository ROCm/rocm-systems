//! In-process RocJitsu daemon hosted by the shared `librocjitsu` lifecycle API.

use std::ffi::CString;
use std::path::{Path, PathBuf};

use mirage_core::emulator::EmulatorDaemon;
use rocjitsu_sys::{Lib, ROCJITSU_STATUS_SUCCESS, RjDaemon, RjDaemonStatus};

/// A running RocJitsu daemon. Dropping it stops the server, joins its threads,
/// destroys the VM, and removes its Unix socket.
pub struct Daemon {
    lib: Lib,
    handle: *mut RjDaemon,
    socket_path: PathBuf,
}

// The C daemon synchronizes lifecycle state internally. Mirage transfers this
// owner to the host shutdown path but never aliases it mutably.
unsafe impl Send for Daemon {}

impl Daemon {
    /// Path of the Unix socket this daemon listens on.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Load `lib_path` and start a daemon on `<runtime_dir>/daemon.sock`.
    pub fn start(
        lib_path: &Path,
        config_path: &Path,
        runtime_dir: &Path,
    ) -> std::result::Result<Self, String> {
        let lib = unsafe { Lib::open(lib_path) }
            .map_err(|e| format!("rocjitsu daemon: cannot load {}: {e}", lib_path.display()))?;
        let config = std::fs::read_to_string(config_path).map_err(|e| {
            format!(
                "rocjitsu daemon: cannot read {}: {e}",
                config_path.display()
            )
        })?;
        let json = CString::new(config)
            .map_err(|e| format!("rocjitsu daemon: configuration contains a NUL byte: {e}"))?;
        let socket_path = runtime_dir.join("daemon.sock");
        let socket = CString::new(socket_path.as_os_str().as_encoded_bytes())
            .map_err(|e| format!("rocjitsu daemon: invalid socket path: {e}"))?;
        let (status, handle) = unsafe { lib.daemon_start(&json, &socket) };
        if status != ROCJITSU_STATUS_SUCCESS || handle.is_null() {
            return Err(format!(
                "rocjitsu daemon: rj_daemon_start({}, {}) failed with status {status}",
                config_path.display(),
                socket_path.display()
            ));
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

    /// Current status reported by `librocjitsu`.
    pub fn status(&self) -> RjDaemonStatus {
        match unsafe { self.lib.daemon_status(self.handle) } {
            Ok(status) => status,
            Err(status) => {
                tracing::error!(status, "librocjitsu returned an invalid daemon status");
                RjDaemonStatus::Error
            }
        }
    }

    /// Tear the daemon down. Idempotent; called by both `stop` and `drop`.
    fn teardown(&mut self) {
        if self.handle.is_null() {
            return;
        }
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
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

impl EmulatorDaemon for Daemon {
    fn stop(mut self: Box<Self>) {
        self.teardown();
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.teardown();
    }
}
