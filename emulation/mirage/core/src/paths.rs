//! XDG-compliant filesystem paths used by mirage.
//!
//! # What is (and is not) on disk
//!
//! Mirage keeps its **configuration** on disk — profiles, topologies and
//! agents are user-authored documents that outlive any process, so they
//! live under `$XDG_CONFIG_HOME` and are read/written on demand.
//!
//! Mirage does **not** keep session or exec state on disk. Sessions,
//! execs, their process table, their output and their health live in
//! memory inside the supervisor daemon (see `mirage_supervisor`), and
//! clients reach them over the daemon's Unix socket. There is no
//! `def.json`, no `status.json`, no pid files, no stdout files and no
//! stdin FIFOs: an earlier design used those as an inter-process
//! communication channel between the CLI and a per-session host process,
//! which made lifecycle and cleanup ambiguous (a crashed writer left
//! state that looked live). The socket has an owner, and when the owner
//! dies the state goes with it.
//!
//! The one runtime directory that remains is a per-session scratch
//! directory ([`session_runtime_dir`]). It is *not* a communication
//! channel between mirage processes: it exists because emulator runtimes
//! are configured by path. rocjitsu's `LD_PRELOAD` interposer, for
//! instance, discovers its `SimulationConfig` by reading a file from
//! `$ROCJITSU_RUNTIME_DIR` and binds its daemon socket in the same place.
//! The supervisor materialises those assets there and removes the whole
//! directory when the session is destroyed.
//!
//! # Layout
//!
//! The layout follows the [XDG Base Directory Specification][xdg]:
//!
//! | Resource                 | Base directory     | Subpath                      |
//! |--------------------------|--------------------|------------------------------|
//! | Profiles                 | `$XDG_CONFIG_HOME` | `mirage/profile/<name>.json` |
//! | Topologies               | `$XDG_CONFIG_HOME` | `mirage/topology/<name>.json`|
//! | Agents                   | `$XDG_CONFIG_HOME` | `mirage/agent/<name>.json`   |
//! | Daemon socket + log      | `$XDG_RUNTIME_DIR` | `mirage/`                    |
//! | Per-session scratch      | `$XDG_RUNTIME_DIR` | `mirage/session/<id>/`       |
//! | Persistent state         | `$XDG_STATE_HOME`  | `mirage/`                    |
//!
//! Three environment variables provide direct overrides for the
//! per-app directories, bypassing the XDG base lookup:
//!
//! * `$MIRAGE_CONFIG` — overrides the mirage config dir (would otherwise
//!   be `$XDG_CONFIG_HOME/mirage`).
//! * `$MIRAGE_STATE` — overrides the mirage state dir (would otherwise
//!   be `$XDG_STATE_HOME/mirage`).
//! * `$MIRAGE_RUNTIME` — overrides the mirage runtime dir (would
//!   otherwise be `$XDG_RUNTIME_DIR/mirage`).
//!
//! ```text
//! $XDG_RUNTIME_DIR/mirage/
//!   mirage.sock       # supervisor daemon control socket
//!   mirage.lock       # exclusive lock held by the running daemon
//!   daemon.log        # daemon stderr when auto-started by the CLI
//!   session/<id>/     # per-session emulator scratch (rocjitsu config, …)
//! ```
//!
//! [xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::session::SessionId;

/// Root namespace under each XDG base directory.
pub const APP_NAMESPACE: &str = "mirage";

/// File name of the supervisor daemon's control socket.
pub const DAEMON_SOCKET_NAME: &str = "mirage.sock";

/// File name of the lock the running supervisor daemon holds exclusively.
pub const DAEMON_LOCK_NAME: &str = "mirage.lock";

/// File name the daemon's stderr is redirected to when the CLI starts it
/// in the background.
pub const DAEMON_LOG_NAME: &str = "daemon.log";

/// Process-wide test override root. When set (via [`set_test_root`]),
/// every directory lookup resolves under this root instead of consulting
/// the environment, keeping tests hermetic without mutating process
/// environment variables. `None` in normal operation.
static TEST_ROOT: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Current test override root, if any.
fn test_root() -> Option<PathBuf> {
    TEST_ROOT.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Returns `$XDG_CONFIG_HOME` (or `$HOME/.config` if unset).
#[must_use]
pub fn xdg_config_home() -> PathBuf {
    if let Some(root) = test_root() {
        return root.join("config");
    }
    if let Ok(p) = std::env::var("XDG_CONFIG_HOME")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    home_dir().join(".config")
}

/// Returns `$XDG_RUNTIME_DIR`.
///
/// Falls back to `$TMPDIR/mirage-<uid>` if unset (per XDG spec note).
#[must_use]
pub fn xdg_runtime_dir() -> PathBuf {
    if let Some(root) = test_root() {
        return root.join("runtime");
    }
    if let Ok(p) = std::env::var("XDG_RUNTIME_DIR")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    let tmp = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    let uid = nix::unistd::getuid().as_raw();
    PathBuf::from(tmp).join(format!("mirage-{uid}"))
}

/// Returns `$XDG_STATE_HOME` (or `$HOME/.local/state`).
#[must_use]
pub fn xdg_state_home() -> PathBuf {
    if let Some(root) = test_root() {
        return root.join("state");
    }
    if let Ok(p) = std::env::var("XDG_STATE_HOME")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    home_dir().join(".local").join("state")
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/"))
}

/// Returns the mirage config directory.
///
/// Honors `$MIRAGE_CONFIG` as a direct override; otherwise returns
/// `$XDG_CONFIG_HOME/mirage`.
#[must_use]
pub fn mirage_config_dir() -> PathBuf {
    if test_root().is_none()
        && let Ok(p) = std::env::var("MIRAGE_CONFIG")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    xdg_config_home().join(APP_NAMESPACE)
}

/// Returns the mirage persistent state directory.
///
/// Honors `$MIRAGE_STATE` as a direct override; otherwise returns
/// `$XDG_STATE_HOME/mirage`.
#[must_use]
pub fn mirage_state_dir() -> PathBuf {
    if test_root().is_none()
        && let Ok(p) = std::env::var("MIRAGE_STATE")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    xdg_state_home().join(APP_NAMESPACE)
}

/// Returns the mirage runtime directory.
///
/// Honors `$MIRAGE_RUNTIME` as a direct override; otherwise returns
/// `$XDG_RUNTIME_DIR/mirage`.
#[must_use]
pub fn mirage_runtime_dir() -> PathBuf {
    if test_root().is_none()
        && let Ok(p) = std::env::var("MIRAGE_RUNTIME")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    xdg_runtime_dir().join(APP_NAMESPACE)
}

/// Path of the supervisor daemon's Unix control socket.
#[must_use]
pub fn daemon_socket_path() -> PathBuf {
    mirage_runtime_dir().join(DAEMON_SOCKET_NAME)
}

/// Path of the lock file the running supervisor daemon holds.
///
/// The lock is what makes "is a daemon already running?" answerable
/// without races: a starting daemon takes an exclusive `flock` on this
/// file and only then binds the socket, so two daemons can never both
/// believe they own the control plane, and a socket left behind by a
/// crashed daemon is recognisable as stale (the lock is free).
#[must_use]
pub fn daemon_lock_path() -> PathBuf {
    mirage_runtime_dir().join(DAEMON_LOCK_NAME)
}

/// Path the daemon's stderr is redirected to when auto-started.
#[must_use]
pub fn daemon_log_path() -> PathBuf {
    mirage_runtime_dir().join(DAEMON_LOG_NAME)
}

/// Root directory for mirage profiles: `<mirage_config_dir>/profile`.
#[must_use]
pub fn profile_root() -> PathBuf {
    mirage_config_dir().join("profile")
}

/// Path to a specific profile file: `<profile_root>/<name>.json`.
///
/// Profile names are case-insensitive and always stored lowercase, so the
/// name is lowercased before building the path.
#[must_use]
pub fn profile_path(name: &str) -> PathBuf {
    profile_root().join(format!("{}.json", name.to_lowercase()))
}

/// Root directory for mirage topologies: `<mirage_config_dir>/topology`.
#[must_use]
pub fn topology_root() -> PathBuf {
    mirage_config_dir().join("topology")
}

/// Path to a specific topology file: `<topology_root>/<name>.json`.
#[must_use]
pub fn topology_path(name: &str) -> PathBuf {
    topology_root().join(format!("{name}.json"))
}

/// Root directory for mirage agents: `<mirage_config_dir>/agent`.
#[must_use]
pub fn agent_root() -> PathBuf {
    mirage_config_dir().join("agent")
}

/// Path to a specific agent file: `<agent_root>/<name>.json`.
///
/// Agent names are case-insensitive and always stored lowercase, so the
/// name is lowercased before building the path.
#[must_use]
pub fn agent_path(name: &str) -> PathBuf {
    agent_root().join(format!("{}.json", name.to_lowercase()))
}

/// Root of the per-session scratch directories:
/// `<mirage_runtime_dir>/session`.
#[must_use]
pub fn session_runtime_root() -> PathBuf {
    mirage_runtime_dir().join("session")
}

/// Per-session scratch directory for emulator runtime assets.
///
/// This holds files an emulator runtime is *configured by path* to find
/// (rocjitsu's synthesised `SimulationConfig`, its `config_path`
/// discovery file, and its daemon socket). It carries no mirage session
/// state and is never read to answer a control-plane query; the
/// supervisor removes it wholesale when the session is destroyed.
#[must_use]
pub fn session_runtime_dir(id: &SessionId) -> PathBuf {
    session_runtime_root().join(id.as_str())
}

/// Override directory resolution for tests.
///
/// When set, all `xdg_*` calls return paths rooted under this override
/// (and the `MIRAGE_*` env overrides are ignored to keep tests
/// hermetic). Specifically, the layout becomes:
///
/// ```text
/// <override>/config/
/// <override>/runtime/
/// <override>/state/
/// ```
///
/// This mutates a process-wide override rather than environment
/// variables, so callers should still hold [`test_env_lock`] for the
/// duration of any operation that touches mirage state on disk to avoid
/// clobbering by parallel tests.
pub fn set_test_root(path: &Path) {
    *TEST_ROOT.write().unwrap_or_else(|e| e.into_inner()) = Some(path.to_path_buf());
}

/// Clear a previously-installed [`set_test_root`] override.
pub fn clear_test_root() {
    *TEST_ROOT.write().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Process-wide lock to use whenever tests redirect mirage directories.
///
/// Tests should hold this for the duration of any operation that
/// touches mirage state on disk to avoid clobbering by parallel tests.
pub fn test_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn test_root_overrides() {
        let _g = test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        set_test_root(tmp.path());
        assert!(xdg_config_home().starts_with(tmp.path()));
        assert!(xdg_runtime_dir().starts_with(tmp.path()));
        assert_eq!(
            profile_path("foo"),
            tmp.path().join("config/mirage/profile/foo.json")
        );
    }

    #[test]
    fn daemon_paths_live_under_the_runtime_dir() {
        let _g = test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        set_test_root(tmp.path());
        let runtime = tmp.path().join("runtime/mirage");
        assert_eq!(daemon_socket_path(), runtime.join("mirage.sock"));
        assert_eq!(daemon_lock_path(), runtime.join("mirage.lock"));
        assert_eq!(daemon_log_path(), runtime.join("daemon.log"));
    }

    #[test]
    fn session_scratch_is_namespaced_per_session() {
        let _g = test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        set_test_root(tmp.path());
        let a = session_runtime_dir(&SessionId::new("a").unwrap());
        let b = session_runtime_dir(&SessionId::new("b").unwrap());
        assert_ne!(a, b);
        assert!(a.starts_with(session_runtime_root()));
    }

    #[test]
    fn profile_and_agent_names_are_lowercased() {
        let _g = test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        set_test_root(tmp.path());
        assert_eq!(profile_path("MI350X"), profile_path("mi350x"));
        assert_eq!(agent_path("MI350X"), agent_path("mi350x"));
        // Topologies are stored verbatim.
        assert_ne!(topology_path("MI350X"), topology_path("mi350x"));
    }
}
