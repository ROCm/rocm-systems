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
//! memory inside the `mirage run` that owns them (see
//! `mirage_supervisor`), and a second terminal reaches them over that
//! run's own socket under `run/`. There is no `def.json`, no
//! `status.json`, no pid files, no stdout files and no stdin FIFOs: an
//! earlier design used those as an inter-process communication channel
//! between the CLI and a per-session host process, which made lifecycle
//! and cleanup ambiguous (a crashed writer left state that looked live).
//! Every socket has exactly one owning process, and when the owner dies
//! the state goes with it.
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
//! | Per-run control socket   | `$XDG_RUNTIME_DIR` | `mirage/run/<session>.sock`  |
//! | Per-session scratch      | `$XDG_RUNTIME_DIR` | `mirage/session/<id>/`       |
//!
//! Two environment variables provide direct overrides for the per-app
//! directories, bypassing the XDG base lookup:
//!
//! * `$MIRAGE_CONFIG` — overrides the mirage config dir (would otherwise
//!   be `$XDG_CONFIG_HOME/mirage`).
//! * `$MIRAGE_RUNTIME` — overrides the mirage runtime dir (would
//!   otherwise be `$XDG_RUNTIME_DIR/mirage`).
//!
//! There is no persistent *state* directory. Mirage writes nothing that
//! has to survive a reboot beyond its configuration: a session and
//! everything in it belongs to the `mirage run` that created it and is
//! gone when that process is.
//!
//! ```text
//! $XDG_RUNTIME_DIR/mirage/
//!   run/<session>.sock  # one socket per live `mirage run`
//!   session/<id>/       # per-session emulator scratch (rocjitsu config, …)
//! ```
//!
//! There is no daemon socket, lock file or daemon log: a run *is* the
//! server for its own session, and the set of sockets in `run/` is the
//! whole registry of what is live.
//!
//! [xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::session::SessionId;

/// Root namespace under each XDG base directory.
pub const APP_NAMESPACE: &str = "mirage";

/// Directory holding one socket per live `mirage run`.
pub const RUN_SOCKET_DIR: &str = "run";

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

/// Directory holding the sockets of every live run:
/// `<mirage_runtime_dir>/run`.
#[must_use]
pub fn run_socket_root() -> PathBuf {
    mirage_runtime_dir().join(RUN_SOCKET_DIR)
}

/// Path of the socket a `mirage run` serves for its session.
///
/// One socket per run, named after the session, rather than one
/// well-known socket for a daemon. That is not a detail: with a single
/// shared socket, "who owns this session?" needs a registry and a lock
/// protocol, and a socket file left behind by a crashed process is
/// indistinguishable from a live one. Here the socket *is* the
/// registration — connecting to it either reaches the owner or fails,
/// and failing is how a stale entry is recognised.
#[must_use]
pub fn run_socket_path(id: &SessionId) -> PathBuf {
    run_socket_root().join(format!("{}.sock", id.as_str()))
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
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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
    fn a_runs_socket_is_named_after_its_session() {
        // Per-run rather than one well-known path: the socket *is* the
        // registration, so two runs can never disagree about who owns a
        // session, and a socket nobody answers on is recognisably stale.
        let _g = test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        set_test_root(tmp.path());
        let runs = tmp.path().join("runtime/mirage/run");
        assert_eq!(run_socket_root(), runs);
        assert_eq!(
            run_socket_path(&SessionId::new("s-1").unwrap()),
            runs.join("s-1.sock")
        );
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
