//! XDG-compliant filesystem paths used by mirage.
//!
//! Mirage stores all of its state on disk so that:
//!
//! * the CLI, host, and any external tooling can read state without
//!   needing IPC,
//! * a crashed daemon/host can recover by re-reading state on restart,
//! * users can inspect what is happening with `ls`/`cat`.
//!
//! The layout follows the [XDG Base Directory Specification][xdg]:
//!
//! | Resource              | Base directory          | Subpath                       |
//! |-----------------------|-------------------------|-------------------------------|
//! | Profiles              | `$XDG_CONFIG_HOME`      | `mirage/profile/<name>.json`  |
//! | Sessions (runtime)    | `$XDG_RUNTIME_DIR`      | `mirage/session/<id>/...`     |
//! | Persistent state      | `$XDG_STATE_HOME`       | `mirage/`                     |
//!
//! Two environment variables provide direct overrides for the
//! per-app directories, bypassing the XDG base lookup:
//!
//! * `$MIRAGE_CONFIG` — overrides the mirage config dir (would otherwise
//!   be `$XDG_CONFIG_HOME/mirage`).
//! * `$MIRAGE_STATE` — overrides the mirage state dir (would otherwise
//!   be `$XDG_STATE_HOME/mirage`).
//! * `$MIRAGE_RUNTIME` — overrides the mirage runtime dir (would
//!   otherwise be `$XDG_RUNTIME_DIR/mirage`); session files live under
//!   `<mirage_runtime_dir>/session/<id>/...`.
//!
//! Per-session structure:
//!
//! ```text
//! $XDG_RUNTIME_DIR/mirage/session/<session>/
//!   def.json          # SessionDef
//!   health.json       # SessionHealth (written by host)
//!   container.json    # ContainerState (only for containerised sessions)
//!   node/             # per-node runtime state (one dir per rank)
//!     <rank>/
//!       pid           # pid of the node's host process
//!       host.log      # the node host's stderr log
//!       cid           # container id (containerised sessions only)
//!   exec/
//!     <exec-id>/
//!       def.json      # ExecDef
//!       status.json   # ExecStatus (started, exit_code, started_at, ended_at)
//!       node/
//!         <node-id>/
//!           stdin     # FIFO (named pipe)
//!           stdout    # plain file (merged stdout+stderr from the PTY)
//!           pid       # pid of the spawned process
//!           exit_code # exit code after the process terminates
//! ```
//!
//! [xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::{exec::ExecId, session::SessionId};

/// Root namespace under each XDG base directory.
pub const APP_NAMESPACE: &str = "mirage";

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
pub fn mirage_runtime_dir() -> PathBuf {
    if test_root().is_none()
        && let Ok(p) = std::env::var("MIRAGE_RUNTIME")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    xdg_runtime_dir().join(APP_NAMESPACE)
}

/// Root directory for mirage profiles: `<mirage_config_dir>/profile`.
pub fn profile_root() -> PathBuf {
    mirage_config_dir().join("profile")
}

/// Path to a specific profile file: `<profile_root>/<name>.json`.
///
/// Profile names are case-insensitive and always stored lowercase, so the
/// name is lowercased before building the path.
pub fn profile_path(name: &str) -> PathBuf {
    profile_root().join(format!("{}.json", name.to_lowercase()))
}

/// Root directory for mirage topologies: `<mirage_config_dir>/topology`.
pub fn topology_root() -> PathBuf {
    mirage_config_dir().join("topology")
}

/// Path to a specific topology file: `<topology_root>/<name>.json`.
pub fn topology_path(name: &str) -> PathBuf {
    topology_root().join(format!("{name}.json"))
}

/// Root directory for mirage agents: `<mirage_config_dir>/agent`.
pub fn agent_root() -> PathBuf {
    mirage_config_dir().join("agent")
}

/// Path to a specific agent file: `<agent_root>/<name>.json`.
///
/// Agent names are case-insensitive and always stored lowercase, so the
/// name is lowercased before building the path.
pub fn agent_path(name: &str) -> PathBuf {
    agent_root().join(format!("{}.json", name.to_lowercase()))
}

/// Root directory for all sessions: `<mirage_runtime_dir>/session`.
pub fn session_root() -> PathBuf {
    mirage_runtime_dir().join("session")
}

/// Per-session directory.
pub fn session_dir(id: &SessionId) -> PathBuf {
    session_root().join(id.as_str())
}

/// Layout helper for files within a session directory.
#[derive(Debug, Clone)]
pub struct SessionLayout {
    pub root: PathBuf,
}

impl SessionLayout {
    pub fn for_id(id: &SessionId) -> Self {
        Self {
            root: session_dir(id),
        }
    }

    pub fn for_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn def(&self) -> PathBuf {
        self.root.join("def.json")
    }
    pub fn health(&self) -> PathBuf {
        self.root.join("health.json")
    }
    /// Root directory holding per-node runtime state: `<session>/node`.
    pub fn node_root(&self) -> PathBuf {
        self.root.join("node")
    }
    /// Per-node runtime directory for `rank`: `<session>/node/<rank>`.
    ///
    /// Each node runs its own host process; this directory records that
    /// node host's `pid`, its `host.log`, and (for containerised
    /// sessions) the backing container's `cid`.
    pub fn node(&self, rank: u32) -> SessionNodeLayout {
        SessionNodeLayout {
            root: self.node_root().join(rank.to_string()),
        }
    }
    /// File recording the provider, network, and per-node containers
    /// backing a containerised session (a [`crate::container::ContainerState`]).
    /// Read by `mirage_core::container::teardown` to remove everything.
    pub fn container_json(&self) -> PathBuf {
        self.root.join("container.json")
    }
    pub fn exec_root(&self) -> PathBuf {
        self.root.join("exec")
    }
    pub fn exec(&self, id: &ExecId) -> ExecLayout {
        ExecLayout {
            root: self.exec_root().join(id.as_str()),
        }
    }
}

/// Layout helper for a single node's session-level runtime directory.
///
/// Each node of a session runs its own host process; this directory
/// (`<session>/node/<rank>`) records that node host's pid, log, and the
/// backing container id when the session is containerised.
#[derive(Debug, Clone)]
pub struct SessionNodeLayout {
    pub root: PathBuf,
}

impl SessionNodeLayout {
    /// Pid of the node's host process.
    pub fn pid(&self) -> PathBuf {
        self.root.join("pid")
    }
    /// The node host's stderr log.
    pub fn host_log(&self) -> PathBuf {
        self.root.join("host.log")
    }
    /// Container id backing this node (containerised sessions only).
    pub fn cid(&self) -> PathBuf {
        self.root.join("cid")
    }
}

/// Layout helper for files within an exec directory.
#[derive(Debug, Clone)]
pub struct ExecLayout {
    pub root: PathBuf,
}

impl ExecLayout {
    pub fn for_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    pub fn def(&self) -> PathBuf {
        self.root.join("def.json")
    }
    pub fn status(&self) -> PathBuf {
        self.root.join("status.json")
    }
    /// Single-shot signal request file. Written by the ctl, consumed
    /// by the host: the host reads the signal number from this file,
    /// forwards it to every node pid, then removes the file.
    pub fn signal(&self) -> PathBuf {
        self.root.join("signal")
    }
    pub fn node_root(&self) -> PathBuf {
        self.root.join("node")
    }
    pub fn node(&self, id: u32) -> NodeLayout {
        NodeLayout {
            root: self.node_root().join(id.to_string()),
        }
    }
}

/// Layout helper for files within a single node directory.
#[derive(Debug, Clone)]
pub struct NodeLayout {
    pub root: PathBuf,
}

impl NodeLayout {
    pub fn stdin(&self) -> PathBuf {
        self.root.join("stdin")
    }
    pub fn stdout(&self) -> PathBuf {
        self.root.join("stdout")
    }
    pub fn pid(&self) -> PathBuf {
        self.root.join("pid")
    }
    pub fn exit_code(&self) -> PathBuf {
        self.root.join("exit_code")
    }
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
}
