//! `MirageCtl`: the control-plane API the CLI uses to drive mirage.
//!
//! There is one "real" implementation, [`FileCtl`], which is backed by
//! the on-disk XDG layout described in [`crate::paths`]. The trait is
//! kept abstract so that tests can stub it out and so an eventual
//! daemon-based RPC implementation can drop in cleanly.

use std::path::Path;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use crate::{
    agent::AgentDef,
    error::{MirageError, Result},
    exec::{ExecDef, ExecId, ExecRef, ExecStatus},
    profile::ProfileDef,
    session::{SessionDef, SessionHealth, SessionId, SessionState},
    topology::TopologyDef,
};

pub type StreamPacketStream = Pin<Box<dyn Stream<Item = StreamPacket> + Send>>;

/// A frame published while attached to an exec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamPacket {
    /// Raw output from one of the streams.
    Output {
        node: u32,
        stream: StdStream,
        data: Vec<u8>,
    },
    /// A node has exited with the given exit code.
    NodeExit { node: u32, exit_code: i32 },
    /// The exec as a whole has finished.
    ExecExit { exit_code: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum StdStream {
    Stdin,
    Stdout,
    Stderr,
}

/// Request used to create a session.
#[derive(Debug, Clone)]
pub struct CreateSessionRequest {
    /// Pre-validated id; if `None` mirage generates one.
    pub id: Option<SessionId>,
    /// Inline or by-name profile reference.
    pub profile: crate::common::MaybeRef<ProfileDef>,
    /// Working directory used as the default cwd for execs.
    pub workdir: String,
    /// Optional container in which to run the session host.
    pub container: Option<crate::container::ContainerizedDef>,
}

/// The control-plane API the CLI talks to.
///
/// All methods are infallible-against-races: re-listing or re-reading is
/// always safe. Implementations may block briefly on filesystem I/O but
/// must not block waiting on processes; long-running operations
/// (attach, run) return streams.
pub trait MirageCtl: Send + Sync {
    /// Current wall-clock time in milliseconds since the unix epoch.
    fn timestamp(&self) -> u64;

    // ---- Profiles -------------------------------------------------------

    /// List the names of all profiles on disk.
    fn profile_list(&self) -> Result<Vec<String>>;
    fn profile_get(&self, name: &str) -> Result<ProfileDef>;
    /// Write a profile. Overwrites any existing profile with the same name.
    fn profile_put(&self, profile: &ProfileDef) -> Result<()>;
    fn profile_delete(&self, name: &str) -> Result<()>;

    // ---- Topologies -----------------------------------------------------

    /// List the names of all topologies on disk.
    fn topology_list(&self) -> Result<Vec<String>>;
    fn topology_get(&self, name: &str) -> Result<TopologyDef>;
    /// Write a topology under `name`. Overwrites any existing one.
    fn topology_put(&self, name: &str, topology: &TopologyDef) -> Result<()>;
    fn topology_delete(&self, name: &str) -> Result<()>;

    // ---- Agents ---------------------------------------------------------

    /// List the names of all agents on disk.
    fn agent_list(&self) -> Result<Vec<String>>;
    fn agent_get(&self, name: &str) -> Result<AgentDef>;
    /// Write an agent under `name`. Overwrites any existing one.
    fn agent_put(&self, name: &str, agent: &AgentDef) -> Result<()>;
    fn agent_delete(&self, name: &str) -> Result<()>;

    // ---- Sessions -------------------------------------------------------

    fn session_list(&self) -> Result<Vec<SessionId>>;
    fn session_state(&self, id: &SessionId) -> Result<SessionState>;
    /// Returns just the health snapshot. Returns a default
    /// "starting" health if the host has not yet written one.
    fn session_health(&self, id: &SessionId) -> Result<SessionHealth>;

    /// Create a new session by writing `def.json` and ensuring the
    /// directory layout exists. Does *not* spawn the host process; the
    /// caller is responsible for that (typically via
    /// `mirage_host::Host::run` or by execing `mirage-host`).
    fn session_create(&self, req: CreateSessionRequest) -> Result<SessionDef>;

    /// Wait until a session's health is `healthy=true` (or `terminal=true`).
    fn session_wait_ready(&self, id: &SessionId, timeout: Duration) -> Result<SessionHealth>;

    /// Tear down a session: signals the host (if any), and removes the
    /// session directory.
    fn session_destroy(&self, id: &SessionId) -> Result<()>;

    // ---- Execs ----------------------------------------------------------

    fn exec_list(&self, session: &SessionId) -> Result<Vec<ExecId>>;
    fn exec_status(&self, r: &ExecRef) -> Result<ExecStatus>;
    fn exec_get(&self, r: &ExecRef) -> Result<ExecDef>;

    /// Start an exec inside a session. Returns the new ref. The host
    /// picks up the request asynchronously; use `attach`/`status` to
    /// follow progress.
    fn session_exec(&self, exec: &ExecDef) -> Result<ExecRef>;

    /// Attach to an exec's streams.
    fn session_attach(&self, exec: &ExecRef) -> Result<StreamPacketStream>;

    /// Write to an exec's stdin (node 0). The underlying stdin is a
    /// FIFO created by the host.
    fn session_stdin(&self, exec: &ExecRef, data: &[u8]) -> Result<()>;

    /// Send a signal to an exec's processes. `sig` follows libc
    /// numbering (e.g. `SIGINT == 2`).
    fn exec_signal(&self, exec: &ExecRef, sig: i32) -> Result<()>;

    /// Remove an exec's directory. If the exec is still running its
    /// node processes (and their descendants) are forcefully terminated
    /// first, so a remove never leaves orphaned children behind.
    fn exec_remove(&self, exec: &ExecRef) -> Result<()>;

    // ---- Daemon ---------------------------------------------------------

    /// Best-effort: stop every session host and exit any running daemon.
    fn daemon_shutdown(&self) -> Result<()>;
}

// =============================================================================
// FileCtl: the file-backed implementation.
// =============================================================================

/// A `MirageCtl` backed by the XDG filesystem layout.
#[derive(Debug, Clone, Default)]
pub struct FileCtl;

impl FileCtl {
    pub fn new() -> Self {
        FileCtl
    }
}

impl MirageCtl for FileCtl {
    fn timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    // ---- Profiles -------------------------------------------------------

    fn profile_list(&self) -> Result<Vec<String>> {
        let root = crate::paths::profile_root();
        if !root.exists() {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&root).map_err(|e| MirageError::Io {
            path: root.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| MirageError::Io {
                path: root.clone(),
                source: e,
            })?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(stem) = name.strip_suffix(".json") {
                out.push(stem.to_string());
            }
        }
        out.sort();
        Ok(out)
    }

    fn profile_get(&self, name: &str) -> Result<ProfileDef> {
        let p = crate::paths::profile_path(name);
        if !p.exists() {
            return Err(MirageError::ProfileNotFound(name.to_string()));
        }
        crate::state::read_json(&p)
    }

    fn profile_put(&self, profile: &ProfileDef) -> Result<()> {
        let p = crate::paths::profile_path(&profile.name);
        crate::state::write_json(&p, profile)
    }

    fn profile_delete(&self, name: &str) -> Result<()> {
        let p = crate::paths::profile_path(name);
        if !p.exists() {
            return Err(MirageError::ProfileNotFound(name.to_string()));
        }
        std::fs::remove_file(&p).map_err(|e| MirageError::Io { path: p, source: e })
    }

    // ---- Topologies -----------------------------------------------------

    fn topology_list(&self) -> Result<Vec<String>> {
        crate::topology::store::list()
    }

    fn topology_get(&self, name: &str) -> Result<TopologyDef> {
        crate::topology::store::get(name)
    }

    fn topology_put(&self, name: &str, topology: &TopologyDef) -> Result<()> {
        crate::topology::store::put(name, topology).map(|_| ())
    }

    fn topology_delete(&self, name: &str) -> Result<()> {
        let p = crate::paths::topology_path(name);
        if !p.exists() {
            return Err(MirageError::Other(format!("topology not found: {name}")));
        }
        std::fs::remove_file(&p).map_err(|e| MirageError::Io { path: p, source: e })
    }

    // ---- Agents ---------------------------------------------------------

    fn agent_list(&self) -> Result<Vec<String>> {
        crate::agent::store::list()
    }

    fn agent_get(&self, name: &str) -> Result<AgentDef> {
        crate::agent::store::get(name)
    }

    fn agent_put(&self, name: &str, agent: &AgentDef) -> Result<()> {
        crate::agent::store::put(name, agent).map(|_| ())
    }

    fn agent_delete(&self, name: &str) -> Result<()> {
        let p = crate::paths::agent_path(name);
        if !p.exists() {
            return Err(MirageError::Other(format!("agent not found: {name}")));
        }
        std::fs::remove_file(&p).map_err(|e| MirageError::Io { path: p, source: e })
    }

    // ---- Sessions -------------------------------------------------------

    fn session_list(&self) -> Result<Vec<SessionId>> {
        let root = crate::paths::session_root();
        if !root.exists() {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&root).map_err(|e| MirageError::Io {
            path: root.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| MirageError::Io {
                path: root.clone(),
                source: e,
            })?;
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if let Ok(id) = SessionId::new(name) {
                out.push(id);
            }
        }
        out.sort();
        Ok(out)
    }

    fn session_state(&self, id: &SessionId) -> Result<SessionState> {
        let layout = crate::paths::SessionLayout::for_id(id);
        if !layout.def().exists() {
            return Err(MirageError::SessionNotFound(id.to_string()));
        }
        let def: SessionDef = crate::state::read_json(&layout.def())?;
        let health = self.session_health(id)?;
        Ok(SessionState { def, health })
    }

    fn session_health(&self, id: &SessionId) -> Result<SessionHealth> {
        let layout = crate::paths::SessionLayout::for_id(id);
        if let Some(h) = crate::state::read_json_opt::<SessionHealth>(&layout.health())? {
            return Ok(h);
        }
        Ok(SessionHealth {
            timestamp: Utc::now(),
            healthy: false,
            state: Some("starting".to_string()),
            terminal: false,
            message: None,
        })
    }

    fn session_create(&self, req: CreateSessionRequest) -> Result<SessionDef> {
        let id = match req.id {
            Some(i) => i,
            None => SessionId::generate(),
        };
        let layout = crate::paths::SessionLayout::for_id(&id);
        if layout.def().exists() {
            return Err(MirageError::SessionExists(id.to_string()));
        }
        std::fs::create_dir_all(&layout.root).map_err(|e| MirageError::Io {
            path: layout.root.clone(),
            source: e,
        })?;
        std::fs::create_dir_all(layout.exec_root()).map_err(|e| MirageError::Io {
            path: layout.exec_root(),
            source: e,
        })?;
        let def = SessionDef {
            id: id.clone(),
            profile: req.profile,
            container: req.container,
            workdir: req.workdir,
            created_at: Utc::now(),
        };
        crate::state::write_json(&layout.def(), &def)?;
        Ok(def)
    }

    fn session_wait_ready(&self, id: &SessionId, timeout: Duration) -> Result<SessionHealth> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let h = self.session_health(id)?;
            if h.healthy || h.terminal {
                return Ok(h);
            }
            if std::time::Instant::now() >= deadline {
                return Err(MirageError::HostStartTimeout(timeout));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn session_destroy(&self, id: &SessionId) -> Result<()> {
        let layout = crate::paths::SessionLayout::for_id(id);
        if !layout.root.exists() {
            return Err(MirageError::SessionNotFound(id.to_string()));
        }

        // If this session is containerised and we know the container
        // id, ask the configured provider to remove it. This is the
        // authoritative shutdown for containerised sessions: killing
        // the container terminates the host, all execs, and their
        // children. We do it before signalling the host so the
        // container runtime sees the request first.
        let def: Option<crate::session::SessionDef> =
            crate::state::read_json_opt(&layout.def())?;
        if let Some(def) = def.as_ref()
            && let Some(container) = def.container.as_ref()
            && let Some(cid) = crate::state::read_small_str(&layout.container_id())?
        {
            stop_container(&container.provider, &cid);
        }

        // signal the host process if any (this is also the fallback
        // path when there is no container, or when the container has
        // already exited).
        if let Some(pid_str) = crate::state::read_small_str(&layout.host_pid())? {
            if let Ok(pid) = pid_str.parse::<i32>() {
                let _ = nix::sys::signal::kill(
                    nix::unistd::Pid::from_raw(pid),
                    nix::sys::signal::Signal::SIGTERM,
                );
                // wait briefly for clean exit
                for _ in 0..50 {
                    if !process_alive(pid) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                if process_alive(pid) {
                    let _ = nix::sys::signal::kill(
                        nix::unistd::Pid::from_raw(pid),
                        nix::sys::signal::Signal::SIGKILL,
                    );
                }
            }
        }

        // Reap any exec node processes (and their descendants) that may
        // still be alive. The graceful host SIGTERM above normally tears
        // these down, but if we had to SIGKILL the host it never got the
        // chance to clean up its children — so kill every node's
        // process-group directly to guarantee nothing is left running.
        let leftover = all_session_node_pids(&layout);
        for &pid in &leftover {
            kill_process_group(pid, nix::sys::signal::Signal::SIGKILL);
        }

        std::fs::remove_dir_all(&layout.root).map_err(|e| MirageError::Io {
            path: layout.root.clone(),
            source: e,
        })?;
        Ok(())
    }

    // ---- Execs ----------------------------------------------------------

    fn exec_list(&self, session: &SessionId) -> Result<Vec<ExecId>> {
        let layout = crate::paths::SessionLayout::for_id(session);
        let root = layout.exec_root();
        if !root.exists() {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&root).map_err(|e| MirageError::Io {
            path: root.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| MirageError::Io {
                path: root.clone(),
                source: e,
            })?;
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if let Ok(id) = ExecId::new(name) {
                out.push(id);
            }
        }
        out.sort();
        Ok(out)
    }

    fn exec_status(&self, r: &ExecRef) -> Result<ExecStatus> {
        let layout = crate::paths::SessionLayout::for_id(&r.session).exec(&r.exec);
        if !layout.root.exists() {
            return Err(MirageError::ExecNotFound(r.exec.to_string()));
        }
        let status: Option<ExecStatus> = crate::state::read_json_opt(&layout.status())?;
        Ok(status.unwrap_or_default())
    }

    fn exec_get(&self, r: &ExecRef) -> Result<ExecDef> {
        let layout = crate::paths::SessionLayout::for_id(&r.session).exec(&r.exec);
        if !layout.def().exists() {
            return Err(MirageError::ExecNotFound(r.exec.to_string()));
        }
        crate::state::read_json(&layout.def())
    }

    fn session_exec(&self, exec: &ExecDef) -> Result<ExecRef> {
        let session_layout = crate::paths::SessionLayout::for_id(&exec.session);
        if !session_layout.def().exists() {
            return Err(MirageError::SessionNotFound(exec.session.to_string()));
        }
        // allocate next exec id by counting existing exec/* directories.
        let exec_id = next_exec_id(&session_layout.exec_root())?;
        let exec_layout = session_layout.exec(&exec_id);
        std::fs::create_dir_all(&exec_layout.root).map_err(|e| MirageError::Io {
            path: exec_layout.root.clone(),
            source: e,
        })?;
        // write the def last (host watches for def.json).
        crate::state::write_json(&exec_layout.def(), exec)?;
        Ok(ExecRef {
            session: exec.session.clone(),
            exec: exec_id,
        })
    }

    fn session_attach(&self, exec: &ExecRef) -> Result<StreamPacketStream> {
        let layout = crate::paths::SessionLayout::for_id(&exec.session).exec(&exec.exec);
        if !layout.root.exists() {
            return Err(MirageError::ExecNotFound(exec.exec.to_string()));
        }
        Ok(crate::attach::attach_stream(layout))
    }

    fn session_stdin(&self, exec: &ExecRef, data: &[u8]) -> Result<()> {
        let layout = crate::paths::SessionLayout::for_id(&exec.session).exec(&exec.exec);
        let stdin = layout.node(0).stdin();
        if !stdin.exists() {
            return Err(MirageError::ExecNotFound(format!(
                "{}/node/0/stdin",
                exec.exec
            )));
        }
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&stdin)
            .map_err(|e| MirageError::Io {
                path: stdin.clone(),
                source: e,
            })?;
        f.write_all(data).map_err(|e| MirageError::Io {
            path: stdin.clone(),
            source: e,
        })
    }

    fn exec_signal(&self, exec: &ExecRef, sig: i32) -> Result<()> {
        let layout = crate::paths::SessionLayout::for_id(&exec.session).exec(&exec.exec);
        if !layout.root.exists() {
            return Err(MirageError::ExecNotFound(exec.exec.to_string()));
        }
        // Validate the signal so we don't ask the host to deliver
        // something that nix would refuse later.
        if nix::sys::signal::Signal::try_from(sig).is_err() {
            return Err(MirageError::other(format!("invalid signal: {sig}")));
        }
        // Write the request file. The host's poll loop picks it up,
        // forwards the signal to every node pid, and removes the file.
        crate::state::write_bytes(&layout.signal(), sig.to_string().as_bytes())
    }

    fn exec_remove(&self, exec: &ExecRef) -> Result<()> {
        let layout = crate::paths::SessionLayout::for_id(&exec.session).exec(&exec.exec);
        let status = self.exec_status(exec)?;
        if status.started && !status.ended {
            // Forcefully terminate the running node processes (and their
            // descendants) so a remove never leaves orphaned children
            // behind. Each node runs in its own session/process-group
            // (the host calls `setsid()`), so killing the group reaps the
            // whole tree. SIGTERM first for a chance to clean up, then
            // SIGKILL anything still alive.
            let pids = exec_node_pids(&layout, &status);
            for &pid in &pids {
                kill_process_group(pid, nix::sys::signal::Signal::SIGTERM);
            }
            for _ in 0..25 {
                if pids.iter().all(|&p| !process_alive(p)) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            for &pid in &pids {
                if process_alive(pid) {
                    kill_process_group(pid, nix::sys::signal::Signal::SIGKILL);
                }
            }
        }
        match std::fs::remove_dir_all(&layout.root) {
            Ok(()) => Ok(()),
            // The host may have already cleaned up the directory (e.g. a
            // `keep=false` exec whose child we just killed); treat an
            // already-gone directory as success.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(MirageError::Io {
                path: layout.root.clone(),
                source: e,
            }),
        }
    }

    fn daemon_shutdown(&self) -> Result<()> {
        for id in self.session_list()? {
            // best effort
            let _ = self.session_destroy(&id);
        }
        Ok(())
    }
}

fn process_alive(pid: i32) -> bool {
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
}

/// Send `sig` to the process group led by `pid`. The host launches each
/// node with `setsid()`, so the node pid is its own process-group leader
/// and signalling the negative pid reaches the whole descendant tree
/// (the program plus anything it spawned). Falls back to signalling the
/// single pid if the group send fails (e.g. the process already exited).
fn kill_process_group(pid: i32, sig: nix::sys::signal::Signal) {
    if pid <= 0 {
        return;
    }
    if nix::sys::signal::kill(nix::unistd::Pid::from_raw(-pid), sig).is_err() {
        let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), sig);
    }
}

/// Collect the node process ids for a single exec. Prefers the pids
/// recorded in `status.json`, falling back to the per-node `pid` files
/// on disk in case the status is stale or incomplete.
fn exec_node_pids(layout: &crate::paths::ExecLayout, status: &ExecStatus) -> Vec<i32> {
    let mut pids: Vec<i32> = status
        .nodes
        .values()
        .filter_map(|n| n.pid)
        .map(|p| p as i32)
        .collect();
    if pids.is_empty()
        && let Ok(nodes) = std::fs::read_dir(layout.node_root())
    {
        for n in nodes.flatten() {
            if let Ok(Some(s)) = crate::state::read_small_str(&n.path().join("pid"))
                && let Ok(pid) = s.parse::<i32>()
            {
                pids.push(pid);
            }
        }
    }
    pids
}

/// Collect every node pid across every exec of a session by walking the
/// on-disk `exec/*/node/*/pid` files. Used as a last-resort reap when
/// tearing a session down.
fn all_session_node_pids(layout: &crate::paths::SessionLayout) -> Vec<i32> {
    let mut pids = Vec::new();
    let Ok(execs) = std::fs::read_dir(layout.exec_root()) else {
        return pids;
    };
    for e in execs.flatten() {
        let exec_layout = crate::paths::ExecLayout::for_root(e.path());
        let Ok(nodes) = std::fs::read_dir(exec_layout.node_root()) else {
            continue;
        };
        for n in nodes.flatten() {
            if let Ok(Some(s)) = crate::state::read_small_str(&n.path().join("pid"))
                && let Ok(pid) = s.parse::<i32>()
            {
                pids.push(pid);
            }
        }
    }
    pids
}

/// Best-effort container removal. Runs `<provider> rm -f <id>` and
/// ignores any error: the caller still proceeds to remove the
/// on-disk session, and signalling the host pid catches any case
/// where the container runtime is unavailable.
fn stop_container(provider: &str, container_id: &str) {
    let _ = std::process::Command::new(provider)
        .args(["rm", "-f", container_id])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

fn next_exec_id(exec_root: &Path) -> Result<ExecId> {
    let mut max: u32 = 0;
    if exec_root.exists() {
        for entry in std::fs::read_dir(exec_root).map_err(|e| MirageError::Io {
            path: exec_root.to_path_buf(),
            source: e,
        })? {
            let entry = entry.map_err(|e| MirageError::Io {
                path: exec_root.to_path_buf(),
                source: e,
            })?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(n) = name.strip_prefix("e-")
                && let Ok(v) = n.parse::<u32>()
                && v >= max
            {
                max = v + 1;
            }
        }
    }
    Ok(ExecId::from_counter(max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::MaybeRef;
    use crate::profile::ProfileDef;

    /// RAII guard: holds the env lock for the duration of a test, and
    /// owns a tempdir whose lifetime must extend until after the lock
    /// is released. Drop order: tempdir first, then lock.
    struct TestEnv {
        _dir: tempfile::TempDir,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    fn fresh_ctl() -> (FileCtl, TestEnv) {
        let guard = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());
        (
            FileCtl::new(),
            TestEnv {
                _dir: dir,
                _guard: guard,
            },
        )
    }

    fn dummy_profile(name: &str) -> ProfileDef {
        ProfileDef {
            name: name.to_string(),
            description: None,
            emulator: crate::emulator::EmulatorDef {
                emulator: "noop".to_string(),
                plugins: Default::default(),
                exec_mode: Default::default(),
                options: Default::default(),
                topology: MaybeRef::Owned(crate::registry::default_topology()),
            },
        }
    }

    #[test]
    fn profiles_round_trip() {
        let (ctl, _env) = fresh_ctl();
        assert!(ctl.profile_list().unwrap().is_empty());
        ctl.profile_put(&dummy_profile("a")).unwrap();
        ctl.profile_put(&dummy_profile("b")).unwrap();
        let names = ctl.profile_list().unwrap();
        assert_eq!(names, vec!["a", "b"]);
        let p = ctl.profile_get("a").unwrap();
        assert_eq!(p.name, "a");
        ctl.profile_delete("a").unwrap();
        assert_eq!(ctl.profile_list().unwrap(), vec!["b"]);
        assert!(matches!(
            ctl.profile_get("a"),
            Err(MirageError::ProfileNotFound(_))
        ));
    }

    #[test]
    fn sessions_create_list_destroy() {
        let (ctl, _env) = fresh_ctl();
        ctl.profile_put(&dummy_profile("p")).unwrap();
        let def = ctl
            .session_create(CreateSessionRequest {
                id: Some(SessionId::new("s1").unwrap()),
                profile: MaybeRef::Ref("p".to_string()),
                workdir: "/tmp".to_string(),
                container: None,
            })
            .unwrap();
        assert_eq!(def.id.as_str(), "s1");
        let list = ctl.session_list().unwrap();
        assert_eq!(list.len(), 1);
        let st = ctl.session_state(&def.id).unwrap();
        assert_eq!(st.def.id, def.id);
        assert!(!st.health.healthy);
        ctl.session_destroy(&def.id).unwrap();
        assert!(ctl.session_list().unwrap().is_empty());
    }

    #[test]
    fn session_create_duplicate() {
        let (ctl, _env) = fresh_ctl();
        let req = || CreateSessionRequest {
            id: Some(SessionId::new("dup").unwrap()),
            profile: MaybeRef::Ref("p".to_string()),
            workdir: "/tmp".to_string(),
            container: None,
        };
        ctl.session_create(req()).unwrap();
        assert!(matches!(
            ctl.session_create(req()),
            Err(MirageError::SessionExists(_))
        ));
    }

    #[test]
    fn exec_allocates_increasing_ids() {
        let (ctl, _env) = fresh_ctl();
        let s = SessionId::new("s").unwrap();
        ctl.session_create(CreateSessionRequest {
            id: Some(s.clone()),
            profile: MaybeRef::Ref("p".to_string()),
            workdir: "/tmp".to_string(),
            container: None,
        })
        .unwrap();
        let def = ExecDef {
            timestamp: Utc::now(),
            session: s.clone(),
            exec: crate::exec::ExecArgs {
                command: "/bin/true".to_string(),
                args: vec![],
                env: Default::default(),
                workdir: None,
            },
            worker_exec: None,
            keep: true,
        };
        let r0 = ctl.session_exec(&def).unwrap();
        let r1 = ctl.session_exec(&def).unwrap();
        assert_eq!(r0.exec.as_str(), "e-000000");
        assert_eq!(r1.exec.as_str(), "e-000001");
        let list = ctl.exec_list(&s).unwrap();
        assert_eq!(list.len(), 2);
    }

    fn make_exec_dir(ctl: &FileCtl, session: &SessionId) -> ExecRef {
        let def = ExecDef {
            timestamp: Utc::now(),
            session: session.clone(),
            exec: crate::exec::ExecArgs {
                command: "/bin/true".to_string(),
                args: vec![],
                env: Default::default(),
                workdir: None,
            },
            worker_exec: None,
            keep: true,
        };
        ctl.session_exec(&def).unwrap()
    }

    #[test]
    fn exec_signal_writes_request_file() {
        let (ctl, _env) = fresh_ctl();
        let s = SessionId::new("sigs").unwrap();
        ctl.session_create(CreateSessionRequest {
            id: Some(s.clone()),
            profile: MaybeRef::Ref("p".to_string()),
            workdir: "/tmp".to_string(),
            container: None,
        })
        .unwrap();
        let r = make_exec_dir(&ctl, &s);
        // Even with no node pids yet, the request must be enqueued
        // for the host to pick up later.
        ctl.exec_signal(&r, libc::SIGTERM).unwrap();
        let layout = crate::paths::SessionLayout::for_id(&s).exec(&r.exec);
        let raw = std::fs::read_to_string(layout.signal()).unwrap();
        assert_eq!(raw.trim().parse::<i32>().unwrap(), libc::SIGTERM);
    }

    #[test]
    fn exec_signal_rejects_invalid_number() {
        let (ctl, _env) = fresh_ctl();
        let s = SessionId::new("sigi").unwrap();
        ctl.session_create(CreateSessionRequest {
            id: Some(s.clone()),
            profile: MaybeRef::Ref("p".to_string()),
            workdir: "/tmp".to_string(),
            container: None,
        })
        .unwrap();
        let r = make_exec_dir(&ctl, &s);
        let err = ctl.exec_signal(&r, 9999).unwrap_err();
        assert!(matches!(err, MirageError::Other(_)), "got {err:?}");
        // and no signal file should have been written
        let layout = crate::paths::SessionLayout::for_id(&s).exec(&r.exec);
        assert!(!layout.signal().exists());
    }

    #[test]
    fn exec_signal_missing_exec_returns_not_found() {
        let (ctl, _env) = fresh_ctl();
        let s = SessionId::new("nope").unwrap();
        ctl.session_create(CreateSessionRequest {
            id: Some(s.clone()),
            profile: MaybeRef::Ref("p".to_string()),
            workdir: "/tmp".to_string(),
            container: None,
        })
        .unwrap();
        let r = ExecRef {
            session: s,
            exec: ExecId::new("e-999999").unwrap(),
        };
        assert!(matches!(
            ctl.exec_signal(&r, libc::SIGTERM),
            Err(MirageError::ExecNotFound(_))
        ));
    }

    #[test]
    fn session_destroy_container_invokes_provider() {
        let (ctl, env) = fresh_ctl();
        // Mock provider script: appends every invocation to a log
        // file so we can assert it was called with "rm -f <id>".
        let tmp_dir = env._dir.path();
        let log = tmp_dir.join("provider.log");
        let provider = tmp_dir.join("mock-provider.sh");
        std::fs::write(
            &provider,
            format!("#!/bin/sh\necho \"$@\" >> {}\n", log.display()),
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();

        let s = SessionId::new("c1").unwrap();
        ctl.session_create(CreateSessionRequest {
            id: Some(s.clone()),
            profile: MaybeRef::Ref("p".to_string()),
            workdir: "/tmp".to_string(),
            container: Some(crate::container::ContainerizedDef {
                provider: provider.to_string_lossy().to_string(),
                image: "ignored:latest".to_string(),
                files: vec![],
                network: crate::container::NetworkConfigDef::None,
            }),
        })
        .unwrap();
        // Simulate the host having recorded a container id.
        let layout = crate::paths::SessionLayout::for_id(&s);
        crate::state::write_bytes(&layout.container_id(), b"mirage-c1-xyz").unwrap();

        ctl.session_destroy(&s).unwrap();

        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            recorded.contains("rm -f mirage-c1-xyz"),
            "expected provider call, got: {recorded:?}"
        );
        assert!(!layout.root.exists(), "session dir should be removed");
    }

    #[test]
    fn session_destroy_without_container_id_skips_provider() {
        let (ctl, env) = fresh_ctl();
        let tmp_dir = env._dir.path();
        let log = tmp_dir.join("provider.log");
        let provider = tmp_dir.join("mock-provider.sh");
        std::fs::write(
            &provider,
            format!("#!/bin/sh\necho \"$@\" >> {}\n", log.display()),
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();

        let s = SessionId::new("c2").unwrap();
        ctl.session_create(CreateSessionRequest {
            id: Some(s.clone()),
            profile: MaybeRef::Ref("p".to_string()),
            workdir: "/tmp".to_string(),
            container: Some(crate::container::ContainerizedDef {
                provider: provider.to_string_lossy().to_string(),
                image: "ignored:latest".to_string(),
                files: vec![],
                network: crate::container::NetworkConfigDef::None,
            }),
        })
        .unwrap();
        // No container.id was ever written → provider must not be
        // called.
        ctl.session_destroy(&s).unwrap();
        assert!(!log.exists(), "provider must not be invoked");
    }
}
