//! [`LocalCtl`]: the control plane for commands that need no daemon.
//!
//! Not every `mirage` command involves a session. Listing profiles,
//! printing paths, and asking which emulator backends are compiled in are
//! all answered from this process — the first two from the config store,
//! the third from a link-time registry that is part of the binary.
//!
//! Routing those through the daemon would mean starting a background
//! process to read a directory. Users notice that, and rightly object to
//! it: a daemon appearing because you ran `mirage profile list` is
//! exactly the kind of surprise that makes a tool feel heavy.
//!
//! Session and exec operations are *not* implemented here. They return a
//! clear error rather than a plausible-looking empty answer, so a command
//! misclassified by [`daemon_need`] fails loudly instead of quietly
//! reporting that no sessions exist.

use async_trait::async_trait;
use mirage_core::agent::AgentDef;
use mirage_core::ctl::{CreateSessionRequest, MirageCtl, StreamPacketStream};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, ExecRef, ExecStatus};
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionDef, SessionHealth, SessionId, SessionState};
use mirage_core::topology::TopologyDef;
use std::time::Duration;

use crate::{CtlCmd, StateCmd};

/// A control plane serving only the daemon-free operations.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalCtl;

/// How a command relates to the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonNeed {
    /// Answer it in this process; never contact or start a daemon.
    Never,
    /// Use a daemon if one is already running, and answer it here
    /// otherwise. Starting one would be self-defeating.
    IfRunning,
    /// Meaningless without a daemon; start one on demand.
    Always,
}

/// How a command relates to the daemon.
///
/// Anything touching a session or an exec needs one. Configuration
/// commands do not: profiles, topologies and agents are files, and the
/// daemon reads them off disk when it needs them, so there is no cached
/// copy to keep coherent.
#[must_use]
pub fn daemon_need(cmd: &CtlCmd) -> DaemonNeed {
    match cmd {
        // Sessions, execs, and anything that streams from one.
        CtlCmd::Session(_)
        | CtlCmd::Exec(_)
        | CtlCmd::Run(_)
        | CtlCmd::Attach(_)
        | CtlCmd::Logs(_) => DaemonNeed::Always,

        // Purge stops every session and the daemon itself, and then
        // deletes the directories. It reaches a *running* daemon, but
        // must never start one: auto-starting a daemon to ask it to exit
        // is absurd on its face, and with `MIRAGE_AUTOSTART=0` — or on a
        // machine whose daemon has already crashed, which is when a user
        // reaches for purge — the connection simply fails and nothing is
        // deleted at all. Every step of `purge` already tolerates a
        // control plane with no sessions, so answering it locally is
        // correct when there is no daemon to talk to.
        //
        // Writing the builtins is pure filesystem work.
        CtlCmd::State(StateCmd::Purge { .. }) => DaemonNeed::IfRunning,
        CtlCmd::State(StateCmd::Builtins) => DaemonNeed::Never,

        CtlCmd::Profile(_)
        | CtlCmd::Topology(_)
        | CtlCmd::Agent(_)
        | CtlCmd::Emulators { .. }
        | CtlCmd::Paths => DaemonNeed::Never,
    }
}

/// The error returned for an operation that genuinely needs the daemon.
fn unsupported(operation: &str) -> MirageError {
    MirageError::daemon(format!(
        "{operation} needs the mirage daemon, but this command was \
         dispatched without one. This is a bug in mirage's command \
         routing; please report it."
    ))
}

#[async_trait]
impl MirageCtl for LocalCtl {
    async fn profile_list(&self) -> Result<Vec<String>> {
        mirage_core::store::profile_list()
    }

    async fn profile_get(&self, name: &str) -> Result<ProfileDef> {
        mirage_core::store::profile_get(name)
    }

    async fn profile_put(&self, profile: &ProfileDef) -> Result<()> {
        mirage_core::store::profile_put(profile)
    }

    async fn profile_delete(&self, name: &str) -> Result<()> {
        mirage_core::store::profile_delete(name)
    }

    async fn topology_list(&self) -> Result<Vec<String>> {
        mirage_core::store::topology_list()
    }

    async fn topology_get(&self, name: &str) -> Result<TopologyDef> {
        mirage_core::store::topology_get(name)
    }

    async fn topology_put(&self, name: &str, topology: &TopologyDef) -> Result<()> {
        mirage_core::store::topology_put(name, topology)
    }

    async fn topology_delete(&self, name: &str) -> Result<()> {
        mirage_core::store::topology_delete(name)
    }

    async fn agent_list(&self) -> Result<Vec<String>> {
        mirage_core::store::agent_list()
    }

    async fn agent_get(&self, name: &str) -> Result<AgentDef> {
        mirage_core::store::agent_get(name)
    }

    async fn agent_put(&self, name: &str, agent: &AgentDef) -> Result<()> {
        mirage_core::store::agent_put(name, agent)
    }

    async fn agent_delete(&self, name: &str) -> Result<()> {
        mirage_core::store::agent_delete(name)
    }

    // ---- Sessions and execs: not available without a daemon ------------

    async fn session_list(&self) -> Result<Vec<SessionId>> {
        Err(unsupported("listing sessions"))
    }

    async fn session_state(&self, _id: &SessionId) -> Result<SessionState> {
        Err(unsupported("reading session state"))
    }

    async fn session_health(&self, _id: &SessionId) -> Result<SessionHealth> {
        Err(unsupported("reading session health"))
    }

    async fn session_create(&self, _req: CreateSessionRequest) -> Result<SessionDef> {
        Err(unsupported("creating a session"))
    }

    async fn session_wait_ready(
        &self,
        _id: &SessionId,
        _timeout: Duration,
    ) -> Result<SessionHealth> {
        Err(unsupported("waiting for a session"))
    }

    async fn session_destroy(&self, _id: &SessionId) -> Result<()> {
        Err(unsupported("destroying a session"))
    }

    async fn exec_list(&self, _session: &SessionId) -> Result<Vec<ExecId>> {
        Err(unsupported("listing execs"))
    }

    async fn exec_status(&self, _r: &ExecRef) -> Result<ExecStatus> {
        Err(unsupported("reading exec status"))
    }

    async fn exec_get(&self, _r: &ExecRef) -> Result<ExecDef> {
        Err(unsupported("reading an exec"))
    }

    async fn session_exec(&self, _exec: &ExecDef) -> Result<ExecRef> {
        Err(unsupported("starting an exec"))
    }

    async fn session_attach(&self, _exec: &ExecRef) -> Result<StreamPacketStream> {
        Err(unsupported("attaching to an exec"))
    }

    async fn session_stdin(&self, _exec: &ExecRef, _data: &[u8]) -> Result<()> {
        Err(unsupported("writing to an exec's stdin"))
    }

    async fn session_stdin_close(&self, _exec: &ExecRef) -> Result<()> {
        Err(unsupported("closing an exec's stdin"))
    }

    async fn exec_resize(&self, _exec: &ExecRef, _rows: u16, _cols: u16) -> Result<()> {
        Err(unsupported("resizing an exec's terminal"))
    }

    async fn exec_signal(&self, _exec: &ExecRef, _sig: i32) -> Result<()> {
        Err(unsupported("signalling an exec"))
    }

    async fn exec_remove(&self, _exec: &ExecRef) -> Result<()> {
        Err(unsupported("removing an exec"))
    }

    async fn daemon_shutdown(&self) -> Result<()> {
        Err(unsupported("stopping the daemon"))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::{AgentCmd, ProfileCmd, SessionCmd, TopologyCmd};

    #[test]
    fn configuration_commands_do_not_need_a_daemon() {
        // Starting a background process to read a directory is exactly the
        // surprise this classification exists to avoid.
        for cmd in [
            CtlCmd::Paths,
            CtlCmd::Emulators { long: false },
            CtlCmd::Profile(ProfileCmd::List { long: false }),
            CtlCmd::Topology(TopologyCmd::List),
            CtlCmd::Agent(AgentCmd::List),
            CtlCmd::State(StateCmd::Builtins),
        ] {
            assert_eq!(daemon_need(&cmd), DaemonNeed::Never, "{cmd:?}");
        }
    }

    #[test]
    fn session_commands_need_a_daemon() {
        assert_eq!(
            daemon_need(&CtlCmd::Session(SessionCmd::List)),
            DaemonNeed::Always
        );
        // Purge stops the daemon and then deletes its directories: it has
        // to reach a running one, but starting one in order to ask it to
        // exit is both absurd and, under `MIRAGE_AUTOSTART=0`, a hard
        // failure that deletes nothing.
        assert_eq!(
            daemon_need(&CtlCmd::State(StateCmd::Purge {
                force: true,
                all: false
            })),
            DaemonNeed::IfRunning
        );
    }

    #[test]
    fn local_config_operations_work_without_a_daemon() {
        // A plain `#[test]` driving its own runtime, rather than
        // `#[tokio::test]`: the path override is guarded by a std mutex,
        // and holding that guard across an await is what the workspace's
        // `await_holding_lock` lint exists to prevent. `block_on` is not
        // an await point in this function, so the guard never spans one.
        let _g = mirage_core::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(dir.path());

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            let ctl = LocalCtl;
            assert!(ctl.profile_list().await.unwrap().is_empty());
            assert!(ctl.agent_list().await.unwrap().is_empty());
        });

        mirage_core::paths::clear_test_root();
    }

    #[tokio::test]
    async fn session_operations_fail_loudly_rather_than_looking_empty() {
        // A misclassified command must not quietly report "no sessions".
        let ctl = LocalCtl;
        let err = ctl.session_list().await.unwrap_err();
        assert!(err.to_string().contains("needs the mirage daemon"), "{err}");
    }
}
