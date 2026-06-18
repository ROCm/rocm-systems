//! Exec: a single command invocation within a session.
//!
//! An exec is started by writing an [`ExecDef`] to
//! `<session>/exec/<exec-id>/def.json`. The session's host picks the
//! file up (via filesystem polling) and spawns one process per node
//! whose stdio is wired through `node/<n>/{stdin,stdout,stderr}`.
//!
//! Lifecycle is published in two granularities:
//!
//! * `exec/<id>/status.json` — the aggregate [`ExecStatus`] for the exec
//!   as a whole (started, ended, overall exit code: the worst exit
//!   across nodes).
//! * `exec/<id>/node/<n>/{pid,exit_code}` — per-node state.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{common::MaybeRef, profile::FileMount, session::SessionId};

/// Concrete process arguments for one program invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecArgs {
    /// The program to run (absolute path or `$PATH`-resolved name).
    pub command: String,

    /// Arguments to the command, e.g. `["-c", "echo hello world"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,

    /// Extra environment variables to set for this run.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,

    /// Working directory; defaults to the session's `workdir`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir: Option<String>,
}

/// Identifier of a single exec within a session.
///
/// Ids are stable and follow the same rules as `SessionId`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ExecId(String);

impl ExecId {
    pub fn new(s: impl Into<String>) -> Result<Self, crate::session::IdError> {
        let s = s.into();
        crate::session::SessionId::new(&s)?; // reuse validator
        Ok(Self(s))
    }

    /// Generate a monotonic-ish id from a counter and the current time.
    pub fn from_counter(n: u32) -> Self {
        Self(format!("e-{n:06}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ExecId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for ExecId {
    type Err = crate::session::IdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl TryFrom<String> for ExecId {
    type Error = crate::session::IdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<ExecId> for String {
    fn from(id: ExecId) -> String {
        id.0
    }
}

/// A fully-qualified reference to an exec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecRef {
    pub session: SessionId,
    pub exec: ExecId,
}

/// A request to start an exec inside an existing session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecDef {
    /// When this exec was requested.
    pub timestamp: DateTime<Utc>,

    /// Session this exec belongs to.
    pub session: SessionId,

    /// What to run on the head node.
    pub exec: ExecArgs,

    /// Optional command to run on worker nodes.  If `None`, workers
    /// don't run any command (single-node exec).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_exec: Option<ExecArgs>,

    /// If `false`, the host removes the exec directory after it exits.
    #[serde(default)]
    pub keep: bool,
}

/// Aggregate status of an exec (all nodes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ExecStatus {
    /// `true` once the host has spawned at least one process.
    pub started: bool,

    /// `true` once every node process has exited.
    pub ended: bool,

    /// Aggregate exit code: `max(|exit_code|)` across all nodes; `0`
    /// if every node exited cleanly. `None` until `ended` is `true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,

    /// Per-node states, indexed by node id.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub nodes: BTreeMap<u32, NodeStatus>,
}

/// Status of a single node's process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct NodeStatus {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
}

/// Modifications to an exec applied by the emulator before launch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InjectionDef {
    pub wrapper: Option<String>,
    pub ld_preload: Option<String>,
    pub files: BTreeMap<String, MaybeRef<Vec<u8>>>,
    pub env: BTreeMap<String, String>,

    /// Host paths the emulator needs bind-mounted into each node's
    /// container so that the injected `LD_PRELOAD`/env paths resolve
    /// inside it. Empty for non-containerised sessions (where the
    /// injected paths are already host paths the workload can see). By
    /// convention these target locations live under `/mnt/mirage`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mounts: Vec<FileMount>,

    /// Host device nodes the emulator needs exposed to each node's
    /// container (`--device`), e.g. `/dev/kfd` and `/dev/dri` for AMD
    /// GPU access. Empty for non-containerised sessions and emulators
    /// that need no device passthrough.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub devices: Vec<String>,

    /// Supplementary groups the emulator needs added inside each node's
    /// container (`--group-add`), e.g. `video`/`render` so the workload
    /// may open the passed-through GPU device nodes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_id_validates() {
        assert!(ExecId::new("e-000001").is_ok());
        assert!(ExecId::new("/bad").is_err());
        assert_eq!(ExecId::from_counter(7).as_str(), "e-000007");
    }
}
