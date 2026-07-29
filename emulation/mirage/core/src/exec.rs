//! Exec: a single command invocation within a session.
//!
//! An exec is requested with an [`ExecDef`] and identified afterwards by
//! an [`ExecRef`]. The supervisor expands it into a process grid —
//! `num_nodes * nproc_per_node` processes, each with piped stdio — and
//! reports back through [`ExecStatus`]:
//!
//! * the aggregate: started, ended, and the overall exit code (the exit
//!   furthest from zero across every process, so a job where one worker
//!   crashed is reported as a failure);
//! * per process: its pid and its own exit code, keyed by global rank.
//!
//! None of this is on disk. An exec used to be started by writing a
//! `def.json` into a directory a per-session host polled, with its
//! output tailed from a file and its completion read from a
//! `status.json`; see [`crate::paths`] for why that changed.

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
        SessionId::new(&s)?; // reuse validator
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
        // `pad` so the CLI's aligned `{:<14}` columns work; `write_str`
        // would ignore the width.
        f.pad(&self.0)
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

    /// Number of workload processes to launch per node. Defaults to `1`.
    /// Values greater than one give each node `nproc_per_node` processes,
    /// each with a distinct `LOCAL_RANK` (`0..nproc_per_node`) and global
    /// `RANK`, so `torch.distributed` runs without a launcher like
    /// `torchrun --nproc-per-node`. The total world size is
    /// `num_nodes * nproc_per_node`.
    #[serde(default = "one_proc", skip_serializing_if = "is_one_proc")]
    pub nproc_per_node: u32,

    /// Run the workload on a pseudo-terminal instead of pipes.
    ///
    /// Interactive programs need a real terminal: a shell only prints a
    /// prompt, echoes input and enables line editing and job control when
    /// `isatty(0)` is true. `mirage run -- bash` is unusable without one.
    ///
    /// The cost is that a PTY has a single stream, so stdout and stderr
    /// are merged and reported as [`StdStream::Stdout`], and the terminal
    /// may translate output (`\n` becomes `\r\n`). Non-interactive runs
    /// therefore default to pipes, where the two streams stay distinct
    /// and output is byte-exact.
    ///
    /// [`StdStream::Stdout`]: crate::ctl::StdStream::Stdout
    #[serde(default)]
    pub tty: bool,

    /// Initial terminal height, in character cells. Ignored without
    /// [`ExecDef::tty`]; `0` means "use a sensible default".
    #[serde(default, skip_serializing_if = "is_zero")]
    pub tty_rows: u16,

    /// Initial terminal width, in character cells. Ignored without
    /// [`ExecDef::tty`]; `0` means "use a sensible default".
    #[serde(default, skip_serializing_if = "is_zero")]
    pub tty_cols: u16,

    /// Whether the client intends to keep this exec after it finishes.
    ///
    /// Advisory: the supervisor never removes an exec on its own, because
    /// doing so would race a client attaching to read its result. The CLI
    /// calls `exec_remove` explicitly when this is `false`. Independently,
    /// a session forgets its oldest *finished* execs once too many
    /// accumulate, so history is bounded either way.
    #[serde(default)]
    pub keep: bool,
}

fn one_proc() -> u32 {
    1
}

fn is_zero(n: &u16) -> bool {
    *n == 0
}

fn is_one_proc(n: &u32) -> bool {
    *n == 1
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

    /// Host paths of shared libraries the workload needs available. For
    /// a containerised session each is bind-mounted into `/mnt/mirage/lib`
    /// (preserving its file name) and that directory is prepended to
    /// `LD_LIBRARY_PATH`, so the dynamic loader prefers them over the
    /// container image's own copies. Used to supply libraries that the
    /// bind-mounted mirage binary and emulator interposers were built
    /// against but the image's system libraries are too old to satisfy
    /// (e.g. a newer `libc.so.6` or `libstdc++.so.6`). Ignored for
    /// non-containerised sessions, where the workload already sees the
    /// host's libraries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub libraries: Vec<String>,

    /// Whether the emulator needs the host's GPUs exposed to each node's
    /// container. When set, every node container is launched with the
    /// host's GPU device nodes (`/dev/kfd`, `/dev/dri`) and the
    /// supplementary groups needed to open them; the group mechanism is
    /// provider-specific (podman inherits the launching user's groups
    /// via `--group-add keep-groups`, docker is given the named GPU
    /// groups explicitly). Only meaningful for containerised sessions.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub host_gpus: bool,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn tty_defaults_to_off_for_documents_that_omit_it() {
        // Pipes are the safe default: byte-exact output with stdout and
        // stderr distinct. A document written before `tty` existed, or by
        // a client that does not set it, must not silently gain a
        // terminal and start rewriting newlines.
        let json = r#"{
            "timestamp": "2026-01-01T00:00:00Z",
            "session": "s",
            "exec": {"command": "/bin/true"}
        }"#;
        let def: ExecDef = serde_json::from_str(json).unwrap();
        assert!(!def.tty);
        assert_eq!(def.nproc_per_node, 1);
    }

    #[test]
    fn exec_id_validates() {
        assert!(ExecId::new("e-000001").is_ok());
        assert!(ExecId::new("/bad").is_err());
        assert_eq!(ExecId::from_counter(7).as_str(), "e-000007");
    }

    #[test]
    fn ids_honour_format_width() {
        // The CLI renders tables with `{:<14}`/`{:<32}`; a Display impl
        // that writes straight to the formatter ignores the width and
        // produces a ragged table.
        assert_eq!(format!("[{:<10}]", ExecId::from_counter(1)), "[e-000001  ]");
        assert_eq!(
            format!("[{:<6}]", SessionId::new("s1").unwrap()),
            "[s1    ]"
        );
    }

    #[test]
    fn counter_ids_sort_lexicographically_in_creation_order() {
        // Exec ids are listed sorted as strings, so the zero padding is
        // what keeps e-000009 before e-000010.
        let mut ids: Vec<String> = (0..12)
            .map(|n| ExecId::from_counter(n).as_str().to_string())
            .collect();
        let expected = ids.clone();
        ids.sort();
        assert_eq!(ids, expected);
    }
}
