use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{common::MaybeRef, session::SessionId};

/// Concrete process arguments for one program invocation.
///
/// Describes the program, its arguments, and any extra environment
/// variables. Used inside [`ExecDef`] and [`ClusterDef`].
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

    /// Optional working directory for this command.  If `None`, the command
    /// will run in the session's default working directory.
    pub workdir: Option<String>,
}

/// A request to start an exec inside an existing session.
///
/// The daemon resolves the session, looks up its simulator, and calls
/// `get_exec_run_def` to let the simulator inject any extra environment
/// variables or wrapper commands before the exec is actually started.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecDef {
    /// a timestamp of when this exec was requested, in milliseconds since the unix epoch
    pub timestamp: u64,

    /// session
    pub session: SessionId,

    /// What to run on the head node.
    pub exec: ExecArgs,

    /// Optional command to run on worker nodes.  If `None`, workers won't
    /// run any command.
    #[serde(default)]
    pub worker_exec: Option<ExecArgs>,

    /// should this exec be kept around after it finishes, or should it be cleaned up immediately?
    pub keep: bool,
}

pub struct ExecStatus {
    // has the exec been started?
    pub started: bool,

    // the exit code of the exec, if it has finished
    pub exit_code: Option<i32>,
}

pub struct ExecId {
    pub session: SessionId,
    pub exec: u32,
}

/// modifcations to an exec
pub struct InjectionDef {
    /// wraper program to launch all programs with
    pub wrapper: Option<String>,

    /// inject a libary to the LD_PRELOAD variable
    pub ld_preload: Option<String>,

    /// ensure that the given files are available in the environment
    pub files: BTreeMap<String, MaybeRef<Vec<u8>>>,

    /// env varibles
    pub env: BTreeMap<String, String>,
}
