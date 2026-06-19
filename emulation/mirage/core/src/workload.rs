//! Workload: a self-contained "profile + exec" describing one run.

use serde::{Deserialize, Serialize};

use crate::{common::MaybeRef, exec::ExecDef, profile::ProfileDef};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workload {
    pub profile: MaybeRef<ProfileDef>,
    pub execution: ExecDef,

    /// Keep the session after the exec finishes.
    pub keep: bool,
}
