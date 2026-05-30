use serde::{Deserialize, Serialize};

use crate::{common::MaybeRef, exec::ExecDef, profile::ProfileDef};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workload {
    pub profile: MaybeRef<ProfileDef>,
    pub execution: ExecDef,
    
    /// keep the session after the exec finishes
    pub keep : bool,
}
