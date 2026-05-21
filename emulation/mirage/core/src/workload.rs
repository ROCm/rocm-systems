use crate::{common::MaybeRef, exec::ExecDef, profile::ProfileDef};

struct Workload {
    profile: MaybeRef<ProfileDef>,
    execution: ExecDef,
}
