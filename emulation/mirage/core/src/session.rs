use serde::{Deserialize, Serialize};

use crate::profile::ProfileDef;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
enum SessionHealth {
    #[default]
    Unknown,
    Healthy,
    Unhealthy,
    Degraded,
}

struct SessionDef {
    /// The name of the session, used for display and logging purposes.
    name: String,

    /// The profile to use for this session.  This is a key that the
    /// daemon uses to look up the profile definition.
    profile: ProfileDef,
}

struct SessionState {
    def: SessionDef,
    health: SessionHealth,
}