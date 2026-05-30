use serde::{Deserialize, Serialize};

use crate::profile::ProfileDef;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SessionHealth {
    #[default]
    Unknown,
    Healthy,
    Unhealthy,
    Degraded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionId(String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDef {
    /// The name of the session, used for display and logging purposes.
    id: SessionId,

    /// The profile to use for this session.  This is a key that the
    /// daemon uses to look up the profile definition.
    profile: ProfileDef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionState {
    /// the session definition
    def: SessionDef,
    /// the health status of the session
    health: SessionHealth,
}
