use serde::{Deserialize, Serialize};

use crate::{common::MaybeRef, container::ContainerizedDef, profile::ProfileDef};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SessionHealth {
    /// a timestamp of this health status, in milliseconds since the unix epoch
    pub timestamp: u64,

    /// whether the session is healthy (i.e. ready to run execs)
    pub healthy: bool,

    /// what the current state of the session is (e.g. "starting", "pulling", "running", "error", etc.)
    pub state: Option<String>,

    /// if this session will never become healthy
    pub terminal: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionId(String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDef {
    /// The name of the session, used for display and logging purposes.
    pub id: SessionId,

    /// The profile to use for this session.
    pub profile: MaybeRef<ProfileDef>,

    /// the container definition for this session
    /// run this session with this image
    pub container: Option<ContainerizedDef>,

    /// the working directory for this session
    pub workdir: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionState {
    /// the session definition
    pub def: SessionDef,
    /// the health status of the session
    pub health: SessionHealth,
}
