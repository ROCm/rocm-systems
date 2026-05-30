use serde::{Deserialize, Serialize};

use crate::emulator::EmulatorDef;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileDef {
    /// the name of the profile
    pub name: String,

    /// the emulator to use for this profile
    pub emulator: EmulatorDef,

    /// run the execs in a container with this image
    pub image: Option<String>,
}
