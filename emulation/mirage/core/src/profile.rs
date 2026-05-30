use serde::{Deserialize, Serialize};

use crate::emulator::EmulatorDef;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProfileDef {
    pub name: String,
    pub emulator: EmulatorDef,
}
