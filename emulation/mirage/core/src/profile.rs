//! Profile: a reusable bundle of emulator configuration.

use serde::{Deserialize, Serialize};

use crate::emulator::EmulatorDef;

/// A profile is a named, on-disk emulator preset that can be referenced
/// by sessions. Profiles live in `$XDG_CONFIG_HOME/mirage/profile/`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileDef {
    /// The profile's name (also used as the filename, `<name>.json`).
    pub name: String,

    /// Free-form description shown in `mirage profile list -l`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The emulator to use for this profile.
    pub emulator: EmulatorDef,
}
