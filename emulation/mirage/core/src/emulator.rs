use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::common::SimpleMap;

pub type PluginsDef = BTreeMap<String, SimpleMap>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmulatorDef {
    pub emulator: String,
    pub plugins: PluginsDef,
    pub options: SimpleMap,
}
