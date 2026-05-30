use std::collections::BTreeMap;

use crate::common::SimpleMap;

pub type PluginsDef = BTreeMap<String, SimpleMap>;

pub struct PluginDef {
    pub name: String,
    pub options: SimpleMap,
    pub version: String,
    pub abi: u32,
}
