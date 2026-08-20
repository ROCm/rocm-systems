//! Emulator plugins: the selection carried on a profile, and the
//! description a backend reports for one it can load.

use std::collections::BTreeMap;

use crate::common::SimpleMap;

/// A profile's plugin selection: plugin name → its argument object.
///
/// An empty argument object means "enable this plugin with its schema
/// defaults", which is what `mirage run --plugin <name>` produces.
pub type PluginsDef = BTreeMap<String, SimpleMap>;

/// A plugin an emulator backend can load, as reported by discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginDef {
    /// The name used to enable the plugin in a [`PluginsDef`].
    pub name: String,
    /// The arguments it accepts.
    pub options: SimpleMap,
    /// The plugin's own version string.
    pub version: String,
    /// The plugin ABI it was built against.
    pub abi: u32,
}
