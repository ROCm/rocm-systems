use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{common::{MaybeRef, SimpleMap}, config::OptionDef, exec::InjectionDef, plugin::PluginsDef, profile::ProfileDef};


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmulatorDef {
    /// name of the emulator, e.g. "rocjitsu"
    pub emulator: String,

    /// plugins to use with the emulator, e.g. "rocjitsu" plugin for AMD GPU simulation
    pub plugins: PluginsDef,
    
    /// extra options to configure the emulator, e.g. {"gpu_model": "cdna3"}
    pub options: SimpleMap,
}


pub trait Emulator {
    /// Creates a new instance of the emulator with the given definition.
    fn new(def: ProfileDef) -> Self;

    /// gets schema for the options that this emulator supports
    fn options() -> OptionDef;
    
    /// shuts down the emulator and releases any resources it holds.
    fn shutdown(self);
 
    /// Returns the definition of the emulator, which includes its name, plugins, and options.
    fn def(&self) -> &EmulatorDef;

    /// Returns true if the emulator is properly installed and can be used.
    fn installed() -> bool;
    
    /// Discovers available plugins for the emulator.
    fn discover_plugins() -> Vec<PluginsDef>;

    /// get healthly status of the emulator, e.g. check if the underlying runtime is responsive
    fn healthly(&self) -> bool;

    /// get extra env varibles and files to inject into the environment
    fn injection_def(&self) -> InjectionDef;
}


