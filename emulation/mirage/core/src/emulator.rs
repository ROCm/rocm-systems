use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{
    common::{MaybeRef, SimpleMap},
    config::OptionDef,
    exec::InjectionDef,
    plugin::PluginsDef,
    profile::ProfileDef,
    session::SessionHealth,
    topology::TopologyDef,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecMode {
    #[default]
    Functional,
    Clocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmulatorDef {
    /// name of the emulator, e.g. "rocjitsu"
    pub emulator: String,

    /// plugins to use with the emulator, e.g. "rocjitsu" plugin for AMD GPU simulation
    pub plugins: PluginsDef,

    pub exec_mode: ExecMode,

    /// extra options to configure the emulator, e.g. {"gpu_model": "cdna3"}
    pub options: SimpleMap,

    /// system topology (rack/node/GPU layout + per-GPU agent).
    pub topology: MaybeRef<TopologyDef>,
}

/// A description of an emulator backend: its identity (name, version,
/// blurb) plus its current runtime status on this machine (whether it
/// is installed and, if so, the resolved path to its runtime library).
///
/// This is the single descriptor the registry exposes for every
/// backend; the generic pass-through `noop` and each emulator-specific
/// crate produce one of these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmulatorDescription {
    pub name: String,
    pub version: String,
    pub description: String,
    /// `true` if this emulator's runtime is present on this machine.
    pub installed: bool,
    /// Resolved path to the emulator's runtime library, when installed
    /// and locatable. `None` for backends without an external runtime
    /// (e.g. `noop`) or when the library could not be found.
    pub path: Option<PathBuf>,
}

pub trait Emulator {
    /// Returns a description of the emulator, including its name, version, and a brief description.
    fn description() -> EmulatorDescription;

    /// Creates a new instance of the emulator with the given definition.
    fn new(def: ProfileDef) -> Self;

    /// gets schema for the options that this emulator supports
    fn options() -> OptionDef;

    /// shuts down the emulator and releases any resources it holds.
    fn shutdown(self);

    fn validate_profile(def: &ProfileDef) -> Result<(), String>;

    /// Returns the definition of the emulator, which includes its name, plugins, and options.
    fn def(&self) -> &EmulatorDef;

    /// Returns true if the emulator is properly installed and can be used.
    fn installed() -> bool;

    /// Discovers available plugins for the emulator.
    fn discover_plugins() -> Vec<PluginsDef>;

    /// get health status of the emulator, e.g. check if the underlying runtime is responsive
    fn health(&self) -> SessionHealth;

    /// get extra env varibles and files to inject into the environment
    fn injection_def(&self) -> InjectionDef;
}
