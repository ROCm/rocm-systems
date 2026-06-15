use std::path::PathBuf;

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{
    common::{MaybeRef, SimpleMap},
    config::OptionDef,
    error::Result,
    exec::InjectionDef,
    plugin::PluginsDef,
    profile::ProfileDef,
    session::{SessionHealth, SessionId},
    topology::TopologyDef,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecMode {
    #[default]
    Functional,
    Clocked,
}

pub type EmulatorKind = String;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmulatorDef {
    /// which emulator backend to use, e.g. "rocjitsu"
    pub emulator: EmulatorKind,

    /// plugins to use with the emulator, e.g. "rocjitsu" plugin for AMD GPU simulation
    pub plugins: PluginsDef,

    pub exec_mode: ExecMode,

    /// extra options to configure the emulator, e.g. {"gpu_model": "cdna3"}
    pub options: SimpleMap,

    /// system topology (rack/node/GPU layout + per-GPU agent).
    pub topology: MaybeRef<TopologyDef>,
}

/// Whether the host's hardware/environment can actually run an
/// emulator. This is distinct from [`EmulatorDescription::installed`]:
/// an emulator can be installed yet unsupported (e.g. HotSwap installed
/// on a machine with no compatible physical GPU), or supported yet not
/// installed. Both signals are surfaced so the UX/CLI can explain
/// exactly what a user needs to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportStatus {
    /// `true` if this host meets the emulator's hardware/environment
    /// requirements.
    pub supported: bool,
    /// Human-readable explanation of the support decision — what was
    /// required and what was found. Always populated so the UX/CLI can
    /// show a reason whether or not the host is supported.
    pub reason: String,
}

impl SupportStatus {
    /// The host meets this emulator's requirements.
    pub fn supported(reason: impl Into<String>) -> Self {
        Self {
            supported: true,
            reason: reason.into(),
        }
    }

    /// The host does not meet this emulator's requirements.
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            supported: false,
            reason: reason.into(),
        }
    }
}

/// A description of an emulator backend: its identity (name, version,
/// blurb)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmulatorDescription {
    pub name: String,
    pub version: String,
    pub description: String,
    pub options_schema: Vec<OptionDef>,
}

pub trait EmulatorBackend : Sync + Send {
    /// Returns a description of the emulator, including its name, version, and a brief description.
    fn description(&self) -> EmulatorDescription;

    /// Creates a new instance of the emulator bound to the given
    /// profile. The profile is retained so instance methods
    /// ([`Emulator::def`], [`Emulator::injection_def`], …) can resolve
    /// against it.
    fn boot(&self, def: &ProfileDef) -> std::result::Result<(), String>;

    /// Schema for the options that this emulator supports. Empty when
    /// the emulator takes no options.
    fn options(&self) -> Vec<OptionDef>;

    /// shuts down the emulator and releases any resources it holds.
    fn shutdown(&self, session: &SessionId);

    /// Validate that `def` can be used with this emulator before it is
    /// persisted. Returns a human-readable reason on rejection.
    fn validate_profile(&self, def: &ProfileDef) -> std::result::Result<(), String>;

    /// Returns true if the emulator is properly installed and can be used.
    fn installed(&self) -> bool;

    /// check if the emulator is supported on this host, i.e. meets the hardware/environment requirements to run. This is a stronger condition than `installed`: an emulator can be installed but unsupported (e.g. HotSwap installed on a machine with no compatible physical GPU), or supported but not installed.
    fn supported(&self) -> SupportStatus;

    /// Discovers available plugins for the emulator.
    fn discover_plugins(&self) -> Vec<PluginsDef>;

    /// get health status of the emulator, e.g. check if the underlying runtime is responsive
    fn health(&self, session: &SessionId) -> SessionHealth;

    /// Compute the env vars / `LD_PRELOAD` / files to inject into a
    /// workload run under this emulator. Returns an error when the
    /// emulator is selected but its runtime library or assets are
    /// missing, so a misconfigured session fails loudly instead of
    /// silently running unemulated.
    ///
    /// `session` is the id of the session the workload runs in, so
    /// emulators can materialise per-session runtime assets under the
    /// session directory.
    fn injection_def(&self, session: &SessionId) -> Result<InjectionDef>;
}

pub struct EmulatorBackendDef {
    pub kind: EmulatorKind,
    pub backend : Box<dyn EmulatorBackend>,
}

inventory::collect!(EmulatorBackendDef);

pub fn get_emulator_backend(kind: &EmulatorKind) -> Option<&'static Box<dyn EmulatorBackend>> {
    for def in inventory::iter::<EmulatorBackendDef> {
        if &def.kind == kind {
            return Some(&def.backend);
        }
    }
    None
}