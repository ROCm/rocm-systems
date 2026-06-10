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

/// The set of emulator backends mirage knows how to drive. Stored as a
/// closed enum (rather than a free-form string) so the host and control
/// plane dispatch over a fixed set of variants and an unknown emulator
/// name is rejected at deserialization time. Serializes as its
/// lowercase canonical name (`"noop"`, `"rocjitsu"`, `"hotswap"`) to
/// keep the on-disk/wire format unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EmulatorKind {
    /// Pass-through: run the workload directly with no GPU emulation.
    #[default]
    Noop,
    /// rocjitsu software GPU emulator.
    Rocjitsu,
    /// HotSwap load-time ISA rewriter.
    Hotswap,
}

impl EmulatorKind {
    /// The canonical lowercase name of this emulator.
    pub fn as_str(&self) -> &'static str {
        match self {
            EmulatorKind::Noop => "noop",
            EmulatorKind::Rocjitsu => "rocjitsu",
            EmulatorKind::Hotswap => "hotswap",
        }
    }
}

impl std::fmt::Display for EmulatorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EmulatorKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "noop" => Ok(EmulatorKind::Noop),
            "rocjitsu" => Ok(EmulatorKind::Rocjitsu),
            "hotswap" => Ok(EmulatorKind::Hotswap),
            other => Err(format!("unknown emulator `{other}`")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmulatorDef {
    /// which emulator backend to use, e.g. [`EmulatorKind::Rocjitsu`]
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
    /// Whether this host's hardware/environment can run the emulator
    /// (some backends require specific physical GPUs). Reported
    /// independently of `installed`.
    pub support: SupportStatus,
}

pub trait Emulator {
    /// Returns a description of the emulator, including its name, version, and a brief description.
    fn description() -> EmulatorDescription;

    /// Creates a new instance of the emulator bound to the given
    /// profile. The profile is retained so instance methods
    /// ([`Emulator::def`], [`Emulator::injection_def`], …) can resolve
    /// against it.
    fn new(def: ProfileDef) -> Self
    where
        Self: Sized;

    /// Schema for the options that this emulator supports. Empty when
    /// the emulator takes no options.
    fn options() -> Vec<OptionDef>;

    /// shuts down the emulator and releases any resources it holds.
    fn shutdown(self);

    /// Validate that `def` can be used with this emulator before it is
    /// persisted. Returns a human-readable reason on rejection.
    fn validate_profile(def: &ProfileDef) -> std::result::Result<(), String>;

    /// Returns the definition of the emulator, which includes its name, plugins, and options.
    fn def(&self) -> &EmulatorDef;

    /// Returns true if the emulator is properly installed and can be used.
    fn installed() -> bool;

    /// Discovers available plugins for the emulator.
    fn discover_plugins() -> Vec<PluginsDef>;

    /// get health status of the emulator, e.g. check if the underlying runtime is responsive
    fn health(&self) -> SessionHealth;

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

/// The built-in pass-through emulator: runs the workload directly with
/// no GPU emulation. Needs no injection and accepts any profile.
pub struct Noop {
    profile: ProfileDef,
}

impl Emulator for Noop {
    fn description() -> EmulatorDescription {
        crate::registry::noop()
    }

    fn new(def: ProfileDef) -> Self {
        Self { profile: def }
    }

    fn options() -> Vec<OptionDef> {
        Vec::new()
    }

    fn shutdown(self) {}

    fn validate_profile(_def: &ProfileDef) -> std::result::Result<(), String> {
        Ok(())
    }

    fn def(&self) -> &EmulatorDef {
        &self.profile.emulator
    }

    fn installed() -> bool {
        true
    }

    fn discover_plugins() -> Vec<PluginsDef> {
        Vec::new()
    }

    fn health(&self) -> SessionHealth {
        SessionHealth {
            healthy: true,
            state: Some("ready".to_string()),
            terminal: false,
            message: None,
            ..Default::default()
        }
    }

    fn injection_def(&self, _session: &SessionId) -> Result<InjectionDef> {
        Ok(InjectionDef::default())
    }
}
