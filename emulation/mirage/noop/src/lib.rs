//! `mirage_noop` — the built-in pass-through emulator backend.
//!
//! The no-op backend runs the workload directly with no GPU emulation:
//! it needs no injection and accepts any profile. It is a normal
//! emulator backend like any other — it lives in its own crate and
//! registers itself into the registry via [`inventory`], so the core
//! and control-plane crates carry no special-casing for it. Disabling
//! this crate's feature simply removes the `noop` backend from the
//! registry.

use mirage_core::config::OptionDef;
use mirage_core::emulator::{
    EmulatorBackend, EmulatorBackendDef, EmulatorDescription, SupportStatus,
};
use mirage_core::error::Result;
use mirage_core::exec::InjectionDef;
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionHealth, SessionId};

/// Canonical name the pass-through backend registers under.
pub const NAME: &str = "noop";

/// The built-in pass-through emulator: runs the workload directly with
/// no GPU emulation. Stateless; a single shared instance is registered
/// in the emulator registry.
pub struct Noop;

impl EmulatorBackend for Noop {
    fn description(&self) -> EmulatorDescription {
        EmulatorDescription {
            name: NAME.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "no-op emulator: runs commands directly with no GPU emulation".to_string(),
            options_schema: Vec::new(),
        }
    }

    fn boot(&self, _def: &ProfileDef) -> std::result::Result<(), String> {
        Ok(())
    }

    fn options(&self) -> Vec<OptionDef> {
        Vec::new()
    }

    fn shutdown(&self, _session: &SessionId) {}

    fn validate_profile(&self, _def: &ProfileDef) -> std::result::Result<(), String> {
        Ok(())
    }

    fn installed(&self) -> bool {
        true
    }

    fn supported(&self) -> SupportStatus {
        SupportStatus::supported("runs commands directly; no special hardware required")
    }

    fn discover_plugins(&self) -> Vec<PluginsDef> {
        Vec::new()
    }

    fn health(&self, _session: &SessionId) -> SessionHealth {
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

inventory::submit! {
    EmulatorBackendDef { kind: NAME, backend: &Noop }
}
