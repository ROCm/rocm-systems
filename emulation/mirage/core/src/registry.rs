//! Emulator registry primitives.
//!
//! [`EmulatorDescription`] is the generic descriptor each emulator
//! backend provides: its identity plus whether its runtime is
//! installed and where. The built-in pass-through [`noop`] lives here
//! because it has no external runtime. Emulator-specific descriptions
//! live in their own crates and are assembled into the full registry
//! by `mirage_ctl`; this module only provides the generic [`find`] /
//! [`default_emulator`] / [`make_def`] helpers that operate over a
//! supplied slice of descriptions.
//!
//! Named hardware agents ([`crate::agent::AgentDef`]) and system
//! topologies ([`crate::topology::TopologyDef`]) live in the
//! `mirage_builtin` crate; the on-disk store policy for them lives in
//! [`crate::agent::store`] / [`crate::topology::store`].

use crate::common::{MaybeRef, SimpleMap};
use crate::emulator::{EmulatorDef, EmulatorDescription, EmulatorKind, ExecMode, SupportStatus};
use crate::topology::TopologyDef;

/// The canonical name of the built-in pass-through emulator.
pub const NOOP_NAME: &str = "noop";

/// Lookup an emulator by its canonical name within `specs`.
pub fn find<'a>(specs: &'a [EmulatorDescription], name: &str) -> Option<&'a EmulatorDescription> {
    specs.iter().find(|e| e.name == name)
}

/// The default emulator for new profiles when the user doesn't pick
/// one explicitly. Picks the first installed, non-noop entry in
/// registration order, falling back to the `noop` entry (or the first
/// entry if `noop` is somehow absent).
pub fn default_emulator(specs: &[EmulatorDescription]) -> &EmulatorDescription {
    specs
        .iter()
        .find(|e| e.name != NOOP_NAME && e.installed)
        .or_else(|| specs.iter().find(|e| e.name == NOOP_NAME))
        .or_else(|| specs.first())
        .expect("emulator registry must not be empty")
}

/// Build an [`EmulatorDef`] for the given registry entry, using the
/// supplied system topology.
pub fn make_def(spec: &EmulatorDescription, topology: TopologyDef) -> EmulatorDef {
    EmulatorDef {
        emulator: spec.name.parse().unwrap_or(EmulatorKind::Noop),
        plugins: Default::default(),
        exec_mode: ExecMode::default(),
        options: SimpleMap::default(),
        topology: MaybeRef::Owned(topology),
    }
}

// =============================================================================
// noop
// =============================================================================

/// Describe the built-in pass-through emulator. Always installed and
/// has no external runtime library.
pub fn noop() -> EmulatorDescription {
    EmulatorDescription {
        name: NOOP_NAME.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "no-op emulator: runs commands directly with no GPU emulation".to_string(),
        installed: true,
        path: None,
        support: SupportStatus::supported("runs commands directly; no special hardware required"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_locates_by_name() {
        let specs = [noop()];
        assert_eq!(find(&specs, "noop").map(|e| e.name.as_str()), Some("noop"));
        assert!(find(&specs, "bogus").is_none());
    }

    #[test]
    fn noop_is_always_installed() {
        assert!(noop().installed);
    }

    #[test]
    fn default_falls_back_to_noop() {
        let specs = [noop()];
        assert_eq!(default_emulator(&specs).name, "noop");
    }
}
