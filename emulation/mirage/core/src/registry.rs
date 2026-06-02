//! Emulator registry primitives.
//!
//! [`EmulatorSpec`] is the generic descriptor each emulator backend
//! provides: it knows how to describe itself and report whether its
//! runtime is installed. The built-in pass-through [`NOOP`] lives here
//! because it has no external runtime. Emulator-specific specs live in
//! their own crates and are assembled into the full registry by
//! `mirage_ctl`; this module only provides the generic [`find`] /
//! [`default_emulator`] / [`make_def`] helpers that operate over a
//! supplied slice of specs.
//!
//! Named hardware agents ([`crate::agent::AgentDef`]) and system
//! topologies ([`crate::topology::TopologyDef`]) live in the
//! `mirage_builtin` crate; the on-disk store policy for them lives in
//! [`crate::agent::store`] / [`crate::topology::store`].

use crate::common::{MaybeRef, SimpleMap};
use crate::emulator::{EmulatorDef, EmulatorDescription, ExecMode};
use crate::topology::TopologyDef;

/// A registered emulator backend.
#[derive(Debug, Clone, Copy)]
pub struct EmulatorSpec {
    /// Canonical name used on disk and on the CLI.
    pub name: &'static str,
    /// Short human description.
    pub description: &'static str,
    /// Returns `true` if this emulator's runtime is present on the
    /// current machine. Cheap (no network, no spawn): used by the
    /// wizard's default picker.
    pub installed: fn() -> bool,
    /// Returns a long-form description (name + version + blurb).
    pub describe: fn() -> EmulatorDescription,
}

/// Lookup an emulator by its canonical name within `specs`.
pub fn find<'a>(specs: &'a [EmulatorSpec], name: &str) -> Option<&'a EmulatorSpec> {
    specs.iter().find(|e| e.name == name)
}

/// The default emulator for new profiles when the user doesn't pick
/// one explicitly. Picks the first installed, non-noop entry in
/// registration order, falling back to [`NOOP`].
pub fn default_emulator(specs: &[EmulatorSpec]) -> &EmulatorSpec {
    specs
        .iter()
        .find(|e| e.name != NOOP.name && (e.installed)())
        .unwrap_or(&NOOP)
}

/// Build an [`EmulatorDef`] for the given registry entry, using the
/// supplied system topology.
pub fn make_def(spec: &EmulatorSpec, topology: TopologyDef) -> EmulatorDef {
    EmulatorDef {
        emulator: spec.name.to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::default(),
        options: SimpleMap::default(),
        topology: MaybeRef::Owned(topology),
    }
}

// =============================================================================
// noop
// =============================================================================

pub const NOOP: EmulatorSpec = EmulatorSpec {
    name: "noop",
    description: "no-op emulator: runs commands directly with no GPU emulation",
    installed: noop_installed,
    describe: noop_describe,
};

fn noop_installed() -> bool {
    true
}

fn noop_describe() -> EmulatorDescription {
    EmulatorDescription {
        name: "noop".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "Pass-through emulator. Useful for orchestration tests.".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_locates_by_name() {
        let specs = [NOOP];
        assert_eq!(find(&specs, "noop").map(|e| e.name), Some("noop"));
        assert!(find(&specs, "bogus").is_none());
    }

    #[test]
    fn noop_is_always_installed() {
        assert!((NOOP.installed)());
    }

    #[test]
    fn default_falls_back_to_noop() {
        let specs = [NOOP];
        assert_eq!(default_emulator(&specs).name, "noop");
    }
}
