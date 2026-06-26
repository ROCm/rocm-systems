//! Emulator registry primitives.
//!
//! Every emulator backend lives in its own crate and registers itself
//! into a global registry via [`inventory`] (see
//! [`crate::emulator::EmulatorBackendDef`]). This module assembles that
//! registry into a list of [`EmulatorInfo`] — each backend's static
//! [`EmulatorDescription`] plus its live runtime status (installed /
//! supported) — and provides the generic [`find`] / [`default_emulator`]
//! / [`make_def`] helpers that operate over a supplied slice of them.
//!
//! No backend is named here: the list is whatever set of backend
//! crates was compiled into the binary, so disabling a backend's
//! feature simply drops it from the registry.
//!
//! Named hardware agents ([`crate::agent::AgentDef`]) and system
//! topologies ([`crate::topology::TopologyDef`]) live in the
//! `mirage_builtin` crate; the on-disk store policy for them lives in
//! [`crate::agent::store`] / [`crate::topology::store`].

use serde::{Deserialize, Serialize};

use crate::common::{MaybeRef, SimpleMap};
use crate::config::OptionDef;
use crate::emulator::{EmulatorBackendDef, EmulatorDef, EmulatorKind, ExecMode, SupportStatus};
use crate::topology::TopologyDef;

/// The canonical name of the built-in pass-through emulator. Only used
/// as a tie-breaker so [`default_emulator`] never picks the no-op
/// backend when a real one is available; the `noop` backend itself
/// lives in the `mirage_noop` crate.
pub const NOOP_NAME: &str = "noop";

/// A registry entry: a backend's static [`EmulatorDescription`]
/// flattened together with its live runtime status on this host.
///
/// [`EmulatorDescription`]: crate::emulator::EmulatorDescription
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmulatorInfo {
    /// Canonical name of the backend (also its [`EmulatorKind`]).
    pub name: String,
    pub version: String,
    pub description: String,
    /// Schema of the options this backend accepts (empty when none).
    pub options_schema: Vec<OptionDef>,
    /// `true` if this backend's runtime is present on this machine.
    pub installed: bool,
    /// Whether this host's hardware/environment can run the backend.
    pub support: SupportStatus,
}

/// Build the full emulator registry by probing every backend that was
/// compiled into the binary. Each backend (registered via
/// [`inventory`]) contributes its description plus a live install /
/// support probe. Entries are returned sorted by name so the order is
/// deterministic regardless of link order.
pub fn registry() -> Vec<EmulatorInfo> {
    let mut out: Vec<EmulatorInfo> = inventory::iter::<EmulatorBackendDef>
        .into_iter()
        .map(|def| {
            let d = def.backend.description();
            EmulatorInfo {
                name: d.name,
                version: d.version,
                description: d.description,
                options_schema: d.options_schema,
                installed: def.backend.installed(),
                support: def.backend.supported(),
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Lookup an emulator by its canonical name within `specs`.
pub fn find<'a>(specs: &'a [EmulatorInfo], name: &str) -> Option<&'a EmulatorInfo> {
    specs.iter().find(|e| e.name == name)
}

/// The default emulator for new profiles when the user doesn't pick
/// one explicitly. Picks the first installed, non-noop entry in name
/// order, falling back to the `noop` entry (or the first entry if
/// `noop` is somehow absent).
pub fn default_emulator(specs: &[EmulatorInfo]) -> &EmulatorInfo {
    specs
        .iter()
        .find(|e| e.name != NOOP_NAME && e.installed)
        .or_else(|| specs.iter().find(|e| e.name == NOOP_NAME))
        .or_else(|| specs.first())
        .expect("emulator registry must not be empty")
}

/// Build an [`EmulatorDef`] for the given registry entry, using the
/// supplied system topology.
pub fn make_def(spec: &EmulatorInfo, topology: TopologyDef) -> EmulatorDef {
    EmulatorDef {
        emulator: EmulatorKind::from(spec.name.clone()),
        plugins: Default::default(),
        exec_mode: ExecMode::default(),
        options: SimpleMap::default(),
        topology: MaybeRef::Owned(topology),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, installed: bool) -> EmulatorInfo {
        EmulatorInfo {
            name: name.to_string(),
            version: "0".to_string(),
            description: String::new(),
            options_schema: Vec::new(),
            installed,
            support: SupportStatus::supported("test"),
        }
    }

    #[test]
    fn find_locates_by_name() {
        let specs = [info("noop", true)];
        assert_eq!(find(&specs, "noop").map(|e| e.name.as_str()), Some("noop"));
        assert!(find(&specs, "bogus").is_none());
    }

    #[test]
    fn default_prefers_installed_non_noop() {
        let specs = [info("noop", true), info("rocjitsu", true)];
        assert_eq!(default_emulator(&specs).name, "rocjitsu");
    }

    #[test]
    fn default_falls_back_to_noop() {
        let specs = [info("noop", true), info("hotswap", false)];
        assert_eq!(default_emulator(&specs).name, "noop");
    }
}
