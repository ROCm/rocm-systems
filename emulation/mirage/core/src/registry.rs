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
    /// Plugin names this backend discovered on the current host.
    pub plugins: Vec<String>,
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
            let mut plugins: Vec<String> = def
                .backend
                .discover_plugins()
                .into_iter()
                .flat_map(|selection| selection.into_keys())
                .collect();
            plugins.sort();
            plugins.dedup();
            EmulatorInfo {
                name: d.name,
                version: d.version,
                description: d.description,
                options_schema: d.options_schema,
                plugins,
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

/// The default emulator for new profiles when the user does not pick one
/// explicitly. Picks the first installed, non-noop entry in name order,
/// falling back to the `noop` entry, then to the first entry.
///
/// Returns `None` only when no backend at all was compiled in, which is
/// reachable: backends are feature-gated, and `--no-default-features`
/// with none selected produces exactly that binary. Reporting it lets the
/// caller say "this build has no emulator backends" instead of panicking
/// somewhere far from the cause.
#[must_use]
pub fn default_emulator(specs: &[EmulatorInfo]) -> Option<&EmulatorInfo> {
    specs
        .iter()
        .find(|e| e.name != NOOP_NAME && e.installed)
        .or_else(|| specs.iter().find(|e| e.name == NOOP_NAME))
        .or_else(|| specs.first())
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn info(name: &str, installed: bool) -> EmulatorInfo {
        EmulatorInfo {
            name: name.to_string(),
            version: "0".to_string(),
            description: String::new(),
            options_schema: Vec::new(),
            plugins: Vec::new(),
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
        assert_eq!(default_emulator(&specs).unwrap().name, "rocjitsu");
    }

    #[test]
    fn default_falls_back_to_noop() {
        let specs = [info("noop", true), info("hotswap", false)];
        assert_eq!(default_emulator(&specs).unwrap().name, "noop");
    }

    #[test]
    fn registry_serializes_discovered_plugins() {
        let mut emulator = info("rocjitsu", true);
        emulator.plugins = vec!["logging".to_string(), "race".to_string()];
        let json = serde_json::to_value(&emulator).unwrap();
        assert_eq!(json["plugins"], serde_json::json!(["logging", "race"]));
    }

    #[test]
    fn an_empty_registry_has_no_default_rather_than_panicking() {
        // Backends are feature-gated, so a build with none selected is a
        // real configuration. It should report the problem, not crash.
        assert!(default_emulator(&[]).is_none());
    }
}
