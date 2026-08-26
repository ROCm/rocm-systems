//! Emulator registry primitives.
//!
//! Every emulator backend lives in its own crate and registers itself
//! into a global registry via [`inventory`] (see
//! [`crate::emulator::EmulatorBackendDef`]). This module assembles that
//! registry into a list of [`EmulatorInfo`] — each backend's static
//! [`crate::emulator::EmulatorDescription`] plus its live runtime status (installed,
//! where its runtime library is or was looked for, and whether this host
//! supports it) — and provides the generic [`find`] /
//! [`default_emulator`] / [`make_def`] helpers that operate over a
//! supplied slice of them.
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
use crate::discovery::RuntimeLocation;
use crate::emulator::{EmulatorBackendDef, EmulatorDef, EmulatorKind, ExecMode, SupportStatus};
use crate::topology::TopologyDef;

/// A registry entry: a backend's static [`crate::emulator::EmulatorDescription`]
/// flattened together with its live runtime status on this host.
///
/// [`crate::emulator::EmulatorDescription`]: crate::emulator::EmulatorDescription
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
    /// Where this backend's runtime library was found on this machine,
    /// or — when it was not — every location that was searched and the
    /// environment variables that would change the answer.
    ///
    /// `installed` says whether mirage can emulate; this says why, which
    /// is what the user actually needs when the answer is no. It is a
    /// fact about this host, gathered in the same search that produced
    /// `installed`, and never a static description of the policy.
    pub runtime: RuntimeLocation,
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
            // One probe for both facts: the runtime search is the
            // expensive part of building this list (a backend that hunts
            // for a build tree beside the mirage binary stats a hundred
            // paths), and asking `installed()` separately would repeat
            // it for every entry.
            let runtime = def.backend.runtime();
            EmulatorInfo {
                name: d.name,
                version: d.version,
                description: d.description,
                options_schema: d.options_schema,
                plugins,
                installed: runtime.installed,
                runtime: runtime.location,
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
/// explicitly: the first *installed* backend in name order, falling back
/// to the first backend of any kind.
///
/// Preferring an installed one matters because selecting a backend whose
/// runtime is missing produces a session that fails at bring-up. Falling
/// back to an uninstalled one is still better than refusing to name a
/// default: the resulting error names the missing runtime, which is more
/// use than "no default emulator".
///
/// Returns `None` only when no backend at all was compiled in, which is
/// reachable: backends are feature-gated, and `--no-default-features`
/// with none selected produces exactly that binary. Reporting it lets the
/// caller say "this build has no emulator backends" instead of panicking
/// somewhere far from the cause.
#[must_use]
pub fn default_emulator(specs: &[EmulatorInfo]) -> Option<&EmulatorInfo> {
    specs.iter().find(|e| e.installed).or_else(|| specs.first())
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
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    use crate::discovery::{LibSearch, locate_emulator_lib};
    use crate::emulator::{EmulatorBackend, EmulatorDescription, RuntimeStatus};

    fn info(name: &str, installed: bool) -> EmulatorInfo {
        EmulatorInfo {
            name: name.to_string(),
            version: "0".to_string(),
            description: String::new(),
            options_schema: Vec::new(),
            plugins: Vec::new(),
            installed,
            runtime: RuntimeLocation::Unknown,
            support: SupportStatus::supported("test"),
        }
    }

    /// The name the not-installed backend below registers under. It
    /// sorts after every real backend's name so it cannot displace one
    /// as the default in another test.
    const MISSING_RUNTIME: &str = "zz-missing-runtime";

    /// The library it looks for, which exists on no machine.
    const MISSING_LIB: &str = "libmirage-no-such-runtime.so";

    /// The environment variable that would point at it.
    const MISSING_LIB_ENV: &str = "MIRAGE_NO_SUCH_RUNTIME_LIB";

    /// A backend whose runtime library is never present, so that the
    /// not-installed reporting can be exercised on any machine —
    /// including one where every real backend happens to be installed.
    #[derive(Debug)]
    struct MissingRuntime;

    fn missing_search() -> LibSearch<'static> {
        LibSearch {
            file_env: &[MISSING_LIB_ENV],
            dir_env: &[],
            home_env: &[],
            lib_name: MISSING_LIB,
            binary_relative_dirs: &["../nowhere"],
            system_fallbacks: true,
        }
    }

    impl EmulatorBackend for MissingRuntime {
        fn description(&self) -> EmulatorDescription {
            EmulatorDescription {
                name: MISSING_RUNTIME.to_string(),
                version: "0".to_string(),
                description: "test-only backend whose runtime is never installed".to_string(),
                options_schema: Vec::new(),
            }
        }

        fn boot(&self, _def: &crate::profile::ProfileDef) -> Result<(), String> {
            Ok(())
        }

        fn options(&self) -> Vec<OptionDef> {
            Vec::new()
        }

        fn shutdown(&self, _ctx: &crate::session::SessionContext) {}

        fn validate_profile(&self, _def: &crate::profile::ProfileDef) -> Result<(), String> {
            Ok(())
        }

        fn runtime(&self) -> RuntimeStatus {
            RuntimeStatus::from_location(locate_emulator_lib(&missing_search()))
        }

        fn supported(&self) -> SupportStatus {
            SupportStatus::supported("nothing is required to not be installed")
        }

        fn discover_plugins(&self) -> Vec<crate::plugin::PluginsDef> {
            Vec::new()
        }

        fn health(&self, _ctx: &crate::session::SessionContext) -> crate::session::SessionHealth {
            crate::session::SessionHealth::phase(false, crate::session::state::FAILED, None)
        }

        fn injection_def(
            &self,
            _ctx: &crate::session::SessionContext,
        ) -> crate::error::Result<crate::exec::InjectionDef> {
            Ok(crate::exec::InjectionDef::default())
        }
    }

    inventory::submit! {
        EmulatorBackendDef { kind: MISSING_RUNTIME, backend: &MissingRuntime }
    }

    /// The entry the registry builds for the never-installed backend.
    fn missing_entry() -> EmulatorInfo {
        find(&registry(), MISSING_RUNTIME)
            .cloned()
            .expect("the test backend is registered")
    }

    /// A backend that reports itself uninstalled must say where mirage
    /// looked. Without it, `mirage emulators` answers the user's
    /// question ("can it emulate?") and refuses the follow-up ("then
    /// what do I install, and where?"), which is the only one they can
    /// act on.
    #[test]
    fn a_not_installed_backend_names_the_locations_it_searched() {
        let entry = missing_entry();

        assert!(!entry.installed);
        let RuntimeLocation::Missing {
            lib_name,
            searched,
            env,
        } = &entry.runtime
        else {
            panic!(
                "a library that exists nowhere cannot be found: {:?}",
                entry.runtime
            );
        };
        assert_eq!(lib_name, MISSING_LIB);
        assert!(
            !searched.is_empty(),
            "a not-installed backend must name the directories it searched"
        );
        assert!(searched.iter().all(|p| p.ends_with(MISSING_LIB)));
        // And the search order is the shared policy's, not a second copy
        // of it: the standard ROCm directory is in the list.
        assert!(searched.contains(&std::path::PathBuf::from("/opt/rocm/lib").join(MISSING_LIB)));
        assert!(env.iter().any(|hint| hint.name == MISSING_LIB_ENV));
    }

    /// Whatever the long text form shows, the JSON must carry — the same
    /// rule the `default` marker is held to. A script reading `--json`
    /// should never have to re-derive a fact the text prints.
    #[test]
    fn the_json_carries_every_fact_the_text_shows() {
        let entry = missing_entry();
        let json = serde_json::to_value(&entry).unwrap();

        assert_eq!(json["installed"], serde_json::json!(false));
        assert_eq!(json["runtime"]["state"], serde_json::json!("missing"));
        assert_eq!(json["runtime"]["lib_name"], serde_json::json!(MISSING_LIB));

        // Every line of the human-readable report is backed by the
        // serialized form: each probed path it prints appears in
        // `searched`, and each variable it tells the user to set appears
        // in `env`. (The text elides the middle of a long list, so the
        // JSON may carry more — never less.)
        let searched = json["runtime"]["searched"].as_array().unwrap().clone();
        let env = json["runtime"]["env"].as_array().unwrap().clone();
        assert_eq!(searched.len(), entry.runtime.searched().len());
        for (key, value) in entry.runtime.report() {
            match key {
                "runtime" => assert!(value.contains(MISSING_LIB), "{value}"),
                "set" | "" if value.contains('=') => {
                    let name = value.split('=').next().unwrap_or_default();
                    assert!(
                        env.iter().any(|hint| hint["name"] == name),
                        "the text offers {name} but the JSON does not list it"
                    );
                }
                _ if value.starts_with('…') => {}
                _ => assert!(
                    searched.iter().any(|p| p == &serde_json::json!(value)),
                    "the text shows {value} but the JSON does not list it"
                ),
            }
        }
    }

    #[test]
    fn find_locates_by_name() {
        let specs = [info("rocjitsu", true)];
        assert_eq!(
            find(&specs, "rocjitsu").map(|e| e.name.as_str()),
            Some("rocjitsu")
        );
        assert!(find(&specs, "bogus").is_none());
    }

    #[test]
    fn default_prefers_an_installed_backend() {
        let specs = [info("hotswap", false), info("rocjitsu", true)];
        assert_eq!(default_emulator(&specs).unwrap().name, "rocjitsu");
    }

    #[test]
    fn default_falls_back_to_an_uninstalled_backend() {
        // Naming one still produces an actionable error at bring-up
        // ("rocjitsu runtime not found"); naming none does not.
        let specs = [info("hotswap", false), info("rocjitsu", false)];
        assert_eq!(default_emulator(&specs).unwrap().name, "hotswap");
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
