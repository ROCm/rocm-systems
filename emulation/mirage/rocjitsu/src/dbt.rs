//! `rocjitsu-dbt` — Dynamic Binary Translation emulator backend.
//!
//! Unlike the `rocjitsu` backend (which emulates the GPU entirely in
//! software via the KMD interposer/daemon), the DBT backend runs the
//! workload on the host's **real** GPU after *translating* its AMDGPU
//! code objects from the architecture they were built for (the *guest*)
//! to the host GPU's architecture (the *target*).
//!
//! The translation happens at code-object load time inside ROCR: the
//! runtime loads rocjitsu's HSA tools hook library
//! ([`HOOKS_LIB_NAME`]) through the `HSA_TOOLS_LIB` environment
//! variable, and the hook invokes the rocjitsu DBT pipeline before ROCR
//! sees a guest code object. The env contract the hook reads is:
//!
//! * `HSA_TOOLS_LIB` — path to `librocjitsu_hooks.so` (mirage injects
//!   this; ROCR `dlopen`s it during `hsa_init()`).
//! * `RJ_DBT_TARGET_ISA` — **required** host GPU ISA the code is
//!   translated *to*, e.g. `gfx1201`.
//! * `RJ_DBT_SOURCE_ISA` — optional guest ISA the code was built *for*,
//!   e.g. `gfx950`. mirage derives this from the profile's agent.
//! * `RJ_DBT_LOG` — optional hook log level (`0`–`3`), forwarded from
//!   the exec environment when set.
//!
//! This is conceptually parallel to the `hotswap` backend (run code
//! built for one GPU on a different physical GPU) but uses rocjitsu's
//! own DBT translator rather than HotSwap's COMGR-based rewriter.

use std::collections::BTreeMap;
use std::path::PathBuf;

use mirage_core::agent::AgentDef;
use mirage_core::common::{MaybeRef, SimpleValue};
use mirage_core::config::OptionDef;
use mirage_core::discovery::{self, LibSearch};
use mirage_core::emulator::{
    EmulatorBackend, EmulatorBackendDef, EmulatorDef, EmulatorDescription, SupportStatus,
};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::InjectionDef;
use mirage_core::hardware::{gfx_name, gpu_gfx_versions};
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::{FileMount, ProfileDef};
use mirage_core::session::{SessionHealth, SessionId};
use mirage_core::topology::TopologyDef;

/// The canonical name the DBT backend registers under (the value stored
/// in [`mirage_core::emulator::EmulatorDef::emulator`]).
pub const NAME: &str = "rocjitsu-dbt";

/// The rocjitsu HSA tools hook library. ROCR loads it through
/// `HSA_TOOLS_LIB` and it runs the DBT translation at code-object load
/// time. Built by the rocjitsu CMake target `rocjitsu_hooks`.
pub const HOOKS_LIB_NAME: &str = "librocjitsu_hooks.so";

/// Emulator option naming the host ISA to translate *to*. Overrides the
/// physically-detected GPU. Value is a gfx name the hook accepts (see
/// [`DBT_ISAS`]).
pub const OPT_TARGET_ISA: &str = "target_isa";

/// Emulator option naming the guest ISA the workload was built *for*.
/// Overrides the value derived from the profile's agent.
pub const OPT_SOURCE_ISA: &str = "source_isa";

/// The ISAs the rocjitsu DBT translator understands, as
/// `gfx_target_version` paired with the conventional `gfx` name the
/// hook's `RJ_DBT_TARGET_ISA`/`RJ_DBT_SOURCE_ISA` accept. The same set
/// is valid for both the guest (source) and the host (target); it
/// mirrors the hook's `kAcceptedConcreteTargetMachs`.
pub const DBT_ISAS: &[(u32, &str)] = &[
    (90402, "gfx942"),
    (90500, "gfx950"),
    (110000, "gfx1100"),
    (120000, "gfx1200"),
    (120001, "gfx1201"),
];

/// The rocjitsu DBT [`EmulatorBackend`]. Stateless; a single shared
/// instance is registered in the emulator registry.
pub struct RocjitsuDbt;

impl EmulatorBackend for RocjitsuDbt {
    fn description(&self) -> EmulatorDescription {
        describe()
    }

    fn boot(&self, _def: &ProfileDef) -> std::result::Result<(), String> {
        Ok(())
    }

    fn options(&self) -> Vec<OptionDef> {
        options_schema()
    }

    fn shutdown(&self, _session: &SessionId) {}

    fn validate_profile(&self, def: &ProfileDef) -> std::result::Result<(), String> {
        // Resolving the guest gfx version exercises the same topology +
        // agent reference resolution the run path needs, so any error
        // here is exactly what would otherwise surface at run time.
        let gfx = guest_gfx_version(&def.emulator)
            .map_err(|e| format!("rocjitsu-dbt cannot use this profile: {e}"))?;
        // A source override (option or env) lets a caller name a guest
        // ISA explicitly; otherwise the agent's gfx must be one the DBT
        // translator can read.
        if source_override(&def.emulator).is_none() && dbt_isa_name(gfx).is_none() {
            return Err(format!(
                "rocjitsu-dbt: guest {} (gfx_target_version {gfx}) is not a translatable \
                 source ISA; supported: {}",
                gfx_name(gfx),
                supported_isa_list(),
            ));
        }
        Ok(())
    }

    fn installed(&self) -> bool {
        is_installed()
    }

    fn supported(&self) -> SupportStatus {
        support_status()
    }

    fn discover_plugins(&self) -> Vec<PluginsDef> {
        Vec::new()
    }

    fn health(&self, _session: &SessionId) -> SessionHealth {
        let installed = is_installed();
        let support = support_status();
        let healthy = installed && support.supported;
        SessionHealth {
            healthy,
            state: Some(if healthy { "ready" } else { "error" }.to_string()),
            terminal: false,
            message: if healthy {
                None
            } else if !installed {
                Some(format!(
                    "rocjitsu-dbt: HSA tools hook library ({HOOKS_LIB_NAME}) not found"
                ))
            } else {
                Some(support.reason)
            },
            ..Default::default()
        }
    }

    fn injection_def(&self, session: &SessionId) -> Result<InjectionDef> {
        // The trait hands us only the session id, so recover the profile
        // (and thus the emulator def) it was started with.
        let profile = mirage_core::session::resolve_profile(session)?;
        let def = &profile.emulator;

        // Refuse to run unemulated: without the hook there is nothing to
        // translate the guest code objects, so the workload would either
        // fail to load or silently run native code. Fail loudly instead.
        let hooks = hooks_preload().ok_or_else(|| {
            MirageError::Other(format!(
                "rocjitsu-dbt: HSA tools hook library ({HOOKS_LIB_NAME}) not found; \
                 cannot translate workload"
            ))
        })?;

        // The translator must know which physical GPU ISA to emit code
        // for. Without a target there is nothing to run on.
        let target = resolve_target_isa(def).ok_or_else(|| {
            MirageError::Other(format!(
                "rocjitsu-dbt: no target ISA; set the `{OPT_TARGET_ISA}` option or \
                 RJ_DBT_TARGET_ISA, or run on a supported GPU (one of: {})",
                supported_isa_list()
            ))
        })?;

        let containerized = profile.containerize.is_some();
        // For a containerised session the workload runs inside a node
        // container that does not share the host filesystem, so the hook
        // library is bind-mounted in under the in-container lib dir (the
        // same convention the `rocjitsu` backend uses for its KMD lib) and
        // `HSA_TOOLS_LIB` must point at the in-container path.
        let hooks_in_workload = if containerized {
            format!("{}/{HOOKS_LIB_NAME}", crate::CONTAINER_LIB_DIR)
        } else {
            hooks.display().to_string()
        };

        let mut env = BTreeMap::new();
        env.insert("HSA_TOOLS_LIB".to_string(), hooks_in_workload.clone());
        env.insert("RJ_DBT_TARGET_ISA".to_string(), target);
        if let Some(source) = resolve_source_isa(def) {
            env.insert("RJ_DBT_SOURCE_ISA".to_string(), source);
        }
        // Forward an explicit hook log level from the exec environment so
        // a user can debug translation without editing the profile.
        if let Ok(log) = std::env::var("RJ_DBT_LOG")
            && !log.is_empty()
        {
            env.insert("RJ_DBT_LOG".to_string(), log);
        }

        let mounts = if containerized {
            vec![FileMount {
                host_path: hooks.display().to_string(),
                container_path: hooks_in_workload,
                read_only: true,
            }]
        } else {
            Default::default()
        };

        Ok(InjectionDef {
            wrapper: None,
            // The hook is loaded by ROCR via HSA_TOOLS_LIB, not LD_PRELOAD.
            ld_preload: None,
            files: Default::default(),
            env,
            mounts,
            libraries: Default::default(),
            // DBT runs the translated code on the host's physical GPU, so
            // a containerised node needs the host GPUs exposed.
            host_gpus: true,
        })
    }
}

/// Describe the rocjitsu-dbt backend for the registry.
pub fn describe() -> EmulatorDescription {
    EmulatorDescription {
        name: NAME.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description:
            "rocjitsu dynamic binary translation: run a GPU's code objects on a different \
             physical GPU by translating them at load time (e.g. gfx1250 on gfx950)"
                .to_string(),
        options_schema: options_schema(),
    }
}

/// The options the DBT backend accepts.
fn options_schema() -> Vec<OptionDef> {
    vec![
        OptionDef {
            name: OPT_TARGET_ISA.to_string(),
            dtype: mirage_core::common::SimpleType::String,
            description: format!(
                "Host GPU ISA to translate code objects to (one of: {}). \
                 Defaults to the first supported physically-present GPU.",
                supported_isa_list()
            ),
            default: None,
        },
        OptionDef {
            name: OPT_SOURCE_ISA.to_string(),
            dtype: mirage_core::common::SimpleType::String,
            description: format!(
                "Guest GPU ISA the workload was built for (one of: {}). \
                 Defaults to the profile agent's gfx target.",
                supported_isa_list()
            ),
            default: None,
        },
    ]
}

inventory::submit! {
    EmulatorBackendDef {
        kind: NAME,
        backend: &RocjitsuDbt,
    }
}

/// Map a packed `gfx_target_version` to the gfx ISA name the DBT hook
/// accepts, or `None` if the translator does not support it.
pub fn dbt_isa_name(gfx_target_version: u32) -> Option<&'static str> {
    DBT_ISAS
        .iter()
        .find(|(version, _)| *version == gfx_target_version)
        .map(|(_, name)| *name)
}

/// Comma-separated list of the gfx ISA names the translator supports,
/// for diagnostics.
fn supported_isa_list() -> String {
    DBT_ISAS
        .iter()
        .map(|(_, name)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Resolve the guest GPU's `gfx_target_version` from `def` by following
/// its topology + agent references. Mirrors the resolution
/// [`crate::kmd_config`] performs.
fn guest_gfx_version(def: &EmulatorDef) -> Result<u32> {
    let topology: TopologyDef = match &def.topology {
        MaybeRef::Owned(t) => t.clone(),
        MaybeRef::Ref(name) => mirage_core::topology::store::get(name)?,
    };
    let agent: AgentDef = match &topology.agent {
        MaybeRef::Owned(a) => a.clone(),
        MaybeRef::Ref(name) => mirage_core::agent::store::get(name)?,
    };
    Ok(agent.vm.gpu.device.gfx_target_version)
}

/// An explicit source-ISA override, from the `source_isa` option or the
/// `RJ_DBT_SOURCE_ISA` environment variable (env wins). `None` when
/// neither is set, in which case the guest is derived from the agent.
fn source_override(def: &EmulatorDef) -> Option<String> {
    if let Ok(value) = std::env::var("RJ_DBT_SOURCE_ISA")
        && !value.is_empty()
    {
        return Some(value);
    }
    string_option(def, OPT_SOURCE_ISA)
}

/// Resolve the host ISA the workload's code objects are translated *to*.
/// Priority: explicit `RJ_DBT_TARGET_ISA` env, then the `target_isa`
/// option, then the first physically-present GPU the translator
/// supports.
pub fn resolve_target_isa(def: &EmulatorDef) -> Option<String> {
    if let Ok(value) = std::env::var("RJ_DBT_TARGET_ISA")
        && !value.is_empty()
    {
        return Some(value);
    }
    if let Some(value) = string_option(def, OPT_TARGET_ISA) {
        return Some(value);
    }
    let present = gpu_gfx_versions();
    DBT_ISAS
        .iter()
        .find(|(version, _)| present.contains(version))
        .map(|(_, name)| (*name).to_string())
}

/// Resolve the guest ISA the workload was built *for*. Priority: an
/// explicit override (option or env), then the profile agent's gfx
/// target. `None` when no override is set and the agent cannot be
/// resolved or its gfx is not translatable.
pub fn resolve_source_isa(def: &EmulatorDef) -> Option<String> {
    if let Some(value) = source_override(def) {
        return Some(value);
    }
    guest_gfx_version(def)
        .ok()
        .and_then(dbt_isa_name)
        .map(str::to_string)
}

/// Read a string-valued emulator option, if present and non-empty.
fn string_option(def: &EmulatorDef, name: &str) -> Option<String> {
    match def.options.get(name) {
        Some(SimpleValue::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Search policy mirage uses to locate the rocjitsu HSA tools hook
/// library. Anchored on `librocjitsu_hooks.so`; mirrors the KMD
/// interposer's discovery (an in-tree monorepo build, then generic
/// ROCm/system fallbacks).
fn hooks_lib_search() -> LibSearch<'static> {
    LibSearch {
        file_env: &["ROCJITSU_HOOKS_LIB"],
        dir_env: &[],
        home_env: &[],
        lib_name: HOOKS_LIB_NAME,
        // rocjitsu's in-tree hooks build output, relative to the mirage
        // binary, so a monorepo build finds a fresh build without extra
        // configuration.
        binary_relative_dirs: &["../../../rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks"],
        system_fallbacks: true,
    }
}

/// Returns the path mirage should pass as `HSA_TOOLS_LIB`, located via
/// the shared discovery search.
pub fn hooks_preload() -> Option<PathBuf> {
    discovery::find_emulator_lib(&hooks_lib_search())
}

/// Returns true if the rocjitsu HSA tools hook library is reachable on
/// this machine.
pub fn is_installed() -> bool {
    discovery::is_lib_installed(&hooks_lib_search())
}

/// Whether this host has a physical GPU the DBT translator can target.
/// DBT runs the translated code on real hardware, so a supported GPU
/// must be present.
pub fn support_status() -> SupportStatus {
    let present = gpu_gfx_versions();
    let matched: Vec<&str> = DBT_ISAS
        .iter()
        .filter(|(version, _)| present.contains(version))
        .map(|(_, name)| *name)
        .collect();
    if !matched.is_empty() {
        return SupportStatus::supported(format!(
            "compatible GPU present for DBT target: {}",
            matched.join(", ")
        ));
    }
    let detected = if present.is_empty() {
        "none".to_string()
    } else {
        present
            .iter()
            .map(|v| gfx_name(*v))
            .collect::<Vec<_>>()
            .join(", ")
    };
    SupportStatus::unsupported(format!(
        "no GPU the DBT translator can target found (requires one of: {}); detected: {detected}",
        supported_isa_list()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirage_core::common::SimpleValue;
    use mirage_core::emulator::ExecMode;

    fn def_with(topology: MaybeRef<TopologyDef>) -> EmulatorDef {
        EmulatorDef {
            emulator: NAME.to_string(),
            plugins: Default::default(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology,
        }
    }

    #[test]
    fn isa_mapping_is_exhaustive_over_table() {
        for (version, name) in DBT_ISAS {
            assert_eq!(dbt_isa_name(*version), Some(*name));
        }
        // gfx1250 (MI450X) is deliberately not a translatable source.
        assert_eq!(dbt_isa_name(120500), None);
    }

    #[test]
    fn registered_under_canonical_name() {
        let backend = mirage_core::emulator::get_emulator_backend(NAME)
            .expect("rocjitsu-dbt backend must be registered");
        assert_eq!(backend.description().name, NAME);
    }

    #[test]
    fn target_isa_prefers_env_override() {
        let def = def_with(MaybeRef::Ref("unused".to_string()));
        // SAFETY: single-threaded test; restored before returning.
        unsafe { std::env::set_var("RJ_DBT_TARGET_ISA", "gfx1201") };
        assert_eq!(resolve_target_isa(&def), Some("gfx1201".to_string()));
        unsafe { std::env::remove_var("RJ_DBT_TARGET_ISA") };
    }

    #[test]
    fn target_isa_uses_option_when_no_env() {
        let mut def = def_with(MaybeRef::Ref("unused".to_string()));
        def.options.insert(
            OPT_TARGET_ISA.to_string(),
            SimpleValue::String("gfx942".to_string()),
        );
        // Ensure no stray env override is present.
        unsafe { std::env::remove_var("RJ_DBT_TARGET_ISA") };
        assert_eq!(resolve_target_isa(&def), Some("gfx942".to_string()));
    }

    #[test]
    fn source_isa_option_overrides_agent() {
        let mut def = def_with(MaybeRef::Ref("does-not-exist".to_string()));
        def.options.insert(
            OPT_SOURCE_ISA.to_string(),
            SimpleValue::String("gfx950".to_string()),
        );
        unsafe { std::env::remove_var("RJ_DBT_SOURCE_ISA") };
        assert_eq!(resolve_source_isa(&def), Some("gfx950".to_string()));
    }

    #[test]
    fn validate_rejects_untranslatable_guest_without_override() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());
        unsafe { std::env::remove_var("RJ_DBT_SOURCE_ISA") };

        // An owned gfx1250 (MI450X) agent: a valid mirage agent, but not
        // a DBT-translatable source.
        let mut agent = AgentDef::default();
        agent.vm.gpu.device.gfx_target_version = 120500;
        let topology = TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Owned(agent),
        };
        let profile = ProfileDef {
            name: "p".to_string(),
            description: None,
            emulator: def_with(MaybeRef::Owned(topology)),
            containerize: None,
        };
        let err = RocjitsuDbt.validate_profile(&profile).unwrap_err();
        assert!(err.contains("not a translatable"), "unexpected: {err}");
    }

    #[test]
    fn validate_accepts_translatable_guest() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());
        unsafe { std::env::remove_var("RJ_DBT_SOURCE_ISA") };

        let mut agent = AgentDef::default();
        agent.vm.gpu.device.gfx_target_version = 90500; // gfx950
        let topology = TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Owned(agent),
        };
        let profile = ProfileDef {
            name: "p".to_string(),
            description: None,
            emulator: def_with(MaybeRef::Owned(topology)),
            containerize: None,
        };
        assert!(RocjitsuDbt.validate_profile(&profile).is_ok());
    }
}
