//! `mirage_rocjitsu` — rocjitsu integration for the mirage binary.
//!
//! This crate exposes helpers the mirage binary needs at runtime:
//!
//! mirage does **not** build or embed the rocjitsu *libraries*
//! (`librocjitsu_kmd.so`, `librocjitsu.so`); they are discovered at
//! runtime from the installed system (see [`kmd_preload`]).
//!
//! Runtime entry points:
//!
//! * [`kmd_config`] synthesises a runtime `SimulationConfig` JSON
//!   from an [`mirage_core::emulator::EmulatorDef`] by resolving its
//!   topology + agent references and wrapping them with rocjitsu's
//!   required runtime fields.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use mirage_core::agent::AgentDef;
use mirage_core::common::MaybeRef;
use mirage_core::config::OptionDef;
use mirage_core::emulator::{Emulator, EmulatorDef, EmulatorDescription, ExecMode, SupportStatus};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::InjectionDef;
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::{FileMount, ProfileDef};
use mirage_core::session::{SessionHealth, SessionId};
use mirage_core::topology::TopologyDef;

/// rocjitsu [`Emulator`] implementation. Bundles the rocjitsu-specific
/// injection (the KMD `LD_PRELOAD` plus the `ROCJITSU_RUNTIME_DIR` env
/// var and the `config_path` discovery file it points at) and profile
/// validation so callers dispatch generically on
/// [`mirage_core::emulator::EmulatorKind`].
pub struct Rocjitsu {
    profile: ProfileDef,
}

impl Emulator for Rocjitsu {
    fn description() -> EmulatorDescription {
        describe()
    }

    fn new(def: ProfileDef) -> Self {
        Self { profile: def }
    }

    fn options() -> Vec<OptionDef> {
        Vec::new()
    }

    fn shutdown(self) {}

    fn validate_profile(def: &ProfileDef) -> std::result::Result<(), String> {
        // Building the kmd config resolves the topology + agent
        // references; any error here is precisely what would otherwise
        // surface at run time. No session exists at validation time, so
        // no per-session config is written.
        kmd_config(&def.emulator, None)
            .map(|_| ())
            .map_err(|e| format!("rocjitsu cannot use this profile: {e}"))
    }

    fn def(&self) -> &EmulatorDef {
        &self.profile.emulator
    }

    fn installed() -> bool {
        is_installed()
    }

    fn discover_plugins() -> Vec<PluginsDef> {
        Vec::new()
    }

    fn health(&self) -> SessionHealth {
        let installed = is_installed();
        SessionHealth {
            healthy: installed,
            state: Some(if installed { "ready" } else { "error" }.to_string()),
            terminal: false,
            message: if installed {
                None
            } else {
                Some(format!("rocjitsu KMD library ({KMD_LIB_NAME}) not found"))
            },
            ..Default::default()
        }
    }

    fn injection_def(&self, session: &SessionId) -> Result<InjectionDef> {
        let def = &self.profile.emulator;
        let config = kmd_config(def, Some(session))?;
        // Refuse to run unemulated: if the KMD interposer can't be
        // located there is nothing to emulate the workload, so fail
        // loudly rather than silently running on real hardware.
        let ld_preload = kmd_preload().ok_or_else(|| {
            MirageError::Other(format!(
                "rocjitsu: KMD preload library ({KMD_LIB_NAME}) not found; \
                 cannot emulate workload"
            ))
        })?;

        // The KMD interposer discovers its `SimulationConfig` by reading a
        // `config_path` file from its per-user runtime directory (resolved
        // as `$ROCJITSU_RUNTIME_DIR`, else `$XDG_RUNTIME_DIR/rocjitsu`, else
        // `/tmp/rocjitsu-<uid>`); the file's contents are the path to the
        // config JSON it then loads via `rj_vm_create`. It does *not* read
        // any configuration environment variable. We therefore point it at
        // a per-session runtime directory and write that discovery file
        // ourselves. Without it the interposer finds no config, never
        // stands up the emulated device, and the workload fails with
        // "Unable to open /dev/kfd ... No such device".
        //
        // `config` is already resolved for whichever filesystem view this
        // injection is computed in — host paths on the orchestrator, and
        // container paths (`/mnt/mirage/...`) when the per-node host
        // re-resolves this injection inside its container — so both the
        // runtime directory and the path written into `config_path` are
        // correct in either context.
        let runtime_dir = write_config_discovery(&config)?;

        let mut env = std::collections::BTreeMap::new();
        env.insert(
            "ROCJITSU_RUNTIME_DIR".to_string(),
            runtime_dir.display().to_string(),
        );

        // For a containerised session the workload runs inside a node
        // container that does *not* share the host filesystem, so the
        // rocjitsu runtime assets (the KMD interposer and the host-side
        // library it may dlopen) must be bind-mounted in. The per-node
        // host *inside* the container re-resolves this injection against
        // its own environment, where `MIRAGE_CACHE=/mnt/mirage/cache`, so
        // it looks for the assets under `<cache>/emulator/rocjitsu/` (see
        // [`asset_dir`]). We mount each host asset to exactly that
        // location so the in-container resolution finds them with no extra
        // configuration. Without these mounts the in-container host fails
        // to locate the KMD library and the exec can never start.
        let mounts = if self.profile.containerize.is_some() {
            // Mirrors `mirage_host`'s `CONTAINER_CACHE_DIR` +
            // `<emulator>/<ASSET_SUBDIR>`; kept as a literal here to avoid
            // a host→emulator crate dependency.
            let container_asset_dir = format!("/mnt/mirage/cache/emulator/{ASSET_SUBDIR}");
            let mut mounts = vec![FileMount {
                host_path: ld_preload.display().to_string(),
                container_path: format!("{container_asset_dir}/{KMD_LIB_NAME}"),
                read_only: true,
            }];
            // The host-side rocjitsu library, if present on disk: the KMD
            // interposer may dlopen it at runtime.
            let host_lib = lib_path();
            if host_lib.exists() {
                mounts.push(FileMount {
                    host_path: host_lib.display().to_string(),
                    container_path: format!("{container_asset_dir}/{LIB_NAME}"),
                    read_only: true,
                });
            }
            mounts
        } else {
            Default::default()
        };

        Ok(InjectionDef {
            wrapper: None,
            ld_preload: Some(ld_preload.display().to_string()),
            files: Default::default(),
            env,
            mounts,
            host_gpus: false,
        })
    }
}

/// Describe the rocjitsu emulator backend for the registry. Owned by
/// this crate (rather than `mirage_core`) so that all rocjitsu-
/// specific policy lives alongside the rocjitsu runtime integration.
/// Reports whether rocjitsu is installed and the resolved path to its
/// runtime KMD library when available.
pub fn describe() -> EmulatorDescription {
    EmulatorDescription {
        name: "rocjitsu".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "ROCm just-in-time GPU emulator (cycle-accurate or functional)".to_string(),
        installed: is_installed(),
        path: kmd_preload(),
        // rocjitsu emulates the GPU in software, so it runs on any
        // host regardless of the physical hardware present.
        support: SupportStatus::supported("software emulator; no special hardware required"),
    }
}
/// Subdirectory under `<MIRAGE_CACHE>/emulator/` where the extracted
/// runtime assets (`librocjitsu_kmd.so`, `librocjitsu.so`) live.
pub const ASSET_SUBDIR: &str = "rocjitsu";

/// Name used for the extracted KMD library on disk.
pub const KMD_LIB_NAME: &str = "librocjitsu_kmd.so";

/// Name used for the extracted host-side library on disk.
pub const LIB_NAME: &str = "librocjitsu.so";

/// Name of the synthesised rocjitsu `SimulationConfig` written into the
/// per-session directory (`<session>/rj_config.json`).
pub const RJ_CONFIG_NAME: &str = "rj_config.json";

/// Directory where extracted runtime assets are stored
/// (`<MIRAGE_CACHE>/emulator/rocjitsu/`).
pub fn asset_dir() -> PathBuf {
    mirage_core::paths::mirage_cache_dir()
        .join("emulator")
        .join(ASSET_SUBDIR)
}

/// On-disk path of the extracted KMD interposer library.
pub fn kmd_lib_path() -> PathBuf {
    asset_dir().join(KMD_LIB_NAME)
}

/// On-disk path of the extracted host-side rocjitsu library.
pub fn lib_path() -> PathBuf {
    asset_dir().join(LIB_NAME)
}

/// On-disk path of the synthesised `SimulationConfig` for `session`
/// (`<MIRAGE_RUNTIME>/session/<id>/rj_config.json`).
pub fn rj_config_path(session: &SessionId) -> PathBuf {
    mirage_core::paths::session_dir(session).join(RJ_CONFIG_NAME)
}

/// Point the KMD interposer at `config` by writing the `config_path`
/// discovery file it reads from `$ROCJITSU_RUNTIME_DIR`, and return that
/// runtime directory (to export as `ROCJITSU_RUNTIME_DIR`).
///
/// The interposer resolves its `SimulationConfig` by reading a
/// `config_path` file from its per-user runtime directory; the file's
/// contents are the path to the config JSON. This derives that runtime
/// directory from `config`'s location, writes the discovery file, and
/// returns the directory.
pub fn write_config_discovery(config: &std::path::Path) -> Result<PathBuf> {
    let runtime_dir = config
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(ASSET_SUBDIR);
    let config_path_file = runtime_dir.join("config_path");
    mirage_core::state::write_bytes(
        &config_path_file,
        format!("{}\n", config.display()).as_bytes(),
    )?;
    Ok(runtime_dir)
}

/// Returns the path mirage should pass as `LD_PRELOAD` to an
/// rocjitsu-emulated workload.
///
/// Prefers the extracted on-disk copy under
/// `<MIRAGE_CACHE>/emulator/rocjitsu/`; falls back to the shared
/// [`mirage_core::discovery`] search (ROCM_HOME, LD_LIBRARY_PATH,
/// next-to-binary, `./emulator/rocjitsu/`, standard system/ROCm dirs,
/// …) for workspaces that rely on a separately-installed rocjitsu.
pub fn kmd_preload() -> Option<PathBuf> {
    let extracted = kmd_lib_path();
    if extracted.exists() {
        return Some(extracted);
    }
    mirage_core::discovery::find_emulator_lib(&kmd_lib_search())
}

/// Shared discovery policy for the KMD interposer (`librocjitsu_kmd.so`).
fn kmd_lib_search() -> mirage_core::discovery::LibSearch<'static> {
    mirage_core::discovery::LibSearch {
        file_env: &["ROCJITSU_KMD_LIB"],
        dir_env: &["ROCJITSU_ROOT"],
        home_env: &[],
        lib_name: KMD_LIB_NAME,
        // rocjitsu's in-tree KMD build output, relative to the mirage
        // binary, so a monorepo `cargo build` finds a fresh build
        // without extra configuration.
        binary_relative_dirs: &["../../../rocjitsu/build/lib/rocjitsu/src/rocjitsu/kmd/linux"],
        // rocjitsu may also be installed separately; keep the generic
        // ROCm/system fallbacks.
        system_fallbacks: true,
    }
}

/// Synthesise a rocjitsu `SimulationConfig` JSON file from the given
/// [`EmulatorDef`] and return its `config_path`. That path is what gets
/// recorded in the rocjitsu `config_path` discovery file so the
/// LD_PRELOAD'd interposer loads it.
///
/// The agent JSON under `<MIRAGE_CONFIG>/agent/` only stores the
/// `vm` + `topology` subset that mirage owns. rocjitsu's KMD shim
/// expects a full `SimulationConfig` (max_ticks, num_threads,
/// exec_mode, vm, topology). This function:
///
/// 1. Resolves `def.topology` (and its inner `agent`), following
///    [`MaybeRef`] references against the on-disk
///    `<MIRAGE_CONFIG>/{topology,agent}/` stores.
/// 2. Wraps the agent's `vm` + `topology` with rocjitsu runtime
///    fields (`exec_mode` is taken from `def.exec_mode`; the other
///    fields use sane defaults).
/// 3. Writes the result to `<session>/rj_config.json` when `session`
///    is supplied (the per-session runtime location, alongside
///    `def.json`/`health.json`). When `session` is `None` — e.g. at
///    profile-validation time, before any session exists — it falls
///    back to a content-addressed
///    `<MIRAGE_CACHE>/emulator/rocjitsu/sim_<hash>.json` so identical
///    configs share a file and stale files are never overwritten
///    in-place.
pub fn kmd_config(
    def: &EmulatorDef,
    session: Option<&SessionId>,
) -> Result<PathBuf> {
    let topology: TopologyDef = match &def.topology {
        MaybeRef::Owned(t) => t.clone(),
        MaybeRef::Ref(name) => mirage_core::topology::store::get(name)?,
    };
    let agent: AgentDef = match &topology.agent {
        MaybeRef::Owned(a) => a.clone(),
        MaybeRef::Ref(name) => mirage_core::agent::store::get(name)?,
    };
    let exec_mode = match def.exec_mode {
        ExecMode::Functional => "functional",
        ExecMode::Clocked => "clocked",
    };
    let sim = serde_json::json!({
        "max_ticks": 100000u64,
        "num_threads": 1u32,
        "exec_mode": exec_mode,
        "vm": agent.vm,
        "topology": agent.topology,
    });
    let bytes = serde_json::to_vec_pretty(&sim).map_err(|e| {
        MirageError::Other(format!("rocjitsu kmd_config: serialize sim config: {e}"))
    })?;
    let cfg = match session {
        // Runtime: write the per-session config alongside the session's
        // other state. One file per session, rewritten each time so it
        // always reflects the current profile.
        Some(id) => {
            let cfg = rj_config_path(id);
            mirage_core::state::write_bytes(&cfg, &bytes)?;
            cfg
        }
        // Validation (no session yet): fall back to a content-addressed
        // cache file so identical configs share a file and stale files
        // are never overwritten in-place.
        None => {
            let mut hasher = DefaultHasher::new();
            bytes.hash(&mut hasher);
            let key = format!("{:016x}", hasher.finish());
            let cfg = asset_dir().join(format!("sim_{key}.json"));
            if !cfg.exists() {
                mirage_core::state::write_bytes(&cfg, &bytes)?;
            }
            cfg
        }
    };
    Ok(cfg)
}

/// Best-effort discovery of the rocjitsu source/install root. Used
/// only as a fallback when the embedded assets are not yet extracted.
pub fn root() -> PathBuf {
    if let Some(root) = std::env::var_os("ROCJITSU_ROOT") {
        return PathBuf::from(root);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("rocjitsu")
}

/// Returns true if rocjitsu is reachable on this machine — i.e. a
/// system install or sibling build of the KMD library is detected.
pub fn is_installed() -> bool {
    mirage_core::discovery::is_lib_installed(&kmd_lib_search())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmd_config_requires_resolvable_topology() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());
        let def = EmulatorDef {
            emulator: mirage_core::emulator::EmulatorKind::Rocjitsu,
            plugins: Default::default(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology: MaybeRef::Ref("does-not-exist".to_string()),
        };
        assert!(kmd_config(&def, None).is_err());
    }
}
