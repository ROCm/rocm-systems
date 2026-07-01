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
use mirage_core::common::{MaybeRef, SimpleValue};
use mirage_core::config::OptionDef;
use mirage_core::emulator::{
    EmulatorBackend, EmulatorBackendDef, EmulatorDaemon, EmulatorDef, EmulatorDescription,
    ExecMode, SupportStatus,
};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::InjectionDef;
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionHealth, SessionId};
use mirage_core::topology::TopologyDef;

pub mod daemon;
pub mod dbt;

/// Overridable default environment for workloads run under rocjitsu.
///
/// These mirror the environment the upstream rocjitsu RCCL collective
/// tests run with (`rocjitsu/tests/daemon_test.cpp`): RCCL must avoid
/// the P2P and shared-memory transports the simulated topology does not
/// model and stay on a single loopback socket, while ROCr must use SDMA
/// copies and skip scratch reclaim. rocprofiler-register is disabled
/// since the simulated GPU does not back it. Applied as defaults in
/// [`Rocjitsu::injection_def`]; the per-exec environment overrides any
/// of them.
const RCCL_ENV_DEFAULTS: &[(&str, &str)] = &[
    ("HSA_ENABLE_SDMA", "1"),
    ("ROCPROFILER_REGISTER_ENABLED", "0"),
    ("HSA_NO_SCRATCH_RECLAIM", "1"),
    ("NCCL_P2P_DISABLE", "1"),
    ("NCCL_SHM_DISABLE", "1"),
    ("NCCL_SOCKET_NTHREADS", "1"),
    ("NCCL_NSOCKS_PERTHREAD", "1"),
    ("NCCL_SOCKET_IFNAME", "lo"),
    ("NCCL_MAX_NCHANNELS", "1"),
    ("NCCL_MIN_NCHANNELS", "1"),
    ("NCCL_NET_GDR_LEVEL", "LOC"),
    ("NCCL_IB_DISABLE", "1"),
    ("NCCL_CUMEM_ENABLE", "0"),
];

/// rocjitsu [`EmulatorBackend`] implementation. Bundles the
/// rocjitsu-specific injection (the KMD `LD_PRELOAD` plus the
/// `ROCJITSU_RUNTIME_DIR` env var and the `config_path` discovery file
/// it points at) and profile validation so callers dispatch generically
/// through [`mirage_core::emulator::get_emulator_backend`]. Stateless; a
/// single shared instance is registered in the emulator registry.
pub struct Rocjitsu;

impl EmulatorBackend for Rocjitsu {
    fn description(&self) -> EmulatorDescription {
        describe()
    }

    fn boot(&self, _def: &ProfileDef) -> std::result::Result<(), String> {
        Ok(())
    }

    fn options(&self) -> Vec<OptionDef> {
        Vec::new()
    }

    fn shutdown(&self, _session: &SessionId) {}

    fn validate_profile(&self, def: &ProfileDef) -> std::result::Result<(), String> {
        // Building the kmd config resolves the topology + agent
        // references; any error here is precisely what would otherwise
        // surface at run time. No session exists at validation time, so
        // no per-session config is written.
        kmd_config(&def.emulator, None)
            .map(|_| ())
            .map_err(|e| format!("rocjitsu cannot use this profile: {e}"))
    }

    fn installed(&self) -> bool {
        is_installed()
    }

    fn supported(&self) -> SupportStatus {
        // rocjitsu emulates the GPU in software, so it runs on any host
        // regardless of the physical hardware present.
        SupportStatus::supported("software emulator; no special hardware required")
    }

    fn discover_plugins(&self) -> Vec<PluginsDef> {
        Vec::new()
    }

    fn health(&self, _session: &SessionId) -> SessionHealth {
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
        // The trait hands us only the session id, so recover the profile
        // (and thus the emulator def) it was started with.
        let profile = mirage_core::session::resolve_profile(session)?;
        let def = &profile.emulator;
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

        // Default runtime tuning the emulated workload needs to behave
        // under rocjitsu. These mirror the environment the upstream
        // rocjitsu RCCL collective tests run with (see
        // `rocjitsu/tests/daemon_test.cpp`): RCCL must avoid the P2P and
        // shared-memory transports the simulated topology does not model
        // and stick to a single loopback socket, and ROCr must use SDMA
        // copies without scratch reclaim. They are *defaults*: the
        // per-exec environment (`mirage run --env KEY=VALUE`) is layered
        // on top in `mirage_host` and overrides any of these, so a user
        // who needs different RCCL/HSA tuning can still set it.
        for (key, value) in RCCL_ENV_DEFAULTS {
            env.insert((*key).to_string(), (*value).to_string());
        }

        // For a containerised session the workload runs inside a node
        // container that does *not* share the host filesystem, so the
        // rocjitsu libraries (the KMD interposer and the host-side library
        // it may dlopen) must be made available inside it. We declare them
        // as `libraries`; the orchestrator bind-mounts each into
        // `CONTAINER_LIB_DIR` (`/mnt/mirage/lib`), preserving its file
        // name, and adds that directory to `LD_LIBRARY_PATH`. The per-node
        // host *inside* the container re-resolves this injection against
        // its own environment, where its discovery also searches
        // `CONTAINER_LIB_DIR`, so the in-container resolution finds the
        // KMD library there with no extra configuration. Without these the
        // in-container host fails to locate the KMD library and the exec
        // can never start.
        let libraries = if profile.containerize.is_some() {
            let mut libraries = vec![ld_preload.display().to_string()];
            // A sibling host-side `librocjitsu.so`, if present next to the
            // KMD interposer: the interposer may dlopen it at runtime.
            if let Some(host_lib) = ld_preload.parent().map(|d| d.join(LIB_NAME))
                && host_lib != ld_preload
                && host_lib.exists()
            {
                libraries.push(host_lib.display().to_string());
            }
            libraries
        } else {
            Default::default()
        };

        Ok(InjectionDef {
            wrapper: None,
            ld_preload: Some(ld_preload.display().to_string()),
            files: Default::default(),
            env,
            mounts: Default::default(),
            libraries,
            host_gpus: false,
        })
    }

    fn start_daemon(&self, session: &SessionId) -> Result<Option<Box<dyn EmulatorDaemon>>> {
        // The per-node host hosts one rocjitsu daemon per node. If the
        // KMD library cannot be located there is nothing to host the
        // emulated device with; return `None` rather than erroring, since
        // the per-exec `injection_def` already fails loudly in that case
        // (and a non-rocjitsu host must not be blocked by a missing
        // rocjitsu library).
        let Some(lib) = kmd_preload() else {
            tracing::warn!(
                "rocjitsu: KMD library ({KMD_LIB_NAME}) not found; \
                 not starting daemon"
            );
            return Ok(None);
        };
        let profile = mirage_core::session::resolve_profile(session)?;
        let config = kmd_config(&profile.emulator, Some(session))?;
        // The daemon binds its socket under the same runtime directory the
        // workload's interposer probes (`$ROCJITSU_RUNTIME_DIR`), which is
        // exactly what `injection_def` exports — so the workload connects
        // to *this* daemon with no extra wiring.
        let runtime_dir = write_config_discovery(&config)?;
        let daemon =
            daemon::Daemon::start(&lib, &config, &runtime_dir).map_err(MirageError::Other)?;
        Ok(Some(Box::new(daemon)))
    }
}

/// Describe the rocjitsu emulator backend for the registry. Owned by
/// this crate (rather than `mirage_core`) so that all rocjitsu-
/// specific policy lives alongside the rocjitsu runtime integration.
pub fn describe() -> EmulatorDescription {
    EmulatorDescription {
        name: "rocjitsu".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "ROCm just-in-time GPU emulator (cycle-accurate or functional)".to_string(),
        options_schema: Vec::new(),
    }
}

inventory::submit! {
    EmulatorBackendDef {
        kind: "rocjitsu",
        backend: &Rocjitsu,
    }
}
/// Subdirectory name used to namespace rocjitsu's per-session runtime
/// directory (the daemon socket + `config_path` discovery file) under
/// the session dir.
pub const RUNTIME_SUBDIR: &str = "rocjitsu";

/// In-container directory where the host-side rocjitsu libraries are
/// bind-mounted for a containerised session. All mirage system mounts
/// live under `/mnt/mirage`; the in-container KMD discovery searches
/// this directory (see [`kmd_search_dirs`]).
pub const CONTAINER_LIB_DIR: &str = "/mnt/mirage/lib";

/// Name used for the KMD interposer library on disk.
pub const KMD_LIB_NAME: &str = "librocjitsu_kmd.so";

/// Name used for the host-side rocjitsu library on disk.
pub const LIB_NAME: &str = "librocjitsu.so";

/// Library file names accepted for the KMD interposer, in priority
/// order. The dedicated KMD interposer (`librocjitsu_kmd.so`) is
/// preferred; `librocjitsu.so` is accepted as a fallback for builds
/// that ship a single combined library.
pub const KMD_LIB_NAMES: &[&str] = &[KMD_LIB_NAME, LIB_NAME];

/// Name of the synthesised rocjitsu `SimulationConfig` written into the
/// per-session directory (`<session>/rj_config.json`).
pub const RJ_CONFIG_NAME: &str = "rj_config.json";

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
        .join(RUNTIME_SUBDIR);
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
/// Searches, in priority order, the in-tree monorepo build output
/// (relative to the mirage binary), `$ROCM_HOME/lib`, the ROCm SDK
/// install root reported by `rocm-sdk path --root`, and the in-container
/// mount directory ([`CONTAINER_LIB_DIR`]). In each location the
/// dedicated KMD interposer (`librocjitsu_kmd.so`) is preferred, falling
/// back to `librocjitsu.so` (see [`KMD_LIB_NAMES`]).
pub fn kmd_preload() -> Option<PathBuf> {
    kmd_search_dirs()
        .iter()
        .find_map(|dir| find_lib_in(dir, KMD_LIB_NAMES))
}

/// Directories searched for the KMD interposer, in priority order:
/// the in-tree monorepo build output (relative to the mirage binary),
/// `$ROCM_HOME/lib`, the ROCm SDK install root reported by `rocm-sdk
/// path --root` (`<root>/lib`), and the in-container mount directory.
fn kmd_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    // rocjitsu's in-tree KMD build output, relative to the mirage
    // binary, so a monorepo `cargo build` finds a fresh build without
    // extra configuration.
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        dirs.extend((0..=3).map(|levels| {
            exe_dir
                .iter()
                .chain(std::iter::repeat("..".as_ref()).take(levels))
                .chain(std::iter::once(
                    "rocjitsu/build/lib/rocjitsu/src/rocjitsu/kmd/linux".as_ref(),
                ))
                .collect::<PathBuf>()
        }));
        // Install layout: a `<prefix>/bin/mirage` finds its sibling
        // `<prefix>/lib/librocjitsu.so` (e.g. both installed to /opt/rocm
        // by scripts/mirage-docker-build.sh). Searched via KMD_LIB_NAMES,
        // so a combined `librocjitsu.so` is picked up here too.
        dirs.push(exe_dir.join("..").join("lib"));
    }
    // ROCm install root.
    if let Some(root) = std::env::var_os("ROCM_HOME").filter(|v| !v.is_empty()) {
        dirs.push(PathBuf::from(root).join("lib"));
    }
    // ROCm SDK install root reported by the `rocm-sdk` CLI (present when
    // a ROCm Python wheel venv is active).
    if let Some(root) = mirage_core::discovery::rocm_sdk_root() {
        dirs.push(root.join("lib"));
    }
    // In-container mount: for a containerised session the host libraries
    // are bind-mounted here (see `injection_def`), and the in-container
    // host re-resolves discovery against this directory.
    dirs.push(PathBuf::from(CONTAINER_LIB_DIR));
    dirs
}

/// First existing entry of `names` inside `dir`, if any.
fn find_lib_in(dir: &std::path::Path, names: &[&str]) -> Option<PathBuf> {
    names
        .iter()
        .map(|name| dir.join(name))
        .find(|candidate| candidate.is_file())
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
///    back to a content-addressed `sim_<hash>.json` in the system temp
///    directory so identical configs share a file and stale files are
///    never overwritten in-place.
pub fn kmd_config(def: &EmulatorDef, session: Option<&SessionId>) -> Result<PathBuf> {
    // Drop-in `--config <path>`: when an explicit rocjitsu simulation
    // config is supplied (mirage being used as a `rocjitsu` replacement)
    // use that file verbatim instead of synthesising one from the
    // profile's topology. This is the `--config` of the upstream
    // `rocjitsu` CLI. (Container path remapping is not applied; the
    // explicit-config path is intended for direct, non-containerised
    // drop-in use.)
    if let Some(SimpleValue::String(path)) = def.options.get("config") {
        let cfg = PathBuf::from(path);
        if !cfg.exists() {
            return Err(MirageError::Other(format!(
                "rocjitsu config not found: {path}"
            )));
        }
        return Ok(cfg);
    }

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
    // Honour the profile's per-node GPU count: rocjitsu's config loader
    // reads `vm.gpu.num_gpus` and synthesises that many KFD devices
    // (deriving per-GPU identities from the single `device` template).
    // Each node's host process emulates the GPUs local to that node, so
    // the per-node `gpus_per_node` is what the config requests.
    let mut vm = agent.vm;
    vm.gpu.num_gpus = topology.gpus_per_node.max(1);
    let sim = serde_json::json!({
        "max_ticks": 100000u64,
        "num_threads": 1u32,
        "exec_mode": exec_mode,
        "vm": vm,
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
        // file in the system temp directory so identical configs share a
        // file and stale files are never overwritten in-place.
        None => {
            let mut hasher = DefaultHasher::new();
            bytes.hash(&mut hasher);
            let key = format!("{:016x}", hasher.finish());
            let cfg = std::env::temp_dir()
                .join(RUNTIME_SUBDIR)
                .join(format!("sim_{key}.json"));
            if !cfg.exists() {
                mirage_core::state::write_bytes(&cfg, &bytes)?;
            }
            cfg
        }
    };
    Ok(cfg)
}

/// Returns true if rocjitsu is reachable on this machine — i.e. a
/// system install or sibling build of the KMD library is detected.
pub fn is_installed() -> bool {
    kmd_preload().is_some()
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
            emulator: "rocjitsu".to_string(),
            plugins: Default::default(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology: MaybeRef::Ref("does-not-exist".to_string()),
        };
        assert!(kmd_config(&def, None).is_err());
    }
}
