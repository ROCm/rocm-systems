//! `mirage_rocjitsu` — rocjitsu integration for the mirage binary.
//!
//! This crate exposes helpers the mirage binary needs at runtime:
//!
//! mirage does **not** build or embed the rocjitsu *library*
//! (`librocjitsu.so`); it is discovered at runtime from the installed
//! system (see [`kmd_preload`]).
//!
//! Runtime entry points:
//!
//! * [`kmd_config`] synthesises a runtime `SimulationConfig` JSON
//!   from an [`mirage_core::emulator::EmulatorDef`] by resolving its
//!   topology + agent references and wrapping them with rocjitsu's
//!   required runtime fields.

use std::path::PathBuf;

use mirage_core::agent::AgentDef;
use mirage_core::common::{MaybeRef, SimpleMap, SimpleValue};
use mirage_core::config::OptionDef;
use mirage_core::discovery::{LibSearch, RuntimeLocation};
use mirage_core::emulator::{
    EmulatorBackend, EmulatorBackendDef, EmulatorDaemon, EmulatorDef, EmulatorDescription,
    ExecMode, RuntimeStatus, SupportStatus,
};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::InjectionDef;
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionContext, SessionHealth, state};
use mirage_core::topology::TopologyDef;

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
#[derive(Debug)]
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

    fn shutdown(&self, _ctx: &SessionContext) {}

    fn validate_profile(&self, def: &ProfileDef) -> std::result::Result<(), String> {
        // Resolving the kmd config follows the topology + agent
        // references and applies rocjitsu's own limits; any error here is
        // precisely what would otherwise surface at run time. No session
        // exists at validation time, so nothing is written — not into the
        // session that does not exist yet, and not into a shared temp
        // directory nobody would ever clean up either.
        check_config(&def.emulator).map_err(|e| format!("rocjitsu cannot use this profile: {e}"))
    }

    fn runtime(&self) -> RuntimeStatus {
        // rocjitsu is installed exactly when its one library is on the
        // machine, so the search that answers "where?" also answers
        // "installed?" — see `runtime_location`. Which is why
        // `installed` is left to the trait: its default reads the flag
        // out of this, and the override that used to sit here was a
        // second route to the same search that could only ever agree or
        // be a bug.
        RuntimeStatus::from_location(runtime_location())
    }

    fn supported(&self) -> SupportStatus {
        // rocjitsu emulates the GPU in software, so it runs on any host
        // regardless of the physical hardware present.
        //
        // Still supported when the located library cannot host a daemon,
        // and deliberately: `--in-process` emulation goes through the
        // interposer and needs none of the daemon API, so calling the
        // host unsupported would refuse a mode that works. What it costs
        // is multi-process sharing of emulated GPU memory — and that is
        // reported, but not from here.
        //
        // Answering it here would mean `dlopen`ing the interposer to
        // answer a question about hardware. `registry()` calls this for
        // every backend, and `registry()` is on the path of every `mirage
        // run` that carries an override flag, `mirage profile create` and
        // `mirage emulators` alike — so a probe in this method maps the
        // KMD interposer into the CLI process of an `--in-process` run
        // that will never host a daemon, which is the very thing the
        // `if ctx.daemon` guard in the supervisor exists to avoid. It
        // would also break the contract this backend's own trait states:
        // `daemon_capability` must be cheap enough for `health` to ask,
        // and `supported` must be cheap enough for a listing.
        //
        // `mirage emulators -l` asks the capability directly instead, so
        // the one command whose job is detail is the one that pays for
        // it. See `emulators_cmd` in `mirage_ctl`.
        SupportStatus::supported("software emulator; no special hardware required")
    }

    fn discover_plugins(&self) -> Vec<PluginsDef> {
        // Report the plugins whose shared objects ship next to the
        // interposer (`librocjitsu_plugin_<name>.so`). Each entry is a
        // ready-to-use selection — the plugin name mapped to an empty
        // argument object — that a caller can merge into a profile's
        // `plugins` to enable it with its schema defaults. Empty when
        // rocjitsu is not installed or ships no plugins.
        let Some(preload) = kmd_preload() else {
            return Vec::new();
        };
        discover_plugin_names(&preload)
            .into_iter()
            .map(|name| PluginsDef::from([(name, SimpleMap::new())]))
            .collect()
    }

    fn health(&self, ctx: &SessionContext) -> SessionHealth {
        // One problem, decided once, so the snapshot is built in one
        // place. Built through `SessionHealth::phase` and not a struct
        // literal: the literal's `..Default::default()` fills `timestamp`
        // with `DateTime::<Utc>::default()`, which is the Unix epoch, so
        // every snapshot rocjitsu reported was stamped 1970-01-01 in the
        // serialized output. `phase` stamps `Utc::now()`.
        let problem = if is_installed() {
            // Located is not the same as usable *for this session*. A
            // library that predates the daemon API emulates a workload
            // in-process perfectly well and cannot host the daemon a
            // multi-process session needs, so a session that wants one is
            // not ready however present the library is.
            ctx.daemon
                .then(|| self.daemon_capability().err())
                .flatten()
                .map(|e| e.to_string())
        } else {
            Some(format!("rocjitsu KMD library ({LIB_NAME}) not found"))
        };
        match problem {
            // Not `state::FAILED`: that one means terminal, and neither of
            // these is — installing the library or updating it makes the
            // same session healthy without recreating it.
            Some(message) => SessionHealth::phase(false, "error", Some(message)),
            None => SessionHealth::phase(true, state::READY, None),
        }
    }

    fn injection_def(&self, ctx: &SessionContext) -> Result<InjectionDef> {
        let def = ctx.emulator();
        let config = kmd_config(def, &ctx.runtime_dir)?;
        // Refuse to run unemulated: if the KMD interposer can't be
        // located there is nothing to emulate the workload, so fail
        // loudly rather than silently running on real hardware.
        let ld_preload = kmd_preload().ok_or_else(|| {
            // The search itself says where it looked, so this cannot
            // drift from it the way the hand-written list did.
            let detail = runtime_location()
                .explain_missing()
                .unwrap_or_else(|| format!("{LIB_NAME} was not found"));
            MirageError::Other(format!(
                "rocjitsu: KMD preload library ({LIB_NAME}) not found; cannot \
                 emulate workload. Install rocjitsu (see docs/building.md) — \
                 {detail}"
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
        // `config` is a host path, and the file records it verbatim.
        // Nothing rewrites file *contents* on the way into a container —
        // only environment values are remapped onto the in-container
        // mounts — so the supervisor bind-mounts the session scratch
        // directory at its host path as well as at
        // `/mnt/mirage/runtime`, and this path resolves in both views.
        // See `plan_container` in `mirage_supervisor::session`. (Before
        // the supervisor existed, a per-node `mirage host` process inside
        // each container re-resolved the whole injection instead.)
        //
        // The runtime directory is the session's, whoever wrote the
        // config: in drop-in `--config` mode `config` is a file of the
        // user's, and deriving the runtime directory from *its* location
        // would leave the discovery file and the daemon socket beside it,
        // outside the session, uncleaned, and shared with any other run
        // pointed at the same config.
        let runtime_dir = write_config_discovery(&ctx.runtime_dir, &config)?;

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
        // rocjitsu library (`librocjitsu.so`) must be made available
        // inside it. We declare it as a `library`; the orchestrator
        // bind-mounts it into `CONTAINER_LIB_DIR` (`/mnt/mirage/lib`),
        // preserving its file name, and adds that directory to
        // `LD_LIBRARY_PATH`. The per-node host *inside* the container
        // re-resolves this injection against its own environment, where
        // its discovery also searches `CONTAINER_LIB_DIR`, so the
        // in-container resolution finds the library there with no extra
        // configuration. Without it the in-container host fails to locate
        // the library and the exec can never start.
        let libraries = if ctx.profile.containerize.is_some() {
            // Bind-mount the interposer plus the shared object for each
            // plugin this profile enables so the in-container plugin loader
            // can resolve it next to the interposer (the loader searches the
            // interposer's own directory / `LD_LIBRARY_PATH`, both of which
            // include `CONTAINER_LIB_DIR`). A plugin the interposer build
            // does not ship is silently skipped here and by the loader at
            // runtime.
            let mut libs = vec![ld_preload.display().to_string()];
            libs.extend(
                enabled_plugin_libs(&ld_preload, &def.plugins)
                    .into_iter()
                    .map(|path| path.display().to_string()),
            );
            libs
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

    fn daemon_capability(&self) -> Result<()> {
        // Answered once per process; see `located_daemon_capability`.
        located_daemon_capability()
            .as_ref()
            .map_or_else(|e| Err(MirageError::Other(e.to_string())), |()| Ok(()))
    }

    fn start_daemon(&self, ctx: &SessionContext) -> Result<Option<Box<dyn EmulatorDaemon>>> {
        // One rocjitsu daemon per session. If the KMD library cannot be
        // located there is nothing to host the emulated device with;
        // return `None` rather than erroring, since the per-exec
        // `injection_def` already fails loudly in that case.
        let Some(lib) = kmd_preload() else {
            tracing::warn!(
                "rocjitsu: KMD library ({LIB_NAME}) not found; \
                 not starting daemon"
            );
            return Ok(None);
        };
        let config = kmd_config(ctx.emulator(), &ctx.runtime_dir)?;
        // The daemon binds its socket under the same runtime directory the
        // workload's interposer probes (`$ROCJITSU_RUNTIME_DIR`), which is
        // exactly what `injection_def` exports — so the workload connects
        // to *this* daemon with no extra wiring. Both live in the
        // session's scratch directory and go away with it.
        let runtime_dir = write_config_discovery(&ctx.runtime_dir, &config)?;
        let daemon = rocjitsu_sys::daemon::Daemon::start(&lib, &config, &runtime_dir)
            .map_err(|e| MirageError::Other(format!("rocjitsu daemon: {e}")))?;
        Ok(Some(Box::new(RocjitsuDaemon(daemon))))
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
/// this directory (see [`kmd_preload`]).
pub const CONTAINER_LIB_DIR: &str = "/mnt/mirage/lib";

/// Name used for the rocjitsu library on disk. A single combined
/// `librocjitsu.so` exports both the KMD interposer (LD_PRELOAD) and the
/// HSA tools hooks (`HSA_TOOLS_LIB`).
pub const LIB_NAME: &str = "librocjitsu.so";

/// Filename prefix of a rocjitsu runtime plugin shared object. Together
/// with [`PLUGIN_LIB_SUFFIX`] it brackets the plugin's `<name>`:
/// `librocjitsu_plugin_<name>.so`. That `<name>` is the key used to
/// enable and configure the plugin in the config file.
pub const PLUGIN_LIB_PREFIX: &str = "librocjitsu_plugin_";
/// Filename suffix of a rocjitsu runtime plugin shared object (see
/// [`PLUGIN_LIB_PREFIX`]).
pub const PLUGIN_LIB_SUFFIX: &str = ".so";

/// Names of the rocjitsu plugins whose shared objects sit next to the
/// interposer `preload` (the `<name>` in `librocjitsu_plugin_<name>.so`),
/// sorted and de-duplicated. Empty when the directory cannot be read.
pub fn discover_plugin_names(preload: &std::path::Path) -> Vec<String> {
    let Some(dir) = preload.parent() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|entry| {
            let file_name = entry.file_name();
            let file_name = file_name.to_str()?;
            file_name
                .strip_prefix(PLUGIN_LIB_PREFIX)?
                .strip_suffix(PLUGIN_LIB_SUFFIX)
                .filter(|name| !name.is_empty())
                .map(str::to_string)
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Shared-object paths for the plugins `plugins` enables that actually
/// exist next to the interposer `preload`. Used to bind-mount the plugin
/// `.so`s into a containerised session alongside the interposer so the
/// in-container plugin loader resolves them (mirage adds the mount dir to
/// `LD_LIBRARY_PATH`). A requested plugin with no matching `.so` on disk
/// is omitted; the loader logs and skips it at runtime.
pub fn enabled_plugin_libs(preload: &std::path::Path, plugins: &PluginsDef) -> Vec<PathBuf> {
    let Some(dir) = preload.parent() else {
        return Vec::new();
    };
    plugins
        .keys()
        .filter_map(|name| {
            find_lib_in(
                dir,
                &format!("{PLUGIN_LIB_PREFIX}{name}{PLUGIN_LIB_SUFFIX}"),
            )
        })
        .collect()
}

/// Name of the synthesised rocjitsu `SimulationConfig` written into a
/// session's scratch directory.
pub const RJ_CONFIG_NAME: &str = "rj_config.json";

/// Path of the synthesised `SimulationConfig` inside a session's scratch
/// directory.
#[must_use]
pub fn rj_config_path(runtime_dir: &std::path::Path) -> PathBuf {
    runtime_dir.join(RJ_CONFIG_NAME)
}

/// Adapter making a [`rocjitsu_sys::daemon::Daemon`] usable as the
/// emulator-agnostic handle mirage's supervisor holds.
#[derive(Debug)]
struct RocjitsuDaemon(rocjitsu_sys::daemon::Daemon);

impl EmulatorDaemon for RocjitsuDaemon {
    fn stop(self: Box<Self>) {
        self.0.stop();
    }
}

/// Point the KMD interposer at `config` by writing the `config_path`
/// discovery file it reads from `$ROCJITSU_RUNTIME_DIR`, and return that
/// runtime directory (to export as `ROCJITSU_RUNTIME_DIR`).
///
/// The interposer resolves its `SimulationConfig` by reading a
/// `config_path` file from its per-user runtime directory; the file's
/// contents are the path to the config JSON.
///
/// The runtime directory is always [`RUNTIME_SUBDIR`] under
/// `session_dir` — the session's own scratch directory — and never
/// derived from where `config` happens to live. The daemon socket lands
/// there too, so both are owned by the session, disappear with it, and
/// stay distinct between two runs. That matters for the drop-in
/// `--config` mode in particular, where `config` is a file of the
/// user's that mirage has no business writing next to and that two
/// concurrent runs may well share.
pub fn write_config_discovery(
    session_dir: &std::path::Path,
    config: &std::path::Path,
) -> Result<PathBuf> {
    let runtime_dir = session_dir.join(RUNTIME_SUBDIR);
    let config_path_file = runtime_dir.join("config_path");
    mirage_core::state::write_bytes(
        &config_path_file,
        format!("{}\n", config.display()).as_bytes(),
    )?;
    Ok(runtime_dir)
}

/// Environment variable naming the KMD interposer directly, as an
/// absolute path to the `.so`. The explicit override that wins over
/// every search location, and the counterpart of the DBT backend's
/// `ROCJITSU_HOOKS_LIB`.
pub const LIB_ENV: &str = "ROCJITSU_LIB";

/// Returns the path mirage should pass as `LD_PRELOAD` to an
/// rocjitsu-emulated workload.
///
/// Discovery goes through the shared [`mirage_core::discovery`] policy,
/// so `$ROCJITSU_LIB`, `$LD_LIBRARY_PATH`, `$ROCM_HOME`/`$ROCM_PATH`, the
/// `rocm-sdk` install root and the standard ROCm/system library
/// directories all locate rocjitsu exactly as they locate every other
/// backend's library — see that module for the order. On top of the
/// shared policy this adds the two locations that are specific to
/// rocjitsu: an in-tree build beside this checkout
/// (`in_tree_relative_dirs`) and, last, the in-container mount
/// directory ([`CONTAINER_LIB_DIR`]).
pub fn kmd_preload() -> Option<PathBuf> {
    runtime_location().path().map(std::path::Path::to_path_buf)
}

/// Return the formatted build identity exported by the installed RocJITsu library.
///
/// The native RocJITsu CLI prints this exact string, so Mirage never maintains
/// a second repository stamp or output format.
///
/// # Errors
///
/// Returns a diagnostic when the library is absent, predates the version API,
/// or cannot be loaded.
pub fn version_string() -> std::result::Result<String, String> {
    let path = kmd_preload().ok_or_else(|| format!("{LIB_NAME} not found"))?;
    rocjitsu_sys::version_string(&path)
}

/// Where `librocjitsu.so` is on this machine, or — when it is not here —
/// every location [`kmd_preload`] probed for it and the environment
/// variables that would change the answer.
///
/// This is the same search [`kmd_preload`] performs, reported rather
/// than reduced to an `Option`, so `mirage emulators -l` can tell a user
/// whose rocjitsu is not found where mirage looked. Deriving the one
/// from the other keeps a single definition of the search: a "we looked
/// here" list assembled separately would be a second thing to keep in
/// step with the policy in [`mirage_core::discovery`].
#[must_use]
pub fn runtime_location() -> RuntimeLocation {
    let located = with_kmd_search(mirage_core::discovery::locate_emulator_lib);
    let RuntimeLocation::Missing {
        lib_name,
        mut searched,
        env,
    } = located
    else {
        return located;
    };
    // A containerised node reaches its bind-mounted copy through
    // `LD_LIBRARY_PATH`; this is the fallback for an in-container
    // process that did not inherit it. It is part of the search, so it
    // belongs in the list of places a failed search reports having
    // looked.
    let in_container = std::path::Path::new(CONTAINER_LIB_DIR).join(LIB_NAME);
    if in_container.is_file() {
        return RuntimeLocation::found(in_container);
    }
    searched.push(in_container);
    RuntimeLocation::Missing {
        lib_name,
        searched,
        env,
    }
}

/// Call `f` with the search policy for the KMD interposer.
///
/// The in-tree build locations are computed rather than listed (see
/// `in_tree_relative_dirs`), so the [`LibSearch`] borrows them and
/// cannot be returned; handing it to a callback is what lets the search
/// and any future "we looked here" guidance share one definition.
fn with_kmd_search<R>(f: impl FnOnce(&LibSearch<'_>) -> R) -> R {
    let in_tree = in_tree_relative_dirs();
    let in_tree: Vec<&str> = in_tree.iter().map(String::as_str).collect();
    f(&LibSearch {
        file_env: &[LIB_ENV],
        dir_env: &[],
        home_env: &[],
        lib_name: LIB_NAME,
        binary_relative_dirs: &in_tree,
        // rocjitsu is an ordinary ROCm-adjacent shared library: unlike
        // HotSwap it does not ship patched copies of the ROCm runtime,
        // so picking it up from `$LD_LIBRARY_PATH` or `/opt/rocm/lib` is
        // exactly what a user who installed it there expects.
        system_fallbacks: true,
    })
}

/// Sub-paths, relative to a project directory, that a rocjitsu build
/// leaves `librocjitsu.so` in.
///
/// `build/` is a plain in-tree `cmake -B build`; `dist/lib` and
/// `stage/lib` are what a superproject build stages into. Listing the
/// shapes rather than one blessed layout is what lets mirage find a
/// freshly built emulator without being told where it is.
const ROCJITSU_BUILD_SHAPES: &[&str] = &["build", "dist/lib", "stage/lib", "lib"];

/// Places a rocjitsu *project* could sit relative to an ancestor of the
/// `mirage` binary.
///
/// Both the sibling-checkout shape (`<root>/rocjitsu`, reached when the
/// ancestor is `emulation/`) and the superproject shape
/// (`<root>/build/emulation/rocjitsu`, reached when it is the repository
/// or its parent), because a CMake build directory is conventionally
/// either inside the checkout or immediately beside it.
const ROCJITSU_PROJECT_DIRS: &[&str] =
    &["rocjitsu", "emulation/rocjitsu", "build/emulation/rocjitsu"];

/// Directories, relative to the `mirage` binary's own directory, holding
/// a rocjitsu build in or beside this checkout.
///
/// This is the location that matters day to day, and it is deliberately
/// generous. A developer who has just built rocjitsu should not then
/// have to tell mirage where it went — the failure mode when they are
/// not told is silent and expensive, because every session test skips
/// and a skipped test still reports `ok`.
///
/// Each ancestor of the binary is tried as a possible repository root:
/// `target/<profile>/mirage` is three levels down, an integration-test
/// binary in `target/<profile>/deps/` is four, and a superproject build
/// directory beside the checkout is further still. Walking rather than
/// counting means none of those has to be enumerated correctly, and a
/// layout nobody anticipated still works.
///
/// Note the limit: this can only find a build that shares an ancestor
/// with the mirage binary. A build directory somewhere else entirely
/// still needs `$ROCJITSU_LIB` or `$ROCM_PATH`.
fn in_tree_relative_dirs() -> Vec<String> {
    const MAX_ANCESTORS: usize = 8;
    let mut dirs = Vec::with_capacity(
        MAX_ANCESTORS * ROCJITSU_PROJECT_DIRS.len() * ROCJITSU_BUILD_SHAPES.len(),
    );
    // The relative prefix for the ancestor being tried: empty for the
    // binary's own directory, then one `..` per level up. Empty rather
    // than `.` because these paths are shown to a user when discovery
    // fails, and `<dir>/./rocjitsu/build` reads as a typo.
    let mut up = String::new();
    for _ in 0..MAX_ANCESTORS {
        for project in ROCJITSU_PROJECT_DIRS {
            for shape in ROCJITSU_BUILD_SHAPES {
                if up.is_empty() {
                    dirs.push(format!("{project}/{shape}"));
                } else {
                    dirs.push(format!("{up}/{project}/{shape}"));
                }
            }
        }
        if up.is_empty() {
            up.push_str("..");
        } else {
            up.push_str("/..");
        }
    }
    dirs
}

/// First existing entry named `name` inside `dir`, if any.
fn find_lib_in(dir: &std::path::Path, name: &str) -> Option<PathBuf> {
    let candidate = dir.join(name);
    candidate.is_file().then_some(candidate)
}

/// Project the profile's plugin selection ([`PluginsDef`]) onto the JSON
/// object shape the rocjitsu config's `plugins` section expects: a map
/// from plugin name to its argument object. mirage's [`SimpleValue`] is
/// an externally-tagged serde enum, so a plugin argument cannot be
/// serialized verbatim (it would render as `{"Boolean": true}`); map each
/// value onto the plain JSON scalar the rocjitsu plugin loader parses.
fn plugins_to_json(plugins: &PluginsDef) -> serde_json::Value {
    let object = plugins
        .iter()
        .map(|(name, args)| {
            let arg_object = args
                .iter()
                .map(|(key, value)| {
                    let scalar = match value {
                        SimpleValue::String(s) => serde_json::Value::from(s.clone()),
                        SimpleValue::Number(n) => serde_json::Value::from(*n),
                        SimpleValue::Boolean(b) => serde_json::Value::from(*b),
                    };
                    (key.clone(), scalar)
                })
                .collect::<serde_json::Map<String, serde_json::Value>>();
            (name.clone(), serde_json::Value::Object(arg_object))
        })
        .collect::<serde_json::Map<String, serde_json::Value>>();
    serde_json::Value::Object(object)
}

/// Largest per-node GPU count mirage will ask rocjitsu to emulate.
///
/// Every GPU in `vm.gpu.num_gpus` becomes a whole software device inside
/// the session — its own KFD node, memory image and queues — built
/// during bring-up and torn down again at exit, so the cost is linear in
/// the count. Past a certain size that stops being a bigger emulated
/// machine and becomes a session that never finishes starting and does
/// not stop when it is asked to, which is the one thing mirage promises
/// cannot happen. Eight GPUs is the widest physical AMD node; this
/// leaves an order of magnitude of headroom above it.
pub const MAX_GPUS_PER_NODE: u32 = 64;

/// The rocjitsu `SimulationConfig` a profile resolves to, before any of
/// it reaches the disk.
#[derive(Debug)]
enum SimConfig {
    /// A config file of the user's own, named by the drop-in `--config`
    /// option and used verbatim. Already on disk; mirage only reads it.
    Supplied(PathBuf),
    /// Config JSON synthesised from the profile's topology + agent,
    /// still to be written into a session's scratch directory.
    Synthesised(Vec<u8>),
}

/// Resolve the rocjitsu `SimulationConfig` `def` asks for, writing
/// nothing.
///
/// The agent JSON under `<MIRAGE_CONFIG>/agent/` only stores the
/// `vm` + `topology` subset that mirage owns. rocjitsu's KMD shim
/// expects a full `SimulationConfig` (max_ticks, num_threads,
/// exec_mode, vm, topology). Unless the profile supplies a config of its
/// own, this:
///
/// 1. Resolves `def.topology` (and its inner `agent`), following
///    [`MaybeRef`] references against the on-disk
///    `<MIRAGE_CONFIG>/{topology,agent}/` stores.
/// 2. Wraps the agent's `vm` + `topology` with rocjitsu runtime
///    fields (`exec_mode` is taken from `def.exec_mode`; the other
///    fields use sane defaults).
///
/// Every way a profile can fail to describe a runnable machine surfaces
/// here, which is what lets [`check_config`] validate one without a
/// session and without leaving a file behind.
fn resolve_sim_config(def: &EmulatorDef) -> Result<SimConfig> {
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
        return Ok(SimConfig::Supplied(cfg));
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
    if topology.gpus_per_node > MAX_GPUS_PER_NODE {
        return Err(MirageError::Other(format!(
            "gpus-per-node {} is more than rocjitsu can emulate; the limit is \
             {MAX_GPUS_PER_NODE} per node. Each GPU is emulated as a whole software \
             device — its own KFD node, memory image and queues — built at bring-up \
             and torn down at exit, so a count this large produces a session that \
             never finishes starting and cannot be stopped promptly. The widest \
             physical AMD node is 8 GPUs. Pass --gpus-per-node {MAX_GPUS_PER_NODE} \
             or fewer, or spread the GPUs over more nodes with --num-nodes.",
            topology.gpus_per_node
        )));
    }
    // Honour the profile's per-node GPU count: rocjitsu's config loader
    // reads `vm.gpu.num_gpus` and synthesises that many KFD devices
    // (deriving per-GPU identities from the single `device` template).
    // Each node's host process emulates the GPUs local to that node, so
    // the per-node `gpus_per_node` is what the config requests.
    let mut vm = agent.vm;
    vm.gpu.num_gpus = topology.gpus_per_node.max(1);
    let mut sim = serde_json::json!({
        "max_ticks": 100000u64,
        "num_threads": 1u32,
        "exec_mode": exec_mode,
        "vm": vm,
        "topology": agent.topology,
    });
    // Carry the profile's plugin selection into the synthesised rocjitsu
    // config so the interposer (local path) and the per-node daemon both
    // enable them through the rocjitsu plugin loader. `def.plugins` maps a
    // plugin name to its argument object — exactly the shape rocjitsu's
    // `plugins` config section expects. Only emit the key when a plugin is
    // actually selected so a plugin-free profile still produces a clean,
    // minimal config (and the near-zero-overhead no-plugin path).
    if !def.plugins.is_empty()
        && let serde_json::Value::Object(map) = &mut sim
    {
        map.insert("plugins".to_string(), plugins_to_json(&def.plugins));
    }
    let bytes = serde_json::to_vec_pretty(&sim).map_err(|e| {
        MirageError::Other(format!("rocjitsu kmd_config: serialize sim config: {e}"))
    })?;
    Ok(SimConfig::Synthesised(bytes))
}

/// Materialise the rocjitsu `SimulationConfig` for `def` in
/// `session_dir` — the session's scratch directory — and return its
/// path. That path is what gets recorded in the rocjitsu `config_path`
/// discovery file so the LD_PRELOAD'd interposer loads it.
///
/// One file per session, rewritten on each call so it always reflects
/// the current profile, and removed with the session. A profile that
/// supplies its own config (drop-in `--config`) is returned as-is and
/// nothing is written.
///
/// # Errors
///
/// Returns an error when the topology or agent references cannot be
/// resolved, the profile asks for more GPUs than rocjitsu will emulate
/// (see [`MAX_GPUS_PER_NODE`]), or the config cannot be written.
pub fn kmd_config(def: &EmulatorDef, session_dir: &std::path::Path) -> Result<PathBuf> {
    match resolve_sim_config(def)? {
        SimConfig::Supplied(cfg) => Ok(cfg),
        SimConfig::Synthesised(bytes) => {
            let cfg = rj_config_path(session_dir);
            mirage_core::state::write_bytes(&cfg, &bytes)?;
            Ok(cfg)
        }
    }
}

/// Check that `def` describes a machine rocjitsu can stand up, without
/// writing anything.
///
/// This is what profile validation needs. It runs long before any
/// session exists — `mirage profile create`, and `mirage run`'s
/// override handling — so it has nowhere of its own to write and must
/// leave nothing behind; it therefore does everything [`kmd_config`]
/// does except the final write.
///
/// # Errors
///
/// Returns an error when the topology or agent references cannot be
/// resolved, the supplied drop-in config does not exist, or the profile
/// asks for more GPUs than rocjitsu will emulate.
pub fn check_config(def: &EmulatorDef) -> Result<()> {
    resolve_sim_config(def).map(|_| ())
}

/// Returns true if rocjitsu is reachable on this machine — i.e. a
/// system install or sibling build of the KMD library is detected.
pub fn is_installed() -> bool {
    kmd_preload().is_some()
}

/// Whether the rocjitsu library at `lib` can host a daemon.
///
/// The daemon entry points are newer than the rest of the C API, so a
/// perfectly good older `librocjitsu.so` loads, emulates in-process, and
/// has no `rj_daemon_start`. Nothing found that out until the daemon was
/// started, which for a containerised session is after the image pull,
/// the network and every container — so the run failed at its last step
/// on a fact about a file.
///
/// Split out from [`Rocjitsu::daemon_capability`] and taking the path
/// explicitly so the answer can be tested against a library known to lack
/// the symbols, with no installed rocjitsu and no environment override to
/// point the search at one.
///
/// # Errors
///
/// A reason phrased for a user who has just been refused a run: what is
/// wrong, what it costs, and the one flag that runs without it — except
/// for a library that will not load at all, which is told the truth
/// instead, because no flag runs without a library.
pub fn daemon_capability_of(lib: &std::path::Path) -> Result<()> {
    rocjitsu_sys::daemon::Daemon::probe(lib).map_err(|e| {
        // Two failures, opposite advice, and only the loader can tell
        // them apart. `--in-process` emulation `LD_PRELOAD`s this very
        // file, so recommending it to somebody whose library does not
        // load sends them to a second failure with the same cause and a
        // less obvious message.
        if e.is_unloadable() {
            return MirageError::Other(format!(
                "the rocjitsu library at {} could not be loaded: {e}\n\
                 This is not a rocjitsu too old for the emulator daemon — \
                 the file is there and the loader will not take it, which \
                 usually means a missing dependency or a build for another \
                 architecture. `--in-process` preloads this same library \
                 and will fail the same way, so reinstalling rocjitsu (see \
                 docs/building.md) is the fix.",
                lib.display()
            ));
        }
        MirageError::Other(format!(
            "the rocjitsu library at {} cannot host the emulator daemon: {e}\n\
             The daemon is what lets several processes share emulated GPU \
             memory, so multi-GPU collectives need it. An installation \
             predating the daemon API looks exactly like this; updating \
             rocjitsu (see docs/building.md) is the fix. Pass `--in-process` \
             to run without it — results from a single process are still \
             correct.",
            lib.display()
        ))
    })
}

/// The one-per-process answer to [`daemon_capability_of`] for the located
/// library.
///
/// `health` is asked on every status request and bring-up asks once more,
/// and the answer costs a `dlopen` of a large shared library whose
/// initialisers run. It cannot change under a running process — the
/// search reads the filesystem and the environment, neither of which this
/// process rewrites — so it is answered once and kept.
///
/// A library that is not installed at all is `Ok`: that is a different
/// problem, already reported by `runtime`, `health` and `injection_def`,
/// and `start_daemon` answers `Ok(None)` to it rather than failing.
/// Saying it again here would refuse an in-process run, which does not
/// need a daemon, for a missing library, with a worse message than the
/// one it is about to get.
fn located_daemon_capability() -> &'static Result<()> {
    static ANSWER: std::sync::OnceLock<Result<()>> = std::sync::OnceLock::new();
    ANSWER.get_or_init(|| match kmd_preload() {
        Some(lib) => daemon_capability_of(&lib),
        None => Ok(()),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn the_installed_flag_and_the_located_library_are_one_answer() {
        // `installed` used to be overridden here with `is_installed()`,
        // a second route to the same search. It agreed, and that is the
        // problem with it: two implementations of "is rocjitsu here?"
        // can only ever agree or be a bug, and the one a caller happens
        // to reach decides which. The override is gone and the trait's
        // default reads the flag out of `runtime`, so there is one
        // search and one verdict.
        let backend = Rocjitsu;
        assert_eq!(backend.installed(), backend.runtime().installed);
        assert_eq!(backend.installed(), is_installed());
    }

    #[test]
    fn a_library_without_the_daemon_api_is_refused_before_anything_is_created() {
        // The regression for the whole point of this check: a library
        // that is *here* and cannot host a daemon. A rocjitsu predating
        // the daemon API is exactly that — it loads, it emulates a
        // workload in-process, and it has no `rj_daemon_start` — and
        // nothing noticed until `start_daemon`, which for a containerised
        // session runs after the image pull, the network and every
        // container.
        //
        // The C library stands in for it: present, loadable, and it has
        // never heard of the rocjitsu C API. Anything with those three
        // properties would do; a host with no glibc `libc.so.6` to borrow
        // cannot run this, which is not a failure of the check.
        let libc = std::path::Path::new("libc.so.6");

        // Whether this host *has* a libc to borrow is decided on the typed
        // error, not by sniffing the rendered message for the loader's
        // wording. The strings this used to match ("cannot open shared
        // object", "No such file") are glibc's, not an API: on musl, in
        // another locale, or after a libloading change the skip would stop
        // firing and this would fail on a host it was written to tolerate
        // — or start firing everywhere and pass vacuously.
        if rocjitsu_sys::daemon::Daemon::probe(libc)
            .err()
            .is_some_and(|e| e.is_unloadable())
        {
            return;
        }

        let Err(e) = daemon_capability_of(libc) else {
            panic!("libc.so.6 hosts a rocjitsu daemon?");
        };
        let msg = e.to_string();

        // The message a user gets is the whole value of failing early, so
        // it has to say which file, what it costs them, and the one flag
        // that runs without it.
        assert!(msg.contains("libc.so.6"), "{msg}");
        assert!(msg.contains("cannot host the emulator daemon"), "{msg}");
        assert!(msg.contains("--in-process"), "{msg}");
    }

    #[test]
    fn a_library_that_will_not_load_is_not_told_to_pass_in_process() {
        // The other half of the diagnosis, and the one that sent a user
        // somewhere that fails again: `--in-process` emulation `LD_PRELOAD`s
        // the very library the loader has just refused, so advising it for
        // a library that cannot be loaded at all is advice to hit the same
        // wall from the other side. Only a library that *loads* and lacks
        // `rj_daemon_start` is an old rocjitsu.
        let missing = std::path::Path::new("/nonexistent/librocjitsu.so");
        let Err(e) = daemon_capability_of(missing) else {
            panic!("a library that is not there hosts a daemon?");
        };
        let msg = e.to_string();
        assert!(msg.contains("could not be loaded"), "{msg}");
        assert!(
            !msg.contains("predating the daemon API"),
            "a library the loader refuses is not a rocjitsu that is merely \
             too old: {msg}"
        );
        assert!(
            !msg.contains("Pass `--in-process`"),
            "`--in-process` preloads this same file and fails the same way, \
             so it must not be offered as the way round: {msg}"
        );
    }

    /// An [`EmulatorDef`] with an owned topology of `gpus_per_node`
    /// GPUs on a default agent, resolvable without touching the stores.
    fn def_with_gpus(gpus_per_node: u32) -> EmulatorDef {
        EmulatorDef {
            emulator: "rocjitsu".to_string(),
            plugins: Default::default(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology: MaybeRef::Owned(TopologyDef {
                num_nodes: 1,
                gpus_per_node,
                agent: MaybeRef::Owned(AgentDef::default()),
            }),
        }
    }

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
        assert!(check_config(&def).is_err());
        assert!(kmd_config(&def, tmp.path()).is_err());
    }

    /// Validating a profile must not write anything: it happens before a
    /// session exists, so whatever it wrote would have no owner and
    /// nothing would ever remove it. Validation used to leave a ~5 KB
    /// `sim_<hash>.json` in a fixed, shared, world-writable directory
    /// under the system temp directory, one per distinct profile,
    /// forever — so that is where this looks.
    #[test]
    fn check_config_writes_nothing() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        let shared = std::env::temp_dir().join(RUNTIME_SUBDIR);
        let entries = || -> std::collections::BTreeSet<std::ffi::OsString> {
            std::fs::read_dir(&shared)
                .map(|dir| dir.flatten().map(|e| e.file_name()).collect())
                .unwrap_or_default()
        };
        let before = entries();

        // A GPU count nothing else here uses, so a file left behind for
        // this profile cannot be one an earlier run already left.
        check_config(&def_with_gpus(47)).expect("an owned topology validates");

        assert_eq!(
            entries(),
            before,
            "profile validation must leave nothing behind in {}",
            shared.display()
        );
    }

    /// The counterpart: with a session directory to write into, the
    /// config lands there and nowhere else.
    #[test]
    fn kmd_config_writes_into_the_session_directory() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());
        let session = tmp.path().join("session");

        let cfg = kmd_config(&def_with_gpus(2), &session).unwrap();

        assert_eq!(cfg, rj_config_path(&session));
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&cfg).unwrap()).unwrap();
        assert_eq!(json["vm"]["gpu"]["num_gpus"], 2);
    }

    /// A drop-in `--config` is used verbatim, but its runtime directory
    /// belongs to the session: the discovery file (and the daemon socket
    /// beside it) must never land next to the user's config file, which
    /// mirage does not own and cannot clean up.
    #[test]
    fn discovery_file_lands_in_the_session_not_beside_the_config() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("mine");
        std::fs::create_dir_all(&user_dir).unwrap();
        let config = user_dir.join("cfg.json");
        std::fs::write(&config, b"{}").unwrap();
        let session = tmp.path().join("session");

        let runtime_dir = write_config_discovery(&session, &config).unwrap();

        assert_eq!(runtime_dir, session.join(RUNTIME_SUBDIR));
        assert_eq!(
            std::fs::read_to_string(runtime_dir.join("config_path")).unwrap(),
            format!("{}\n", config.display())
        );
        assert_eq!(
            std::fs::read_dir(&user_dir).unwrap().count(),
            1,
            "nothing may be written beside the user's own config file"
        );
    }

    #[test]
    fn a_node_wider_than_the_limit_is_refused_with_the_limit_named() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        // The limit itself is fine; one more is not.
        check_config(&def_with_gpus(MAX_GPUS_PER_NODE)).expect("the limit itself is allowed");
        let err = check_config(&def_with_gpus(MAX_GPUS_PER_NODE + 1)).unwrap_err();

        let msg = err.to_string();
        // A good message names the offending input, the limit, and the
        // way out.
        assert!(msg.contains(&(MAX_GPUS_PER_NODE + 1).to_string()), "{msg}");
        assert!(msg.contains(&MAX_GPUS_PER_NODE.to_string()), "{msg}");
        assert!(msg.contains("--num-nodes"), "{msg}");
        // And it must be refused before anything is written for it.
        assert!(kmd_config(&def_with_gpus(1_000_000), tmp.path()).is_err());
        assert!(!rj_config_path(tmp.path()).exists());
    }

    /// rocjitsu's discovery must go through the shared search policy, so
    /// the documented locations (`$LD_LIBRARY_PATH`, `$ROCM_PATH`, the
    /// standard ROCm directories) find it like any other backend's
    /// library.
    #[test]
    fn discovery_uses_the_shared_search_policy() {
        with_kmd_search(|search| {
            assert_eq!(search.lib_name, LIB_NAME);
            assert!(search.system_fallbacks, "the ROCm/system locations count");
            assert!(search.file_env.contains(&LIB_ENV));
            // The in-tree build shapes are still probed, relative to the
            // mirage binary, so a fresh sibling build is found untold.
            let candidates = search.candidate_paths();
            assert!(
                candidates
                    .iter()
                    .any(|p| p.ends_with("emulation/rocjitsu/build/librocjitsu.so")),
                "a sibling monorepo build must remain discoverable"
            );
            assert!(
                candidates.contains(&PathBuf::from("/opt/rocm/lib").join(LIB_NAME)),
                "the standard ROCm library directories must be searched"
            );
        });
    }

    /// What `mirage emulators` reports and what a workload actually
    /// gets preloaded must be the same file, on whichever kind of host
    /// this runs: a report that named a different library than the one
    /// mirage loads would be worse than no report at all.
    #[test]
    fn the_reported_location_is_the_library_mirage_preloads() {
        let location = runtime_location();
        assert_eq!(location.path(), kmd_preload().as_deref());
        assert_eq!(location.is_found(), is_installed());
        if let RuntimeLocation::Missing {
            lib_name, searched, ..
        } = &location
        {
            assert_eq!(lib_name, LIB_NAME);
            // Including the in-container mount, which is part of this
            // backend's search on top of the shared policy and would
            // otherwise be a location mirage probed without saying so.
            assert!(
                searched.contains(&std::path::Path::new(CONTAINER_LIB_DIR).join(LIB_NAME)),
                "the in-container fallback is searched, so it must be reported: {searched:?}"
            );
        }
    }

    #[test]
    fn plugins_to_json_projects_simple_values_to_plain_json() {
        let mut args = SimpleMap::new();
        args.insert("verbose".to_string(), SimpleValue::Boolean(true));
        args.insert(
            "path".to_string(),
            SimpleValue::String("/tmp/x".to_string()),
        );
        args.insert("level".to_string(), SimpleValue::Number(3));
        let plugins = PluginsDef::from([
            ("race".to_string(), SimpleMap::new()),
            ("logging".to_string(), args),
        ]);

        let json = plugins_to_json(&plugins);

        // An empty-arg plugin renders as an empty object, not null.
        assert_eq!(json["race"], serde_json::json!({}));
        // SimpleValue must project onto plain JSON scalars, NOT the
        // externally-tagged enum form ({"Boolean": true}) that a naive
        // serialization of SimpleValue would otherwise emit — the rocjitsu
        // plugin loader parses plain values.
        assert_eq!(
            json["logging"],
            serde_json::json!({"verbose": true, "path": "/tmp/x", "level": 3})
        );
    }

    #[test]
    fn discover_plugin_names_lists_plugin_sos_next_to_interposer() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let preload = dir.join(LIB_NAME);
        std::fs::write(&preload, b"").unwrap();

        // No plugin shared objects present yet.
        assert!(discover_plugin_names(&preload).is_empty());

        // Two real plugins, a non-plugin sibling sharing the `librocjitsu_`
        // prefix (ignored), and a degenerate empty-name file (ignored).
        std::fs::write(dir.join("librocjitsu_plugin_race.so"), b"").unwrap();
        std::fs::write(dir.join("librocjitsu_plugin_logging.so"), b"").unwrap();
        std::fs::write(dir.join("librocjitsu_hooks.so"), b"").unwrap();
        std::fs::write(dir.join("librocjitsu_plugin_.so"), b"").unwrap();

        // Sorted + de-duplicated names, prefix/suffix stripped.
        assert_eq!(
            discover_plugin_names(&preload),
            vec!["logging".to_string(), "race".to_string()]
        );
    }

    #[test]
    fn enabled_plugin_libs_returns_only_existing_enabled() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let preload = dir.join(LIB_NAME);
        std::fs::write(&preload, b"").unwrap();
        std::fs::write(dir.join("librocjitsu_plugin_race.so"), b"").unwrap();

        // Enable race (present on disk) and logging (absent). Only the
        // present one is returned for bind-mounting; the loader logs and
        // skips the missing plugin at runtime.
        let plugins = PluginsDef::from([
            ("race".to_string(), SimpleMap::new()),
            ("logging".to_string(), SimpleMap::new()),
        ]);
        let libs = enabled_plugin_libs(&preload, &plugins);
        assert_eq!(libs.len(), 1);
        assert_eq!(
            libs[0].file_name().and_then(|n| n.to_str()),
            Some("librocjitsu_plugin_race.so")
        );
    }
}
