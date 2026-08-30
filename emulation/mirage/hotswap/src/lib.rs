//! `mirage_hotswap` — HotSwap integration for the mirage binary.
//!
//! HotSwap is a load-time ISA rewriter that runs a workload built for
//! one AMD GPU architecture on a different physical GPU (e.g.
//! `gfx1250` on `gfx942`/`gfx950`) by rewriting device code as it is
//! loaded.
//!
//! HotSwap is **not** a single HSA tools library. It is a set of
//! co-installed artifacts (see the reference Docker recipe), staged
//! into one tree:
//!
//! * `libhotswap_intercept.so` — the HIP intercept, `LD_PRELOAD`ed.
//! * `libhsa-runtime64.so`   — the HotSwap-patched ROCR runtime.
//! * `libamd_comgr.so`         — the COMGR transpiler.
//! * `llvm-tools/`             — `llc`/`llvm-mc`/`lld` the transpiler
//!   shells out to, plus a `runtime/hotswap_py/` python runtime.
//!
//! mirage does not build HotSwap by default (the `MIRAGE_BUILD_HOTSWAP`
//! CMake flag does an opt-in source build). This crate *discovers* an
//! installed HotSwap tree — anchored on `libhotswap_intercept.so` via
//! the shared [`mirage_core::discovery`] search policy — and wires it
//! into a workload through the HotSwap env contract (`HSA_HOTSWAP_*`
//! vars + the preloaded intercept). See [`../README.md`](../README.md).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mirage_core::config::OptionDef;
use mirage_core::discovery::{self, LibSearch};
use mirage_core::emulator::{
    EmulatorBackend, EmulatorBackendDef, EmulatorDescription, RuntimeStatus, SupportStatus,
};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::InjectionDef;
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::{FileMount, ProfileDef};
use mirage_core::session::{SessionContext, SessionHealth};

/// The HIP intercept library HotSwap ships. It is the artifact mirage
/// anchors discovery on (its directory is the HotSwap lib dir) and the
/// library the env contract `LD_PRELOAD`s.
pub const LIB_NAME: &str = "libhotswap_intercept.so";

/// The HotSwap-patched ROCR runtime, expected alongside [`LIB_NAME`].
pub const ROCR_LIB: &str = "libhsa-runtime64.so";

/// The COMGR transpiler, expected alongside [`LIB_NAME`].
pub const COMGR_LIB: &str = "libamd_comgr.so";

/// Subdirectory name used to namespace a HotSwap install (e.g. under an
/// `./emulator/` tree).
pub const ASSET_SUBDIR: &str = "hotswap";

/// In-container mount point for the HotSwap runtime tree (lib,
/// `llvm-tools/`, `runtime/hotswap_py/`). All mirage system mounts live
/// under `/mnt/mirage`; the HotSwap install root is bind-mounted here so
/// the injected `LD_PRELOAD`/`HOTSWAP_HOME`/`PYTHONPATH` paths resolve
/// inside each node's container.
pub const CONTAINER_HOTSWAP_DIR: &str = "/mnt/mirage/emulator/hotswap";

/// Human-facing name used in guidance messages.
pub const DISPLAY_NAME: &str = "HotSwap";

/// Default source GPU target the workload was built for, in the
/// `gfx<arch>:<wave>` form the HotSwap env contract expects. Injected
/// as `HSA_HOTSWAP_SOURCE_TARGET`; overridable from the exec environment.
pub const DEFAULT_SOURCE_TARGET: &str = "gfx1250:32";

/// Default HotSwap adapter policy (`HSA_HOTSWAP_BACKEND_ADAPTER_POLICY`).
pub const DEFAULT_ADAPTER_POLICY: &str = "compile";

/// The recognised HotSwap adapter policies, mirroring env_contract.py's
/// `ADAPTER_POLICIES`. Each maps to a set of backend adapters via
/// `adapter_backends_for_policy`.
pub const ADAPTER_POLICIES: &[&str] = &["none", "env", "native_build", "triton", "compile", "full"];

/// The physical GPU architectures HotSwap can retarget code *onto*,
/// as KFD `gfx_target_version` values paired with their conventional
/// `gfx` name. HotSwap rewrites device code at load time so a workload
/// built for one architecture runs on one of these cards (e.g.
/// `gfx1250` code on a `gfx942`/`gfx950` GPU). Without one of these
/// GPUs physically present there is nothing for HotSwap to run on.
pub const SUPPORTED_GPUS: &[(u32, &str)] = &[(90402, "gfx942"), (90500, "gfx950")];

/// HotSwap [`EmulatorBackend`] implementation. HotSwap is wired into a
/// workload via the env contract: the patched ROCR + COMGR shadow the
/// system copies (`LD_LIBRARY_PATH`), the HIP intercept is `LD_PRELOAD`ed,
/// and the `HSA_HOTSWAP_*` variables select the source target and policy.
/// Stateless; a single shared instance is registered in the emulator
/// registry.
#[derive(Debug)]
pub struct Hotswap;

impl EmulatorBackend for Hotswap {
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

    fn validate_profile(&self, _def: &ProfileDef) -> std::result::Result<(), String> {
        // HotSwap is not bundled or built by mirage; it must be
        // installed separately. Surface actionable guidance now, at
        // profile-creation time, rather than only when a session is
        // later started — and say which of its parts is missing, since
        // a half-staged tree is the likelier mistake than none at all.
        installed_lib_dir().map(|_| ())
    }

    fn runtime(&self) -> RuntimeStatus {
        // Finding the intercept is not the same as being installed here:
        // HotSwap is three co-located libraries, and a tree missing one
        // of them cannot emulate. Both answers come out of this one
        // search, so `mirage emulators` still stats the candidates once.
        //
        // Which is also why the trait's default `installed` is left in
        // place rather than overridden with [`is_installed`]. The two
        // are the same walk — locate the intercept, take its directory,
        // insist the directory is complete — and the reason this backend
        // is the one worth checking is that the walk has a second half
        // an override could quietly lose. It did once: `installed` was
        // "the intercept is here" while the injection demanded the whole
        // tree, so a half-staged install was reported missing, injected
        // anyway, and ran on the host GPU unemulated. `is_installed`
        // stays because callers outside the trait use it.
        let location = discovery::locate_emulator_lib(&lib_search());
        let installed = location
            .path()
            .and_then(Path::parent)
            .is_some_and(|dir| complete_tree(dir.to_path_buf()).is_ok());
        RuntimeStatus::new(installed, location)
    }

    fn supported(&self) -> SupportStatus {
        support_status()
    }

    fn discover_plugins(&self) -> Vec<PluginsDef> {
        Vec::new()
    }

    fn health(&self, _ctx: &SessionContext) -> SessionHealth {
        let support = support_status();
        let install = installed_lib_dir();
        let healthy = install.is_ok() && support.supported;
        SessionHealth {
            healthy,
            state: Some(if healthy { "ready" } else { "error" }.to_string()),
            terminal: false,
            // The install problem comes first when there is one: it is
            // the one the user can fix without changing machines, and it
            // names exactly which part of the tree is absent.
            message: match install {
                Err(problem) => Some(problem),
                Ok(_) if !support.supported => Some(support.reason),
                Ok(_) => None,
            },
            ..Default::default()
        }
    }

    fn injection_def(&self, ctx: &SessionContext) -> Result<InjectionDef> {
        self.injection_def_for(ctx, installed_lib_dir(), physical_target_gfx())
    }
}

impl Hotswap {
    /// [`EmulatorBackend::injection_def`] against an explicit view of
    /// the machine: the HotSwap install to wire in (as
    /// `installed_lib_dir` reports it) and the GPU to retarget onto.
    ///
    /// Both are threaded in for the reason `mirage_rocjitsu::dbt` threads
    /// its environment lookup: they come from `$HOTSWAP_HOME` and the
    /// host's own GPUs, and a test can change neither — Rust 2024 makes
    /// `set_var` unsafe, and the GPU is whatever the machine has — so a
    /// test that could not supply them could only ever assert about the
    /// host it happened to run on.
    ///
    /// # Errors
    ///
    /// Returns an error when the HotSwap install is unusable or the host
    /// has no GPU to retarget onto. Either way the workload would run
    /// unemulated, so neither is allowed to pass quietly.
    pub fn injection_def_for(
        &self,
        ctx: &SessionContext,
        install: std::result::Result<PathBuf, String>,
        target_gfx: Option<String>,
    ) -> Result<InjectionDef> {
        // Refuse to run unemulated. Anything short of a complete HotSwap
        // tree means the workload runs on the host GPU untouched and
        // exits 0, which is the worst outcome available: a green result
        // that never went near the emulator. This is the same predicate
        // `installed()` reports, so what mirage says about the backend
        // and what it will actually do cannot disagree.
        let dir = install.map_err(MirageError::Other)?;

        // Likewise for the hardware: HotSwap rewrites device code to run
        // on a real, compatible GPU, so without one there is nothing to
        // retarget onto.
        let target_gfx = target_gfx.ok_or_else(|| {
            MirageError::Other(format!(
                "hotswap: {}. {DISPLAY_NAME} rewrites device code to run on a real \
                 GPU, so there is nothing here to retarget onto and the workload \
                 would run unemulated. Run this on a supported card, or set \
                 HSA_HOTSWAP_ISA_OVERRIDE to the gfx name to target if you know \
                 this host can run it.",
                support_status().reason
            ))
        })?;

        let containerized = ctx.profile.containerize.is_some();
        let (ld_preload, env, mounts) = build_hotswap_env(&dir, containerized, &target_gfx);

        // HotSwap retargets device code onto a *physical* GPU, so the
        // workload always needs host GPU access. The container engine
        // turns this into the host GPU device nodes plus the
        // provider-specific group passthrough; it is a no-op for a
        // non-containerised session (which already sees the host GPUs).
        Ok(InjectionDef {
            wrapper: None,
            ld_preload: Some(ld_preload),
            files: Default::default(),
            env,
            mounts,
            libraries: Default::default(),
            host_gpus: true,
        })
    }
}

/// The backend adapters a policy enables, mirroring env_contract.py's
/// `_ADAPTERS_BY_POLICY`. Any unrecognised value falls back to the
/// default `compile` set.
fn adapter_backends_for_policy(policy: &str) -> &'static [&'static str] {
    match policy {
        "none" => &[],
        "env" => &["extension_jit"],
        "native_build" => &["extension_jit", "native_build"],
        "triton" => &["extension_jit", "triton"],
        "full" => &["extension_jit", "native_build", "triton", "inductor"],
        // "compile" (the default) and any unknown value.
        _ => &["extension_jit", "triton", "inductor"],
    }
}

/// The `gfx` arch portion of a `gfx<arch>:<wave>` source-target spec
/// (env_contract.py's `parse_target_spec(...)[0]`). `gfx1250:32` →
/// `gfx1250`.
fn source_arch_of(source_target: &str) -> &str {
    source_target.split(':').next().unwrap_or(source_target)
}

/// The physical GPU HotSwap retargets code *onto*, as
/// `HSA_HOTSWAP_ISA_OVERRIDE`: a caller's explicit `override_env`, else
/// the first [`SUPPORTED_GPUS`] entry this host actually has.
///
/// `None` when neither applies. It used to fall back to the first
/// *supported* arch instead, which names a card that is not in the
/// machine: HotSwap would then rewrite every kernel for a gfx942 that
/// nothing here can run, and the honest diagnosis — that this host has
/// no GPU HotSwap can target, which `mirage emulators` already reports —
/// would be buried under whatever the transpiler or ROCr failed with
/// several steps later.
fn target_gfx_for(present: &[u32], override_env: Option<String>) -> Option<String> {
    if let Some(value) = override_env.filter(|v| !v.is_empty()) {
        return Some(value);
    }
    SUPPORTED_GPUS
        .iter()
        .find(|(version, _)| present.contains(version))
        .map(|(_, name)| (*name).to_string())
}

/// [`target_gfx_for`] against this host and process environment.
fn physical_target_gfx() -> Option<String> {
    target_gfx_for(
        &mirage_core::hardware::gpu_gfx_versions(),
        std::env::var("HSA_HOTSWAP_ISA_OVERRIDE").ok(),
    )
}

/// Build the HotSwap env contract for a workload, mirroring
/// env_contract.py's `build_hotswap_env`: the `HSA_HOTSWAP_*` variables,
/// the python `sitecustomize` on `PYTHONPATH`, and the source-arch
/// overrides selected by the adapter policy. Returns the `LD_PRELOAD`
/// value (patched ROCR + intercept) separately so the host can merge it
/// with any user-supplied preload.
fn build_hotswap_env(
    dir: &Path,
    containerized: bool,
    target_gfx: &str,
) -> (String, BTreeMap<String, String>, Vec<FileMount>) {
    // Canonicalize to an absolute path: `dir` may be discovered via a
    // relative probe (e.g. `../../build/hotswap/lib`), and the workload
    // runs from a different cwd, so every path derived from it (LD_PRELOAD,
    // LD_LIBRARY_PATH, HOTSWAP_HOME) must be absolute.
    let host_dir = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    let source_target = std::env::var("HSA_HOTSWAP_SOURCE_TARGET")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_SOURCE_TARGET.to_string());
    let policy = std::env::var("HSA_HOTSWAP_BACKEND_ADAPTER_POLICY")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_ADAPTER_POLICY.to_string());
    let backends = adapter_backends_for_policy(&policy);
    let source_arch = source_arch_of(&source_target).to_string();

    // Establish host→workload path mappings. For a containerised session
    // every host tree HotSwap relies on is bind-mounted under
    // `/mnt/mirage` and the workload must reference the in-container path;
    // otherwise it sees the host paths directly. The install root is
    // mounted read-only (immutable runtime libs/tools).
    let mut mounts: Vec<FileMount> = Vec::new();
    let mut mappings: Vec<(PathBuf, PathBuf)> = Vec::new();
    if containerized && let Some(home) = host_dir.parent() {
        mounts.push(FileMount {
            host_path: home.display().to_string(),
            container_path: CONTAINER_HOTSWAP_DIR.to_string(),
            read_only: true,
        });
        mappings.push((home.to_path_buf(), PathBuf::from(CONTAINER_HOTSWAP_DIR)));
    }
    // Rewrite a host path to the path the workload sees: any path under a
    // mounted host root maps to its container target; everything else is
    // left as the host path (correct for non-containerised runs).
    let remap = |p: &Path| -> PathBuf {
        for (host_root, container_root) in &mappings {
            if let Ok(rel) = p.strip_prefix(host_root) {
                return container_root.join(rel);
            }
        }
        p.to_path_buf()
    };

    let dir = remap(&host_dir);
    let dir = dir.as_path();

    // env_contract.py preloads the patched ROCR first, then the intercept.
    let libhsa = dir.join(ROCR_LIB);
    let intercept = dir.join(LIB_NAME);
    let ld_preload = format!("{}:{}", libhsa.display(), intercept.display());

    let mut env: BTreeMap<String, String> = BTreeMap::new();
    env.insert("HSA_HOTSWAP_CACHE_DEBUG".into(), "1".into());
    env.insert("HSA_HOTSWAP_ISA_OVERRIDE".into(), target_gfx.to_string());
    env.insert("HSA_HOTSWAP_IR_RAISER".into(), "1".into());
    env.insert("HSA_HOTSWAP_STRICT".into(), "1".into());
    env.insert("HSA_HOTSWAP_SOURCE_TARGET".into(), source_target.clone());
    env.insert("HSA_HOTSWAP_BACKEND_ADAPTER_POLICY".into(), policy.clone());
    env.insert("HSA_HOTSWAP_BACKEND_ADAPTERS".into(), backends.join(","));

    // Force-compile framework caches so a swapped target never reuses a
    // host-targeted artifact.
    env.insert("TRITON_ALWAYS_COMPILE".into(), "1".into());

    // The patched ROCR + COMGR shadow the system copies via the loader
    // search path, so name the HotSwap lib dir explicitly. A workload
    // inherits the caller's environment, and the supervisor treats
    // `LD_LIBRARY_PATH` as a search list: this entry is prepended to
    // whatever the caller exported rather than replacing it, so the
    // patched libraries win the search without deleting the user's.
    env.insert("LD_LIBRARY_PATH".into(), dir.display().to_string());
    // Expose the install root so the workload/runtime can resolve the
    // sibling `llvm-tools/` and `runtime/hotswap_py/` trees.
    if let Some(home) = dir.parent() {
        env.insert("HOTSWAP_HOME".into(), home.display().to_string());
    }
    if let Some(py) = py_dir_in(&host_dir) {
        // `sitecustomize.py` lives at the root of the python runtime, so
        // putting it on PYTHONPATH auto-activates the adapter layer.
        env.insert("PYTHONPATH".into(), remap(&py).display().to_string());
    }

    // Steer the frameworks' own arch selection at the source arch so they
    // emit code HotSwap can raise and re-target.
    if policy != "none" {
        for key in [
            "PYTORCH_ROCM_ARCH",
            "PYTORCH_ROCM_ARCH_OVERRIDE",
            "GPU_ARCHS",
            "AITER_GPU_ARCHS",
            "TRITON_OVERRIDE_ARCH",
        ] {
            env.insert(key.into(), source_arch.clone());
        }
    }
    if backends.contains(&"native_build") {
        for key in [
            "HIP_ARCHITECTURES",
            "CMAKE_HIP_ARCHITECTURES",
            "AMDGPU_TARGETS",
            "HCC_AMDGPU_TARGET",
            "ROCM_TARGETS",
        ] {
            env.insert(key.into(), source_arch.clone());
        }
    }
    if backends.contains(&"triton") {
        env.insert("TRITON_CORPUS_FORCE_TARGET".into(), source_target);
    }

    (ld_preload, env, mounts)
}

/// Describe the hotswap emulator backend for the registry. Owned by
/// this crate (rather than `mirage_core`) so that all hotswap-specific
/// policy lives alongside the hotswap discovery integration.
pub fn describe() -> EmulatorDescription {
    EmulatorDescription {
        name: "hotswap".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "load-time ISA rewriter: run a GPU's code on a different GPU (e.g. gfx1250 on gfx942/gfx950)".to_string(),
        options_schema: Vec::new(),
    }
}

inventory::submit! {
    EmulatorBackendDef {
        kind: "hotswap",
        backend: &Hotswap,
    }
}

/// Determine whether this host has a physical GPU HotSwap can retarget
/// onto. HotSwap needs a real, compatible GPU present (it rewrites code
/// to run on the hardware), so this inspects the host GPUs reported by
/// the kernel and matches them against [`SUPPORTED_GPUS`].
pub fn support_status() -> SupportStatus {
    let present = mirage_core::hardware::gpu_gfx_versions();
    let matched: Vec<&str> = SUPPORTED_GPUS
        .iter()
        .filter(|(version, _)| present.contains(version))
        .map(|(_, name)| *name)
        .collect();

    if !matched.is_empty() {
        return SupportStatus::supported(format!("compatible GPU present: {}", matched.join(", ")));
    }

    let required: Vec<&str> = SUPPORTED_GPUS.iter().map(|(_, name)| *name).collect();
    let detected = if present.is_empty() {
        "none".to_string()
    } else {
        present
            .iter()
            .map(|v| mirage_core::hardware::gfx_name(*v))
            .collect::<Vec<_>>()
            .join(", ")
    };
    SupportStatus::unsupported(format!(
        "no compatible GPU found (HotSwap requires one of: {}); detected: {detected}",
        required.join(", ")
    ))
}

/// Search policy mirage uses to locate the HotSwap install. Discovery
/// is anchored on `libhotswap_intercept.so`; its directory is the
/// HotSwap lib dir (where the patched ROCR and COMGR also live).
///
/// Kept deliberately small (see the crate README's "Where mirage
/// looks"): an explicit `$HOTSWAP_HOME` install root, then the in-tree
/// `MIRAGE_BUILD_HOTSWAP` build output. No generic system fallbacks —
/// HotSwap ships its own patched ROCR/COMGR, so picking libs up from
/// `$LD_LIBRARY_PATH` or `/opt/rocm/lib` would be wrong.
pub fn lib_search() -> LibSearch<'static> {
    LibSearch {
        file_env: &[],
        dir_env: &[],
        // The HotSwap install root: `$HOTSWAP_HOME/lib` holds the libs.
        // This is what the `MIRAGE_BUILD_HOTSWAP` source build stages to
        // (under `build/hotswap`), and the var mirage injects.
        home_env: &["HOTSWAP_HOME"],
        lib_name: LIB_NAME,
        // The in-tree `MIRAGE_BUILD_HOTSWAP` source build stages to
        // `build/hotswap/lib`. Relative to the mirage binary at
        // `target/<profile>/mirage`, that is `../../build/hotswap/lib`,
        // so a monorepo build finds it without extra configuration.
        binary_relative_dirs: &["../../build/hotswap/lib"],
        // HotSwap's discovery contract is just the two locations above.
        system_fallbacks: false,
    }
}

/// Locate the HotSwap intercept library, returning its path if HotSwap
/// is installed anywhere mirage knows to look.
pub fn lib_path() -> Option<PathBuf> {
    discovery::find_emulator_lib(&lib_search())
}

/// The HotSwap lib dir: the directory containing the intercept, the
/// patched ROCR, and the COMGR transpiler. The env contract points
/// `LD_LIBRARY_PATH` here and exposes its parent as `HOTSWAP_HOME`.
pub fn lib_dir() -> Option<PathBuf> {
    lib_path().and_then(|p| p.parent().map(Path::to_path_buf))
}

/// The HotSwap install root — the parent of the lib dir — under which
/// the `llvm-tools/` and `runtime/hotswap_py/` siblings live.
fn install_root() -> Option<PathBuf> {
    lib_dir().and_then(|d| d.parent().map(Path::to_path_buf))
}

/// The python adapter runtime directory, if present: `runtime/hotswap_py`
/// under the install root.
pub fn py_dir() -> Option<PathBuf> {
    install_root()
        .map(|r| r.join("runtime/hotswap_py"))
        .filter(|p| p.is_dir())
}

/// The python adapter runtime directory belonging to `lib_dir`, if
/// present.
///
/// The env contract derives it from the lib dir it is wiring in rather
/// than from [`py_dir`], so every path it emits comes from one install:
/// the two differ for a containerised session, where the lib dir has
/// been canonicalized in order to be remapped onto the bind mount and a
/// separately-resolved python directory would not match the mapping and
/// would leak a host path into the container.
fn py_dir_in(lib_dir: &Path) -> Option<PathBuf> {
    lib_dir
        .parent()
        .map(|root| root.join("runtime/hotswap_py"))
        .filter(|p| p.is_dir())
}

/// Returns `true` if a usable HotSwap install is present on this
/// machine (the intercept, patched ROCR, and COMGR all co-located).
pub fn is_installed() -> bool {
    installed_lib_dir().is_ok()
}

/// The lib dir of a *complete* HotSwap install, or a message explaining
/// what is missing.
///
/// This is the single answer to "can this machine emulate?", used both
/// to report the backend as installed and to build the injection. They
/// were once two different predicates — the injection needed only the
/// intercept — and a half-staged tree passed one and failed the other:
/// mirage said HotSwap was not installed, built the injection anyway,
/// and the workload ran on the host GPU unemulated and exited 0.
fn installed_lib_dir() -> std::result::Result<PathBuf, String> {
    let Some(dir) = lib_dir() else {
        return Err(format!(
            "{DISPLAY_NAME} is not installed: {LIB_NAME} was not found, so the \
             workload cannot be emulated.\n{}",
            install_guidance()
        ));
    };
    complete_tree(dir)
}

/// `dir` when it holds every library HotSwap needs, else a message
/// naming the ones it does not. Split out from `installed_lib_dir` so
/// the check can be made against a staged tree rather than whatever this
/// machine happens to have installed.
fn complete_tree(dir: PathBuf) -> std::result::Result<PathBuf, String> {
    let missing: Vec<&str> = [ROCR_LIB, COMGR_LIB]
        .into_iter()
        .filter(|lib| !dir.join(lib).is_file())
        .collect();
    if missing.is_empty() {
        return Ok(dir);
    }
    Err(format!(
        "{DISPLAY_NAME} is installed at {dir} but the tree is incomplete: {missing} \
         {verb} missing next to {LIB_NAME}. {DISPLAY_NAME} is not one library — the \
         intercept rewrites device code, the patched ROCR runtime loads it and COMGR \
         transpiles it — so all three must sit in the same directory; without them \
         the workload would run on the host GPU unemulated. Stage the full install \
         and point $HOTSWAP_HOME at its root.",
        dir = dir.display(),
        missing = missing.join(" and "),
        verb = if missing.len() == 1 { "is" } else { "are" },
    ))
}

/// Multi-line, user-facing guidance describing where mirage looked for
/// HotSwap and how to make it discoverable.
pub fn install_guidance() -> String {
    discovery::install_guidance(DISPLAY_NAME, &lib_search())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn the_installed_flag_accounts_for_the_whole_tree() {
        // The backend worth checking, because HotSwap's install is three
        // co-located libraries and "is it here?" is not "was the
        // intercept found". The override that used to sit on `installed`
        // called `is_installed()`, which walks the same three steps
        // — locate the intercept, take its directory, insist the
        // directory is complete — and agreeing was the most it could do.
        // The trait's default now reads the flag `runtime` computed, so
        // a tree missing its ROCR or COMGR cannot be reported installed
        // by one path and missing by the other.
        let backend = Hotswap;
        assert_eq!(backend.installed(), backend.runtime().installed);
        assert_eq!(backend.installed(), is_installed());
        // And the location came with the verdict, whichever way it went.
        let status = backend.runtime();
        assert_eq!(status.location.is_found(), status.location.path().is_some());
    }

    #[test]
    fn search_targets_the_intercept_lib() {
        let s = lib_search();
        assert_eq!(s.lib_name, "libhotswap_intercept.so");
        // Discovery is scoped to the HotSwap install root and the
        // in-tree build output — no generic system fallbacks.
        assert!(s.home_env.contains(&"HOTSWAP_HOME"));
        assert!(!s.system_fallbacks);
    }

    #[test]
    fn guidance_mentions_the_library() {
        assert!(install_guidance().contains("libhotswap_intercept.so"));
    }

    #[test]
    fn source_arch_strips_the_wave_suffix() {
        assert_eq!(source_arch_of("gfx1250:32"), "gfx1250");
        assert_eq!(source_arch_of("gfx942"), "gfx942");
    }

    #[test]
    fn adapter_policies_map_to_backends() {
        assert!(adapter_backends_for_policy("none").is_empty());
        assert_eq!(adapter_backends_for_policy("env"), &["extension_jit"]);
        // The default and any unknown value resolve to the `compile` set.
        let compile = ["extension_jit", "triton", "inductor"];
        assert_eq!(
            adapter_backends_for_policy(DEFAULT_ADAPTER_POLICY),
            &compile
        );
        assert_eq!(adapter_backends_for_policy("bogus"), &compile);
        // Every named policy is recognised (no fallthrough surprises).
        assert!(ADAPTER_POLICIES.contains(&DEFAULT_ADAPTER_POLICY));
    }

    /// A directory under the system temp directory, removed on drop.
    ///
    /// These tests need a HotSwap tree on disk to stage half-installed
    /// and fully-installed shapes into, and staging one is the whole of
    /// what a temp-directory crate would do for them; this crate has no
    /// dev-dependencies to pull one in with.
    #[derive(Debug)]
    struct StagedTree(PathBuf);

    impl StagedTree {
        /// An empty directory named after `tag` and this process, so
        /// concurrent test binaries cannot collide.
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "mirage-hotswap-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("stage a HotSwap tree");
            Self(dir)
        }

        /// Place an empty stand-in for library `name`. Discovery only
        /// ever asks whether the file is there.
        fn with(self, name: &str) -> Self {
            std::fs::write(self.0.join(name), b"").expect("stage a library");
            self
        }

        fn path(&self) -> PathBuf {
            self.0.clone()
        }
    }

    impl Drop for StagedTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn session_ctx() -> SessionContext {
        SessionContext {
            id: mirage_core::session::SessionId::new("hotswap-test").unwrap(),
            profile: ProfileDef {
                name: "hotswap-test".to_string(),
                description: None,
                emulator: mirage_core::emulator::EmulatorDef {
                    emulator: "hotswap".to_string(),
                    plugins: Default::default(),
                    exec_mode: mirage_core::emulator::ExecMode::Functional,
                    options: Default::default(),
                    topology: mirage_core::common::MaybeRef::Ref("unused".to_string()),
                },
                containerize: None,
            },
            runtime_dir: std::env::temp_dir(),
            daemon: false,
        }
    }

    /// A tree with the intercept but not the runtime it needs cannot
    /// emulate anything, and must not produce an injection. Left to run,
    /// the workload goes to the host GPU untouched and exits 0 — a green
    /// result that never met the emulator.
    #[test]
    fn a_partial_install_is_refused_and_says_what_is_missing() {
        let tree = StagedTree::new("partial").with(LIB_NAME);

        let err = complete_tree(tree.path()).unwrap_err();
        assert!(err.contains(ROCR_LIB), "{err}");
        assert!(err.contains(COMGR_LIB), "{err}");
        assert!(err.contains(&tree.path().display().to_string()), "{err}");

        // And the injection refuses on exactly that verdict rather than
        // building one anyway.
        let injection = Hotswap.injection_def_for(
            &session_ctx(),
            complete_tree(tree.path()),
            Some("gfx942".to_string()),
        );
        assert_eq!(injection.unwrap_err().to_string(), err);
    }

    /// One missing library is still a partial install, and the message
    /// names the one that is missing rather than the pair.
    #[test]
    fn a_tree_missing_only_comgr_is_still_refused() {
        let tree = StagedTree::new("no-comgr").with(LIB_NAME).with(ROCR_LIB);

        let err = complete_tree(tree.path()).unwrap_err();
        assert!(err.contains(COMGR_LIB), "{err}");
        assert!(!err.contains(ROCR_LIB), "{err}");
    }

    /// The complete tree is what the two predicates agree on: it passes
    /// the install check and yields an injection preloading both the
    /// patched ROCR and the intercept out of that directory.
    #[test]
    fn a_complete_install_yields_an_injection() {
        let tree = StagedTree::new("complete")
            .with(LIB_NAME)
            .with(ROCR_LIB)
            .with(COMGR_LIB);

        let dir = complete_tree(tree.path()).expect("a complete tree is installed");
        let injection = Hotswap
            .injection_def_for(&session_ctx(), Ok(dir), Some("gfx950".to_string()))
            .expect("a complete install with a target GPU can be injected");

        let preload = injection
            .ld_preload
            .expect("HotSwap preloads two libraries");
        assert!(preload.contains(ROCR_LIB), "{preload}");
        assert!(preload.contains(LIB_NAME), "{preload}");
        assert_eq!(
            injection
                .env
                .get("HSA_HOTSWAP_ISA_OVERRIDE")
                .map(String::as_str),
            Some("gfx950")
        );
    }

    /// Without a GPU HotSwap can retarget onto, the injection must fail
    /// rather than name one that is not in the machine.
    #[test]
    fn no_compatible_gpu_is_refused_rather_than_fabricated() {
        let tree = StagedTree::new("no-gpu")
            .with(LIB_NAME)
            .with(ROCR_LIB)
            .with(COMGR_LIB);

        // Nothing present, and no override: there is no honest value for
        // HSA_HOTSWAP_ISA_OVERRIDE, so there must be no injection.
        assert_eq!(target_gfx_for(&[], None), None);
        let err = Hotswap
            .injection_def_for(&session_ctx(), complete_tree(tree.path()), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("gfx942") && err.contains("gfx950"), "{err}");
        assert!(err.contains("HSA_HOTSWAP_ISA_OVERRIDE"), "{err}");

        // A GPU that *is* present is named, and an explicit override
        // wins over detection — the escape hatch the error points at.
        assert_eq!(target_gfx_for(&[90402], None).as_deref(), Some("gfx942"));
        assert_eq!(
            target_gfx_for(&[], Some("gfx1201".to_string())).as_deref(),
            Some("gfx1201")
        );
        // An empty override reads as unset, not as an empty ISA name.
        assert_eq!(target_gfx_for(&[], Some(String::new())), None);
    }

    #[test]
    fn support_status_always_has_a_reason() {
        // Whatever this host looks like, the support check must produce
        // a non-empty, human-readable reason for the UX/CLI to show.
        let status = support_status();
        assert!(!status.reason.is_empty());
        // The required architectures should be named in the reason so
        // the user knows what HotSwap needs.
        if !status.supported {
            assert!(status.reason.contains("gfx942"));
            assert!(status.reason.contains("gfx950"));
        }
    }
}
