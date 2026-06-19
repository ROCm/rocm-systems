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
use mirage_core::emulator::{Emulator, EmulatorDef, EmulatorDescription, SupportStatus};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::InjectionDef;
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::{FileMount, ProfileDef};
use mirage_core::session::{SessionHealth, SessionId};

/// The HIP intercept library HotSwap ships. It is the artifact mirage
/// anchors discovery on (its directory is the HotSwap lib dir) and the
/// library the env contract `LD_PRELOAD`s.
pub const LIB_NAME: &str = "libhotswap_intercept.so";

/// The HotSwap-patched ROCR runtime, expected alongside [`LIB_NAME`].
pub const ROCR_LIB: &str = "libhsa-runtime64.so";

/// The COMGR transpiler, expected alongside [`LIB_NAME`].
pub const COMGR_LIB: &str = "libamd_comgr.so";

/// Subdirectory under the mirage cache / `./emulator/` where a
/// HotSwap install may be dropped.
pub const ASSET_SUBDIR: &str = "hotswap";

/// In-container mount point for the HotSwap runtime tree (lib,
/// `llvm-tools/`, `runtime/hotswap_py/`). All mirage system mounts live
/// under `/mnt/mirage`; the HotSwap install root is bind-mounted here so
/// the injected `LD_PRELOAD`/`HOTSWAP_HOME`/`PYTHONPATH` paths resolve
/// inside each node's container.
pub const CONTAINER_HOTSWAP_DIR: &str = "/mnt/mirage/emulator/hotswap";

/// In-container mount point for HotSwap's read-write cache tree
/// (translation + per-framework caches), bind-mounted from the host
/// cache dir so artifacts persist across runs.
pub const CONTAINER_HOTSWAP_CACHE: &str = "/mnt/mirage/cache/hotswap";

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
/// [`adapter_backends_for_policy`].
pub const ADAPTER_POLICIES: &[&str] = &["none", "env", "native_build", "triton", "compile", "full"];

/// The physical GPU architectures HotSwap can retarget code *onto*,
/// as KFD `gfx_target_version` values paired with their conventional
/// `gfx` name. HotSwap rewrites device code at load time so a workload
/// built for one architecture runs on one of these cards (e.g.
/// `gfx1250` code on a `gfx942`/`gfx950` GPU). Without one of these
/// GPUs physically present there is nothing for HotSwap to run on.
pub const SUPPORTED_GPUS: &[(u32, &str)] = &[(90402, "gfx942"), (90500, "gfx950")];

/// HotSwap [`Emulator`] implementation. HotSwap is wired into a
/// workload via the env contract: the patched ROCR + COMGR shadow the
/// system copies (`LD_LIBRARY_PATH`), the HIP intercept is `LD_PRELOAD`ed,
/// and the `HSA_HOTSWAP_*` variables select the source target and policy.
pub struct Hotswap {
    profile: ProfileDef,
}

impl Emulator for Hotswap {
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

    fn validate_profile(_def: &ProfileDef) -> std::result::Result<(), String> {
        // HotSwap is not bundled or built by mirage; it must be
        // installed separately. Surface actionable guidance now, at
        // profile-creation time, rather than only when a session is
        // later started.
        if is_installed() {
            Ok(())
        } else {
            Err(install_guidance())
        }
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
        let support = support_status();
        let installed = is_installed();
        let healthy = installed && support.supported;
        SessionHealth {
            healthy,
            state: Some(if healthy { "ready" } else { "error" }.to_string()),
            terminal: false,
            message: if healthy {
                None
            } else if !installed {
                Some(format!("{LIB_NAME} not found"))
            } else {
                Some(support.reason)
            },
            ..Default::default()
        }
    }

    fn injection_def(&self, _session: &SessionId) -> Result<InjectionDef> {
        // Refuse to run unemulated: without the HotSwap tree the
        // workload would silently run on real hardware, so fail loudly
        // with guidance instead.
        let dir = lib_dir().ok_or_else(|| {
            MirageError::Other(format!(
                "hotswap: {LIB_NAME} not found; workload cannot be emulated.\n{}",
                install_guidance()
            ))
        })?;

        let containerized = self.profile.containerize.is_some();
        let (ld_preload, env, mounts) = build_hotswap_env(&dir, containerized);

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

/// The physical GPU HotSwap retargets code *onto*: the first present
/// [`SUPPORTED_GPUS`] entry, else the first supported arch as a
/// fallback. Drives `HSA_HOTSWAP_ISA_OVERRIDE`.
fn physical_target_gfx() -> String {
    let present = mirage_core::hardware::gpu_gfx_versions();
    SUPPORTED_GPUS
        .iter()
        .find(|(version, _)| present.contains(version))
        .or_else(|| SUPPORTED_GPUS.first())
        .map(|(_, name)| (*name).to_string())
        .unwrap_or_default()
}

/// Scratch root for HotSwap's translation/framework caches, under the
/// mirage cache dir. Mirrors env_contract.py's `branch_root`.
fn hotswap_branch_root() -> PathBuf {
    mirage_core::paths::mirage_cache_dir().join("hotswap")
}

/// Cache subdirectories created under [`hotswap_branch_root`].
const HOTSWAP_CACHE_SUBDIRS: &[&str] = &[
    "hotswap_translation_cache",
    "triton_cache",
    "torchinductor_cache",
    "torch_extensions",
    "pytorch_kernel_cache",
    "miopen_user_db",
    "miopen_cache",
];

/// Build the HotSwap env contract for a workload, mirroring
/// env_contract.py's `build_hotswap_env`: the `HSA_HOTSWAP_*` variables,
/// the per-framework cache redirects, the python `sitecustomize` on
/// `PYTHONPATH`, and the source-arch overrides selected by the adapter
/// policy. Returns the `LD_PRELOAD` value (patched ROCR + intercept)
/// separately so the host can merge it with any user-supplied preload.
fn build_hotswap_env(
    dir: &Path,
    containerized: bool,
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
    let target_gfx = physical_target_gfx();

    let root = hotswap_branch_root();
    let _ = ensure_cache_dirs(&root);
    let host_cache_root = std::fs::canonicalize(&root).unwrap_or(root);

    // Establish host→workload path mappings. For a containerised session
    // every host tree HotSwap relies on is bind-mounted under
    // `/mnt/mirage` and the workload must reference the in-container path;
    // otherwise it sees the host paths directly. The install root is
    // mounted read-only (immutable runtime libs/tools), the cache root
    // read-write so framework caches persist.
    let mut mounts: Vec<FileMount> = Vec::new();
    let mut mappings: Vec<(PathBuf, PathBuf)> = Vec::new();
    if containerized {
        if let Some(home) = host_dir.parent() {
            mounts.push(FileMount {
                host_path: home.display().to_string(),
                container_path: CONTAINER_HOTSWAP_DIR.to_string(),
                read_only: true,
            });
            mappings.push((home.to_path_buf(), PathBuf::from(CONTAINER_HOTSWAP_DIR)));
        }
        mounts.push(FileMount {
            host_path: host_cache_root.display().to_string(),
            container_path: CONTAINER_HOTSWAP_CACHE.to_string(),
            read_only: false,
        });
        mappings.push((
            host_cache_root.clone(),
            PathBuf::from(CONTAINER_HOTSWAP_CACHE),
        ));
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

    let cache = |name: &str| remap(&host_cache_root.join(name)).display().to_string();

    let mut env: BTreeMap<String, String> = BTreeMap::new();
    env.insert(
        "HSA_HOTSWAP_CACHE_DIR".into(),
        cache("hotswap_translation_cache"),
    );
    env.insert("HSA_HOTSWAP_CACHE_DEBUG".into(), "1".into());
    env.insert("HSA_HOTSWAP_ISA_OVERRIDE".into(), target_gfx);
    env.insert("HSA_HOTSWAP_IR_RAISER".into(), "1".into());
    env.insert("HSA_HOTSWAP_STRICT".into(), "1".into());
    env.insert("HSA_HOTSWAP_SOURCE_TARGET".into(), source_target.clone());
    env.insert("HSA_HOTSWAP_BACKEND_ADAPTER_POLICY".into(), policy.clone());
    env.insert("HSA_HOTSWAP_BACKEND_ADAPTERS".into(), backends.join(","));

    // Force-compile framework caches so a swapped target never reuses a
    // host-targeted artifact.
    env.insert("TRITON_ALWAYS_COMPILE".into(), "1".into());
    env.insert("TRITON_CACHE_DIR".into(), cache("triton_cache"));
    env.insert(
        "TORCHINDUCTOR_CACHE_DIR".into(),
        cache("torchinductor_cache"),
    );
    env.insert("TORCH_EXTENSIONS_DIR".into(), cache("torch_extensions"));
    env.insert(
        "PYTORCH_KERNEL_CACHE_PATH".into(),
        cache("pytorch_kernel_cache"),
    );
    env.insert("MIOPEN_USER_DB_PATH".into(), cache("miopen_user_db"));
    env.insert("MIOPEN_CUSTOM_CACHE_DIR".into(), cache("miopen_cache"));

    // The patched ROCR + COMGR shadow the system copies via the loader
    // search path. The host launcher inherits a minimal env (no
    // LD_LIBRARY_PATH), so set it explicitly to the HotSwap lib dir; an
    // exec-level override still wins (applied afterwards).
    env.insert("LD_LIBRARY_PATH".into(), dir.display().to_string());
    // Expose the install root so the workload/runtime can resolve the
    // sibling `llvm-tools/` and `runtime/hotswap_py/` trees.
    if let Some(home) = dir.parent() {
        env.insert("HOTSWAP_HOME".into(), home.display().to_string());
    }
    if let Some(py) = py_dir() {
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

/// Best-effort creation of the HotSwap cache subdirectories.
fn ensure_cache_dirs(root: &Path) -> std::io::Result<()> {
    for name in HOTSWAP_CACHE_SUBDIRS {
        std::fs::create_dir_all(root.join(name))?;
    }
    Ok(())
}

/// Describe the hotswap emulator backend for the registry. Owned by
/// this crate (rather than `mirage_core`) so that all hotswap-specific
/// policy lives alongside the hotswap discovery integration. Reports
/// whether hotswap is installed, the resolved path to its runtime
/// library when available, and whether this host has a physical GPU
/// HotSwap can actually run on.
pub fn describe() -> EmulatorDescription {
    EmulatorDescription {
        name: "hotswap".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "load-time ISA rewriter: run a GPU's code on a different GPU (e.g. gfx1250 on gfx942/gfx950)".to_string(),
        installed: is_installed(),
        path: lib_path(),
        support: support_status(),
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

/// Returns `true` if a usable HotSwap install is present on this
/// machine (the intercept, patched ROCR, and COMGR all co-located).
pub fn is_installed() -> bool {
    match lib_dir() {
        Some(dir) => dir.join(ROCR_LIB).is_file() && dir.join(COMGR_LIB).is_file(),
        None => false,
    }
}

/// Multi-line, user-facing guidance describing where mirage looked for
/// HotSwap and how to make it discoverable.
pub fn install_guidance() -> String {
    discovery::install_guidance(DISPLAY_NAME, &lib_search())
}

#[cfg(test)]
mod tests {
    use super::*;

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
