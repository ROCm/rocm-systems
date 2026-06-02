//! `mirage_hotswap` — HotSwap integration for the mirage binary.
//!
//! HotSwap is a load-time ISA rewriter that runs a workload built for
//! one AMD GPU architecture on a different physical GPU (e.g.
//! `gfx1250` on `gfx942`/`gfx950`) by rewriting device code as it is
//! loaded. It plugs into the ROCm runtime as an HSA tools library: the
//! runtime loads it when `HSA_TOOLS_LIB` points at
//! `libhsa-hotswap.so`.
//!
//! mirage does **not** build or bundle HotSwap. This crate only
//! *discovers* an existing `libhsa-hotswap.so` install (using the
//! shared [`mirage_core::discovery`] search policy) and exposes the
//! `HSA_TOOLS_LIB` value plus install guidance. See
//! [`../README.md`](../README.md) for the user-facing docs.

use std::path::PathBuf;

use mirage_core::discovery::{self, LibSearch};
use mirage_core::emulator::{EmulatorDescription, SupportStatus};

/// The HSA tools library file name HotSwap ships as.
pub const LIB_NAME: &str = "libhsa-hotswap.so";

/// Subdirectory under the mirage cache / `./emulator/` where a
/// HotSwap library may be dropped.
pub const ASSET_SUBDIR: &str = "hotswap";

/// Human-facing name used in guidance messages.
pub const DISPLAY_NAME: &str = "HotSwap";

/// The physical GPU architectures HotSwap can retarget code *onto*,
/// as KFD `gfx_target_version` values paired with their conventional
/// `gfx` name. HotSwap rewrites device code at load time so a workload
/// built for one architecture runs on one of these cards (e.g.
/// `gfx1250` code on a `gfx942`/`gfx950` GPU). Without one of these
/// GPUs physically present there is nothing for HotSwap to run on.
pub const SUPPORTED_GPUS: &[(u32, &str)] = &[(90402, "gfx942"), (90500, "gfx950")];

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

/// Search policy mirage uses to locate `libhsa-hotswap.so`.
pub fn lib_search() -> LibSearch<'static> {
    LibSearch {
        // An explicit path to the `.so`. `HSA_TOOLS_LIB` is the ROCm
        // runtime's own variable, so honouring it here keeps mirage
        // consistent with a manually-exported environment.
        file_env: &["HOTSWAP_LIB", "HSA_TOOLS_LIB"],
        dir_env: &["HOTSWAP_LIB_DIR"],
        lib_name: LIB_NAME,
        binary_relative_dirs: &[],
    }
}

/// Locate `libhsa-hotswap.so`, returning its path if HotSwap is
/// installed anywhere mirage knows to look.
pub fn lib_path() -> Option<PathBuf> {
    discovery::find_emulator_lib(&lib_search())
}

/// The value mirage should set for `HSA_TOOLS_LIB` when running a
/// workload under HotSwap, if the library can be found.
pub fn hsa_tools_lib() -> Option<String> {
    lib_path().map(|p| p.display().to_string())
}

/// Returns `true` if a usable `libhsa-hotswap.so` is present on this
/// machine.
pub fn is_installed() -> bool {
    discovery::is_lib_installed(&lib_search())
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
    fn search_targets_the_hsa_tools_lib() {
        let s = lib_search();
        assert_eq!(s.lib_name, "libhsa-hotswap.so");
        assert!(s.file_env.contains(&"HSA_TOOLS_LIB"));
    }

    #[test]
    fn guidance_mentions_the_library() {
        assert!(install_guidance().contains("libhsa-hotswap.so"));
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
