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
use mirage_core::emulator::EmulatorDescription;
use mirage_core::registry::EmulatorSpec;

/// The HSA tools library file name HotSwap ships as.
pub const LIB_NAME: &str = "libhsa-hotswap.so";

/// Subdirectory under the mirage cache / `./emulator/` where a
/// HotSwap library may be dropped.
pub const ASSET_SUBDIR: &str = "hotswap";

/// Human-facing name used in guidance messages.
pub const DISPLAY_NAME: &str = "HotSwap";

/// Registry entry describing the hotswap emulator backend. Owned by
/// this crate (rather than `mirage_core`) so that all hotswap-specific
/// policy lives alongside the hotswap discovery integration.
pub const SPEC: EmulatorSpec = EmulatorSpec {
    name: "hotswap",
    description: "load-time ISA rewriter: run a GPU's code on a different GPU (e.g. gfx1250 on gfx942/gfx950)",
    installed: is_installed,
    describe: spec_describe,
};

fn spec_describe() -> EmulatorDescription {
    EmulatorDescription {
        name: "hotswap".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "Load-time ISA rewriter loaded via HSA_TOOLS_LIB; \
                      runs one GPU architecture's code on another real GPU."
            .to_string(),
    }
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
}
