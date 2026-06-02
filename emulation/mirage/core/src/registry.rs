//! Emulator registry.
//!
//! [`builtins`] is the append-only list of registered emulator
//! backends. Each [`EmulatorSpec`] knows how to describe itself and
//! report whether its runtime is installed.
//!
//! Named hardware agents ([`crate::agent::AgentDef`]) and system
//! topologies ([`crate::topology::TopologyDef`]) live in the
//! `mirage_builtin` crate; the on-disk store policy for them lives in
//! [`crate::agent::store`] / [`crate::topology::store`].

use crate::common::{MaybeRef, SimpleMap};
use crate::emulator::{EmulatorDef, EmulatorDescription, ExecMode};
use crate::topology::TopologyDef;

/// A registered emulator backend.
#[derive(Debug, Clone, Copy)]
pub struct EmulatorSpec {
    /// Canonical name used on disk and on the CLI.
    pub name: &'static str,
    /// Short human description.
    pub description: &'static str,
    /// Returns `true` if this emulator's runtime is present on the
    /// current machine. Cheap (no network, no spawn): used by the
    /// wizard's default picker.
    pub installed: fn() -> bool,
    /// Returns a long-form description (name + version + blurb).
    pub describe: fn() -> EmulatorDescription,
}

/// Built-in emulator registry. Append-only by intent.
pub fn builtins() -> &'static [EmulatorSpec] {
    &[NOOP, ROCJITSU, HOTSWAP]
}

/// Lookup an emulator by its canonical name.
pub fn find(name: &str) -> Option<&'static EmulatorSpec> {
    builtins().iter().find(|e| e.name == name)
}

/// The default emulator for new profiles when the user doesn't pick
/// one explicitly. Picks the first installed entry in registration
/// order, falling back to `noop`.
pub fn default_emulator() -> &'static EmulatorSpec {
    builtins()
        .iter()
        .find(|e| !std::ptr::eq(*e, &NOOP) && (e.installed)())
        .unwrap_or(&NOOP)
}

/// Build an [`EmulatorDef`] for the given registry entry, using the
/// supplied system topology.
pub fn make_def(spec: &EmulatorSpec, topology: TopologyDef) -> EmulatorDef {
    EmulatorDef {
        emulator: spec.name.to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::default(),
        options: SimpleMap::default(),
        topology: MaybeRef::Owned(topology),
    }
}

// =============================================================================
// noop
// =============================================================================

pub const NOOP: EmulatorSpec = EmulatorSpec {
    name: "noop",
    description: "no-op emulator: runs commands directly with no GPU emulation",
    installed: noop_installed,
    describe: noop_describe,
};

fn noop_installed() -> bool {
    true
}

fn noop_describe() -> EmulatorDescription {
    EmulatorDescription {
        name: "noop".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "Pass-through emulator. Useful for orchestration tests.".to_string(),
    }
}

// =============================================================================
// rocjitsu
// =============================================================================

pub const ROCJITSU: EmulatorSpec = EmulatorSpec {
    name: "rocjitsu",
    description: "ROCm just-in-time GPU emulator (cycle-accurate or functional)",
    installed: rocjitsu_installed,
    describe: rocjitsu_describe,
};

/// rocjitsu is "installed" if the dynamic library `librocjitsu.so`
/// can be found in the mirage emulator cache (extracted by
/// `mirage_rocjitsu::ensure_assets`), via the loader's standard
/// search path, or if the user has set `ROCJITSU_LIB_DIR` /
/// `ROCJITSU_ROOT`.
fn rocjitsu_installed() -> bool {
    crate::discovery::is_lib_installed(&rocjitsu_lib_search())
}

/// Shared discovery policy for `librocjitsu.so`.
fn rocjitsu_lib_search() -> crate::discovery::LibSearch<'static> {
    crate::discovery::LibSearch {
        file_env: &["ROCJITSU_LIB"],
        dir_env: &["ROCJITSU_LIB_DIR", "ROCJITSU_ROOT"],
        lib_name: "librocjitsu.so",
    }
}

fn rocjitsu_describe() -> EmulatorDescription {
    EmulatorDescription {
        name: "rocjitsu".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "ROCm GPU simulator (decodes AMDGPU/RISC-V ISA, event-driven PDES core)."
            .to_string(),
    }
}

// =============================================================================
// hotswap
// =============================================================================

pub const HOTSWAP: EmulatorSpec = EmulatorSpec {
    name: "hotswap",
    description: "load-time ISA rewriter: run a GPU's code on a different GPU (e.g. gfx1250 on gfx942/gfx950)",
    installed: hotswap_installed,
    describe: hotswap_describe,
};

/// Shared discovery policy for `libhsa-hotswap.so`. Mirrors
/// `mirage_hotswap::lib_search` so the registry's "installed" check
/// and the runtime injection agree on where to look.
fn hotswap_lib_search() -> crate::discovery::LibSearch<'static> {
    crate::discovery::LibSearch {
        file_env: &["HOTSWAP_LIB", "HSA_TOOLS_LIB"],
        dir_env: &["HOTSWAP_LIB_DIR"],
        lib_name: "libhsa-hotswap.so",
    }
}

/// hotswap is "installed" if `libhsa-hotswap.so` can be located in any
/// of the standard discovery locations (see `crate::discovery`).
fn hotswap_installed() -> bool {
    crate::discovery::is_lib_installed(&hotswap_lib_search())
}

fn hotswap_describe() -> EmulatorDescription {
    EmulatorDescription {
        name: "hotswap".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "Load-time ISA rewriter loaded via HSA_TOOLS_LIB; \
                      runs one GPU architecture's code on another real GPU."
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lookup() {
        assert_eq!(find("noop").map(|e| e.name), Some("noop"));
        assert_eq!(find("rocjitsu").map(|e| e.name), Some("rocjitsu"));
        assert!(find("bogus").is_none());
    }

    #[test]
    fn noop_is_always_installed() {
        assert!((NOOP.installed)());
    }

    #[test]
    fn default_matches_installation_state() {
        if (ROCJITSU.installed)() {
            assert_eq!(default_emulator().name, "rocjitsu");
        } else {
            assert_eq!(default_emulator().name, "noop");
        }
    }
}
