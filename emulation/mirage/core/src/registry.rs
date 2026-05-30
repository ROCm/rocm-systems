//! Emulator registry.
//!
//! A small in-process table of available emulator backends. Each
//! entry can:
//!
//! * describe itself ([`EmulatorSpec::description`]),
//! * declare whether it is installed on this machine
//!   ([`EmulatorSpec::installed`]) — used by the CLI wizards and the
//!   `default_emulator()` selector,
//! * produce a default topology to use when the user creates a
//!   profile without supplying one
//!   ([`EmulatorSpec::default_topology`]).
//!
//! New backends register themselves by adding a new
//! [`EmulatorSpec`] to [`builtins()`]. The list is the
//! authoritative source for both the CLI (`--emulator <name>`) and
//! the wizards (which only offer registered names).

use crate::common::{MaybeRef, SimpleMap};
use crate::emulator::{EmulatorDef, EmulatorDescription, ExecMode};
use crate::topology::{ComponentDef, TopologyDef};

/// A registered emulator backend.
///
/// Methods are function pointers so the registry stays a plain `&'static`
/// slice; no trait objects, no globals to initialise.
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
    /// Returns this emulator's default topology. Used when the user
    /// asks for a profile but doesn't supply a `--topology` (or via
    /// the wizard's "use default" answer).
    pub default_topology: fn(nodes: u32, gpus_per_node: u32) -> TopologyDef,
    /// Returns a long-form description (name + version + blurb).
    pub describe: fn() -> EmulatorDescription,
}

/// Built-in registry. Append-only by intent.
pub fn builtins() -> &'static [EmulatorSpec] {
    &[NOOP, ROCJITSU]
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

/// Build an [`EmulatorDef`] for the given registry entry.
pub fn make_def(spec: &EmulatorSpec, nodes: u32, gpus_per_node: u32) -> EmulatorDef {
    EmulatorDef {
        emulator: spec.name.to_string(),
        plugins: Default::default(),
        nodes,
        gpus_per_node,
        exec_mode: ExecMode::default(),
        options: SimpleMap::default(),
        topology: MaybeRef::Owned((spec.default_topology)(nodes, gpus_per_node)),
    }
}

// =============================================================================
// noop
// =============================================================================

pub const NOOP: EmulatorSpec = EmulatorSpec {
    name: "noop",
    description: "no-op emulator: runs commands directly with no GPU emulation",
    installed: noop_installed,
    default_topology: noop_topology,
    describe: noop_describe,
};

fn noop_installed() -> bool {
    true
}

fn noop_topology(_nodes: u32, _gpus_per_node: u32) -> TopologyDef {
    TopologyDef {
        root: ComponentDef {
            name: "noop".to_string(),
            r#type: "noop".to_string(),
            ..Default::default()
        },
        links: vec![],
    }
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
    default_topology: rocjitsu_topology,
    describe: rocjitsu_describe,
};

/// rocjitsu is "installed" if the dynamic library `librocjitsu.so` can
/// be found via the loader's standard search path *or* if the user
/// has set `ROCJITSU_LIB_DIR` / `ROCJITSU_ROOT`. We intentionally do
/// not link to it from here — installation is a runtime question.
fn rocjitsu_installed() -> bool {
    if std::env::var_os("ROCJITSU_LIB_DIR").is_some()
        || std::env::var_os("ROCJITSU_ROOT").is_some()
    {
        return true;
    }
    let candidates = [
        "/usr/local/lib/librocjitsu.so",
        "/usr/lib/librocjitsu.so",
        "/usr/lib/x86_64-linux-gnu/librocjitsu.so",
        "/opt/rocm/lib/librocjitsu.so",
    ];
    candidates.iter().any(|p| std::path::Path::new(p).exists())
}

/// Default CDNA-style topology for a `nodes`x`gpus_per_node` system:
/// one `node` per node, one `gpu` per GPU under each node. This is
/// just enough for orchestration code; the real topology comes from
/// rocjitsu's own JSON when the emulator boots.
fn rocjitsu_topology(nodes: u32, gpus_per_node: u32) -> TopologyDef {
    let mut node_children = Vec::with_capacity(nodes as usize);
    for n in 0..nodes {
        let mut gpus = Vec::with_capacity(gpus_per_node as usize);
        for g in 0..gpus_per_node {
            gpus.push(ComponentDef {
                name: format!("gpu{g}"),
                r#type: "gpu".to_string(),
                ..Default::default()
            });
        }
        node_children.push(ComponentDef {
            name: format!("node{n}"),
            r#type: "node".to_string(),
            children: gpus,
            ..Default::default()
        });
    }
    TopologyDef {
        root: ComponentDef {
            name: "cluster".to_string(),
            r#type: "cluster".to_string(),
            children: node_children,
            ..Default::default()
        },
        links: vec![],
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
    fn default_falls_back_to_noop() {
        // Unset every variable that could make rocjitsu look installed.
        // The known library paths are unlikely to exist in CI; the
        // env vars are scoped to this test.
        // SAFETY: single-threaded by the test runner default; we
        // restore after.
        let prev_lib = std::env::var_os("ROCJITSU_LIB_DIR");
        let prev_root = std::env::var_os("ROCJITSU_ROOT");
        unsafe {
            std::env::remove_var("ROCJITSU_LIB_DIR");
            std::env::remove_var("ROCJITSU_ROOT");
        }
        // We can't control which side wins if rocjitsu actually is
        // present on this machine, so only assert when it isn't.
        if !(ROCJITSU.installed)() {
            assert_eq!(default_emulator().name, "noop");
        }
        unsafe {
            if let Some(v) = prev_lib {
                std::env::set_var("ROCJITSU_LIB_DIR", v);
            }
            if let Some(v) = prev_root {
                std::env::set_var("ROCJITSU_ROOT", v);
            }
        }
    }

    #[test]
    fn default_topologies_match_node_count() {
        let t = rocjitsu_topology(2, 4);
        assert_eq!(t.root.children.len(), 2);
        for n in &t.root.children {
            assert_eq!(n.children.len(), 4);
        }
    }
}
