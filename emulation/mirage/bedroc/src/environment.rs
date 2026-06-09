//! The execution environment: the implicit constraints of where the
//! user is running.
//!
//! A request is only meaningful relative to what is actually available:
//! which physical GPUs are present and which tools are installed. The
//! [`Environment`] captures those facts and gates which manifests are
//! usable. Tools whose requirements the environment cannot meet are
//! excluded from planning but reported, so a [`crate::proof::Proof`] can
//! explain precisely why a goal is unsupported *here*.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// What the host offers: installed tools and present GPUs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Environment {
    /// Names of installed tools (e.g. `rocjitsu`, `hotswap`). A manifest
    /// whose `needs_tools` are not all present is unavailable.
    pub installed_tools: BTreeSet<String>,
    /// Canonical `gfxNNN` architectures of physical GPUs on the host.
    pub gpus: BTreeSet<String>,
}

impl Environment {
    /// An empty environment: nothing installed, no GPUs. Useful as a
    /// baseline and for tests.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Probe the local host for present GPUs via the kernel KFD
    /// topology. Installed tools are *not* probed here — that requires
    /// the emulator registry, which lives above this crate; callers
    /// (the CLI/daemon) inject installed tool names via
    /// [`Environment::with_tool`].
    pub fn probe() -> Self {
        let gpus = mirage_core::hardware::gpu_gfx_versions()
            .into_iter()
            .map(mirage_core::hardware::gfx_name)
            .collect();
        Self {
            installed_tools: BTreeSet::new(),
            gpus,
        }
    }

    /// Builder: record an installed tool.
    pub fn with_tool(mut self, name: impl Into<String>) -> Self {
        self.installed_tools.insert(name.into());
        self
    }

    /// Builder: record a present GPU architecture.
    pub fn with_gpu(mut self, gfx: impl Into<String>) -> Self {
        self.gpus.insert(gfx.into());
        self
    }

    /// Whether `tool` is installed.
    pub fn has_tool(&self, tool: &str) -> bool {
        self.installed_tools.contains(tool)
    }

    /// Whether any physical GPU is present.
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    /// Environment facts seeded into the initial planning state: one
    /// `installed:<tool>` per installed tool and one `host_gpu:<gfx>`
    /// per present GPU. These let manifests optionally key off the
    /// host's own hardware.
    pub fn facts(&self) -> Vec<String> {
        let mut facts = Vec::new();
        for t in &self.installed_tools {
            facts.push(format!("installed:{t}"));
        }
        for g in &self.gpus {
            facts.push(format!("host_gpu:{g}"));
        }
        facts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_and_queries() {
        let env = Environment::empty()
            .with_tool("rocjitsu")
            .with_gpu("gfx950");
        assert!(env.has_tool("rocjitsu"));
        assert!(!env.has_tool("hotswap"));
        assert!(env.has_gpu());
        let facts = env.facts();
        assert!(facts.contains(&"installed:rocjitsu".to_string()));
        assert!(facts.contains(&"host_gpu:gfx950".to_string()));
    }

    #[test]
    fn empty_environment_has_no_gpu() {
        assert!(!Environment::empty().has_gpu());
    }
}
