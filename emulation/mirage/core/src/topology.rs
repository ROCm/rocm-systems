//! System topology definitions.
//!
//! A [`TopologyDef`] describes the *system layout* — how many nodes
//! and how many GPUs per node — together with the
//! [`crate::agent::AgentDef`] used for each GPU slot.
//!
//! The agent is referenced via [`MaybeRef`]: callers can either
//! embed a full agent definition inline or refer to a named entry
//! from `<MIRAGE_CONFIG>/agent/`.
//!
//! Topologies themselves live at `<MIRAGE_CONFIG>/topology/<name>.json`.

use serde::{Deserialize, Serialize};

use crate::agent::AgentDef;
use crate::common::MaybeRef;

fn one() -> u32 {
    1
}

/// System-level topology: node/GPU counts plus the agent
/// definition each GPU instantiates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyDef {
    /// Number of nodes. Defaults to 1.
    #[serde(default = "one")]
    pub num_nodes: u32,

    /// Number of GPUs per node. Defaults to 1.
    #[serde(default = "one")]
    pub gpus_per_node: u32,

    /// Hardware agent for each GPU slot. Either an inline
    /// [`AgentDef`] or a name resolvable under `<MIRAGE_CONFIG>/agent/`.
    pub agent: MaybeRef<AgentDef>,
}

impl TopologyDef {
    /// Total number of nodes.
    pub fn total_nodes(&self) -> u32 {
        self.num_nodes
    }

    /// Total number of GPUs across the whole system.
    pub fn total_gpus(&self) -> u32 {
        self.total_nodes().saturating_mul(self.gpus_per_node)
    }
}

/// On-disk topology store backed by `<MIRAGE_CONFIG>/topology/`.
pub mod store {
    use super::TopologyDef;
    use crate::error::{MirageError, Result};
    use std::path::PathBuf;

    /// List the names of all topology files on disk.
    pub fn list() -> Result<Vec<String>> {
        let root = crate::paths::topology_root();
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&root).map_err(|e| MirageError::Io {
            path: root.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| MirageError::Io {
                path: root.clone(),
                source: e,
            })?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(stem) = name.strip_suffix(".json") {
                out.push(stem.to_string());
            }
        }
        out.sort();
        Ok(out)
    }

    /// Read a topology by name.
    pub fn get(name: &str) -> Result<TopologyDef> {
        let p = crate::paths::topology_path(name);
        crate::state::read_json(&p)
    }

    /// Write a topology to disk.
    pub fn put(name: &str, topology: &TopologyDef) -> Result<PathBuf> {
        let p = crate::paths::topology_path(name);
        crate::state::write_json(&p, topology)?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totals() {
        let t = TopologyDef {
            num_nodes: 4,
            gpus_per_node: 8,
            agent: MaybeRef::Ref("noop".to_string()),
        };
        assert_eq!(t.total_nodes(), 4);
        assert_eq!(t.total_gpus(), 32);
    }
}
