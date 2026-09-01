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
///
/// Unknown fields are rejected; see [`crate::profile`] for why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
///
/// [`crate::store::topology_get`] is where a `MaybeRef::Ref` on a profile is
/// followed, so it is
/// also where that reference is checked: a name that arrived inside a
/// document, rather than from a command line, is interpolated into
/// `<config>/topology/<name>.json` by exactly the same rule and escapes
/// the config directory just as easily. It is also where a reference that
/// resolves to nothing is reported as the dangling reference it is.
pub mod store {
    use super::TopologyDef;
    use crate::error::{MirageError, Result};
    use crate::store::{DocKind, Referrer, dangling_ref, validate_name};
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

    /// Read a topology by name, for a caller that cannot say which
    /// profile sent it.
    ///
    /// Prefer [`get_referred_by`] wherever the referring profile is in
    /// scope: a user with a dozen profiles cannot act on "a profile
    /// refers to a topology that is not there" without grepping for the
    /// one that does.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not a single path component, if
    /// there is no such topology — reported as the dangling reference it
    /// is, since a profile is what brought the name here — or if the
    /// document is malformed.
    pub fn get(name: &str) -> Result<TopologyDef> {
        get_referred_by(Referrer::anonymous(DocKind::Profile), name)
    }

    /// Read a topology by name on behalf of the document that named it.
    ///
    /// The referrer is carried in rather than assumed because it is the
    /// half of a dangling-reference error the reader recognises — usually
    /// the very word they typed on the command line — and it is known one
    /// frame up from here and nowhere else.
    ///
    /// # Errors
    ///
    /// As [`get`], with the referring document named in a dangling
    /// reference.
    pub fn get_referred_by(referrer: Referrer<'_>, name: &str) -> Result<TopologyDef> {
        validate_name(DocKind::Topology, name)?;
        let p = crate::paths::topology_path(name);
        if !p.exists() {
            return Err(dangling_ref(referrer, DocKind::Topology, name));
        }
        crate::state::read_json(&p)
    }

    /// Write a topology to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not a single path component or the
    /// document cannot be written.
    pub fn put(name: &str, topology: &TopologyDef) -> Result<PathBuf> {
        validate_name(DocKind::Topology, name)?;
        let p = crate::paths::topology_path(name);
        crate::state::write_json(&p, topology)?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn totals() {
        let t = TopologyDef {
            num_nodes: 4,
            gpus_per_node: 8,
            agent: MaybeRef::Ref("MI350X".to_string()),
        };
        assert_eq!(t.total_nodes(), 4);
        assert_eq!(t.total_gpus(), 32);
    }
}
