use serde::{Deserialize, Serialize};

fn one() -> u32 {
    1
}

/// Key-value pair for component configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ConfigEntry {
    pub key: String,

    /// All values as strings, parsed by the factory.
    pub value: String,
}

/// Port definition for dynamic ports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PortDef {
    pub name: String,

    /// "in" or "out".
    pub direction: String,

    /// "untyped", "memory_req", "memory_resp", "dispatch", etc.
    pub protocol: String,
}

/// Component definition (recursive for hierarchy).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ComponentDef {
    /// Name or range pattern like "xcd[0:7]".
    pub name: String,

    /// Registry type: "compute_unit", "l2_cache", etc.
    #[serde(rename = "type")]
    pub r#type: String,

    /// Component-specific parameters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config: Vec<ConfigEntry>,

    /// Child components (recursive).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ComponentDef>,

    /// Optional dynamic ports.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ports: Vec<PortDef>,
}

/// Range variable for link pattern expansion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ForRange {
    /// Variable name: "i", "j", "k".
    pub var_name: String,

    /// Range start (inclusive).
    pub start: u32,

    /// Range end (exclusive).
    pub end: u32,
}

/// Link definition (direct or pattern-based).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkDef {
    /// Direct source: "soc.xcd0.l2.hbm_out".
    #[serde(default)]
    pub src: String,

    /// Direct destination.
    #[serde(default)]
    pub dst: String,

    /// Pattern: "soc.xcd[i].l2 -> soc.iod[i/4].msc".
    #[serde(default)]
    pub pattern: String,

    /// Loop variables.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub for_ranges: Vec<ForRange>,

    /// Filter: "i != j".
    #[serde(default)]
    pub where_expr: String,

    #[serde(default = "one")]
    pub latency: u32,

    #[serde(default = "one")]
    pub weight: u32,
}

impl Default for LinkDef {
    fn default() -> Self {
        Self {
            src: String::new(),
            dst: String::new(),
            pattern: String::new(),
            for_ranges: Vec::new(),
            where_expr: String::new(),
            latency: 1,
            weight: 1,
        }
    }
}

/// Top-level topology definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TopologyDef {
    pub root: ComponentDef,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<LinkDef>,
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

    /// Write all builtin topologies to disk.
    ///
    /// If `force` is true, existing files are overwritten. Otherwise
    /// only missing topologies are written.
    ///
    /// Returns the list of `(name, written)` entries: `written` is
    /// true if the file was created or overwritten on this call.
    pub fn ensure_builtins(force: bool) -> Result<Vec<(String, bool)>> {
        let mut report = Vec::new();
        for (name, build) in crate::registry::builtin_topologies() {
            let p = crate::paths::topology_path(name);
            let exists = p.exists();
            if exists && !force {
                report.push((name.to_string(), false));
                continue;
            }
            crate::state::write_json(&p, &build())?;
            report.push((name.to_string(), true));
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::store;

    #[test]
    fn ensure_builtins_writes_then_skips() {
        let _g = crate::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(tmp.path());

        let first = store::ensure_builtins(false).unwrap();
        assert!(!first.is_empty(), "expected at least one builtin topology");
        assert!(first.iter().all(|(_, w)| *w), "first run should write every builtin");

        let names: Vec<_> = first.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(store::list().unwrap(), {
            let mut s = names.clone();
            s.sort();
            s
        });

        let second = store::ensure_builtins(false).unwrap();
        assert!(second.iter().all(|(_, w)| !*w), "second run should not rewrite existing builtins");

        let forced = store::ensure_builtins(true).unwrap();
        assert!(forced.iter().all(|(_, w)| *w), "force should rewrite every builtin");

        for name in &names {
            assert!(store::get(name).is_ok(), "{name} should be readable");
        }
    }
}
