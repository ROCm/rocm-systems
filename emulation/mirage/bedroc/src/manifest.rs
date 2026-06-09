//! Tool manifests: the data-defined catalogue of correctness tools.
//!
//! A [`ToolManifest`] is a pure-data description of one correctness
//! tool: what it needs, what it establishes, what it costs, and what it
//! requires from the environment. Manifests are JSON documents — tools
//! are **never hardcoded** — so the catalogue is extended by dropping a
//! new `.json` file in a tools directory (see `docs/bedroc.md`).
//!
//! Manifests are *templates*: a fact like `compiled:${target}` is
//! expanded once per requested target by the [`crate::engine::Engine`]
//! into concrete [`mirage_solver::Step`]s.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The manifest schema version this crate understands. Manifests must
/// declare a matching `schema_version` so the format can evolve.
pub const SCHEMA_VERSION: u32 = 1;

/// An optional concrete command a tool can run. Bedroc's planner and
/// the default executor are command-agnostic (they simulate execution
/// for hermetic, fast operation); this field lets a deployment wire a
/// real invocation without changing any code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSpec {
    /// Program to execute.
    pub program: String,
    /// Arguments; `${target}` and `${source}` are substituted.
    #[serde(default)]
    pub args: Vec<String>,
}

/// A data-defined correctness tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolManifest {
    /// Must equal [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Stable unique identifier, e.g. `waitcheck`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// What the tool does and what guarantee it provides.
    pub description: String,
    /// Coarse grouping for display, e.g. `compile`, `analyze`,
    /// `emulate`, `reference`.
    pub category: String,
    /// Relative cost of running the tool (see [`mirage_solver::Step::cost`]).
    pub cost: u64,
    /// Whether the tool is idempotent and its result may be cached.
    #[serde(default)]
    pub cacheable: bool,
    /// Precondition fact templates (e.g. `compiled:${target}`).
    #[serde(default)]
    pub requires: Vec<String>,
    /// Effect fact templates the tool establishes.
    pub produces: Vec<String>,
    /// When `true`, the manifest is expanded once per requested target,
    /// substituting `${target}`. When `false`, it is expanded once.
    #[serde(default)]
    pub per_target: bool,
    /// Names of tools that must be installed in the environment for this
    /// manifest to be usable (e.g. `rocjitsu`).
    #[serde(default)]
    pub needs_tools: Vec<String>,
    /// Whether a physical GPU must be present in the environment.
    #[serde(default)]
    pub needs_gpu: bool,
    /// Optional real command (unused by the default simulated executor).
    #[serde(default)]
    pub command: Option<CommandSpec>,
}

impl ToolManifest {
    /// Validate the manifest's invariants.
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ManifestError::SchemaVersion {
                id: self.id.clone(),
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.id.trim().is_empty() {
            return Err(ManifestError::EmptyId);
        }
        if self.produces.is_empty() {
            return Err(ManifestError::NoEffects { id: self.id.clone() });
        }
        // A per-target manifest must actually reference ${target}
        // somewhere, otherwise expansion produces duplicate steps.
        if self.per_target
            && !self
                .requires
                .iter()
                .chain(self.produces.iter())
                .any(|f| f.contains("${target}"))
        {
            return Err(ManifestError::PerTargetWithoutTemplate { id: self.id.clone() });
        }
        Ok(())
    }

    /// Substitute `${target}` in `template`.
    pub fn substitute(template: &str, target: &str) -> String {
        template.replace("${target}", target)
    }
}

/// A validated set of tool manifests, keyed by id.
#[derive(Debug, Clone, Default)]
pub struct ToolCatalog {
    tools: BTreeMap<String, ToolManifest>,
}

impl ToolCatalog {
    /// An empty catalogue.
    pub fn new() -> Self {
        Self::default()
    }

    /// The built-in catalogue embedded in the crate. These cover the
    /// tools called out in Project Bedroc: a compiler front, the static
    /// Wait Check / hazard detector, the rocjitsu emulator, LdsSan, an
    /// FPSan numeric check, and a reference oracle. They are ordinary
    /// JSON manifests embedded via `include_str!` — identical in form to
    /// anything a user drops into a tools directory.
    pub fn builtin() -> Self {
        let mut catalog = Self::new();
        for raw in BUILTIN_MANIFESTS {
            let m: ToolManifest =
                serde_json::from_str(raw).expect("builtin manifest must be valid JSON");
            catalog
                .insert(m)
                .expect("builtin manifest must pass validation");
        }
        catalog
    }

    /// Insert a manifest, validating it and rejecting duplicate ids.
    pub fn insert(&mut self, manifest: ToolManifest) -> Result<(), ManifestError> {
        manifest.validate()?;
        if self.tools.contains_key(&manifest.id) {
            return Err(ManifestError::Duplicate {
                id: manifest.id.clone(),
            });
        }
        self.tools.insert(manifest.id.clone(), manifest);
        Ok(())
    }

    /// Load and merge every `*.json` manifest in `dir` into this
    /// catalogue. A missing directory is not an error (it simply adds
    /// nothing), so user tool directories are optional.
    pub fn load_dir(&mut self, dir: &Path) -> Result<usize, ManifestError> {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => {
                return Err(ManifestError::Io {
                    path: dir.to_path_buf(),
                    message: e.to_string(),
                });
            }
        };
        let mut count = 0;
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        paths.sort();
        for path in paths {
            let bytes = std::fs::read(&path).map_err(|e| ManifestError::Io {
                path: path.clone(),
                message: e.to_string(),
            })?;
            let manifest: ToolManifest =
                serde_json::from_slice(&bytes).map_err(|e| ManifestError::Parse {
                    path: path.clone(),
                    message: e.to_string(),
                })?;
            self.insert(manifest)?;
            count += 1;
        }
        Ok(count)
    }

    /// Iterate manifests in deterministic (id) order.
    pub fn iter(&self) -> impl Iterator<Item = &ToolManifest> {
        self.tools.values()
    }

    /// Look up a manifest by id.
    pub fn get(&self, id: &str) -> Option<&ToolManifest> {
        self.tools.get(id)
    }

    /// Number of tools in the catalogue.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the catalogue is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Errors from loading or validating manifests.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ManifestError {
    /// The manifest declared an unsupported schema version.
    #[error("tool `{id}` has schema_version {found}, expected {expected}")]
    SchemaVersion {
        /// Offending tool id.
        id: String,
        /// Version found in the manifest.
        found: u32,
        /// Version this crate supports.
        expected: u32,
    },
    /// A manifest had an empty id.
    #[error("tool manifest has an empty id")]
    EmptyId,
    /// A manifest produced no effects.
    #[error("tool `{id}` produces nothing")]
    NoEffects {
        /// Offending tool id.
        id: String,
    },
    /// A per-target manifest never referenced `${target}`.
    #[error("tool `{id}` is per_target but never references ${{target}}")]
    PerTargetWithoutTemplate {
        /// Offending tool id.
        id: String,
    },
    /// Two manifests shared an id.
    #[error("duplicate tool id `{id}`")]
    Duplicate {
        /// The duplicated id.
        id: String,
    },
    /// Failed to read a manifest file.
    #[error("io error reading {path}: {message}")]
    Io {
        /// Path involved.
        path: std::path::PathBuf,
        /// Underlying message.
        message: String,
    },
    /// Failed to parse a manifest file.
    #[error("failed to parse {path}: {message}")]
    Parse {
        /// Path involved.
        path: std::path::PathBuf,
        /// Underlying message.
        message: String,
    },
}

/// The embedded built-in manifests, in catalogue order.
const BUILTIN_MANIFESTS: &[&str] = &[
    include_str!("tools/compile-hip.json"),
    include_str!("tools/compile-asm.json"),
    include_str!("tools/waitcheck.json"),
    include_str!("tools/rocjitsu-emulate.json"),
    include_str!("tools/ldssan.json"),
    include_str!("tools/fpsan.json"),
    include_str!("tools/reference-run.json"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalogue_loads_and_validates() {
        let cat = ToolCatalog::builtin();
        assert!(cat.len() >= 6);
        assert!(cat.get("waitcheck").is_some());
        for m in cat.iter() {
            m.validate().unwrap();
        }
    }

    #[test]
    fn rejects_bad_schema_version() {
        let m = ToolManifest {
            schema_version: 999,
            id: "x".into(),
            name: "x".into(),
            description: String::new(),
            category: "analyze".into(),
            cost: 1,
            cacheable: false,
            requires: vec![],
            produces: vec!["y".into()],
            per_target: false,
            needs_tools: vec![],
            needs_gpu: false,
            command: None,
        };
        assert!(matches!(
            m.validate(),
            Err(ManifestError::SchemaVersion { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_ids() {
        let mut cat = ToolCatalog::new();
        let m = ToolManifest {
            schema_version: SCHEMA_VERSION,
            id: "dup".into(),
            name: "dup".into(),
            description: String::new(),
            category: "analyze".into(),
            cost: 1,
            cacheable: false,
            requires: vec![],
            produces: vec!["y".into()],
            per_target: false,
            needs_tools: vec![],
            needs_gpu: false,
            command: None,
        };
        cat.insert(m.clone()).unwrap();
        assert!(matches!(cat.insert(m), Err(ManifestError::Duplicate { .. })));
    }

    #[test]
    fn per_target_must_template() {
        let m = ToolManifest {
            schema_version: SCHEMA_VERSION,
            id: "pt".into(),
            name: "pt".into(),
            description: String::new(),
            category: "analyze".into(),
            cost: 1,
            cacheable: false,
            requires: vec!["compiled".into()],
            produces: vec!["done".into()],
            per_target: true,
            needs_tools: vec![],
            needs_gpu: false,
            command: None,
        };
        assert!(matches!(
            m.validate(),
            Err(ManifestError::PerTargetWithoutTemplate { .. })
        ));
    }

    #[test]
    fn load_dir_merges_and_skips_missing() {
        let dir = tempfile::tempdir().unwrap();
        let mut cat = ToolCatalog::new();
        // Missing dir is fine.
        assert_eq!(cat.load_dir(&dir.path().join("nope")).unwrap(), 0);
        std::fs::write(
            dir.path().join("custom.json"),
            r#"{
                "schema_version": 1,
                "id": "custom-check",
                "name": "Custom Check",
                "description": "demo",
                "category": "analyze",
                "cost": 5,
                "produces": ["custom_ok:${target}"],
                "per_target": true
            }"#,
        )
        .unwrap();
        assert_eq!(cat.load_dir(dir.path()).unwrap(), 1);
        assert!(cat.get("custom-check").is_some());
    }
}
