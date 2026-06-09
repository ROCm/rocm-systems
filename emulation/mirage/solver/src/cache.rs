//! Memoization of idempotent steps.
//!
//! A [`Cache`] records the outcome of running a cacheable [`crate::Step`]
//! keyed by the step id plus a fingerprint of the inputs that determine
//! its result. When the planner (or an executor) sees a cache hit it can
//! reuse the prior result instead of re-running the step, and the
//! planner discounts the step's cost accordingly.
//!
//! The cache is deliberately simple and serializable so it can be
//! persisted to disk (one JSON document) and shared across runs.

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::state::Fact;
use crate::step::Step;

/// A content-addressed key identifying a memoized step result.
///
/// The key combines the step id with the sorted set of precondition
/// facts that were true when it ran, so the same step run against the
/// same inputs maps to the same entry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CacheKey(pub String);

impl CacheKey {
    /// Derive a cache key for `step` given the `inputs` (typically the
    /// step's satisfied preconditions). Inputs are sorted so ordering
    /// does not affect the key.
    pub fn for_step(step: &Step, inputs: &[Fact]) -> Self {
        let mut sorted = inputs.to_vec();
        sorted.sort();
        let mut hasher = DefaultHasher::new();
        step.id.hash(&mut hasher);
        for i in &sorted {
            i.hash(&mut hasher);
        }
        CacheKey(format!("{}@{:016x}", step.id, hasher.finish()))
    }
}

/// The stored outcome of a memoized step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedResult {
    /// Facts the step produced.
    pub produced: Vec<Fact>,
    /// Opaque artifact fingerprint (e.g. a hash of a compiled binary),
    /// for callers that want to detect divergence across runs.
    pub fingerprint: String,
}

/// A serializable map of memoized step results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Cache {
    entries: BTreeMap<CacheKey, CachedResult>,
}

impl Cache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a memoized result.
    pub fn get(&self, key: &CacheKey) -> Option<&CachedResult> {
        self.entries.get(key)
    }

    /// Whether a result is memoized for `key`.
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Store a memoized result, replacing any prior entry.
    pub fn put(&mut self, key: CacheKey, result: CachedResult) {
        self.entries.insert(key, result);
    }

    /// Number of memoized entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Load a cache from a JSON file. A missing file yields an empty
    /// cache rather than an error, so first runs work transparently.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        match std::fs::read(path) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)
                .unwrap_or_default()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::new()),
            Err(e) => Err(e),
        }
    }

    /// Persist the cache to a JSON file, creating parent directories.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_stable_under_input_reordering() {
        let step = Step::new("compile", ["source"], ["compiled"], 1);
        let a = CacheKey::for_step(&step, &["x".into(), "y".into()]);
        let b = CacheKey::for_step(&step, &["y".into(), "x".into()]);
        assert_eq!(a, b);
    }

    #[test]
    fn key_differs_by_step_and_inputs() {
        let s1 = Step::new("a", [] as [&str; 0], ["o"], 1);
        let s2 = Step::new("b", [] as [&str; 0], ["o"], 1);
        assert_ne!(
            CacheKey::for_step(&s1, &["i".into()]),
            CacheKey::for_step(&s2, &["i".into()])
        );
        assert_ne!(
            CacheKey::for_step(&s1, &["i".into()]),
            CacheKey::for_step(&s1, &["j".into()])
        );
    }

    #[test]
    fn roundtrips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("cache.json");
        let mut c = Cache::new();
        let step = Step::new("s", [] as [&str; 0], ["o"], 1);
        let key = CacheKey::for_step(&step, &[]);
        c.put(
            key.clone(),
            CachedResult {
                produced: vec!["o".into()],
                fingerprint: "abc".into(),
            },
        );
        c.save(&path).unwrap();
        let loaded = Cache::load(&path).unwrap();
        assert_eq!(loaded.get(&key).unwrap().fingerprint, "abc");
    }

    #[test]
    fn missing_file_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let c = Cache::load(&dir.path().join("nope.json")).unwrap();
        assert!(c.is_empty());
    }
}
