//! Built-in agents and topologies that mirage preloads into
//! `<MIRAGE_CONFIG>/{agent,topology}/`.
//!
//! Historically these shipped as `agents/*.json` files embedded into
//! `mirage_core` at build time and parsed at runtime. They now live
//! here as strongly-typed [`mirage_core::agent::AgentDef`] /
//! [`mirage_core::topology::TopologyDef`] constructors so the data is
//! validated by the compiler instead of by a runtime parse.
//!
//! This crate owns both the builtin *data* and the policy for writing
//! it to disk. It relies on `mirage_core` only for the low-level path
//! resolution ([`mirage_core::paths`]) and JSON serialization
//! ([`mirage_core::state::write_json`]).

pub mod agents;
pub mod profiles;
pub mod topologies;

use mirage_core::error::Result;

pub use agents::{agents, mi300x, mi350x, mi450x};
pub use profiles::profiles;
pub use topologies::{default_topology, topologies};

/// Write all builtin agents to disk.
///
/// If `force` is true existing files are overwritten, otherwise only
/// missing agents are written. Returns `(name, written)` per agent.
pub fn ensure_agents(force: bool) -> Result<Vec<(String, bool)>> {
    let mut report = Vec::new();
    for (name, agent) in agents() {
        let p = mirage_core::paths::agent_path(name);
        if p.exists() && !force {
            report.push((name.to_string(), false));
            continue;
        }
        mirage_core::state::write_json(&p, &agent)?;
        report.push((name.to_string(), true));
    }
    Ok(report)
}

/// Write all builtin topologies to disk.
///
/// If `force` is true existing files are overwritten, otherwise only
/// missing topologies are written. Returns `(name, written)` per entry.
pub fn ensure_topologies(force: bool) -> Result<Vec<(String, bool)>> {
    let mut report = Vec::new();
    for (name, topology) in topologies() {
        let p = mirage_core::paths::topology_path(name);
        if p.exists() && !force {
            report.push((name.to_string(), false));
            continue;
        }
        mirage_core::state::write_json(&p, &topology)?;
        report.push((name.to_string(), true));
    }
    Ok(report)
}

/// Write all builtin profiles to disk.
///
/// If `force` is true existing files are overwritten, otherwise only
/// missing profiles are written. Returns `(name, written)` per entry.
pub fn ensure_profiles(force: bool) -> Result<Vec<(String, bool)>> {
    let mut report = Vec::new();
    for (name, profile) in profiles() {
        let p = mirage_core::paths::profile_path(name);
        if p.exists() && !force {
            report.push((name.to_string(), false));
            continue;
        }
        mirage_core::state::write_json(&p, &profile)?;
        report.push((name.to_string(), true));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_agents_writes_then_skips() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        let first = ensure_agents(false).unwrap();
        assert!(!first.is_empty());
        assert!(
            first.iter().all(|(_, w)| *w),
            "first run should write every builtin"
        );

        let mut names: Vec<String> = first.iter().map(|(n, _)| n.clone()).collect();
        names.sort();
        assert_eq!(mirage_core::agent::store::list().unwrap(), names);

        assert!(
            ensure_agents(false).unwrap().iter().all(|(_, w)| !*w),
            "second run should not rewrite existing builtins"
        );
        assert!(
            ensure_agents(true).unwrap().iter().all(|(_, w)| *w),
            "force should rewrite every builtin"
        );

        for name in &names {
            assert!(
                mirage_core::agent::store::get(name).is_ok(),
                "{name} should be readable"
            );
        }
    }

    #[test]
    fn ensure_topologies_writes_then_skips() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        let first = ensure_topologies(false).unwrap();
        assert!(!first.is_empty());
        assert!(first.iter().all(|(_, w)| *w));
        assert!(first.iter().any(|(n, _)| n == "MI350X-1x1"));

        assert!(
            ensure_topologies(false).unwrap().iter().all(|(_, w)| !*w),
            "second run should not rewrite existing builtins"
        );
    }
}
