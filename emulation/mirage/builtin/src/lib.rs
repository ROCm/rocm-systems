//! Built-in agents and topologies that mirage preloads into
//! `<MIRAGE_CONFIG>/{agent,topology}/`.
//!
//! Historically these shipped as `agents/*.json` files embedded into
//! `mirage_core` at build time and parsed at runtime. They now live
//! here as strongly-typed [`mirage_core::agent::AgentDef`] /
//! [`mirage_core::topology::TopologyDef`] constructors so the data is
//! validated by the compiler instead of by a runtime parse.
//!
//! `mirage_core` owns the on-disk *store* logic
//! ([`mirage_core::agent::store::write_builtins`]); this crate owns
//! the *data* and provides thin `ensure_*` helpers that feed the
//! store.

pub mod agents;
pub mod topologies;

use mirage_core::error::Result;

pub use agents::{agents, mi300x, mi350x};
pub use topologies::{default_topology, topologies};

/// Write all builtin agents to disk.
///
/// If `force` is true existing files are overwritten, otherwise only
/// missing agents are written. Returns `(name, written)` per agent.
pub fn ensure_agents(force: bool) -> Result<Vec<(String, bool)>> {
    mirage_core::agent::store::write_builtins(&agents(), force)
}

/// Write all builtin topologies to disk.
pub fn ensure_topologies(force: bool) -> Result<Vec<(String, bool)>> {
    mirage_core::topology::store::write_builtins(&topologies(), force)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_writes_then_skips() {
        let _g = mirage_core::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(tmp.path());

        let first = ensure_agents(false).unwrap();
        assert!(!first.is_empty());
        assert!(first.iter().all(|(_, w)| *w));
        assert!(ensure_agents(false).unwrap().iter().all(|(_, w)| !*w));
        assert!(ensure_agents(true).unwrap().iter().all(|(_, w)| *w));

        let topos = ensure_topologies(false).unwrap();
        assert!(topos.iter().any(|(n, _)| n == "MI350X-1x1"));
    }
}
