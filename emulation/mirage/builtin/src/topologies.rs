//! Strongly-typed builtin [`TopologyDef`]s (system layouts).
//!
//! Each topology references a builtin agent by name; see
//! [`crate::agents`] for the agents themselves.

use mirage_core::common::MaybeRef;
use mirage_core::topology::TopologyDef;

/// All builtin topologies, keyed by the name written to disk.
pub fn topologies() -> Vec<(&'static str, TopologyDef)> {
    vec![
        ("MI350X-1x1", layout(1, 1, "MI350X")),
        ("MI350X-1x8", layout(1, 8, "MI350X")),
        ("MI350X-2x8", layout(2, 8, "MI350X")),
        ("MI300X-1x8", layout(1, 8, "MI300X")),
    ]
}

/// Default topology referenced by a fresh profile: a single GPU
/// driven by the `MI350X` agent.
pub fn default_topology() -> TopologyDef {
    layout(1, 1, "MI350X")
}

fn layout(num_nodes: u32, gpus_per_node: u32, agent: &str) -> TopologyDef {
    TopologyDef {
        num_nodes,
        gpus_per_node,
        agent: MaybeRef::Ref(agent.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_default_layouts() {
        assert!(topologies().iter().any(|(n, _)| *n == "MI350X-1x1"));
        assert_eq!(default_topology().total_gpus(), 1);
    }
}
