//! Container runtime types shared across mirage crates.
//!
//! The *static* configuration of containerisation (image, mounts,
//! provider) lives on the profile as [`crate::profile::ContainerizedDef`].
//! This module holds the *runtime* pieces:
//!
//! * [`ContainerState`] — the record the host writes to
//!   `<session>/container/state.json` once it has launched a session's
//!   per-node containers and virtual network. It is the single source
//!   of truth used to tear everything down again.
//! * Naming helpers ([`container_name`], [`network_name`]) so every
//!   crate derives the same deterministic names.
//! * Provider resolution ([`detect_provider`], [`resolve_provider`])
//!   implementing the "prefer podman, fall back to docker" policy.
//! * [`teardown`] — a dependency-free, best-effort removal of a
//!   session's containers and network, used both by the host on
//!   shutdown and by the control plane when a session is destroyed.
//!
//! The richer orchestration (pulling images, creating the network,
//! starting containers, building `exec` argv) lives in the
//! `mirage_container` crate, which builds on these shared types.

use std::path::Path;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::session::SessionId;

/// Environment variable carrying a node's rank (0 = head). Always set
/// on every node process, containerised or not.
pub const ENV_RANK: &str = "MIRAGE_RANK";

/// Environment variable carrying the head node's address. Set on every
/// node, including the head (rank 0), which gets `localhost`.
pub const ENV_HEAD_ADDR: &str = "MIRAGE_HEAD_ADDR";

/// Environment variable carrying the port the head node may listen on.
/// Always set on every node process.
pub const ENV_HEAD_PORT: &str = "MIRAGE_HEAD_PORT";

/// Deterministic container name for a node of a session.
///
/// Doubles as the container's network hostname, so other nodes can
/// reach it by this name on the shared per-session network.
pub fn container_name(session: &SessionId, rank: u32) -> String {
    format!("mirage-{}-node-{rank}", session.as_str())
}

/// Deterministic virtual-network name connecting a session's nodes.
pub fn network_name(session: &SessionId) -> String {
    format!("mirage-{}", session.as_str())
}

/// Auto-detect a container provider on `PATH`, preferring podman.
///
/// Returns the provider's bare name (`"podman"` / `"docker"`) when
/// found, or `None` if neither is installed.
pub fn detect_provider() -> Option<String> {
    for candidate in ["podman", "docker"] {
        if which_on_path(candidate).is_some() {
            return Some(candidate.to_string());
        }
    }
    None
}

/// Resolve the provider to use, in priority order:
///
/// 1. an explicit value from the profile (`"podman"`, `"docker"`, or a path),
/// 2. the `MIRAGE_CONTAINER_PROVIDER` environment override,
/// 3. auto-detection ([`detect_provider`]).
///
/// Returns `None` only when no provider was specified and none could be
/// detected.
pub fn resolve_provider(explicit: Option<&str>) -> Option<String> {
    if let Some(p) = explicit
        && !p.is_empty()
    {
        return Some(p.to_string());
    }
    if let Ok(p) = std::env::var("MIRAGE_CONTAINER_PROVIDER")
        && !p.is_empty()
    {
        return Some(p);
    }
    detect_provider()
}

/// One container backing one node of a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeContainer {
    /// Node rank (0 = head).
    pub rank: u32,
    /// Container name/id (also its hostname on the network).
    pub name: String,
}

/// Persisted record of the containers + network backing a session.
///
/// Written by the host to `<session>/container/state.json` after it
/// launches a containerised session, and consumed by [`teardown`] to
/// remove everything again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ContainerState {
    /// Resolved provider binary used to manage these containers.
    pub provider: String,
    /// Image every node was launched from.
    pub image: String,
    /// Per-session virtual network the nodes are joined to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<String>,
    /// Port the head node may listen on (carried in `MIRAGE_HEAD_PORT`).
    pub head_port: u16,
    /// The per-node containers, in rank order.
    #[serde(default)]
    pub nodes: Vec<NodeContainer>,
}

/// Best-effort teardown of a session's containers and network.
///
/// Reads the [`ContainerState`] at `state_path` (a no-op if it is
/// absent, e.g. a non-containerised session) and asks the recorded
/// provider to `rm -f` every node container, then remove the network.
/// Every command is best-effort: failures are ignored so a missing or
/// already-removed container/network never blocks session cleanup, and
/// the operation is idempotent.
pub fn teardown(state_path: &Path) {
    let Ok(Some(state)) = crate::state::read_json_opt::<ContainerState>(state_path) else {
        return;
    };
    for node in &state.nodes {
        run_quiet(&state.provider, &["rm", "-f", &node.name]);
    }
    if let Some(network) = &state.network {
        run_quiet(&state.provider, &["network", "rm", network]);
    }
}

/// Run `<provider> <args...>` discarding all stdio, ignoring failures.
fn run_quiet(provider: &str, args: &[&str]) {
    let _ = Command::new(provider)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// Locate an executable named `name` on `PATH`.
fn which_on_path(name: &str) -> Option<std::path::PathBuf> {
    // An explicit path (absolute or containing a separator) is used
    // as-is when it points at a real file.
    if name.contains('/') {
        let p = std::path::PathBuf::from(name);
        return p.is_file().then_some(p);
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn mock_provider(dir: &Path, log: &Path) -> std::path::PathBuf {
        let provider = dir.join("mock-provider.sh");
        std::fs::write(
            &provider,
            format!("#!/bin/sh\necho \"$@\" >> {}\n", log.display()),
        )
        .unwrap();
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();
        provider
    }

    #[test]
    fn names_are_deterministic() {
        let s = SessionId::new("abc").unwrap();
        assert_eq!(container_name(&s, 0), "mirage-abc-node-0");
        assert_eq!(container_name(&s, 3), "mirage-abc-node-3");
        assert_eq!(network_name(&s), "mirage-abc");
    }

    #[test]
    fn resolve_provider_prefers_explicit() {
        assert_eq!(resolve_provider(Some("docker")), Some("docker".to_string()));
    }

    #[test]
    fn teardown_noop_when_state_missing() {
        let dir = tempfile::tempdir().unwrap();
        // Must not panic / must be a no-op.
        teardown(&dir.path().join("nope.json"));
    }

    #[test]
    fn teardown_removes_every_container_and_network() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("provider.log");
        let provider = mock_provider(dir.path(), &log);

        let state = ContainerState {
            provider: provider.to_string_lossy().to_string(),
            image: "img:latest".to_string(),
            network: Some("mirage-s1".to_string()),
            head_port: 12345,
            nodes: vec![
                NodeContainer {
                    rank: 0,
                    name: "mirage-s1-node-0".to_string(),
                },
                NodeContainer {
                    rank: 1,
                    name: "mirage-s1-node-1".to_string(),
                },
            ],
        };
        let state_path = dir.path().join("state.json");
        crate::state::write_json(&state_path, &state).unwrap();

        teardown(&state_path);

        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(recorded.contains("rm -f mirage-s1-node-0"), "{recorded:?}");
        assert!(recorded.contains("rm -f mirage-s1-node-1"), "{recorded:?}");
        assert!(recorded.contains("network rm mirage-s1"), "{recorded:?}");
    }
}
