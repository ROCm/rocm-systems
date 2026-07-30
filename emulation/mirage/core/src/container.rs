//! Container runtime types shared across mirage crates.
//!
//! The *static* configuration of containerisation (image, mounts,
//! provider) lives on the profile as [`crate::profile::ContainerizedDef`].
//! This module holds the *runtime* pieces:
//!
//! * [`ContainerState`] — the record the supervisor builds once it has
//!   launched a session's per-node containers and virtual network. It is
//!   held in memory alongside the session and is the single source of
//!   truth used to tear everything down again.
//! * Naming helpers ([`container_name`], [`network_name`]) so every
//!   crate derives the same deterministic names.
//! * Provider resolution ([`detect_provider`], [`resolve_provider`])
//!   implementing the "prefer podman, fall back to docker" policy.
//! * [`teardown`] — a dependency-free, best-effort removal of a
//!   session's containers and network, run during session teardown and
//!   again as a backstop when the daemon shuts down.
//!
//! The richer orchestration (pulling images, creating the network,
//! starting containers, building `exec` argv) lives in the
//! `mirage_container` crate, which builds on these shared types.

use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::session::SessionId;

/// Environment variable carrying a node's rank (0 = head). Always set
/// on every node process, containerised or not.
pub const ENV_RANK: &str = "MIRAGE_RANK";
/// `torch.distributed` global rank: the process's index across the whole
/// job, in `0..WORLD_SIZE`. Distinct from [`ENV_RANK`] (`MIRAGE_RANK`),
/// which identifies the *node*: with `--nproc-per-node > 1` several
/// processes share a node (and thus a `MIRAGE_RANK`) but each gets a
/// unique `RANK`. With the default of one process per node it equals
/// `MIRAGE_RANK`. Set so PyTorch's `env://` init method (and `torchrun`)
/// can read the rank under its standard name without the workload having
/// to translate mirage's own variables.
pub const ENV_TORCH_RANK: &str = "RANK";

/// Environment variable carrying the head node's address. Set on every
/// node, including the head (rank 0), which gets `localhost`.
pub const ENV_HEAD_ADDR: &str = "MIRAGE_HEAD_ADDR";

/// Environment variable carrying the port the head node may listen on.
/// Always set on every node process.
pub const ENV_HEAD_PORT: &str = "MIRAGE_HEAD_PORT";

/// `torch.distributed` rendezvous address. Aliases [`ENV_HEAD_ADDR`] so
/// PyTorch's `env://` init method (and `torchrun --rdzv-endpoint`) work
/// out of the box on every node without the workload having to translate
/// mirage's own variables.
pub const ENV_MASTER_ADDR: &str = "MASTER_ADDR";

/// `torch.distributed` rendezvous port. Aliases [`ENV_HEAD_PORT`].
pub const ENV_MASTER_PORT: &str = "MASTER_PORT";

/// `torch.distributed` world size: the total number of ranks in the
/// job, i.e. `num_nodes * nproc_per_node`. With the default of one
/// workload process per node this is just the session's node count. Set
/// on every process so PyTorch's `env://` init method works without a
/// launcher like `torchrun`.
pub const ENV_WORLD_SIZE: &str = "WORLD_SIZE";

/// `torch.distributed` local rank: the process's index *within* its
/// node, in `0..nproc_per_node`. With the default of one workload
/// process per node this is always `0`. Set on every process so
/// PyTorch's `env://` init method works without a launcher like
/// `torchrun`.
pub const ENV_LOCAL_RANK: &str = "LOCAL_RANK";

/// RCCL/NCCL host identifier. Normally NCCL derives this from the real
/// machine's hostname to group ranks that share a host (so it can use
/// fast intra-host transports and reject two ranks claiming the same
/// physical GPU). In mirage's non-containerised multi-node mode every
/// emulated node runs on the *same* real host and is synthesised from an
/// identical per-node config, so each rank's GPU reports the same
/// `location_id`. NCCL then aborts with "Duplicate GPU detected: rank X
/// and rank Y both on CUDA device …". Setting a distinct `NCCL_HOSTID`
/// per emulated node makes NCCL treat them as separate hosts, which is
/// the correct model here (one emulated GPU per node). Ranks sharing a
/// node keep the same value and disambiguate by local GPU index.
pub const ENV_NCCL_HOSTID: &str = "NCCL_HOSTID";

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

/// Label stamped on every container and network mirage creates.
///
/// Names alone are not ownership. `mirage-s1-node-0` is a name any user
/// can give a container of their own, and mirage's cleanup paths do
/// `rm -f <name>` — so without a mark that says "mirage made this", a
/// collision means destroying someone else's container. The label is
/// checked before every removal.
pub const LABEL_OWNER: &str = "mirage.owner";

/// Value [`LABEL_OWNER`] carries. Anything else is not ours.
pub const LABEL_OWNER_VALUE: &str = "mirage";

/// Label recording which session a resource belongs to.
///
/// This is what makes orphans recoverable. Session state lives in the
/// supervisor's memory, so a daemon that is `SIGKILL`ed takes its record
/// of every container with it; the label survives on the resource itself,
/// so [`reclaim_orphans`] can still find them and say which session they
/// came from.
pub const LABEL_SESSION: &str = "mirage.session";

/// The labels every resource belonging to `session` carries.
#[must_use]
pub fn owner_labels(session: &SessionId) -> Vec<(String, String)> {
    vec![
        (LABEL_OWNER.to_string(), LABEL_OWNER_VALUE.to_string()),
        (LABEL_SESSION.to_string(), session.as_str().to_string()),
    ]
}

/// Read one label off a container, or `None` if it is absent.
fn container_label(provider: &str, name: &str, label: &str) -> Option<String> {
    inspect_label(provider, &["inspect"], name, &format!("{{{{index .Config.Labels {label:?}}}}}"))
}

/// Read one label off a network, or `None` if it is absent.
fn network_label(provider: &str, name: &str, label: &str) -> Option<String> {
    inspect_label(
        provider,
        &["network", "inspect"],
        name,
        &format!("{{{{index .Labels {label:?}}}}}"),
    )
}

/// `<provider> <verb…> --format <template> <name>`, trimmed.
///
/// Go templates render a missing key as `<no value>`, and an empty label
/// as the empty string; both mean "not ours" and become `None`.
fn inspect_label(provider: &str, verb: &[&str], name: &str, template: &str) -> Option<String> {
    let out = retrying_etxtbsy(|| {
        Command::new(provider)
            .args(verb)
            .arg("--format")
            .arg(template)
            .arg(name)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
    })
    .ok()?;
    if !out.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if value.is_empty() || value == "<no value>" {
        return None;
    }
    Some(value)
}

/// Whether `name` is a container mirage created.
#[must_use]
pub fn container_is_ours(provider: &str, name: &str) -> bool {
    container_label(provider, name, LABEL_OWNER).as_deref() == Some(LABEL_OWNER_VALUE)
}

/// Whether `name` is a network mirage created.
#[must_use]
pub fn network_is_ours(provider: &str, name: &str) -> bool {
    network_label(provider, name, LABEL_OWNER).as_deref() == Some(LABEL_OWNER_VALUE)
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

/// Record of the containers + network backing a session.
///
/// Built by the supervisor after it launches a containerised session,
/// held in memory for the session's lifetime, and consumed by
/// [`teardown`] to remove everything again.
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
/// Asks the recorded provider to `rm -f` every node container, then to
/// remove the network. Every command is best-effort: failures are ignored
/// so a missing or already-removed container never blocks session
/// cleanup, and the operation is idempotent — running it twice, or
/// against a session whose containers are already gone, is a no-op.
///
/// Each resource's [`LABEL_OWNER`] is checked first. Container names are
/// derived from the session id, so `mirage-s1-node-0` is a name a user
/// can perfectly well have given a container of their own; `rm -f` on a
/// name alone would then destroy it. Removing only what carries mirage's
/// label makes cleanup safe to run against a shared engine.
///
/// This blocks on the provider binary, so async callers must run it on a
/// blocking task.
pub fn teardown(state: &ContainerState) {
    for node in &state.nodes {
        if container_is_ours(&state.provider, &node.name) {
            run_quiet(&state.provider, &["rm", "-f", &node.name]);
        } else {
            tracing::warn!(
                container = %node.name,
                "refusing to remove a container that is not labelled {LABEL_OWNER}={LABEL_OWNER_VALUE}"
            );
        }
    }
    if let Some(network) = &state.network
        && network_is_ours(&state.provider, network)
    {
        run_quiet(&state.provider, &["network", "rm", network]);
    }
}

/// Remove every mirage-owned container and network whose session is not
/// in `live`, returning what was removed.
///
/// This is the recovery path for a supervisor that died without tearing
/// its sessions down. Session state is deliberately in-memory, so a
/// `SIGKILL`ed daemon leaves containers with no record anywhere that a
/// later mirage could read — except the resources themselves, which carry
/// [`LABEL_SESSION`]. Filtering by [`LABEL_OWNER`] first is what keeps
/// this safe to run on an engine shared with other work: a container
/// mirage did not create is never a candidate, whatever it is called.
///
/// Blocks on the provider binary.
pub fn reclaim_orphans(provider: &str, live: &[SessionId]) -> Vec<String> {
    let live: std::collections::HashSet<&str> = live.iter().map(SessionId::as_str).collect();
    let mut removed = Vec::new();

    for (name, session) in labelled(provider, &["ps", "--all"]) {
        if live.contains(session.as_str()) {
            continue;
        }
        run_quiet(provider, &["rm", "-f", &name]);
        removed.push(name);
    }
    for (name, session) in labelled(provider, &["network", "ls"]) {
        if live.contains(session.as_str()) {
            continue;
        }
        run_quiet(provider, &["network", "rm", &name]);
        removed.push(name);
    }
    removed
}

/// `(name, session)` for every mirage-owned resource the given listing
/// verb reports.
///
/// The owner filter is applied by the engine rather than by us, so a
/// resource without the label is never even named here.
fn labelled(provider: &str, verb: &[&str]) -> Vec<(String, String)> {
    let filter = format!("label={LABEL_OWNER}={LABEL_OWNER_VALUE}");
    let template = format!("{{{{.Name}}}}\t{{{{index .Labels {LABEL_SESSION:?}}}}}");
    let Ok(out) = retrying_etxtbsy(|| {
        Command::new(provider)
            .args(verb)
            .arg("--filter")
            .arg(&filter)
            .arg("--format")
            .arg(&template)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
    }) else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let (name, session) = line.split_once('\t')?;
            let (name, session) = (name.trim(), session.trim());
            if name.is_empty() || session.is_empty() || session == "<no value>" {
                return None;
            }
            Some((name.to_string(), session.to_string()))
        })
        .collect()
}

/// Run `<provider> <args...>` discarding all stdio.
///
/// Teardown is best-effort by design: a container that is already gone,
/// or a provider that reports failure removing it, must not block the
/// rest of session cleanup. Spawn failures are the one thing worth
/// retrying — `ExecutableFileBusy` is transient and means the provider
/// binary was still being written when we tried to exec it, so giving up
/// on it would silently skip a removal and leak the container. Real
/// providers (podman/docker) are stable binaries that never hit this;
/// the guard matters for a freshly-written wrapper script.
fn run_quiet(provider: &str, args: &[&str]) {
    if let Err(e) = retrying_etxtbsy(|| {
        Command::new(provider)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .output()
    }) {
        tracing::debug!(provider, ?args, "teardown command could not run: {e}");
    }
}

/// Run a provider command, retrying while the binary reports `ETXTBSY`.
///
/// `ETXTBSY` means the executable was still open for writing when we
/// tried to exec it — transient, and never seen with a real
/// podman/docker install, but routine for a freshly-written wrapper
/// script. Giving up on it silently skips whatever the command was going
/// to do, which for an ownership probe means concluding "not ours" about
/// a container that *is* ours and then declining to remove it.
fn retrying_etxtbsy<F>(mut run: F) -> std::io::Result<std::process::Output>
where
    F: FnMut() -> std::io::Result<std::process::Output>,
{
    const MAX_ATTEMPTS: u32 = 50;
    const BACKOFF: std::time::Duration = std::time::Duration::from_millis(10);

    let mut attempts = 0;
    loop {
        match run() {
            Err(e)
                if e.kind() == std::io::ErrorKind::ExecutableFileBusy
                    && attempts < MAX_ATTEMPTS =>
            {
                attempts += 1;
                std::thread::sleep(BACKOFF);
            }
            other => return other,
        }
    }
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use std::path::Path;
    use std::os::unix::fs::PermissionsExt;

    /// A provider that records its argv and claims every resource is
    /// mirage's.
    fn mock_provider(dir: &Path, log: &Path) -> std::path::PathBuf {
        mock_provider_owned_by(dir, log, LABEL_OWNER_VALUE)
    }

    /// A provider whose `inspect` reports `owner` as the value of
    /// [`LABEL_OWNER`], so a test can stand up resources mirage does
    /// *not* own.
    fn mock_provider_owned_by(dir: &Path, log: &Path, owner: &str) -> std::path::PathBuf {
        let provider = dir.join("mock-provider.sh");
        std::fs::write(
            &provider,
            format!(
                "#!/bin/sh\n\
                 echo \"$@\" >> {log}\n\
                 case \"$1 $2\" in\n\
                 \"inspect --format\"|\"network inspect\") printf '%s' '{owner}' ;;\n\
                 esac\n\
                 exit 0\n",
                log = log.display(),
                owner = owner,
            ),
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
    fn teardown_of_an_empty_state_is_a_no_op() {
        // A non-containerised session has nothing to remove; teardown must
        // not invent a provider invocation (or panic) for it.
        teardown(&ContainerState::default());
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
        teardown(&state);

        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(recorded.contains("rm -f mirage-s1-node-0"), "{recorded:?}");
        assert!(recorded.contains("rm -f mirage-s1-node-1"), "{recorded:?}");
        assert!(recorded.contains("network rm mirage-s1"), "{recorded:?}");
    }

    #[test]
    fn teardown_spares_a_container_mirage_did_not_create() {
        // Container names are derived from the session id, so
        // `mirage-s1-node-0` is a name any user can give a container of
        // their own — and a session with a colliding id would then have
        // mirage `rm -f` it. Ownership is the label, not the name.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("provider.log");
        let provider = mock_provider_owned_by(dir.path(), &log, "someone-else");

        let state = ContainerState {
            provider: provider.to_string_lossy().to_string(),
            image: "img:latest".to_string(),
            network: Some("mirage-s1".to_string()),
            head_port: 1,
            nodes: vec![NodeContainer {
                rank: 0,
                name: "mirage-s1-node-0".to_string(),
            }],
        };
        teardown(&state);

        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            !recorded.contains("rm -f"),
            "mirage removed a container it does not own:\n{recorded}"
        );
        assert!(
            !recorded.contains("network rm"),
            "mirage removed a network it does not own:\n{recorded}"
        );
    }

    #[test]
    fn owner_labels_name_the_session() {
        let labels = owner_labels(&SessionId::new("s1").unwrap());
        assert_eq!(
            labels,
            vec![
                (LABEL_OWNER.to_string(), LABEL_OWNER_VALUE.to_string()),
                (LABEL_SESSION.to_string(), "s1".to_string()),
            ]
        );
    }

    #[test]
    fn reclaim_removes_only_orphans_and_only_ours() {
        // The recovery path after a supervisor died without tearing down:
        // the only surviving record of a container is the label on the
        // container itself.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("provider.log");
        let provider = dir.path().join("mock-provider.sh");
        // `ps`/`network ls` report one live session and one orphan; the
        // owner filter is applied by the engine, so everything listed is
        // already ours.
        std::fs::write(
            &provider,
            format!(
                "#!/bin/sh\n\
                 echo \"$@\" >> {log}\n\
                 case \"$1 $2\" in\n\
                 \"ps --all\") printf 'mirage-live-node-0\\tlive\\nmirage-dead-node-0\\tdead\\n' ;;\n\
                 \"network ls\") printf 'mirage-dead\\tdead\\n' ;;\n\
                 esac\n\
                 exit 0\n",
                log = log.display(),
            ),
        )
        .unwrap();
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();
        let provider = provider.to_string_lossy().to_string();

        let live = vec![SessionId::new("live").unwrap()];
        let removed = reclaim_orphans(&provider, &live);

        assert_eq!(removed, vec!["mirage-dead-node-0", "mirage-dead"]);
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            recorded.contains(&format!("--filter label={LABEL_OWNER}={LABEL_OWNER_VALUE}")),
            "the listing must be filtered to mirage's own resources:\n{recorded}"
        );
        assert!(recorded.contains("rm -f mirage-dead-node-0"), "{recorded}");
        assert!(recorded.contains("network rm mirage-dead"), "{recorded}");
        assert!(
            !recorded.contains("mirage-live-node-0\n") || !recorded.contains("rm -f mirage-live"),
            "a container belonging to a live session was reclaimed:\n{recorded}"
        );
    }

    #[test]
    fn teardown_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("provider.log");
        let provider = mock_provider(dir.path(), &log);
        let state = ContainerState {
            provider: provider.to_string_lossy().to_string(),
            image: "img:latest".to_string(),
            network: Some("mirage-s2".to_string()),
            head_port: 1,
            nodes: vec![NodeContainer {
                rank: 0,
                name: "mirage-s2-node-0".to_string(),
            }],
        };
        // Running teardown twice must not error; the second pass simply
        // asks the provider to remove things that are already gone. This
        // is relied upon: the session teardown path and the daemon's
        // shutdown backstop can both fire for the same session.
        teardown(&state);
        teardown(&state);
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert_eq!(
            recorded.matches("rm -f mirage-s2-node-0").count(),
            2,
            "{recorded:?}"
        );
    }
}
