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
//!   implementing the "prefer podman, fall back to docker" policy, and
//!   [`installed_providers`] for the callers that have to ask every
//!   engine on the machine rather than the one a new session would use.
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

/// Environment variable naming the session a process belongs to.
///
/// Set on every workload process and on every container provider client
/// mirage spawns, and inherited by anything they fork. It is what makes a
/// stranded process recoverable: session state lives in the owning run's
/// memory, so a run that was `SIGKILL`ed leaves no record anywhere that a
/// later mirage could read — except on the processes themselves. This is
/// the process-side counterpart of [`LABEL_SESSION`], and
/// [`crate::reclaim`] is what reads it.
pub const ENV_SESSION: &str = "MIRAGE_SESSION";

/// Environment variable naming the mirage runtime directory that owns a
/// process.
///
/// [`crate::paths::mirage_runtime_dir`] already reads this as an *input*
/// override, and setting it on a workload is coherent with that reading:
/// a nested `mirage` inside a session sees the same state directory as
/// the run that owns it. What makes it a *marker* is that reclamation
/// reads it back. [`ENV_SESSION`] alone says "some mirage started this",
/// which is not enough on a machine where two mirages are running under
/// different runtime directories: the other one's sessions are not in
/// this one's registry of live runs, so without this variable they look
/// exactly like the wreckage of a crash. See [`crate::reclaim`], and
/// [`LABEL_RUNTIME`] for the container-side counterpart.
pub const ENV_RUNTIME: &str = "MIRAGE_RUNTIME";

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

/// RCCL/NCCL network interface selection. Accepts a comma-separated list
/// of interface names, a `=` prefix for an exact match, or a `^` prefix
/// for exclusion (`^lo` is "anything but loopback").
///
/// Named here because a session's *shape* decides whether the emulator's
/// own answer to this can stand. An emulator that models no fabric pins
/// it to `lo`, which is right for every rank running as a process on one
/// host — and wrong the moment the ranks are in separate containers,
/// where the peer a rank dials is a name on the session's bridge network
/// and loopback cannot reach it. See
/// `mirage_supervisor::spec` for where that override is applied.
pub const ENV_NCCL_SOCKET_IFNAME: &str = "NCCL_SOCKET_IFNAME";

/// The loopback-only value an emulator pins [`ENV_NCCL_SOCKET_IFNAME`]
/// to for a job whose ranks all share a host.
pub const NCCL_IFNAME_LOOPBACK: &str = "lo";

/// [`ENV_NCCL_SOCKET_IFNAME`] for ranks that must reach each other over
/// a container network: everything except loopback.
///
/// Spelled as an exclusion rather than as an interface name because the
/// name is the engine's to choose — `eth0` under a netavark or docker
/// bridge, something else under another network backend — while "not the
/// interface that cannot reach another container" is true whatever it
/// picked.
pub const NCCL_IFNAME_NOT_LOOPBACK: &str = "^lo";

/// The directory inside every node container that belongs to mirage.
///
/// A node container is not only the user's image. Mirage bind-mounts its
/// own binary, the session's scratch directory, the config directory and
/// the emulator's libraries into it, and every one of those lands under
/// this prefix so that exactly one path inside the container is spoken
/// for. The individual destinations are spelled out by
/// `mirage_supervisor::session`, which builds them from this root.
///
/// A *user* mount at or above this path is therefore a collision, and
/// [`covers_mirage_dir`] is the test for it.
pub const CONTAINER_MIRAGE_DIR: &str = "/mnt/mirage";

/// Whether a bind mount at `container_path` would be laid over
/// [`CONTAINER_MIRAGE_DIR`].
///
/// True for the reserved directory itself and for every ancestor of it
/// (`/mnt`, `/`): those are the mounts that put a host directory
/// *underneath* mirage's own, which is how a user's `--mount` target came
/// back holding root-owned `bin`, `config`, `lib` and `runtime` entries
/// the container wrote through it.
///
/// A path strictly *below* the reserved directory is deliberately not
/// this function's business: everything mirage itself mounts is below it,
/// so a test that answered "yes" there could not tell mirage's own mounts
/// from a user's and would refuse every containerised session.
///
/// # The spelling is not the path
///
/// Both sides are reduced to their components first, by
/// `normalise_container_path`, because a mount destination has many
/// spellings and the engine acts on the location rather than on the
/// text. `/mnt/mirage/../mirage` names the reserved directory and used
/// to pass this test — the comparison was textual-by-component, `..` is
/// a component like any other, and the original bug came straight back
/// through a path that merely looked like it was somewhere else.
#[must_use]
pub fn covers_mirage_dir(container_path: &str) -> bool {
    // An empty container path is not a mount over anything; it is also
    // not a path, and normalising it would make it the root.
    if container_path.is_empty() {
        return false;
    }
    let reserved = normalise_container_path(CONTAINER_MIRAGE_DIR);
    reserved.starts_with(&normalise_container_path(container_path))
}

/// Whether two container paths name overlapping locations.
///
/// True when they are the same location, and when either is a directory
/// prefix of the other. Both are the cases that matter for a bind mount:
/// mounting at a path *above* another mount buries it, and mounting at a
/// path *below* one overlays a piece of it — and the second is the one
/// that looks harmless.
///
/// Component-wise via `normalise_container_path`, so `/mnt/mirage` and
/// `/mnt/mirage/../mirage/runtime` overlap and `/mnt/mirage` and
/// `/mnt/mirage-of-my-own` do not. A textual `starts_with` would get the
/// second wrong in the direction that refuses honest mounts.
#[must_use]
pub fn container_paths_overlap(a: &str, b: &str) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let (a, b) = (normalise_container_path(a), normalise_container_path(b));
    a.starts_with(&b) || b.starts_with(&a)
}

/// The components a container path names, with `.`, `..`, empty segments
/// and any trailing separator resolved away.
///
/// Purely lexical: the path names a location inside a container that
/// does not exist yet, so there is no filesystem to ask and no symlink
/// to follow. That is also why `..` is resolved by dropping the
/// preceding component rather than by resolving the parent — the two
/// differ only through symlinks, and the engine's own mount destination
/// handling is lexical in the same way.
///
/// A path with no leading separator is read from the root as well. A
/// mount destination is always an absolute location to the engine (both
/// refuse a relative one outright), so `mnt/mirage` describes the
/// reserved directory just as `/mnt/mirage` does, and reading it any
/// other way would let a relative spelling walk past the check on its
/// way to being refused for an unrelated reason.
///
/// `..` above the root is dropped, as it is by the kernel: `/..` is `/`.
fn normalise_container_path(path: &str) -> Vec<&str> {
    let mut components: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                components.pop();
            }
            name => components.push(name),
        }
    }
    components
}

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

/// Label recording which mirage runtime directory created a resource.
///
/// [`LABEL_OWNER`] says "a mirage made this" and [`LABEL_SESSION`] says
/// which session, but neither says *which* mirage — and two mirages
/// running under different `MIRAGE_RUNTIME` directories keep separate
/// registries of what is live. Each therefore sees the other's sessions
/// as belonging to no live run at all, which is indistinguishable from
/// the wreckage this module's reclamation exists to remove. This label is
/// what tells them apart; it is the resource-side counterpart of
/// [`ENV_RUNTIME`].
pub const LABEL_RUNTIME: &str = "mirage.runtime";

/// The labels every resource belonging to `session` carries.
#[must_use]
pub fn owner_labels(session: &SessionId) -> Vec<(String, String)> {
    vec![
        (LABEL_OWNER.to_string(), LABEL_OWNER_VALUE.to_string()),
        (LABEL_SESSION.to_string(), session.as_str().to_string()),
        (LABEL_RUNTIME.to_string(), owning_runtime()),
    ]
}

/// The labels stamped on a derived image mirage builds.
///
/// [`LABEL_SESSION`] is deliberately absent, and its absence is the
/// difference between an image and a container: a derived image is
/// keyed by its base plus the hacks applied to it, built once, and then
/// reused by every later session that asks for the same combination.
/// Stamping it with the session that happened to build it first would
/// name an owner that has been gone for days.
///
/// What it does carry is the same two marks everything else mirage
/// creates does — that a mirage made it, and which runtime directory's
/// mirage that was — because an image is host state like any other and
/// the alternative is an untagged `mirage-hack-…` nobody can attribute.
#[must_use]
pub fn image_labels() -> Vec<(String, String)> {
    vec![
        (LABEL_OWNER.to_string(), LABEL_OWNER_VALUE.to_string()),
        (LABEL_RUNTIME.to_string(), owning_runtime()),
    ]
}

/// The runtime directory to record on everything this process creates.
///
/// Resolved rather than copied out of the environment: `MIRAGE_RUNTIME`
/// is one of three ways the directory can be named (see
/// [`crate::paths::mirage_runtime_dir`]), and a user who set none of them
/// would otherwise leave the marker empty. Canonicalised where possible
/// so that `/tmp/rt`, `/tmp/rt/` and a symlink to either all record the
/// same string; where it is not — the directory does not exist yet — an
/// absolute path is the next best thing, because a relative one means
/// something different to every process that reads it.
#[must_use]
pub fn owning_runtime() -> String {
    let dir = crate::paths::mirage_runtime_dir();
    dir.canonicalize()
        .or_else(|_| std::path::absolute(&dir))
        .unwrap_or(dir)
        .to_string_lossy()
        .into_owned()
}

/// Whether a recorded runtime marker names the same directory as `ours`.
///
/// Both sides are canonicalised first, so a resource stamped before its
/// runtime directory was resolvable still matches one scanned after, and
/// a symlinked or trailing-slashed path matches the path it names.
///
/// Canonicalisation fails when a path no longer exists — a purged runtime
/// directory, a tempdir the test that made it has removed — and that must
/// not be read as agreement, because *every* removed path would then
/// match every other. The fallback is a literal comparison of the two
/// paths, which is exact: it can only report a difference that is not
/// really there, and the consequence of that is a resource left alone,
/// never one destroyed.
#[must_use]
pub fn same_runtime(recorded: &str, ours: &str) -> bool {
    let recorded = std::path::Path::new(recorded);
    let ours = std::path::Path::new(ours);
    match (recorded.canonicalize(), ours.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        // `Path`'s equality is by component, so this already ignores a
        // trailing separator and a doubled one.
        _ => recorded == ours,
    }
}

/// Read one label off a container, or `None` if it is absent.
fn container_label(provider: &str, name: &str, label: &str) -> Option<String> {
    inspect_label(
        provider,
        &["inspect"],
        name,
        &format!("{{{{index .Config.Labels {label:?}}}}}"),
    )
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

/// Whether the provider positively reports that `name` belongs to
/// somebody else.
///
/// The distinction from `!container_is_ours` is the whole point: an
/// *unanswerable* question is not an answer. `inspect_label` folds a
/// provider that could not be spawned, one that exited non-zero, and a
/// container that no longer exists all into `None`, so treating "not
/// ours" as "leave it alone" turns a transient engine failure during
/// shutdown — a restarting rootless service, a process at its fork limit
/// — into containers and a network that survive the run silently.
///
/// Removal is therefore gated on a *positive* foreign label. The safety
/// property that gate exists for is preserved: a container a user named
/// `mirage-s1-node-0` themselves carries no mirage owner label, so it
/// still has to be labelled by somebody else to be protected — and it is
/// only ever consulted for names this session recorded at bring-up.
#[must_use]
pub fn container_is_foreign(provider: &str, name: &str) -> bool {
    matches!(container_label(provider, name, LABEL_OWNER), Some(v) if v != LABEL_OWNER_VALUE)
}

/// Whether the provider positively reports that network `name` belongs to
/// somebody else. See [`container_is_foreign`].
#[must_use]
pub fn network_is_foreign(provider: &str, name: &str) -> bool {
    matches!(network_label(provider, name, LABEL_OWNER), Some(v) if v != LABEL_OWNER_VALUE)
}

/// Environment variable naming the container engine to use.
///
/// Read as a *default*: it outranks autodetection but not a provider
/// something asked for by name. See [`resolve_provider`] for the whole
/// order and why it is that way round.
pub const ENV_PROVIDER: &str = "MIRAGE_CONTAINER_PROVIDER";

/// The container engines mirage knows how to drive, in the order it
/// prefers them. podman first: it is rootless by default, so a session
/// started by an ordinary user does not need a daemon socket they may
/// not be allowed to open.
const KNOWN_PROVIDERS: [&str; 2] = ["podman", "docker"];

/// Auto-detect a container provider on `PATH`, preferring podman.
///
/// Returns the provider's bare name (`"podman"` / `"docker"`) when
/// found, or `None` if neither is installed.
pub fn detect_provider() -> Option<String> {
    KNOWN_PROVIDERS
        .into_iter()
        .find(|candidate| which_on_path(candidate).is_some())
        .map(str::to_string)
}

/// Every container engine installed here, not just the one a new session
/// would use.
///
/// [`detect_provider`] answers "which engine should this session run
/// on?" and answers it with the first match, which is the wrong question
/// for anything that has to *find* what earlier sessions left behind: a
/// session created with `--container-provider docker` on a host that also
/// has podman leaves containers a sweep that only asked podman never sees
/// — and then reports success, which is the one answer that stops a user
/// from looking.
///
/// [`ENV_PROVIDER`] still names one engine deliberately and is then the
/// only candidate, which is how a test (or a user with a wrapper script)
/// points a sweep at something that is not on the usual list at all.
///
/// This does not see an engine named only by an absolute path inside a
/// profile: nothing here reads profiles. A caller that has one should add
/// it to what this returns.
#[must_use]
pub fn installed_providers() -> Vec<String> {
    if let Ok(p) = std::env::var(ENV_PROVIDER)
        && !p.is_empty()
    {
        return vec![p];
    }
    KNOWN_PROVIDERS
        .into_iter()
        .filter(|engine| which_on_path(engine).is_some())
        .map(str::to_string)
        .collect()
}

/// Where a provider name or path resolves to a file, if anywhere.
///
/// A name is looked up on `PATH`; anything containing a separator is
/// taken as the path it plainly is. Exposed so the engine can tell "you
/// named something that is not there" from "the engine is there and
/// refused", which are different errors with different fixes.
#[must_use]
pub fn provider_binary(name: &str) -> Option<std::path::PathBuf> {
    which_on_path(name)
}

/// Resolve the provider to use, in priority order:
///
/// 1. an explicit value (`"podman"`, `"docker"`, or a path) — either
///    `--container-provider` or the provider pinned on the profile,
///    which reach this function as the same argument,
/// 2. the [`ENV_PROVIDER`] environment default,
/// 3. auto-detection ([`detect_provider`]).
///
/// Returns `None` only when no provider was specified and none could be
/// detected.
///
/// # Why the environment variable does not outrank the profile
///
/// The conventional order is flag, then environment, then configuration
/// file — and a profile is a configuration file, so putting the variable
/// second looks backwards. The reason it is not is that mirage cannot
/// tell those two apart by the time it gets here: `--container-provider`
/// is applied to the profile before the session is built, so a
/// profile-pinned provider and a flag-given one are the same string in
/// the same field. Ranking the variable above it would therefore let an
/// exported `MIRAGE_CONTAINER_PROVIDER` silently beat a flag the user
/// typed on the command line, which is a worse surprise than the one it
/// would fix.
///
/// So the variable is what its name in `docs/cli.md` says: the default
/// when nothing names an engine. What it must not be is *silent* about
/// it, which is what a set-but-unused variable used to be — hence the
/// warning below, naming both values and which one won.
pub fn resolve_provider(explicit: Option<&str>) -> Option<String> {
    let from_env = std::env::var(ENV_PROVIDER).ok().filter(|p| !p.is_empty());
    if let Some(p) = explicit
        && !p.is_empty()
    {
        if let Some(env) = from_env.filter(|env| env != p) {
            tracing::warn!(
                "{ENV_PROVIDER}={env} was ignored: this session names the container \
                 provider `{p}` (on its profile, or with --container-provider), and that \
                 wins over the environment. Pass `--container-provider {env}` to use it \
                 for this run."
            );
        }
        return Some(p.to_string());
    }
    from_env.or_else(detect_provider)
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
/// name alone would then destroy it. Skipping anything positively
/// labelled as somebody else's makes cleanup safe to run against a shared
/// engine — see [`container_is_foreign`] for why the check is phrased
/// that way round rather than as "only what is ours".
///
/// This blocks on the provider binary, so async callers must run it on a
/// blocking task.
pub fn teardown(state: &ContainerState) {
    for node in &state.nodes {
        if container_is_foreign(&state.provider, &node.name) {
            tracing::warn!(
                container = %node.name,
                "refusing to remove a container labelled as somebody else's"
            );
        } else {
            run_quiet(&state.provider, &["rm", "-f", &node.name]);
        }
    }
    if let Some(network) = &state.network {
        if network_is_foreign(&state.provider, network) {
            tracing::warn!(
                network = %network,
                "refusing to remove a network labelled as somebody else's"
            );
        } else {
            run_quiet(&state.provider, &["network", "rm", network]);
        }
    }
}

/// One mirage-owned container or network whose session is not live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orphan {
    /// The resource's id or name.
    pub name: String,
    /// The session it was created for.
    pub session: String,
    /// Whether it is a network rather than a container.
    pub is_network: bool,
}

/// Every container and network this runtime directory created whose
/// session is not in `live`, containers first.
///
/// Split out from [`reclaim_orphans`] so `mirage cleanup --dry-run` can
/// name what it would remove without removing it, and so the ordering —
/// containers before the network they are attached to — is stated once.
///
/// Filtering by [`LABEL_OWNER`] is what keeps this safe on an engine
/// shared with other work: a container mirage did not create is never a
/// candidate, whatever it is called. Filtering by [`LABEL_RUNTIME`] is
/// what keeps it safe on an engine shared with *another mirage*: `live`
/// is this runtime directory's registry of live runs, so a container
/// belonging to a healthy session of some other runtime directory would
/// otherwise be an orphan by every test applied here.
///
/// Blocks on the provider binary.
#[must_use]
pub fn orphans(provider: &str, live: &[SessionId]) -> Vec<Orphan> {
    let live: std::collections::HashSet<&str> = live.iter().map(SessionId::as_str).collect();
    let ours = owning_runtime();
    let mut found = Vec::new();
    for (verb, is_network) in [
        (&["ps", "--all"][..], false),
        (&["network", "ls"][..], true),
    ] {
        for resource in labelled(provider, verb) {
            if live.contains(resource.session.as_str()) {
                continue;
            }
            // An unlabelled runtime cannot be attributed to anybody, and
            // removing what cannot be attributed is the failure mode this
            // filter exists to prevent — so it is left alone. The cost is
            // that a container created by a mirage older than this label
            // is never reclaimed by this one; the alternative is
            // destroying a healthy container belonging to somebody else,
            // which is strictly worse.
            if !resource
                .runtime
                .is_some_and(|recorded| same_runtime(&recorded, &ours))
            {
                continue;
            }
            found.push(Orphan {
                name: resource.name,
                session: resource.session,
                is_network,
            });
        }
    }
    found
}

/// Remove each of `orphans`, best-effort.
///
/// Takes the list rather than re-deriving it, so a caller that has
/// already reported what it found removes exactly that — see
/// [`crate::reclaim::reap`], which is the same split for processes.
///
/// No ownership re-check: [`orphans`] filters on [`LABEL_OWNER`] at the
/// engine, so nothing that reaches here was somebody else's.
///
/// Blocks on the provider binary.
pub fn remove_orphans(provider: &str, orphans: &[Orphan]) {
    for orphan in orphans {
        if orphan.is_network {
            run_quiet(provider, &["network", "rm", &orphan.name]);
        } else {
            run_quiet(provider, &["rm", "-f", &orphan.name]);
        }
    }
}

/// Find and remove every mirage-owned container and network whose session
/// is not in `live`, returning what was removed.
///
/// This is the recovery path for a supervisor that died without tearing
/// its sessions down. Session state is deliberately in-memory, so a
/// `SIGKILL`ed run leaves containers with no record anywhere that a later
/// mirage could read — except the resources themselves, which carry
/// [`LABEL_SESSION`] and [`LABEL_RUNTIME`]. See [`crate::reclaim`] for the
/// same idea applied to the workload processes such a run also strands.
///
/// Blocks on the provider binary.
pub fn reclaim_orphans(provider: &str, live: &[SessionId]) -> Vec<String> {
    let found = orphans(provider, live);
    remove_orphans(provider, &found);
    found.into_iter().map(|o| o.name).collect()
}

/// One mirage-owned resource, as a listing plus its labels describe it.
struct Labelled {
    /// The resource's id.
    name: String,
    /// The session it was created for ([`LABEL_SESSION`]).
    session: String,
    /// The runtime directory that created it ([`LABEL_RUNTIME`]), absent
    /// on anything made before that label existed.
    runtime: Option<String>,
}

/// Every mirage-owned resource the given listing verb reports, with the
/// labels that say who it belongs to.
///
/// The owner filter is applied by the engine rather than by us, so a
/// resource without the label is never even named here.
fn labelled(provider: &str, verb: &[&str]) -> Vec<Labelled> {
    let filter = format!("label={LABEL_OWNER}={LABEL_OWNER_VALUE}");
    // `{{.ID}}` only, and the session label read back per resource with
    // `inspect`.
    //
    // The listing verbs are where the two providers' Go templates diverge:
    // `podman ps` exposes `.Name` and a `.Labels` *map*, while `docker ps`
    // exposes `.Names` and renders `.Labels` as a comma-joined *string* —
    // so `{{index .Labels "…"}}` makes docker exit non-zero, `labelled`
    // returns nothing, and `reclaim_orphans` silently reclaims nothing at
    // all on a docker host. `.ID` is common to both listing verbs, and
    // `inspect` (which already backs `container_is_ours`) exposes the
    // labels as a map on both.
    let Ok(out) = retrying_etxtbsy(|| {
        Command::new(provider)
            .args(verb)
            .arg("--filter")
            .arg(&filter)
            .arg("--format")
            .arg("{{.ID}}")
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
    }) else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let network = verb.first() == Some(&"network");
    let label = |id: &str, label: &str| {
        if network {
            network_label(provider, id, label)
        } else {
            container_label(provider, id, label)
        }
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let id = line.trim();
            if id.is_empty() {
                return None;
            }
            let session = label(id, LABEL_SESSION)?;
            if session.is_empty() || session == "<no value>" {
                return None;
            }
            Some(Labelled {
                name: id.to_string(),
                session,
                runtime: label(id, LABEL_RUNTIME),
            })
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

/// Whether a spawn failure is worth trying again.
///
/// All three of these say "not right now" rather than "not ever":
///
/// * `ExecutableFileBusy` — the executable was still open for writing
///   when we tried to exec it. Never seen with a real podman/docker
///   install, routine for a freshly-written wrapper script.
/// * `WouldBlock` (`EAGAIN`) — `fork` hit the process or thread limit.
/// * `OutOfMemory` (`ENOMEM`) — the kernel could not set up the new
///   process right now.
///
/// The last two are what a heavily loaded machine produces, and they are
/// worth retrying here for the same reason as the first: giving up
/// silently skips whatever the command was going to do, which for an
/// ownership probe means concluding "not ours" about a container that
/// *is* ours and then declining to remove it. A leak is a much worse
/// outcome than a few milliseconds of backoff.
fn worth_retrying(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::ExecutableFileBusy
            | std::io::ErrorKind::WouldBlock
            | std::io::ErrorKind::OutOfMemory
    )
}

/// Run a provider command, retrying transient spawn failures.
///
/// See `worth_retrying` for which failures are treated as transient and
/// why the bias is towards retrying. Generic over the result so the same
/// policy covers both `output()` and `spawn()`; `mirage_container` calls
/// it for the latter rather than keeping a second copy that had already
/// drifted to a narrower predicate.
pub fn retrying_etxtbsy<T, F>(mut run: F) -> std::io::Result<T>
where
    F: FnMut() -> std::io::Result<T>,
{
    const MAX_ATTEMPTS: u32 = 50;
    const BACKOFF: std::time::Duration = std::time::Duration::from_millis(10);

    let mut attempts = 0;
    loop {
        match run() {
            Err(e) if worth_retrying(&e) && attempts < MAX_ATTEMPTS => {
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

/// Holds the runtime directory still for the duration of a test.
///
/// Both [`owning_runtime`] and everything that compares against it read
/// the process-wide directory resolution, and other tests in this crate
/// move that resolution under a temporary root while they run
/// ([`crate::paths::set_test_root`]). A resource stamped before one of
/// those starts would be compared against a different directory
/// afterwards — the comparison behaving exactly as designed, and the test
/// measuring nothing. Holding [`crate::paths::test_env_lock`] excludes
/// them for as long as the guard lives.
///
/// Lives outside the test modules because both this module's and
/// [`crate::reclaim`]'s need it.
#[cfg(test)]
// A test fixture: a failure to set one up is a failed test, and there is
// nobody to return an error to.
#[allow(clippy::unwrap_used)]
pub(crate) struct PinnedRuntime {
    _lock: std::sync::MutexGuard<'static, ()>,
    _root: tempfile::TempDir,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
impl PinnedRuntime {
    pub(crate) fn new() -> Self {
        let lock = crate::paths::test_env_lock();
        let root = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(root.path());
        // Materialised, so the marker is stamped and compared as a
        // canonical path — the case that runs in production.
        std::fs::create_dir_all(crate::paths::mirage_runtime_dir()).unwrap();
        Self {
            _lock: lock,
            _root: root,
        }
    }
}

#[cfg(test)]
impl Drop for PinnedRuntime {
    fn drop(&mut self) {
        // Before the tempdir goes, so nothing left behind resolves paths
        // under a directory that no longer exists.
        crate::paths::clear_test_root();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

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
    fn every_installed_engine_is_a_candidate_and_not_just_the_first() {
        // `detect_provider` stops at the first match, which is right for
        // "what should this session run on" and wrong for anything that
        // has to find what an earlier session left: on a host with both
        // engines, a docker session's containers are invisible to a sweep
        // that only ever asked podman.
        //
        // `MIRAGE_CONTAINER_PROVIDER` overrides the whole list, and a
        // test cannot set an environment variable in a workspace that
        // forbids `unsafe` — so when the machine running the suite has
        // one exported, only the shape of the answer is checked.
        let candidates = installed_providers();
        assert!(candidates.iter().all(|c| !c.is_empty()), "{candidates:?}");
        if std::env::var_os(ENV_PROVIDER).is_some() {
            assert_eq!(candidates.len(), 1, "{candidates:?}");
            return;
        }
        assert!(
            candidates.iter().all(|c| provider_binary(c).is_some()),
            "an engine that is not installed was named a candidate: {candidates:?}"
        );
        match detect_provider() {
            Some(first) => {
                assert_eq!(candidates.first(), Some(&first));
                assert!(candidates.contains(&first), "{candidates:?}");
            }
            None => assert!(candidates.is_empty(), "{candidates:?}"),
        }
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
    fn owner_labels_name_the_session_and_the_runtime_that_made_it() {
        let _runtime = PinnedRuntime::new();
        let labels = owner_labels(&SessionId::new("s1").unwrap());
        assert_eq!(
            labels,
            vec![
                (LABEL_OWNER.to_string(), LABEL_OWNER_VALUE.to_string()),
                (LABEL_SESSION.to_string(), "s1".to_string()),
                (LABEL_RUNTIME.to_string(), owning_runtime()),
            ]
        );
    }

    #[test]
    fn the_recorded_runtime_is_absolute() {
        // A relative path names a different directory to every process
        // that reads it back, and the readers are `mirage cleanup`
        // invocations started from wherever the user happened to be.
        let _runtime = PinnedRuntime::new();
        assert!(Path::new(&owning_runtime()).is_absolute());

        // And again for a directory that cannot be canonicalised because
        // it is not there — the `std::path::absolute` fallback, which is
        // the branch this test's own comment is about and the one that
        // runs on the first containerised run under a fresh
        // `MIRAGE_RUNTIME`. A *relative* one, because that is the only
        // shape where the fallback changes the answer: it is what a
        // `MIRAGE_RUNTIME=rt` in somebody's shell profile resolves to,
        // and recording it verbatim would name a different directory to
        // every `mirage cleanup` started from somewhere else. Nothing is
        // created for it; the test lock this fixture holds is what keeps
        // the process-wide root ours to move.
        crate::paths::set_test_root(Path::new("mirage-runtime-that-is-not-there"));
        assert!(
            Path::new(&owning_runtime()).is_absolute(),
            "a runtime directory that does not exist yet was recorded as \
             the relative path it was written as: {}",
            owning_runtime()
        );
    }

    #[test]
    fn a_mount_over_mirages_own_directory_is_recognised() {
        // The reserved tree itself and everything above it: those are the
        // mounts that put a user's host directory underneath the binary,
        // config and scratch mounts mirage adds to every node container.
        assert!(covers_mirage_dir(CONTAINER_MIRAGE_DIR));
        assert!(covers_mirage_dir("/mnt/mirage/"));
        assert!(covers_mirage_dir("/mnt//mirage"));
        assert!(covers_mirage_dir("/mnt/./mirage"));
        assert!(covers_mirage_dir("/mnt"));
        assert!(covers_mirage_dir("/"));

        // Everything else is the user's to mount, including the paths
        // *below* the reserved tree — mirage's own mounts are all down
        // there, so a check that caught them would refuse every
        // containerised session there is.
        assert!(!covers_mirage_dir("/mnt/mirage/lib"));
        assert!(!covers_mirage_dir("/mnt/mine"));
        assert!(!covers_mirage_dir("/data"));
        assert!(!covers_mirage_dir(""));
    }

    #[test]
    fn an_overlapping_mount_is_recognised_in_both_directions() {
        // Above: the case `covers_mirage_dir` already had. A mount at
        // `/mnt` buries everything mirage put under `/mnt/mirage`.
        assert!(container_paths_overlap("/mnt", "/mnt/mirage/runtime"));
        // Below: the case it could not have, and the one that reads as
        // harmless. `covers_mirage_dir` answers "no" for every path under
        // the reserved tree, because mirage's own mounts are all down
        // there — so a user mount at
        // `/mnt/mirage/runtime/rj_config.json` passed it, overlaid the
        // emulator's config inside the scratch mount, and left the
        // workload running unemulated against the real device.
        assert!(container_paths_overlap(
            "/mnt/mirage/runtime/rj_config.json",
            "/mnt/mirage/runtime"
        ));
        // The same location, however it is spelled.
        assert!(container_paths_overlap("/mnt/mirage", "/mnt/mirage/"));
        assert!(container_paths_overlap(
            "/mnt/mirage",
            "/mnt/mirage/../mirage"
        ));

        // A shared textual prefix is not a shared location, and a check
        // that confused the two would refuse honest mounts.
        assert!(!container_paths_overlap(
            "/mnt/mirage",
            "/mnt/mirage-of-mine"
        ));
        assert!(!container_paths_overlap("/data", "/mnt/mirage"));
        // An empty path is not a mount over anything.
        assert!(!container_paths_overlap("", "/mnt/mirage"));
    }

    #[test]
    fn a_mount_that_reaches_the_reserved_directory_through_dot_dot_is_recognised() {
        // The check went in to stop a `--mount` from being laid over
        // mirage's own tree, and a reviewer walked the original bug back
        // through it with a `..`: the comparison was by literal
        // component, so `/mnt/mirage/../mirage` was two components
        // longer than the reserved path and read as somewhere else
        // entirely — while naming exactly the directory the check
        // exists to protect.
        assert!(covers_mirage_dir("/mnt/mirage/../mirage"));
        assert!(covers_mirage_dir("/mnt/mirage/lib/.."));
        assert!(covers_mirage_dir("/mnt/mirage/lib/../.."));
        assert!(covers_mirage_dir("/data/../mnt/mirage"));
        // `..` past the root is the root, which is the widest cover
        // there is.
        assert!(covers_mirage_dir("/mnt/../.."));
        assert!(covers_mirage_dir("/.."));
        assert!(covers_mirage_dir("/data/.."));

        // The other spellings of the same path, each of which a provider
        // resolves before it mounts anything.
        assert!(covers_mirage_dir("/mnt/mirage/"));
        assert!(covers_mirage_dir("/mnt//mirage"));
        assert!(covers_mirage_dir("/mnt/./mirage"));
        assert!(covers_mirage_dir("//mnt/mirage//"));
        assert!(covers_mirage_dir("/mnt/."));

        // A destination with no leading separator is read from the root
        // as well: it is the only reading under which it names a
        // location at all, and it is the reading that keeps the check
        // from being sidestepped by dropping one character.
        assert!(covers_mirage_dir("mnt/mirage"));
        assert!(covers_mirage_dir("mnt/mirage/../mirage"));

        // And a `..` that genuinely leads somewhere else is still the
        // user's to mount, including back below the reserved tree, where
        // mirage's own mounts live.
        assert!(!covers_mirage_dir("/mnt/mirage/../mine"));
        assert!(!covers_mirage_dir("/mnt/mirage/lib/../lib"));
    }

    #[test]
    fn a_derived_image_is_marked_as_mirages_but_not_as_a_sessions() {
        let _runtime = PinnedRuntime::new();
        assert_eq!(
            image_labels(),
            vec![
                (LABEL_OWNER.to_string(), LABEL_OWNER_VALUE.to_string()),
                (LABEL_RUNTIME.to_string(), owning_runtime()),
            ],
            "a derived image outlives the session that built it, so it \
             must be attributable without naming one"
        );
    }

    #[test]
    fn spellings_of_one_directory_all_compare_equal() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("rt");
        std::fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let canonical = real.to_string_lossy().into_owned();
        assert!(same_runtime(&canonical, &canonical));
        assert!(same_runtime(&format!("{canonical}/"), &canonical));
        assert!(same_runtime(&format!("{canonical}//"), &canonical));
        assert!(same_runtime(&link.to_string_lossy(), &canonical));
        assert!(!same_runtime(
            &dir.path().join("other").to_string_lossy(),
            &canonical
        ));
    }

    #[test]
    fn a_runtime_that_no_longer_exists_matches_only_itself() {
        // Canonicalisation fails for a directory that has been removed —
        // a purged runtime, a tempdir the run that owned it took with it.
        // Folding that into "matches" would make every removed path equal
        // to every other, which is the cross-runtime kill this comparison
        // exists to prevent.
        let gone = "/tmp/mirage-no-such-runtime-a";
        let also_gone = "/tmp/mirage-no-such-runtime-b";
        assert!(same_runtime(gone, gone));
        assert!(!same_runtime(gone, also_gone));
    }

    /// A provider whose listings report one resource of each kind
    /// reclamation has to tell apart.
    ///
    /// The owner filter is applied by the engine, so everything listed is
    /// already mirage's; what distinguishes these is the session and
    /// runtime labels, which the listing does not carry. It yields ids
    /// only and the labels are read back per resource with `inspect` —
    /// the one shape podman and docker agree on.
    fn mock_reclaim_provider(dir: &Path, log: &Path) -> String {
        let provider = dir.join("mock-provider.sh");
        std::fs::write(
            &provider,
            format!(
                "#!/bin/sh\n\
                 echo \"$@\" >> {log}\n\
                 for a in \"$@\"; do last=$a; done\n\
                 case \"$1 $2\" in\n\
                 \"ps --all\") printf 'mirage-live-node-0\\nmirage-dead-node-0\\n\
                 mirage-elsewhere-node-0\\nmirage-unmarked-node-0\\n' ;;\n\
                 \"network ls\") printf 'mirage-dead\\nmirage-elsewhere\\n' ;;\n\
                 \"inspect --format\"|\"network inspect\")\n\
                 case \"$*\" in\n\
                 *{runtime_label}*)\n\
                 case \"$last\" in\n\
                 mirage-unmarked-node-0) printf '<no value>' ;;\n\
                 mirage-elsewhere-node-0|mirage-elsewhere) echo /somewhere/else ;;\n\
                 *) echo '{ours}' ;;\n\
                 esac ;;\n\
                 *)\n\
                 case \"$last\" in\n\
                 mirage-live-node-0) echo live ;;\n\
                 mirage-dead-node-0|mirage-dead) echo dead ;;\n\
                 mirage-elsewhere-node-0|mirage-elsewhere) echo elsewhere ;;\n\
                 mirage-unmarked-node-0) echo unmarked ;;\n\
                 esac ;;\n\
                 esac ;;\n\
                 esac\n\
                 exit 0\n",
                log = log.display(),
                runtime_label = LABEL_RUNTIME,
                ours = owning_runtime(),
            ),
        )
        .unwrap();
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();
        provider.to_string_lossy().into_owned()
    }

    #[test]
    fn reclaim_removes_only_orphans_and_only_ours() {
        // The recovery path after a supervisor died without tearing down:
        // the only surviving record of a container is the label on the
        // container itself.
        let _runtime = PinnedRuntime::new();
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("provider.log");
        let provider = mock_reclaim_provider(dir.path(), &log);

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
    fn a_resource_of_another_runtime_directory_is_never_an_orphan() {
        // The session belongs to a mirage running under a different
        // `MIRAGE_RUNTIME`, so it is absent from *this* runtime's list of
        // live runs — and every other test here would call it an orphan.
        // It is somebody's healthy session.
        let _runtime = PinnedRuntime::new();
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("provider.log");
        let provider = mock_reclaim_provider(dir.path(), &log);

        let found = orphans(&provider, &[]);
        assert!(
            !found.iter().any(|o| o.name.contains("elsewhere")),
            "a resource created by another runtime directory was reported: {found:?}"
        );

        reclaim_orphans(&provider, &[]);
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            !recorded.contains("rm -f mirage-elsewhere-node-0"),
            "another runtime directory's container was removed:\n{recorded}"
        );
        assert!(
            !recorded.contains("network rm mirage-elsewhere"),
            "another runtime directory's network was removed:\n{recorded}"
        );
    }

    #[test]
    fn a_resource_with_no_runtime_label_is_left_alone() {
        // Made by a mirage older than `mirage.runtime`, so nothing says
        // which runtime directory owns it. Unattributable is not the same
        // as unowned, and this one is skipped: the cost is a container
        // that has to be removed by hand, against the alternative of
        // destroying a container that some other mirage is using.
        let _runtime = PinnedRuntime::new();
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("provider.log");
        let provider = mock_reclaim_provider(dir.path(), &log);

        let found = orphans(&provider, &[]);
        assert!(
            !found.iter().any(|o| o.name.contains("unmarked")),
            "an unattributable resource was reported as an orphan: {found:?}"
        );

        reclaim_orphans(&provider, &[]);
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            !recorded.contains("rm -f mirage-unmarked-node-0"),
            "an unattributable container was removed:\n{recorded}"
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
