//! Container engine: drives the `docker`/`podman` CLI to realise the
//! containerised parts of a mirage session.
//!
//! The *static* configuration ([`mirage_core::profile::ContainerizedDef`],
//! [`mirage_core::profile::FileMount`]) and the *runtime* record
//! ([`mirage_core::container::ContainerState`], naming + provider
//! resolution, and dependency-free [`teardown`](mirage_core::container::teardown))
//! live in `mirage_core`. This crate adds the imperative orchestration:
//! pulling images, creating the per-session network, launching one
//! container per node, and building the `exec` argv used to run a
//! command inside a node's container.
//!
//! The design keeps a clean split:
//!
//! * **argv builders** ([`Engine::run_argv`], [`Engine::exec_argv`]) are
//!   pure functions of their inputs and fully unit-tested without a real
//!   runtime.
//! * **side-effecting methods** ([`Engine::pull`], [`Engine::ensure_network`],
//!   [`Engine::launch_node`], …) invoke the provider and are exercised in
//!   tests with a mock provider shell script.

use std::process::{Command, Stdio};

use mirage_core::container::{ContainerState, NodeContainer};
use mirage_core::profile::{ContainerizedDef, FileMount, PortMapping};

/// Foreground process of a node container.
///
/// The container needs a PID 1 that simply stays alive: workloads are
/// started from outside with `provider exec`, so the container's own
/// entrypoint has no work to do. An earlier design ran
/// `mirage host --rank N` here, which meant every container had a second
/// mirage process inside it polling a bind-mounted directory.
pub const CONTAINER_IDLE_COMMAND: &[&str] = &["sleep", "infinity"];

/// How long a node container gets to report itself running before
/// bring-up gives up on it.
///
/// Generous: the image is already pulled by this point, but a cold
/// container runtime on a loaded machine can still take seconds to set up
/// namespaces, mounts and the network.
pub const NODE_START_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// How long a node container has to keep running, after it has reported
/// itself up, before bring-up believes it.
///
/// A container that stops inside this window never really started, and
/// without the wait mirage reports that in the worst possible way. The
/// provider answers `Running: true` while the image's entrypoint is
/// still dying — a `sleep` the image's shell cannot spell, a musl image
/// handed a glibc `LD_PRELOAD` it cannot relocate — so bring-up
/// succeeds, the session goes ready, and the *first exec* fails with the
/// engine's own words about a container id the user has never seen.
///
/// Half a second because that is the measured gap: podman reported
/// `Running: true` some 380ms before its client exited for an alpine
/// image whose entrypoint could not load its preloaded library. It is
/// paid once per bring-up, not once per node, against a phase that
/// already takes seconds.
pub const NODE_SETTLE: std::time::Duration = std::time::Duration::from_millis(500);

/// How often a cancellable wait looks at its [`Cancel`] switch.
///
/// Short enough that Ctrl-C during an image pull feels immediate, long
/// enough that a poll costs nothing next to the provider it is watching.
const CANCEL_POLL: std::time::Duration = std::time::Duration::from_millis(25);

/// How long to let a dead client's last words catch up with it.
///
/// A process's exit is observable before what it wrote is: `try_wait`
/// reaps the client while the thread reading its pipe still has bytes to
/// deliver, so an error built the instant the exit is noticed can quote
/// nothing at all. Bounded, and small, because a pipe can also be held
/// open by something the provider left behind — a diagnostic is worth a
/// moment's wait and not a hang.
const OUTPUT_FLUSH: std::time::Duration = std::time::Duration::from_millis(200);

/// A bounded ring of the last lines a child process wrote.
type LineRing = std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>;

/// How many drain threads are still delivering into a [`LineRing`].
type Draining = std::sync::Arc<std::sync::atomic::AtomicUsize>;

/// A provider client owning one node container.
///
/// The container runs for exactly as long as this value is alive. That is
/// the whole point of launching it in the foreground: there is no step
/// where a container exists without something responsible for it, so a
/// `mirage run` that dies — cleanly, by `SIGKILL`, or with its terminal —
/// cannot leave one behind.
#[derive(Debug)]
pub struct NodeClient {
    /// Rank this container hosts.
    pub rank: u32,
    /// Its container name.
    pub name: String,
    /// The provider client process. `None` once it has been killed.
    child: Option<std::process::Child>,
    /// How the client ended, once it has. `None` while it is running.
    ///
    /// Recorded rather than recomputed because it can only be collected
    /// once: `try_wait` reaps the client, and every later caller — the
    /// error that has to say *why* a container is gone, most of all —
    /// would find nothing to ask.
    exit: Option<std::process::ExitStatus>,
    /// The last few lines the client wrote, on either stream.
    ///
    /// Both streams, because the two things that speak through them are
    /// different: the provider writes its own refusals to stderr, while
    /// the container's process writes wherever it likes, and the reason a
    /// node died is whichever of them got there first.
    ///
    /// Drained on a thread rather than left in the pipe, because the
    /// client outlives the container and a pipe nobody reads eventually
    /// blocks the writer. Bounded for the same reason a log is: a client
    /// that chatters for hours must not grow this without limit.
    output: LineRing,
    /// How many of those drains are still running. Zero means everything
    /// the client ever wrote is in `output`.
    draining: Draining,
}

/// How many lines of a provider client's output to keep.
///
/// Enough for a refusal with a little context around it; the interesting
/// part of `podman run` failing is always its last words.
const CLIENT_OUTPUT_LINES: usize = 20;

/// A new, empty ring for [`drain_lines`] to fill.
fn line_ring() -> LineRing {
    std::sync::Arc::new(std::sync::Mutex::new(
        std::collections::VecDeque::with_capacity(CLIENT_OUTPUT_LINES),
    ))
}

/// Drain `reader` on a thread, keeping its last [`CLIENT_OUTPUT_LINES`]
/// lines in `sink`.
///
/// `draining` counts the threads still doing this, so a reader of the
/// ring can tell "it said nothing" from "it has not all arrived yet".
fn drain_lines(reader: impl std::io::Read + Send + 'static, sink: LineRing, draining: &Draining) {
    use std::io::{BufRead as _, BufReader};
    use std::sync::atomic::Ordering;
    draining.fetch_add(1, Ordering::SeqCst);
    let draining = draining.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(reader)
            .lines()
            .map_while(std::result::Result::ok)
        {
            // Recovered from rather than panicked on: this is a ring of
            // plain strings with no invariant a panic could have broken,
            // and losing the diagnostics this thread exists to collect is
            // the worse outcome.
            let mut sink = sink.lock().unwrap_or_else(|e| e.into_inner());
            if sink.len() == CLIENT_OUTPUT_LINES {
                sink.pop_front();
            }
            sink.push_back(line);
        }
        draining.fetch_sub(1, Ordering::SeqCst);
    });
}

/// Wait, briefly, for the threads counted by `draining` to finish.
///
/// Worth doing once the process they are reading from has exited, which
/// is the moment its pipes reach an end of file and its last partial
/// line becomes a line. See [`OUTPUT_FLUSH`].
fn wait_for_drains(draining: &Draining) {
    use std::sync::atomic::Ordering;
    let deadline = std::time::Instant::now() + OUTPUT_FLUSH;
    while draining.load(Ordering::SeqCst) > 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// What a drained stream has said so far, most recent lines last.
fn tail_of(ring: &LineRing) -> String {
    ring.lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .join("; ")
        .trim()
        .to_string()
}

/// How a process ended, in words: shared by everything that has to
/// report a dead provider client, so the two never drift.
///
/// The number is not decoration. A client that exits 125 is the provider
/// itself refusing; 127 is the image's own entrypoint failing to start;
/// a signal is something outside the session killing it — three
/// different fixes behind what mirage used to report identically as
/// "stopped immediately".
fn exit_phrase(code: &Option<i32>, signal: &Option<i32>) -> String {
    match (code, signal) {
        (Some(code), _) => format!("exit status {code}"),
        (None, Some(signal)) => format!("killed by signal {signal}"),
        (None, None) => "no exit status".to_string(),
    }
}

impl NodeClient {
    /// Adopt a freshly-spawned provider client, draining what it writes.
    fn adopt(rank: u32, name: String, mut child: std::process::Child) -> Self {
        let output = line_ring();
        let draining = Draining::default();
        if let Some(pipe) = child.stdout.take() {
            drain_lines(pipe, output.clone(), &draining);
        }
        if let Some(pipe) = child.stderr.take() {
            drain_lines(pipe, output.clone(), &draining);
        }
        Self {
            rank,
            name,
            child: Some(child),
            exit: None,
            output,
            draining,
        }
    }

    /// Wait, briefly, for everything the client wrote to arrive.
    fn settle_output(&self) {
        wait_for_drains(&self.draining);
    }

    /// What the provider client has said, most recent lines last.
    ///
    /// Empty when it said nothing, which is the normal case: a healthy
    /// `podman run` of an idling container is silent for its whole life.
    #[must_use]
    pub fn output_tail(&self) -> String {
        tail_of(&self.output)
    }

    /// Stop the container by killing its provider client, and reap the
    /// client so it does not become a zombie.
    ///
    /// Idempotent, and safe to call from a `Drop`: it never blocks on
    /// anything but the client's own exit, which follows immediately from
    /// the kill.
    pub fn kill(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        let _ = child.kill();
        if let Ok(status) = child.wait() {
            self.exit.get_or_insert(status);
        }
    }

    /// Whether the provider client is still running.
    ///
    /// A client that has exited on its own means the container died
    /// underneath the session — an OOM kill, an external `podman stop`,
    /// a crashed engine — which the session reports as unhealthy rather
    /// than discovering later through a failing exec.
    pub fn alive(&mut self) -> bool {
        match self.child.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(None) => true,
                Ok(Some(status)) => {
                    self.exit.get_or_insert(status);
                    false
                }
                Err(_) => false,
            },
            None => false,
        }
    }

    /// Why this container is gone, for a caller that has just found it
    /// is: its own exit status and its last words.
    ///
    /// `None` while the client is still running. The container's own
    /// reason is the whole value of this: "a node container has exited"
    /// describes an event the caller had already noticed, whereas
    /// "exit status 127: Error relocating …: symbol not found" names the
    /// image that cannot host a node and why.
    pub fn death_report(&mut self) -> Option<String> {
        if self.alive() {
            return None;
        }
        self.settle_output();
        let (code, signal) = self.exit_codes();
        let phrase = exit_phrase(&code, &signal);
        let said = self.output_tail();
        Some(if said.is_empty() {
            format!("container `{}` stopped ({phrase})", self.name)
        } else {
            format!("container `{}` stopped ({phrase}): {said}", self.name)
        })
    }

    /// The exit code and terminating signal of a client that has ended.
    fn exit_codes(&self) -> (Option<i32>, Option<i32>) {
        use std::os::unix::process::ExitStatusExt as _;
        match self.exit {
            Some(status) => (status.code(), status.signal()),
            None => (None, None),
        }
    }

    /// The error describing a client that ended before its container
    /// could be used, having lasted `waited`.
    fn exited(&self, waited: std::time::Duration) -> ContainerError {
        self.settle_output();
        let (code, signal) = self.exit_codes();
        ContainerError::ClientExited {
            name: self.name.clone(),
            waited,
            code,
            signal,
            output: self.output_tail(),
        }
    }
}

impl Drop for NodeClient {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Errors raised while driving a container provider.
#[derive(Debug, thiserror::Error)]
pub enum ContainerError {
    /// No provider was configured and none could be auto-detected.
    #[error(
        "no container provider found; install podman or docker, or set MIRAGE_CONTAINER_PROVIDER"
    )]
    NoProvider,

    /// The provider binary could not be spawned.
    #[error("failed to spawn `{provider} {}`: {source}", args.join(" "))]
    Spawn {
        /// Provider binary that failed to spawn.
        provider: String,
        /// Arguments passed to the provider.
        args: Vec<String>,
        /// Underlying OS error.
        source: std::io::Error,
    },

    /// The provider ran but exited non-zero.
    #[error("`{provider} {}` failed (exit {code}): {stderr}", args.join(" "))]
    Command {
        /// Provider binary.
        provider: String,
        /// Arguments passed to the provider.
        args: Vec<String>,
        /// Exit code (or -1 when terminated by a signal).
        code: i32,
        /// Captured stderr, trimmed.
        stderr: String,
    },

    /// A container was launched but never reported itself running.
    #[error("container `{name}` did not start within {waited:?}")]
    NotRunning {
        /// Name of the container that failed to come up.
        name: String,
        /// How long mirage waited for it.
        waited: std::time::Duration,
    },

    /// The provider client exited before its container could be used.
    ///
    /// Distinct from [`ContainerError::NotRunning`] because it is a
    /// different event with a different fix. A timeout means the engine
    /// is slow or wedged; this means it refused, and it usually said why
    /// — a bound port, a device that does not exist, an entrypoint the
    /// image cannot run. That reason is the whole value of the variant:
    /// reporting "did not start within 543ms" against a sixty-second
    /// budget describes neither what happened nor what to do about it.
    ///
    /// The exit status is carried alongside the words because the two
    /// answer different questions. A client that refused says so on
    /// stderr and exits 125; a client whose *container* died says
    /// whatever the container said and exits with the container's own
    /// status. When it said nothing at all the status is all there is,
    /// and 127 from an image mirage only ever asked to run
    /// [`CONTAINER_IDLE_COMMAND`] is already the diagnosis.
    #[error("container `{name}` stopped immediately (after {waited:?}, {}){}",
        exit_phrase(.code, .signal),
        if .output.is_empty() {
            format!(
                "; it said nothing, and the only thing mirage asked it to run was `{}`, \
                 which this image may be unable to",
                CONTAINER_IDLE_COMMAND.join(" ")
            )
        } else {
            format!(": {}", .output)
        })]
    ClientExited {
        /// Name of the container that failed to come up.
        name: String,
        /// How long the provider client lasted.
        waited: std::time::Duration,
        /// The client's exit code, or `None` when a signal ended it.
        code: Option<i32>,
        /// The signal that ended the client, when one did.
        signal: Option<i32>,
        /// What the client wrote, on either stream, trimmed. Empty when
        /// it said nothing.
        output: String,
    },

    /// A bind mount names a host path no container can be given.
    ///
    /// Checked by mirage rather than left to the provider, because the
    /// providers disagree about what a missing host path means: docker
    /// creates it as a root-owned directory on the host and carries on,
    /// podman refuses the container. One `--mount` must not mean two
    /// things, and neither of those two is what the user asked for.
    #[error("--mount {spec}: the host path `{path}` {problem}")]
    Mount {
        /// The mount as the provider would have been given it
        /// (`HOST:CONTAINER[:ro]`).
        spec: String,
        /// The host path mirage resolved it to.
        path: String,
        /// What is wrong with that path, and what to do about it.
        problem: String,
    },

    /// The caller asked for the operation to stop before it finished.
    ///
    /// Bring-up is a chain of blocking provider invocations, and an image
    /// pull is minutes of it. A user who changes their mind in the middle
    /// wants their prompt back, which means the provider running right
    /// now has to be ended rather than waited for — see [`Cancel`].
    #[error("interrupted while {what}")]
    Cancelled {
        /// What was in flight, phrased to follow "interrupted while".
        what: String,
    },
}

/// A shared "stop what you are doing" switch for a bring-up in flight.
///
/// The container engine is a chain of blocking child processes, so
/// nothing outside it can hurry it along: `podman pull` is not a future
/// that can be dropped, it is a process that has to be killed. A
/// [`Cancel`] is how the owner of a bring-up — the `mirage run` whose
/// user just pressed Ctrl-C — says so. Every long step polls it, kills
/// whatever provider it is waiting on, and returns
/// [`ContainerError::Cancelled`], so an interrupt during a ten-minute
/// pull costs the caller a poll interval rather than the rest of the
/// pull.
///
/// Cloning shares the switch; flipping it is one-way, because there is no
/// version of "actually, carry on" that a half-torn-down session could
/// honour.
#[derive(Debug, Clone, Default)]
pub struct Cancel(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Cancel {
    /// A switch nobody has flipped yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask whatever is using this switch to stop.
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether the switch has been flipped.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Result alias for container operations.
pub type Result<T> = std::result::Result<T, ContainerError>;

/// Whether `provider` resolves to podman (by binary name or path
/// basename). podman supports `--group-add keep-groups`, which docker
/// does not, so callers branch their GPU group passthrough on this.
fn provider_is_podman(provider: &str) -> bool {
    std::path::Path::new(provider)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(provider)
        .contains("podman")
}

/// Host AMD GPU device nodes to expose to a node container (`--device`)
/// when host GPU access is requested: the KFD compute device
/// (`/dev/kfd`) and the DRM render nodes (`/dev/dri`). Only paths that
/// actually exist on the host are returned, so a host missing one simply
/// omits it.
fn host_gpu_devices() -> Vec<String> {
    ["/dev/kfd", "/dev/dri"]
        .iter()
        .filter(|p| std::path::Path::new(p).exists())
        .map(|p| (*p).to_string())
        .collect()
}

/// A `Stdio` that writes to mirage's own stderr.
///
/// Used where a child's *stdout* has to be shown to the user. Inheriting
/// would be the obvious thing and is the wrong one: it hands the child
/// mirage's stdout, which belongs to the workload alone — the promise
/// that `mirage run … > out.txt` is byte-exact is only worth as much as
/// the number of things allowed to write to that file. Falls back to
/// discarding the stream if the descriptor cannot be duplicated, which
/// loses provider chatter rather than misdirecting it.
fn mirage_stderr() -> Stdio {
    use std::os::fd::AsFd as _;
    std::io::stderr()
        .as_fd()
        .try_clone_to_owned()
        .map_or_else(|_| Stdio::null(), Stdio::from)
}

/// Resolve the host side of every bind mount, refusing the ones no
/// container should be given.
///
/// Two shapes of host path are trouble, and each is trouble differently
/// on the two engines mirage drives:
///
/// * **Relative** (`--mount data:/data`). Neither engine reads that as a
///   path. A `-v` source with no leading separator is a *named volume*,
///   so both quietly create a persistent volume called `data` and mount
///   it empty: the directory the user meant is not in the container, and
///   the volume outlives the session, the run, and `mirage cleanup`,
///   which reclaims containers and networks and has never heard of it.
///   Resolved against the working directory instead, which is what the
///   spec plainly means.
/// * **Nonexistent**. docker creates it on the host as a root-owned
///   directory and starts the container; podman refuses with
///   `statfs …: no such file or directory`. Mirage rejects it on both,
///   before anything is created, naming the path — see
///   [`ContainerError::Mount`].
fn resolve_mounts(mounts: &[FileMount]) -> Result<Vec<FileMount>> {
    mounts.iter().map(resolve_mount).collect()
}

/// Resolve and check one bind mount's host path.
fn resolve_mount(mount: &FileMount) -> Result<FileMount> {
    let refuse = |path: &std::path::Path, problem: String| ContainerError::Mount {
        spec: mount.to_volume_arg(),
        path: path.display().to_string(),
        problem,
    };

    let host = std::path::Path::new(&mount.host_path);
    let host = if host.is_absolute() {
        host.to_path_buf()
    } else {
        // Lexical, not canonical: this answers "which path did the user
        // mean", and resolving symlinks as well would hand the provider
        // a path the user never wrote and cannot recognise in an error.
        std::path::absolute(host).map_err(|e| {
            refuse(
                host,
                format!("could not be resolved against the working directory: {e}"),
            )
        })?
    };

    match host.try_exists() {
        Ok(true) => Ok(FileMount {
            host_path: host.display().to_string(),
            ..mount.clone()
        }),
        Ok(false) => Err(refuse(
            &host,
            "does not exist. Create it before the run, or correct the path — mirage will \
             not create it for you, because docker would make it a root-owned directory on \
             the host while podman would refuse to start the container"
                .to_string(),
        )),
        Err(e) => Err(refuse(&host, format!("could not be read: {e}"))),
    }
}

/// Supplementary groups that own the host GPU device nodes
/// (`--group-add`). `video` and `render` are the conventional owners of
/// `/dev/kfd` and the `/dev/dri/render*` nodes on ROCm hosts; docker is
/// given these explicitly (podman inherits them via `keep-groups`).
fn host_gpu_groups() -> Vec<String> {
    vec!["video".to_string(), "render".to_string()]
}

/// A phase of container bring-up, reported to the `progress` callback of
/// [`Engine::bring_up`] so the host can surface detailed, live status to
/// clients as a session starts.
///
/// Each variant maps to a `(state, message)` pair via [`Self::health`],
/// keeping the full set of bring-up conditions described in one place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BringUpPhase {
    /// The derived image already exists locally, so the build is skipped.
    ImageBuilt { image: String },
    /// Building a derived image (applying profile hacks); can take a
    /// while as it runs package-manager commands inside the build.
    BuildingImage { base: String, image: String },
    /// The image is already present locally, so the pull is skipped.
    ImagePresent { image: String },
    /// Pulling the image from its registry (can take a while).
    Pulling { image: String },
    /// The image pull finished successfully.
    Pulled { image: String },
    /// Reusing a per-session network that already exists.
    NetworkExists { network: String },
    /// Creating the per-session network.
    CreatingNetwork { network: String },
    /// Starting node container `rank` (0-based) of `total`.
    LaunchingNode { rank: u32, total: u32, name: String },
    /// Node container `rank` (0-based) of `total` has started.
    NodeStarted { rank: u32, total: u32, name: String },
}

impl BringUpPhase {
    /// The lifecycle `state` slug for this phase: one of `"pulling"`,
    /// `"networking"`, or `"starting"`. Stable enough for clients to key
    /// off while [`message`](Self::message) carries the human detail.
    pub fn state(&self) -> &'static str {
        match self {
            BringUpPhase::ImageBuilt { .. } | BringUpPhase::BuildingImage { .. } => "building",
            BringUpPhase::ImagePresent { .. }
            | BringUpPhase::Pulling { .. }
            | BringUpPhase::Pulled { .. } => "pulling",
            BringUpPhase::NetworkExists { .. } | BringUpPhase::CreatingNetwork { .. } => {
                "networking"
            }
            BringUpPhase::LaunchingNode { .. } | BringUpPhase::NodeStarted { .. } => "starting",
        }
    }

    /// A detailed, human-readable description of this phase, suitable for
    /// surfacing directly to the user as the session's status message.
    pub fn message(&self) -> String {
        match self {
            BringUpPhase::ImageBuilt { image } => {
                format!("derived image {image} already built; skipping build")
            }
            BringUpPhase::BuildingImage { base, image } => {
                format!("building derived image {image} from {base} (this can take a while)…")
            }
            BringUpPhase::ImagePresent { image } => {
                format!("image {image} already present locally; skipping pull")
            }
            BringUpPhase::Pulling { image } => {
                format!("pulling image {image} (this can take a while)…")
            }
            BringUpPhase::Pulled { image } => format!("image {image} ready"),
            BringUpPhase::NetworkExists { network } => {
                format!("reusing existing session network {network}")
            }
            BringUpPhase::CreatingNetwork { network } => {
                format!("creating session network {network}")
            }
            BringUpPhase::LaunchingNode { rank, total, name } => {
                format!("starting node {}/{total} ({name})", rank + 1)
            }
            BringUpPhase::NodeStarted { rank, total, name } => {
                format!("node {}/{total} ({name}) started", rank + 1)
            }
        }
    }

    /// Convenience pairing of [`state`](Self::state) and
    /// [`message`](Self::message).
    pub fn health(&self) -> (&'static str, String) {
        (self.state(), self.message())
    }
}

/// The shape every node container in a session shares, borrowed from the
/// profile that describes it.
///
/// Bring-up derives these once from the [`ContainerizedDef`] and then
/// launches one identical container per rank, so they are exactly the
/// inputs that do *not* vary across the loop — only the container's name,
/// its rank, and its environment do. Passing them as one value says that
/// in the signature, and stops eight identical arguments being threaded
/// through [`Engine::launch_node`] into [`Engine::run_argv`] on every
/// iteration.
///
/// Grouping them is also the only thing that makes the pair safe to call.
/// `devices` and `groups` are both `&[String]` and mean entirely
/// different things — one becomes `--device`, the other `--group-add` —
/// so as positional parameters they could be exchanged at a call site
/// without the compiler noticing, and the mistake would surface only as a
/// container the provider refuses to start. As named fields they cannot.
#[derive(Debug)]
pub struct NodeSpec<'a> {
    /// The session these containers belong to.
    ///
    /// Not part of the container's shape — nothing in `run_argv` reads
    /// it — but part of its *provenance*: the provider client mirage
    /// spawns is marked with it, so a client stranded by a `SIGKILL`ed
    /// run can be found and reaped by [`mirage_core::reclaim`] later.
    pub session: &'a mirage_core::session::SessionId,
    /// Image to run.
    pub image: &'a str,
    /// Network to attach to, or `None` to leave the provider's default.
    pub network: Option<&'a str>,
    /// Whether the container needs host GPU access.
    pub host_gpus: bool,
    /// Host paths bind-mounted into the container.
    pub mounts: &'a [FileMount],
    /// Ports published from the container to the host.
    pub ports: &'a [PortMapping],
    /// Device nodes passed through (`--device`).
    pub devices: &'a [String],
    /// Supplementary groups granted on docker (`--group-add`); ignored on
    /// podman, which inherits the launching user's groups instead.
    pub groups: &'a [String],
    /// Ownership labels stamped on the container so teardown and orphan
    /// reclamation can prove it is mirage's before removing it.
    pub labels: &'a [(String, String)],
}

/// A resolved container provider plus the operations mirage performs on
/// it. Cheap to clone; holds only the provider binary name/path.
#[derive(Debug, Clone)]
pub struct Engine {
    provider: String,
    /// The switch that ends whatever this engine is waiting on, when the
    /// caller has installed one. `None` means "there is nobody to
    /// interrupt this", which is every call outside a session bring-up.
    cancel: Option<Cancel>,
}

impl Engine {
    /// Resolve an engine for a containerised profile, applying the
    /// "explicit > `MIRAGE_CONTAINER_PROVIDER` > autodetect (podman then
    /// docker)" policy. Errors with [`ContainerError::NoProvider`] when
    /// nothing is available.
    pub fn resolve(def: &ContainerizedDef) -> Result<Self> {
        let provider = mirage_core::container::resolve_provider(def.provider.as_deref())
            .ok_or(ContainerError::NoProvider)?;
        Ok(Self {
            provider,
            cancel: None,
        })
    }

    /// Build an engine around an explicit provider binary (name or
    /// path). Primarily for tests and callers that already resolved a
    /// provider.
    pub fn with_provider(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            cancel: None,
        }
    }

    /// Give this engine a switch its caller can flip to end whatever it
    /// is waiting on. See [`Cancel`].
    #[must_use]
    pub fn with_cancel(mut self, cancel: Cancel) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// The resolved provider binary (`"podman"`, `"docker"`, or a path).
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Whether the caller has asked this engine to stop.
    fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(Cancel::is_cancelled)
    }

    /// `Err(Cancelled)` if the caller has asked this engine to stop,
    /// phrased to follow "interrupted while".
    fn check_cancelled(&self, what: &str) -> Result<()> {
        if self.cancelled() {
            return Err(ContainerError::Cancelled {
                what: what.to_string(),
            });
        }
        Ok(())
    }

    // ---- argv builders (pure) -------------------------------------

    /// Build the argv (after the provider binary) for launching a
    /// detached node container.
    ///
    /// `command` is the container's foreground process (PID 1). Mirage
    /// runs each node's own `mirage host --session <id> --rank <n>` here
    /// so the container hosts its node directly; an empty `command`
    /// leaves the image's default entrypoint in place.
    ///
    /// The first element of `command` is passed as `--entrypoint` so it
    /// *replaces* the image's default `ENTRYPOINT` rather than being
    /// appended to it (the remaining elements become the entrypoint's
    /// arguments after the image). Without this, images that ship their
    /// own entrypoint (e.g. `vllm/vllm-openai`) would run that entrypoint
    /// with `mirage host …` tacked on as arguments instead of running
    /// mirage.
    ///
    /// When `spec.host_gpus` is set, the container is launched with the
    /// supplementary groups needed to open the passed-through GPU device
    /// nodes. The mechanism depends on `provider`: podman inherits the
    /// launching user's groups via `--group-add keep-groups`, while
    /// docker (which has no `keep-groups`) is given the named
    /// `spec.groups` explicitly. When `host_gpus` is unset no group
    /// passthrough is emitted, which keeps plain (non-GPU) containers
    /// working on docker — `keep-groups` is a podman-only feature and
    /// docker rejects it.
    ///
    /// The container is named and given a matching hostname so peers can
    /// resolve it by name on the shared network.
    pub fn run_argv(
        provider: &str,
        name: &str,
        spec: &NodeSpec<'_>,
        env: &[(String, String)],
        command: &[String],
    ) -> Vec<String> {
        let &NodeSpec {
            // The session marks the *client process* mirage spawns, not
            // the container it asks for; see [`Self::launch_node`]. It is
            // named here rather than elided with `..` so that a field
            // added to the spec later cannot be silently ignored by the
            // argv this whole crate exists to build.
            session: _,
            image,
            network,
            host_gpus,
            mounts,
            ports,
            devices,
            groups,
            labels,
        } = spec;
        let mut argv = vec![
            "run".to_string(),
            // Not detached. The provider client stays in the foreground
            // and mirage owns it as a child process, so the container's
            // lifetime is bounded by the `mirage run` that asked for it
            // rather than by whoever remembers to remove it later.
            //
            // `--rm` closes the other half: the container is deleted the
            // moment it stops, however it stops. Between the two there is
            // no state left behind by a run that crashed, was `SIGKILL`ed,
            // or had its terminal closed.
            "--rm".to_string(),
            "--name".to_string(),
            name.to_string(),
            "--hostname".to_string(),
            name.to_string(),
        ];
        // Stamp ownership on the container itself. The name is derived
        // from the session id and is not proof of anything — teardown and
        // orphan reclamation both check this label before removing
        // anything, so a user's own `mirage-s1-node-0` is safe.
        for (k, v) in labels {
            argv.push("--label".to_string());
            argv.push(format!("{k}={v}"));
        }
        if host_gpus {
            // Run the GPU device nodes unconfined and grant the container
            // the supplementary groups that own `/dev/kfd` and
            // `/dev/dri/*`, so the workload can open them.
            argv.push("--security-opt".to_string());
            argv.push("seccomp=unconfined".to_string());
            if provider_is_podman(provider) {
                // podman inherits the launching user's supplementary
                // groups (including `video`/`render`) rather than naming
                // them. It also rejects combining `keep-groups` with any
                // other `--group-add`, so the named groups are dropped.
                argv.push("--group-add".to_string());
                argv.push("keep-groups".to_string());
            } else {
                // docker has no `keep-groups`; add the named GPU groups
                // explicitly so the workload can open the device nodes.
                for g in groups {
                    argv.push("--group-add".to_string());
                    argv.push(g.clone());
                }
            }
        }
        if let Some(net) = network {
            argv.push("--network".to_string());
            argv.push(net.to_string());
        }
        for (k, v) in env {
            argv.push("-e".to_string());
            argv.push(format!("{k}={v}"));
        }
        for m in mounts {
            argv.push("-v".to_string());
            argv.push(m.to_volume_arg());
        }
        for p in ports {
            argv.push("-p".to_string());
            argv.push(p.to_publish_arg());
        }
        for d in devices {
            argv.push("--device".to_string());
            argv.push(d.clone());
        }
        // The container's foreground process. Mirage hosts the node from
        // inside the container, so this is normally `mirage host ...`.
        // The first element overrides the image ENTRYPOINT (so it runs
        // mirage rather than the image's own entrypoint); the rest become
        // its arguments after the image. An empty `command` leaves the
        // image's default entrypoint in place.
        if let Some((entrypoint, args)) = command.split_first() {
            argv.push("--entrypoint".to_string());
            argv.push(entrypoint.clone());
            argv.push(image.to_string());
            argv.extend(args.iter().cloned());
        } else {
            argv.push(image.to_string());
        }
        argv
    }

    /// Build the argv (after the provider binary) for executing a
    /// command inside an already-running node container.
    ///
    /// `-i` keeps stdin open so input reaches the workload exactly as it
    /// would for a non-containerised exec. Environment is injected
    /// explicitly with `-e` rather than inherited from the host.
    ///
    /// `tty` adds `-t`, asking the provider to allocate a pseudo-terminal
    /// inside the container. Mirage still allocates none of its own, and
    /// for a workload running *on the host* it does not need to: the
    /// child inherits the caller's real file descriptors, so if the
    /// caller is on a terminal then so is the workload.
    ///
    /// That reasoning does not survive the container boundary, and
    /// assuming it did is why no interactive program worked in a
    /// containerised session. `provider exec` does not hand the caller's
    /// descriptors to the in-container process — it cannot, they are in
    /// different namespaces — it proxies the streams over its own socket
    /// and gives the process pipes. `isatty(0)` is then false however
    /// good the caller's terminal is: `bash` prints no prompt and runs no
    /// job control, and anything ncurses refuses to start.
    ///
    /// The flag is not unconditional because `-t` merges stderr into
    /// stdout — a pseudo-terminal has one stream. That is invisible when
    /// every stream is the same terminal anyway, and destroys
    /// `… -- job > out 2> err` when they are not, so the caller decides
    /// from the shape of the exec and the state of its own streams; see
    /// `mirage_supervisor::spec`.
    pub fn exec_argv(
        container: &str,
        workdir: Option<&str>,
        env: &[(String, String)],
        command: &str,
        args: &[String],
        tty: bool,
    ) -> Vec<String> {
        let mut argv = vec!["exec".to_string(), "-i".to_string()];
        if tty {
            argv.push("-t".to_string());
        }
        if let Some(wd) = workdir {
            argv.push("-w".to_string());
            argv.push(wd.to_string());
        }
        for (k, v) in env {
            argv.push("-e".to_string());
            argv.push(format!("{k}={v}"));
        }
        argv.push(container.to_string());
        argv.push(command.to_string());
        argv.extend(args.iter().cloned());
        argv
    }

    /// Full argv including the provider binary for executing a command
    /// inside a node container. Convenience for callers that build a
    /// `Command` from a single vector.
    pub fn exec_command_line(
        &self,
        container: &str,
        workdir: Option<&str>,
        env: &[(String, String)],
        command: &str,
        args: &[String],
        tty: bool,
    ) -> Vec<String> {
        let mut full = vec![self.provider.clone()];
        full.extend(Self::exec_argv(container, workdir, env, command, args, tty));
        full
    }

    // ---- side-effecting operations --------------------------------

    /// Pull `image` so node launches don't race on an implicit pull.
    ///
    /// The slowest step in bring-up by a wide margin, and the one a user
    /// most needs to see. What they see depends on where mirage's stderr
    /// goes:
    ///
    /// * **A terminal** — the provider's output is passed straight
    ///   through, so `podman pull`'s layer-by-layer progress renders
    ///   exactly as it does when run by hand. That cannot be reproduced
    ///   by capturing: a progress bar is `\r`-driven, and a line reader
    ///   holds every update until a newline that never comes.
    /// * **Anything else** (a CI log, `2>file`) — captured, so a
    ///   failure's stderr is carried in the error rather than scattered
    ///   into whatever the caller redirected to.
    ///
    /// The trade on the first branch is deliberate: a failed pull's
    /// output is not repeated in [`ContainerError::Command`], because the
    /// user just watched it go past.
    ///
    /// # Provider chatter never lands on mirage's stdout
    ///
    /// Not even the provider's *own* stdout, which is why the pass-
    /// through branch duplicates mirage's stderr onto the child's stdout
    /// rather than inheriting. mirage's stdout belongs to the workload
    /// and to nothing else: `mirage run … > out.txt` from a terminal has
    /// to produce a byte-exact `out.txt`, and inheriting put the pull's
    /// progress and the pulled image's digest in it. Sent to stderr, the
    /// digest is still on screen for a user watching the pull, and still
    /// out of the way of a user redirecting the run.
    pub fn pull(&self, image: &str) -> Result<()> {
        use std::io::IsTerminal as _;

        let args = vec!["pull".to_string(), image.to_string()];
        let passthrough = std::io::stderr().is_terminal();
        let mut child = spawn_retrying_etxtbsy(|| {
            Command::new(&self.provider)
                .args(&args)
                .stdin(Stdio::null())
                .stdout(if passthrough {
                    mirage_stderr()
                } else {
                    Stdio::null()
                })
                .stderr(if passthrough {
                    Stdio::inherit()
                } else {
                    Stdio::piped()
                })
                .spawn()
        })
        .map_err(|source| ContainerError::Spawn {
            provider: self.provider.clone(),
            args: args.clone(),
            source,
        })?;

        let said = line_ring();
        let draining = Draining::default();
        if let Some(pipe) = child.stderr.take() {
            drain_lines(pipe, said.clone(), &draining);
        }
        // Waited on a poll rather than with `output()`, so that a Ctrl-C
        // ten seconds into a ten-minute pull ends the pull instead of
        // being noticed after it. Keeping only the tail of a captured
        // stderr is part of the same trade: a failing pull's last words
        // are the ones worth carrying in an error, and a full transcript
        // would have to be buffered whole.
        let status = self.wait_cancellable(&mut child, &args, &format!("pulling image {image}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(ContainerError::Command {
                provider: self.provider.clone(),
                args,
                code: status.code().unwrap_or(-1),
                stderr: if passthrough {
                    "see the provider's output above".to_string()
                } else {
                    // The pull has exited; its last words may still be in
                    // flight. See [`OUTPUT_FLUSH`].
                    wait_for_drains(&draining);
                    tail_of(&said)
                },
            })
        }
    }

    /// Wait for a provider child, ending it if the caller cancels.
    ///
    /// The child must not have any pipe left for this thread to drain:
    /// waiting is a poll here, so nothing is reading, and a provider that
    /// filled such a pipe would block forever against a loop that only
    /// ever asks whether it has exited.
    fn wait_cancellable(
        &self,
        child: &mut std::process::Child,
        args: &[String],
        what: &str,
    ) -> Result<std::process::ExitStatus> {
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => {}
                Err(source) => {
                    return Err(ContainerError::Spawn {
                        provider: self.provider.clone(),
                        args: args.to_vec(),
                        source,
                    });
                }
            }
            if self.cancelled() {
                // Killed and reaped here rather than left to the caller:
                // this is the one path that abandons a running provider,
                // and `std::process::Child` has no `Drop` to catch it.
                let _ = child.kill();
                let _ = child.wait();
                return Err(ContainerError::Cancelled {
                    what: what.to_string(),
                });
            }
            std::thread::sleep(CANCEL_POLL);
        }
    }

    /// Build an image tagged `tag` from the given `dockerfile` contents,
    /// streamed to the provider's `build` over stdin (`build -t <tag> -`,
    /// which both podman and docker accept for a context-less build).
    ///
    /// Used to realise profile [hacks](mirage_core::profile::Hack): a
    /// derivative image is built once from the base image and then run in
    /// place of it. The provider's build output (which can take a while —
    /// apt updates, package installs, …) is streamed line by line to the
    /// log at INFO so progress is visible live; the captured lines are
    /// also retained and, on failure, surfaced in the error so a broken
    /// `RUN` step is actionable.
    ///
    /// When mirage's stderr is a terminal the lines are echoed there as
    /// well. The log alone is off unless the user passed `-v`, which made
    /// the other multi-minute phase of bring-up — see [`Self::pull`] —
    /// look identical to a hang. Line-buffered rather than inherited,
    /// unlike the pull: a build's output is lines, not a progress bar,
    /// and they are wanted in the error too.
    pub fn build_image(&self, tag: &str, dockerfile: &str) -> Result<()> {
        use std::io::IsTerminal as _;
        let echo = std::io::stderr().is_terminal();
        let args = vec![
            "build".to_string(),
            "-t".to_string(),
            tag.to_string(),
            "-".to_string(),
        ];
        let mut child = spawn_retrying_etxtbsy(|| {
            Command::new(&self.provider)
                .args(&args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        })
        .map_err(|source| ContainerError::Spawn {
            provider: self.provider.clone(),
            args: args.clone(),
            source,
        })?;
        use std::io::{BufRead, BufReader, Write};

        // Drain stdout and stderr concurrently, logging each line at INFO
        // as it arrives and retaining it so a failing build's output can
        // be surfaced in the error. Build providers write most progress
        // to stderr, so both streams are followed.
        fn log_stream<R: std::io::Read + Send + 'static>(
            reader: Option<R>,
            tag: String,
            echo: bool,
        ) -> (
            std::sync::Arc<std::sync::Mutex<Vec<String>>>,
            Option<std::thread::JoinHandle<()>>,
        ) {
            let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
            let lines_for_thread = lines.clone();
            let handle = reader.map(|r| {
                std::thread::spawn(move || {
                    for line in BufReader::new(r).lines().map_while(std::result::Result::ok) {
                        tracing::info!(image = %tag, "{line}");
                        if echo {
                            eprintln!("mirage: {tag}: {line}");
                        }
                        // Recover from poisoning rather than panicking:
                        // the buffer is plain data with no invariant a
                        // panic could have broken, and taking down the
                        // build over a poisoned log buffer would lose the
                        // diagnostics this thread exists to collect.
                        lines_for_thread
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(line);
                    }
                })
            });
            (lines, handle)
        }
        let (out_lines, out_handle) = log_stream(child.stdout.take(), tag.to_string(), echo);
        let (err_lines, err_handle) = log_stream(child.stderr.take(), tag.to_string(), echo);

        // Only now stream the Dockerfile in, then close stdin so the
        // provider proceeds.
        //
        // Order matters, and getting it wrong deadlocks bring-up with no
        // timeout: both providers interleave `STEP`/pull progress on
        // stderr while they are still reading the build context, so
        // writing first meant the provider could fill its output pipe
        // — nobody was reading it yet — and block, while this thread
        // blocked writing to an input pipe the provider had stopped
        // reading. The drains above are already running, so neither side
        // can stall the other.
        if let Some(mut stdin) = child.stdin.take()
            && let Err(source) = stdin.write_all(dockerfile.as_bytes())
        {
            // The provider is still running and owns two pipes we are
            // about to stop reading. Ending it and reaping it here is
            // what keeps `?` from orphaning it: `std::process::Child`
            // has no `Drop`, so returning would leave the build running
            // unattended and unwaited-for.
            let _ = child.kill();
            let _ = child.wait();
            return Err(ContainerError::Spawn {
                provider: self.provider.clone(),
                args: args.clone(),
                source,
            });
        }

        let status = child.wait().map_err(|source| ContainerError::Spawn {
            provider: self.provider.clone(),
            args: args.clone(),
            source,
        })?;
        if let Some(h) = out_handle {
            let _ = h.join();
        }
        if let Some(h) = err_handle {
            let _ = h.join();
        }

        if status.success() {
            Ok(())
        } else {
            // Prefer stderr (where build errors land); fall back to stdout.
            let mut captured = err_lines
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .join("\n");
            if captured.trim().is_empty() {
                captured = out_lines
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .join("\n");
            }
            Err(ContainerError::Command {
                provider: self.provider.clone(),
                args,
                code: status.code().unwrap_or(-1),
                stderr: captured.trim().to_string(),
            })
        }
    }

    /// Whether `image` is already present locally.
    pub fn image_present(&self, image: &str) -> bool {
        self.status(&[
            "image".to_string(),
            "inspect".to_string(),
            image.to_string(),
        ])
        .unwrap_or(false)
    }

    /// Whether a network named `name` already exists.
    pub fn network_exists(&self, name: &str) -> bool {
        self.status(&[
            "network".to_string(),
            "inspect".to_string(),
            name.to_string(),
        ])
        .unwrap_or(false)
    }

    /// Create the per-session network if it does not already exist.
    pub fn ensure_network(&self, name: &str, labels: &[(String, String)]) -> Result<()> {
        if self.network_exists(name) {
            return Ok(());
        }
        let mut argv = vec!["network".to_string(), "create".to_string()];
        for (k, v) in labels {
            argv.push("--label".to_string());
            argv.push(format!("{k}={v}"));
        }
        argv.push(name.to_string());
        self.checked(&argv)
    }

    /// Launch a node container and return the provider client running it.
    ///
    /// The client is *not* detached: it is a child of this process, and
    /// the caller owns it for as long as the container should live.
    /// Dropping or killing it stops the container, and `--rm` then
    /// removes it.
    ///
    /// Its stdin goes to `/dev/null`: the container's foreground process
    /// is an idle placeholder — workloads arrive later via
    /// `provider exec` — so nothing in there reads.
    ///
    /// Both output streams are *captured* rather than inherited or
    /// discarded, and drained into the returned [`NodeClient`].
    /// Inheriting would interleave provider chatter with the workload
    /// output the user asked for; discarding is what mirage used to do,
    /// and it meant a `podman run` that refused instantly — a bound
    /// port, a device that does not exist, an entrypoint the image
    /// cannot run — said so into nothing, and mirage reported only that
    /// the container "did not start". stdout is kept for the same reason
    /// stderr is: the words that explain a node's death are the
    /// *container's*, and a container writes them to whichever stream it
    /// likes.
    ///
    /// The client is also marked with the session and runtime directory
    /// it belongs to, which is a promise
    /// [`mirage_core::container::ENV_SESSION`] makes on its behalf.
    /// Marking matters exactly when nothing else survives: a `SIGKILL`ed
    /// run leaves this process reparented to init, still holding a
    /// container, and its environment is then the only evidence of whose
    /// it was — which is what `mirage cleanup` reads.
    ///
    /// Returns as soon as the client has been spawned. The container is
    /// not necessarily running yet; use [`Self::await_running`] for that.
    ///
    /// `spec.host_gpus` requests host GPU access for the container; the
    /// group passthrough it implies is provider-specific (see
    /// [`Self::run_argv`]).
    pub fn launch_node(
        &self,
        name: &str,
        spec: &NodeSpec<'_>,
        env: &[(String, String)],
        rank: u32,
    ) -> Result<NodeClient> {
        let command: Vec<String> = CONTAINER_IDLE_COMMAND
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let argv = Self::run_argv(&self.provider, name, spec, env, &command);
        let child = spawn_retrying_etxtbsy(|| {
            Command::new(&self.provider)
                .args(&argv)
                .env(mirage_core::container::ENV_SESSION, spec.session.as_str())
                .env(
                    mirage_core::container::ENV_RUNTIME,
                    mirage_core::container::owning_runtime(),
                )
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        })
        .map_err(|source| ContainerError::Spawn {
            provider: self.provider.clone(),
            args: argv.clone(),
            source,
        })?;
        Ok(NodeClient::adopt(rank, name.to_string(), child))
    }

    /// Whether a container named `name` is currently running.
    pub fn container_running(&self, name: &str) -> bool {
        match self.output(&[
            "inspect".to_string(),
            "-f".to_string(),
            "{{.State.Running}}".to_string(),
            name.to_string(),
        ]) {
            Ok(out) => String::from_utf8_lossy(&out).trim() == "true",
            Err(_) => false,
        }
    }

    /// Block until `name` reports itself running, or `timeout` elapses.
    ///
    /// A detached `run -d` returned only once the container existed, so
    /// the next `exec` was guaranteed a target. A foreground client
    /// returns immediately and the container comes up behind it, so that
    /// guarantee has to be re-established explicitly — otherwise the
    /// first exec races bring-up and fails with "no such container".
    ///
    /// # Errors
    ///
    /// Returns [`ContainerError::NotRunning`] if the container has not
    /// come up within `timeout`, [`ContainerError::ClientExited`] if its
    /// client gave up first, or [`ContainerError::Cancelled`] if the
    /// caller stopped waiting (see [`Cancel`]).
    pub fn await_running(
        &self,
        client: &mut NodeClient,
        timeout: std::time::Duration,
    ) -> Result<()> {
        const POLL: std::time::Duration = std::time::Duration::from_millis(50);
        let name = client.name.clone();
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if self.container_running(&name) {
                return Ok(());
            }
            // The client is the container's lifetime, so a client that
            // has already exited means the container is never coming.
            //
            // Waiting the full timeout for it is worse than slow, it is
            // misleading: a `podman run` that fails instantly — a bound
            // port, a device that does not exist, a name already in use,
            // an entrypoint the image cannot run — would be reported
            // after 60s of polling as a container that "did not start",
            // when the engine had said exactly what was wrong in the
            // first millisecond. Its stderr is captured for precisely
            // this moment. Checking the client also stops a *leftover*
            // container of the same name from being adopted as though
            // this run had created it.
            if !client.alive() {
                return Err(client.exited(
                    std::time::Instant::now().saturating_duration_since(deadline - timeout),
                ));
            }
            self.check_cancelled(&format!("starting container `{name}`"))?;
            if std::time::Instant::now() >= deadline {
                return Err(ContainerError::NotRunning {
                    name,
                    waited: timeout,
                });
            }
            std::thread::sleep(POLL);
        }
    }

    /// Best-effort removal of a single container.
    pub fn rm(&self, name: &str) {
        let _ = self.status(&["rm".to_string(), "-f".to_string(), name.to_string()]);
    }

    /// Best-effort removal of a network.
    pub fn network_rm(&self, name: &str) {
        let _ = self.status(&["network".to_string(), "rm".to_string(), name.to_string()]);
    }

    /// Pull the image, create the network, and launch one container per
    /// rank, returning the [`ContainerState`] describing them plus the
    /// provider clients that own them.
    ///
    /// The caller must keep the returned clients alive for as long as the
    /// session lasts: each one *is* its container's lifetime. Dropping
    /// them stops the containers, and `--rm` removes them.
    ///
    /// `host_gpus` requests host GPU access for every node container
    /// (the provider-specific group passthrough described on
    /// [`Self::run_argv`]); the emulator decides whether its workload
    /// needs it.
    ///
    /// `node_env(rank)` yields the environment for the node of that rank
    /// (mirage injects `MIRAGE_RANK`/`MIRAGE_HEAD_ADDR`/`MIRAGE_HEAD_PORT`
    /// there). `progress(phase)` is invoked before/after each step
    /// ([`BringUpPhase`]) so callers can surface detailed live status.
    /// On any failure the partially-created containers and network are
    /// torn down before returning the error, so a failed bring-up never
    /// leaks resources — including the failure that is a caller flipping
    /// this engine's [`Cancel`], which is the same rollback rather than a
    /// second path that would have to be kept honest separately.
    #[allow(clippy::too_many_arguments)]
    pub fn bring_up<F, P>(
        &self,
        session: &mirage_core::session::SessionId,
        def: &ContainerizedDef,
        host_gpus: bool,
        node_count: u32,
        head_port: u16,
        mut node_env: F,
        mut progress: P,
    ) -> Result<(ContainerState, Vec<NodeClient>)>
    where
        F: FnMut(u32) -> Vec<(String, String)>,
        P: FnMut(BringUpPhase),
    {
        let network = mirage_core::container::network_name(session);
        // Every resource this call creates carries mirage's ownership
        // label plus the session it belongs to, so teardown can prove a
        // resource is ours before removing it and `reclaim_orphans` can
        // find what a crashed supervisor left behind.
        let labels = mirage_core::container::owner_labels(session);

        // Before anything is created, and before the pull above all: a
        // mistyped `--mount` is the cheapest failure in bring-up to
        // diagnose and the most expensive to wait for, and finding it
        // after ten minutes of pulling an image would be nobody's idea
        // of a good error.
        let mounts = resolve_mounts(&def.mounts)?;

        // Nothing has been created yet, so an interrupt that arrived
        // before this point costs the caller nothing to honour.
        self.check_cancelled("bringing up the session's containers")?;

        // Pull the image unless it is already present locally; pulling a
        // large image is the slowest, most visible step, so report it.
        if self.image_present(&def.image) {
            progress(BringUpPhase::ImagePresent {
                image: def.image.clone(),
            });
        } else {
            progress(BringUpPhase::Pulling {
                image: def.image.clone(),
            });
            self.pull(&def.image)?;
            progress(BringUpPhase::Pulled {
                image: def.image.clone(),
            });
        }

        let mut state = ContainerState {
            provider: self.provider.clone(),
            image: def.image.clone(),
            network: Some(network.clone()),
            head_port,
            nodes: Vec::new(),
        };
        let mut clients: Vec<NodeClient> = Vec::new();

        // Whether this bring-up is the one that created the network. A
        // rollback must remove only what it made: the network may have
        // been there already — left by a run that was `SIGKILL`ed, or
        // created by something else entirely — and removing it would
        // disconnect whatever is using it.
        let network_existed = self.network_exists(&network);

        // Helper that removes anything created so far on failure.
        // Killing the clients first stops the containers; `rm -f` then
        // cleans up any that `--rm` has not caught up with yet.
        //
        // Removal goes through the same ownership check as
        // [`mirage_core::container::teardown`]: a container's name is
        // derived from the session id and is not proof that mirage
        // created it, and this is the one removal path that can run
        // against a resource this bring-up did not make.
        let rollback = |engine: &Engine, nodes: &[NodeContainer], clients: &mut Vec<NodeClient>| {
            for c in clients.iter_mut() {
                c.kill();
            }
            let state = ContainerState {
                provider: engine.provider.clone(),
                image: def.image.clone(),
                network: (!network_existed).then(|| network.clone()),
                head_port,
                nodes: nodes.to_vec(),
            };
            mirage_core::container::teardown(&state);
        };

        if network_existed {
            progress(BringUpPhase::NetworkExists {
                network: network.clone(),
            });
        } else {
            progress(BringUpPhase::CreatingNetwork {
                network: network.clone(),
            });
            self.ensure_network(&network, &labels)?;
        }

        // When the emulator requested host GPU access, expose the host's
        // GPU device nodes and the groups that own them on top of any
        // devices/groups the profile already configured. The group
        // passthrough mechanism itself is provider-specific and handled
        // in `run_argv`.
        let (devices, groups) = if host_gpus {
            let mut devices = def.devices.clone();
            devices.extend(host_gpu_devices());
            let mut groups = def.groups.clone();
            groups.extend(host_gpu_groups());
            (devices, groups)
        } else {
            (def.devices.clone(), def.groups.clone())
        };

        // Every node in the session gets the same container, so this is
        // built once and borrowed by each launch below; only the name,
        // the rank and the environment differ per rank.
        let spec = NodeSpec {
            session,
            image: &def.image,
            network: Some(&network),
            host_gpus,
            mounts: &mounts,
            ports: &def.ports,
            devices: &devices,
            groups: &groups,
            labels: &labels,
        };

        for rank in 0..node_count {
            let name = mirage_core::container::container_name(session, rank);
            progress(BringUpPhase::LaunchingNode {
                rank,
                total: node_count,
                name: name.clone(),
            });
            let env = node_env(rank);
            let launched = self
                .check_cancelled(&format!("starting node {} of {node_count}", rank + 1))
                .and_then(|()| self.launch_node(&name, &spec, &env, rank))
                .and_then(|mut client| {
                    // The client is spawned; the container is not up yet.
                    // Wait for it here rather than letting the first exec
                    // discover the race.
                    //
                    // `launch_node` hands back an owning `NodeClient`
                    // rather than a bare `Child`, so a failure here always
                    // has something that owns the process. Returning the
                    // `Child` and dropping it on the error path left the
                    // provider client running and unreaped —
                    // `std::process::Child` has no `Drop` — and its
                    // container out of `state.nodes`, which is the only
                    // list rollback removes. A slow node therefore leaked
                    // exactly the orphan container and zombie client this
                    // crate exists to prevent.
                    match self.await_running(&mut client, NODE_START_TIMEOUT) {
                        Ok(()) => Ok(client),
                        Err(e) => {
                            client.kill();
                            self.rm(&name);
                            Err(e)
                        }
                    }
                });
            match launched {
                Ok(client) => {
                    progress(BringUpPhase::NodeStarted {
                        rank,
                        total: node_count,
                        name: name.clone(),
                    });
                    clients.push(client);
                    state.nodes.push(NodeContainer { rank, name });
                }
                Err(e) => {
                    rollback(self, &state.nodes, &mut clients);
                    return Err(e);
                }
            }
        }

        // Every node says it is up. Wait a moment and ask again, because
        // the provider answers "running" about a container whose process
        // is still deciding whether it can run at all; see
        // [`NODE_SETTLE`]. A node that dies in this window is reported
        // here, with its own exit status and last words, instead of
        // reaching the caller as a healthy session whose first exec
        // fails with the engine's words about a missing container.
        if !clients.is_empty() {
            std::thread::sleep(NODE_SETTLE);
            let mut died = None;
            for client in &mut clients {
                if !client.alive() {
                    died = Some(client.exited(NODE_SETTLE));
                    break;
                }
            }
            if let Some(e) = died {
                rollback(self, &state.nodes, &mut clients);
                return Err(e);
            }
        }

        Ok((state, clients))
    }

    // ---- private command plumbing ---------------------------------

    /// Run the provider with `args`, succeeding only on a zero exit.
    fn checked(&self, args: &[String]) -> Result<()> {
        let output = spawn_retrying_etxtbsy(|| {
            Command::new(&self.provider)
                .args(args)
                .stdin(Stdio::null())
                .output()
        })
        .map_err(|source| ContainerError::Spawn {
            provider: self.provider.clone(),
            args: args.to_vec(),
            source,
        })?;
        if output.status.success() {
            Ok(())
        } else {
            Err(ContainerError::Command {
                provider: self.provider.clone(),
                args: args.to_vec(),
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            })
        }
    }

    /// Run the provider with `args` and return whether it exited zero.
    fn status(&self, args: &[String]) -> Result<bool> {
        let status = spawn_retrying_etxtbsy(|| {
            Command::new(&self.provider)
                .args(args)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
        })
        .map_err(|source| ContainerError::Spawn {
            provider: self.provider.clone(),
            args: args.to_vec(),
            source,
        })?;
        Ok(status.success())
    }

    /// Run the provider with `args`, returning captured stdout on a zero
    /// exit.
    fn output(&self, args: &[String]) -> Result<Vec<u8>> {
        let output = spawn_retrying_etxtbsy(|| {
            Command::new(&self.provider)
                .args(args)
                .stdin(Stdio::null())
                .output()
        })
        .map_err(|source| ContainerError::Spawn {
            provider: self.provider.clone(),
            args: args.to_vec(),
            source,
        })?;
        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(ContainerError::Command {
                provider: self.provider.clone(),
                args: args.to_vec(),
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            })
        }
    }
}

/// Spawn a command, transparently retrying the transient spawn failures.
///
/// Delegates to [`mirage_core::container::retrying_etxtbsy`] rather than
/// keeping a second copy of the policy. The two had already diverged: this
/// one retried only `ETXTBSY`, so a `fork` that failed with `EAGAIN` under
/// the process-table pressure of a wide emulated job failed the whole
/// bring-up, while the identical call shape in `mirage_core` retried and
/// succeeded. Bring-up is the path that can least afford it.
fn spawn_retrying_etxtbsy<T>(run: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    mirage_core::container::retrying_etxtbsy(run)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

    /// The session every test in here brings up.
    fn session() -> &'static mirage_core::session::SessionId {
        static SESSION: std::sync::OnceLock<mirage_core::session::SessionId> =
            std::sync::OnceLock::new();
        SESSION.get_or_init(|| mirage_core::session::SessionId::new("s").unwrap())
    }

    /// The ownership labels bring-up stamps on every resource.
    fn labels() -> Vec<(String, String)> {
        mirage_core::container::owner_labels(session())
    }

    fn mount(spec: &str) -> FileMount {
        FileMount::parse(spec).unwrap()
    }

    fn port(spec: &str) -> PortMapping {
        PortMapping::parse(spec).unwrap()
    }

    /// The plainest node there is: an image and nothing else. Tests that
    /// care about one flag override just that field with `..bare_spec()`,
    /// so what a case is actually about is the only thing written down.
    fn bare_spec() -> NodeSpec<'static> {
        NodeSpec {
            session: session(),
            image: "img",
            network: None,
            host_gpus: false,
            mounts: &[],
            ports: &[],
            devices: &[],
            groups: &[],
            labels: &[],
        }
    }

    /// Mock provider: logs every invocation to `log`, exits non-zero for
    /// `network inspect` (so `ensure_network` takes the create path) and
    /// `image inspect` (so callers think the image is absent), and prints
    /// a fake id on stdout for everything else.
    ///
    /// A `run` also logs the ownership marks it was *given* — the
    /// environment of the client process, which no argv records and which
    /// is the only thing a reclaim can read off a stranded client — and
    /// then *stays in the foreground*, as a real client does. A `run`
    /// that returned immediately would be a container that stopped the
    /// instant it started, which is a thing bring-up now (rightly)
    /// refuses to call a healthy session.
    fn mock_provider(dir: &Path, log: &Path) -> std::path::PathBuf {
        let provider = dir.join("mock-provider.sh");
        let script = format!(
            "#!/bin/sh\necho \"$@\" >> {log}\n\
             if [ \"$1\" = run ]; then \
             echo \"client-env MIRAGE_SESSION=${{MIRAGE_SESSION-unset}} \
             MIRAGE_RUNTIME=${{MIRAGE_RUNTIME-unset}}\" >> {log}; \
             echo fake-cid-123; exec sleep 30; fi\n\
             if [ \"$1\" = network ] && [ \"$2\" = inspect ]; then exit 1; fi\n\
             if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then exit 1; fi\n\
             if [ \"$1\" = inspect ]; then echo true; exit 0; fi\n\
             echo fake-cid-123\n",
            log = log.display()
        );
        std::fs::write(&provider, script).unwrap();
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();
        provider
    }

    /// Wait for the provider log to contain `needle`, and return it.
    ///
    /// The client writes its line and then blocks for the rest of its
    /// life, so a test that reads the log the moment `launch_node`
    /// returns races the `echo` rather than the container.
    fn log_containing(log: &Path, needle: &str) -> String {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let recorded = std::fs::read_to_string(log).unwrap_or_default();
            if recorded.contains(needle) {
                return recorded;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the provider never recorded {needle:?}:\n{recorded}"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// A provider script that is whatever the test needs it to be.
    fn scripted_provider(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let provider = dir.join(name);
        std::fs::write(&provider, format!("#!/bin/sh\n{body}")).unwrap();
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();
        provider
    }

    /// A containerised profile pointing at `provider`, with `mounts`.
    fn def_with(provider: &Path, mounts: Vec<FileMount>) -> ContainerizedDef {
        ContainerizedDef {
            provider: Some(provider.to_string_lossy().to_string()),
            image: "img:latest".to_string(),
            mounts,
            ports: vec![],
            devices: vec![],
            groups: vec![],
            hacks: vec![],
        }
    }

    /// Bring up a one-node session on `engine` with `def`, discarding the
    /// progress reports.
    fn bring_up_one(engine: &Engine, def: &ContainerizedDef) -> Result<()> {
        engine
            .bring_up(session(), def, false, 1, 6000, |_| vec![], |_| {})
            .map(|_| ())
    }

    #[test]
    fn run_argv_includes_network_env_and_mounts() {
        let env = vec![
            ("MIRAGE_RANK".to_string(), "0".to_string()),
            ("MIRAGE_HEAD_PORT".to_string(), "5000".to_string()),
        ];
        let mounts = vec![mount("/data:/data:ro"), mount("/h:/c")];
        let ports = vec![port("8080:8000"), port("53:53/udp")];
        let devices = vec!["/dev/kfd".to_string(), "/dev/dri".to_string()];
        let groups = vec!["video".to_string(), "render".to_string()];
        let command = vec![
            "/mnt/mirage/bin/mirage".to_string(),
            "host".to_string(),
            "--session".to_string(),
            "s".to_string(),
            "--rank".to_string(),
            "0".to_string(),
        ];
        let argv = Engine::run_argv(
            "podman",
            "mirage-s-node-0",
            &NodeSpec {
                session: session(),
                image: "img:latest",
                network: Some("mirage-s"),
                host_gpus: true,
                mounts: &mounts,
                ports: &ports,
                devices: &devices,
                groups: &groups,
                labels: &labels(),
            },
            &env,
            &command,
        );

        let joined = argv.join(" ");
        assert!(joined.starts_with("run --rm --name mirage-s-node-0 --hostname mirage-s-node-0"));
        assert!(joined.contains("--security-opt seccomp=unconfined"));
        assert!(joined.contains("--group-add keep-groups"));
        assert!(joined.contains("--network mirage-s"));
        assert!(joined.contains("-e MIRAGE_RANK=0"));
        assert!(joined.contains("-e MIRAGE_HEAD_PORT=5000"));
        assert!(joined.contains("-v /data:/data:ro"));
        assert!(joined.contains("-v /h:/c"));
        assert!(joined.contains("-p 8080:8000"));
        assert!(joined.contains("-p 53:53/udp"));
        assert!(joined.contains("--device /dev/kfd"));
        assert!(joined.contains("--device /dev/dri"));
        // On podman the named groups are dropped: `--group-add
        // keep-groups` cannot be combined with other `--group-add`
        // options, and already inherits them from the host.
        assert!(!joined.contains("--group-add video"));
        assert!(!joined.contains("--group-add render"));
        // The first command element overrides the image ENTRYPOINT; the
        // rest are its arguments after the image.
        assert!(joined.contains("--entrypoint /mnt/mirage/bin/mirage"));
        assert!(joined.ends_with("img:latest host --session s --rank 0"));
    }

    /// Pins the *whole* argv, element by element, for a spec that
    /// exercises every list `run_argv` can emit.
    ///
    /// The other `run_argv` tests each assert one property and would all
    /// still pass if two same-typed lists — `devices` and `groups`, say —
    /// swapped places, or if the flags moved relative to each other. This
    /// one would not: it is the regression net for any change that is
    /// supposed to leave the command handed to podman/docker alone.
    #[test]
    fn run_argv_is_pinned_element_by_element() {
        // docker rather than podman: it is the branch that emits the
        // named groups, so `--group-add` and `--device` — two `&[String]`
        // lists that a transposition would silently exchange — both
        // appear in the pinned argv.
        let mounts = vec![mount("/data:/data:ro")];
        let ports = vec![port("8080:8000")];
        let devices = vec!["/dev/kfd".to_string()];
        let groups = vec!["render".to_string()];
        let env = vec![("MIRAGE_RANK".to_string(), "0".to_string())];
        let labels = vec![("mirage.owner".to_string(), "mirage".to_string())];
        let command = vec!["/bin/mirage".to_string(), "host".to_string()];

        let argv = Engine::run_argv(
            "docker",
            "mirage-s-node-0",
            &NodeSpec {
                session: session(),
                image: "img:latest",
                network: Some("mirage-s"),
                host_gpus: true,
                mounts: &mounts,
                ports: &ports,
                devices: &devices,
                groups: &groups,
                labels: &labels,
            },
            &env,
            &command,
        );

        assert_eq!(
            argv,
            vec![
                "run",
                "--rm",
                "--name",
                "mirage-s-node-0",
                "--hostname",
                "mirage-s-node-0",
                "--label",
                "mirage.owner=mirage",
                "--security-opt",
                "seccomp=unconfined",
                "--group-add",
                "render",
                "--network",
                "mirage-s",
                "-e",
                "MIRAGE_RANK=0",
                "-v",
                "/data:/data:ro",
                "-p",
                "8080:8000",
                "--device",
                "/dev/kfd",
                "--entrypoint",
                "/bin/mirage",
                "img:latest",
                "host",
            ]
        );
    }

    #[test]
    fn run_argv_docker_host_gpus_adds_named_groups() {
        // docker has no `keep-groups`; the named GPU groups are added
        // explicitly so the workload can open the device nodes.
        let groups = vec!["video".to_string(), "render".to_string()];
        let argv = Engine::run_argv(
            "docker",
            "n",
            &NodeSpec {
                host_gpus: true,
                groups: &groups,
                ..bare_spec()
            },
            &[],
            &[],
        );
        let joined = argv.join(" ");
        assert!(joined.contains("--security-opt seccomp=unconfined"));
        assert!(!joined.contains("keep-groups"));
        assert!(joined.contains("--group-add video"));
        assert!(joined.contains("--group-add render"));
    }

    #[test]
    fn run_argv_without_host_gpus_omits_group_passthrough() {
        // Plain (non-GPU) containers emit no group passthrough at all, so
        // docker — which rejects `keep-groups` — keeps working.
        let groups = vec!["video".to_string()];
        let argv = Engine::run_argv(
            "docker",
            "n",
            &NodeSpec {
                groups: &groups,
                ..bare_spec()
            },
            &[],
            &[],
        );
        let joined = argv.join(" ");
        assert!(!joined.contains("--group-add"));
        assert!(!joined.contains("keep-groups"));
        assert!(!joined.contains("seccomp=unconfined"));
    }

    #[test]
    fn run_argv_omits_network_when_none() {
        let command = vec!["sleep".to_string(), "infinity".to_string()];
        let argv = Engine::run_argv("podman", "n", &bare_spec(), &[], &command);
        assert!(!argv.iter().any(|a| a == "--network"));
        assert_eq!(argv.last().map(String::as_str), Some("infinity"));
        // `sleep` overrides the entrypoint; `infinity` is its argument.
        assert!(argv.iter().any(|a| a == "--entrypoint"));
        let ep = argv.iter().position(|a| a == "--entrypoint").unwrap();
        assert_eq!(argv[ep + 1], "sleep");
    }

    #[test]
    fn exec_argv_has_workdir_env_and_command() {
        let env = vec![("K".to_string(), "V".to_string())];
        let argv = Engine::exec_argv(
            "mirage-s-node-1",
            Some("/work"),
            &env,
            "/bin/echo",
            &["hi".to_string(), "there".to_string()],
            false,
        );
        assert_eq!(
            argv,
            vec![
                "exec",
                "-i",
                "-w",
                "/work",
                "-e",
                "K=V",
                "mirage-s-node-1",
                "/bin/echo",
                "hi",
                "there",
            ]
        );
    }

    #[test]
    fn an_interactive_exec_asks_the_provider_for_a_terminal() {
        // `provider exec` gives the in-container process pipes, not the
        // caller's descriptors — they are in different namespaces — so
        // without `-t` `isatty(0)` is false however good the caller's
        // terminal is, and `bash` prints no prompt at all.
        let argv = Engine::exec_argv("c", None, &[], "bash", &[], true);
        assert_eq!(argv, vec!["exec", "-i", "-t", "c", "bash"]);
    }

    #[test]
    fn a_non_interactive_exec_gets_no_terminal() {
        // `-t` merges stderr into stdout, so it may only be asked for
        // when every stream was going to the same terminal anyway.
        let argv = Engine::exec_argv("c", None, &[], "bash", &[], false);
        assert_eq!(argv, vec!["exec", "-i", "c", "bash"]);
        assert!(!argv.contains(&"-t".to_string()), "{argv:?}");
    }

    #[test]
    fn exec_command_line_prefixes_provider() {
        let engine = Engine::with_provider("podman");
        let line = engine.exec_command_line("c", None, &[], "ls", &[], false);
        assert_eq!(line, vec!["podman", "exec", "-i", "c", "ls"]);
        let line = engine.exec_command_line("c", None, &[], "ls", &[], true);
        assert_eq!(line, vec!["podman", "exec", "-i", "-t", "c", "ls"]);
    }

    #[test]
    fn resolve_uses_explicit_provider() {
        let def = ContainerizedDef {
            provider: Some("docker".to_string()),
            image: "img".to_string(),
            mounts: vec![],
            ports: vec![],
            devices: vec![],
            groups: vec![],
            hacks: vec![],
        };
        let engine = Engine::resolve(&def).unwrap();
        assert_eq!(engine.provider(), "docker");
    }

    #[test]
    fn pull_invokes_provider() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        engine.pull("img:latest").unwrap();
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(recorded.contains("pull img:latest"), "{recorded:?}");
    }

    #[test]
    fn image_present_false_when_inspect_fails() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());
        assert!(!engine.image_present("img"));
    }

    #[test]
    fn ensure_network_creates_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        engine.ensure_network("mirage-s", &labels()).unwrap();
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            recorded.contains("network inspect mirage-s"),
            "{recorded:?}"
        );
        // Labelled, so teardown can prove the network is mirage's before
        // removing it and orphan reclamation can find it later.
        assert!(
            recorded.contains("network create --label mirage.owner=mirage")
                && recorded.contains("--label mirage.session=s")
                && recorded.contains(" mirage-s"),
            "{recorded:?}"
        );
    }

    #[test]
    fn launch_node_owns_the_client_and_never_detaches() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let mut client = engine
            .launch_node(
                "mirage-s-node-0",
                &NodeSpec {
                    network: Some("mirage-s"),
                    labels: &labels(),
                    ..bare_spec()
                },
                &[],
                0,
            )
            .unwrap();
        assert_eq!(client.rank, 0);
        assert_eq!(client.name, "mirage-s-node-0");
        // The client is still there, holding its container, until
        // somebody stops it — which is the whole ownership model.
        let recorded = log_containing(&log, "run --rm");
        assert!(client.alive(), "the client must own the container's life");
        assert!(
            recorded.contains("run --rm --name mirage-s-node-0"),
            "the container must be launched attached and self-removing: {recorded:?}"
        );
        assert!(
            !recorded.split_whitespace().any(|a| a == "-d"),
            "a detached container outlives the run that owns it: {recorded:?}"
        );
    }

    #[test]
    fn killing_a_node_client_is_idempotent() {
        // Teardown kills explicitly and `Drop` kills again; the second
        // call must not panic or block on an already-reaped child.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let mut client = engine
            .launch_node(
                "n",
                &NodeSpec {
                    labels: &labels(),
                    ..bare_spec()
                },
                &[],
                0,
            )
            .unwrap();
        client.kill();
        assert!(!client.alive());
        client.kill();
    }

    #[test]
    fn the_provider_client_carries_the_session_that_owns_it() {
        // The client is the one process that survives a `SIGKILL`ed run
        // still holding a container, and nothing on disk records that it
        // exists. Its environment is the only evidence of whose it was,
        // which is exactly what `mirage cleanup` reads — and the marker
        // is documented as being set on "every container provider client
        // mirage spawns", so an unmarked client makes that a lie and the
        // stranded container unattributable.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let _client = engine
            .launch_node("mirage-s-node-0", &bare_spec(), &[], 0)
            .unwrap();

        let recorded = log_containing(&log, "client-env");
        assert!(
            recorded.contains(&format!(
                "client-env MIRAGE_SESSION={} MIRAGE_RUNTIME={}",
                session().as_str(),
                mirage_core::container::owning_runtime()
            )),
            "the provider client was spawned unmarked:\n{recorded}"
        );
    }

    #[test]
    fn a_client_that_refuses_says_why() {
        // The reason a `podman run` fails is the single most useful thing
        // about a failed bring-up, and it used to go to `/dev/null`: the
        // user was told the container "did not start" and nothing else.
        let dir = tempfile::tempdir().unwrap();
        let provider = dir.path().join("refusing-provider.sh");
        std::fs::write(
            &provider,
            "#!/bin/sh\n\
             if [ \"$1\" = run ]; then echo 'Error: no such device /dev/kfd' >&2; exit 125; fi\n\
             if [ \"$1\" = inspect ]; then echo false; exit 0; fi\n\
             exit 0\n",
        )
        .unwrap();
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let mut client = engine
            .launch_node(
                "n",
                &NodeSpec {
                    labels: &labels(),
                    ..bare_spec()
                },
                &[],
                0,
            )
            .unwrap();
        let err = engine
            .await_running(&mut client, std::time::Duration::from_secs(30))
            .unwrap_err();

        // Fails fast rather than polling out the full timeout, and the
        // engine's own words come back with it.
        assert!(
            matches!(err, ContainerError::ClientExited { .. }),
            "a client that exited must not be reported as a timeout: {err}"
        );
        assert!(
            err.to_string().contains("no such device /dev/kfd"),
            "the provider's reason was lost: {err}"
        );
        // And its exit status, which is the half that distinguishes a
        // provider refusing (125) from a container that ran and died.
        assert!(
            err.to_string().contains("exit status 125"),
            "the client's exit status was lost: {err}"
        );
    }

    #[test]
    fn a_node_that_dies_just_after_reporting_up_fails_the_bring_up() {
        // The provider answers "running" about a container whose process
        // is still deciding whether it can run: an alpine image handed a
        // glibc `LD_PRELOAD` reports itself up some 380ms before its
        // entrypoint gives up. Bring-up used to believe the first
        // answer, so the session went ready and the *first exec* failed
        // with the engine's words about a container id nobody had seen —
        // exit 255, no mention of the image, the mount or the preload.
        //
        // The container's last words here go to stdout, which mirage
        // discarded until it was pointed out that a dying process writes
        // wherever it likes.
        let dir = tempfile::tempdir().unwrap();
        let provider = scripted_provider(
            dir.path(),
            "dying-provider.sh",
            "case \"$1\" in\n\
             run) sleep 0.1; echo 'Error relocating /mnt/mirage/lib/librocjitsu.so: \
             symbol not found'; exit 127 ;;\n\
             inspect) echo true; exit 0 ;;\n\
             *) exit 0 ;;\n\
             esac\n",
        );
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let err = bring_up_one(&engine, &def_with(&provider, vec![])).unwrap_err();
        let message = err.to_string();
        assert!(
            matches!(err, ContainerError::ClientExited { .. }),
            "a node that died must fail bring-up rather than reach the first exec: {message}"
        );
        assert!(
            message.contains("mirage-s-node-0")
                && message.contains("exit status 127")
                && message.contains("Error relocating"),
            "the error must name the container, how it ended and what it said: {message}"
        );
    }

    #[test]
    fn a_container_that_dies_mid_session_can_say_why() {
        // Nothing re-publishes a session's health once it is ready, so
        // the death of a node container is discovered by whoever next
        // asks — and all that caller has to go on is this. "A node
        // container has exited" is a restatement of the question; the
        // engine's status and the container's last words are the answer.
        let dir = tempfile::tempdir().unwrap();
        let provider = scripted_provider(
            dir.path(),
            "oom-provider.sh",
            "case \"$1\" in\n\
             run) echo 'container mirage-s-node-0 was killed by the OOM killer' >&2; \
             exit 137 ;;\n\
             *) exit 0 ;;\n\
             esac\n",
        );
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let mut client = engine
            .launch_node("mirage-s-node-0", &bare_spec(), &[], 0)
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let report = loop {
            if let Some(report) = client.death_report() {
                break report;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the client never exited"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        assert!(
            report.contains("mirage-s-node-0")
                && report.contains("exit status 137")
                && report.contains("OOM killer"),
            "the report must name the container, how it ended and what it said: {report}"
        );
        // And a client that is still there has nothing to report, which
        // is what makes this safe to ask on the healthy path.
        let log = dir.path().join("log");
        let healthy = Engine::with_provider(
            mock_provider(dir.path(), &log)
                .to_string_lossy()
                .to_string(),
        );
        let mut client = healthy
            .launch_node("mirage-s-node-1", &bare_spec(), &[], 1)
            .unwrap();
        assert_eq!(client.death_report(), None);
    }

    #[test]
    fn bring_up_refuses_a_mount_whose_host_path_does_not_exist() {
        // docker creates the missing path as a root-owned directory on
        // the host and starts the container; podman refuses. Mirage
        // decides, before either of them is asked, and says which path
        // it means.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());
        let missing = dir.path().join("not-here");

        let err = bring_up_one(
            &engine,
            &def_with(
                &provider,
                vec![mount(&format!("{}:/data", missing.display()))],
            ),
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(
            matches!(err, ContainerError::Mount { .. }),
            "a missing host path must be mirage's refusal, not the provider's: {message}"
        );
        assert!(
            message.contains(&missing.display().to_string()) && message.contains("does not exist"),
            "the error must name the path that is missing: {message}"
        );
        // And nothing was created on the way to finding out — not even
        // the pull, which is minutes the user would have waited before
        // being told about a typo.
        let recorded = std::fs::read_to_string(&log).unwrap_or_default();
        assert!(
            recorded.is_empty(),
            "the provider was asked to do something before the mount was checked:\n{recorded}"
        );
        assert!(
            !missing.exists(),
            "mirage created the host path it refused to mount"
        );
    }

    #[test]
    fn bring_up_makes_a_relative_mount_absolute() {
        // A `-v` source with no leading separator is a *named volume* to
        // both engines, so `--mount data:/data` used to mount an empty
        // volume called `data` instead of the directory the user meant —
        // and leave it behind afterwards, where `mirage cleanup` (which
        // knows about containers and networks) never looks.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        // A relative path that exists: this crate's own manifest, next to
        // the working directory `cargo test` runs a unit test in.
        bring_up_one(
            &engine,
            &def_with(&provider, vec![mount("Cargo.toml:/m:ro")]),
        )
        .unwrap();

        let expected = std::env::current_dir().unwrap().join("Cargo.toml");
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            recorded.contains(&format!("-v {}:/m:ro", expected.display())),
            "the host path reached the provider unresolved:\n{recorded}"
        );
        assert!(
            !recorded.contains("-v Cargo.toml:"),
            "a relative host path became a named volume:\n{recorded}"
        );
    }

    #[test]
    fn a_pull_stops_when_the_caller_cancels() {
        // Pulling an image is minutes of a blocking child process, and a
        // user who changes their mind wants the prompt back now. Nothing
        // outside the engine can hurry it: `podman pull` is not a future
        // that can be dropped, it is a process that has to be killed.
        let dir = tempfile::tempdir().unwrap();
        let provider = scripted_provider(
            dir.path(),
            "slow-provider.sh",
            "case \"$1\" in\n\
             pull) sleep 30; exit 0 ;;\n\
             *) exit 0 ;;\n\
             esac\n",
        );
        let cancel = Cancel::new();
        let engine = Engine::with_provider(provider.to_string_lossy().to_string())
            .with_cancel(cancel.clone());

        let started = std::time::Instant::now();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            cancel.cancel();
        });
        let err = engine.pull("img:latest").unwrap_err();
        let waited = started.elapsed();

        assert!(
            matches!(err, ContainerError::Cancelled { .. }),
            "a cancelled pull must say so: {err}"
        );
        assert!(
            err.to_string().contains("pulling image img:latest"),
            "the error must name what was interrupted: {err}"
        );
        assert!(
            waited < std::time::Duration::from_secs(5),
            "the pull was waited out rather than ended: {waited:?}"
        );
    }

    #[test]
    fn a_cancelled_bring_up_creates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let cancel = Cancel::new();
        cancel.cancel();
        let engine =
            Engine::with_provider(provider.to_string_lossy().to_string()).with_cancel(cancel);

        let err = bring_up_one(&engine, &def_with(&provider, vec![])).unwrap_err();
        assert!(
            matches!(err, ContainerError::Cancelled { .. }),
            "a cancelled bring-up must say so: {err}"
        );
        let recorded = std::fs::read_to_string(&log).unwrap_or_default();
        assert!(
            !recorded.contains("run --rm"),
            "a cancelled bring-up started a container anyway:\n{recorded}"
        );
    }

    #[test]
    fn bring_up_pulls_creates_network_and_launches_each_node() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let session = mirage_core::session::SessionId::new("s").unwrap();
        let def = ContainerizedDef {
            provider: Some(provider.to_string_lossy().to_string()),
            image: "img:latest".to_string(),
            mounts: vec![],
            ports: vec![],
            devices: vec![],
            groups: vec![],
            hacks: vec![],
        };

        let mut phases: Vec<BringUpPhase> = Vec::new();
        let (state, clients) = engine
            .bring_up(
                &session,
                &def,
                false,
                2,
                6000,
                |rank| vec![("MIRAGE_RANK".to_string(), rank.to_string())],
                |phase| phases.push(phase),
            )
            .unwrap();

        assert_eq!(state.image, "img:latest");
        assert_eq!(state.network.as_deref(), Some("mirage-s"));
        assert_eq!(state.head_port, 6000);
        assert_eq!(state.nodes.len(), 2);
        assert_eq!(state.nodes[0].name, "mirage-s-node-0");
        assert_eq!(state.nodes[1].name, "mirage-s-node-1");
        // One owned provider client per rank: the session's containers
        // have an owner from the moment they exist.
        assert_eq!(clients.len(), 2);
        assert_eq!(clients[0].rank, 0);
        assert_eq!(clients[0].name, "mirage-s-node-0");
        assert_eq!(clients[1].rank, 1);
        assert_eq!(clients[1].name, "mirage-s-node-1");

        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(recorded.contains("pull img:latest"), "{recorded:?}");
        // Labelled, so teardown can prove the network is mirage's before
        // removing it and orphan reclamation can find it later.
        assert!(
            recorded.contains("network create --label mirage.owner=mirage")
                && recorded.contains("--label mirage.session=s")
                && recorded.contains(" mirage-s"),
            "{recorded:?}"
        );
        assert!(
            recorded.contains("run --rm --name mirage-s-node-0"),
            "{recorded:?}"
        );
        assert!(
            recorded.contains("run --rm --name mirage-s-node-1"),
            "{recorded:?}"
        );
        assert!(recorded.contains("-e MIRAGE_RANK=0"), "{recorded:?}");
        assert!(recorded.contains("-e MIRAGE_RANK=1"), "{recorded:?}");

        // The progress callback reports each bring-up phase in order: the
        // mock image-inspect fails, so the image is pulled, the network
        // is created, and each node is launched then confirmed started.
        assert_eq!(
            phases,
            vec![
                BringUpPhase::Pulling {
                    image: "img:latest".to_string()
                },
                BringUpPhase::Pulled {
                    image: "img:latest".to_string()
                },
                BringUpPhase::CreatingNetwork {
                    network: "mirage-s".to_string()
                },
                BringUpPhase::LaunchingNode {
                    rank: 0,
                    total: 2,
                    name: "mirage-s-node-0".to_string()
                },
                BringUpPhase::NodeStarted {
                    rank: 0,
                    total: 2,
                    name: "mirage-s-node-0".to_string()
                },
                BringUpPhase::LaunchingNode {
                    rank: 1,
                    total: 2,
                    name: "mirage-s-node-1".to_string()
                },
                BringUpPhase::NodeStarted {
                    rank: 1,
                    total: 2,
                    name: "mirage-s-node-1".to_string()
                },
            ]
        );
    }
}
