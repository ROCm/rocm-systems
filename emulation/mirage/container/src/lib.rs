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
            // Blank lines are dropped on the way in, because the ring is
            // read back as one `; `-joined line. An engine's refusal is
            // several paragraphs — `docker run` follows its error with an
            // empty line and `See 'docker run --help'` — and keeping the
            // gaps turned that into `…; ; See 'docker run --help'`, which
            // reads as though mirage had lost something.
            .filter(|line| !line.trim().is_empty())
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

/// The host port an engine's refusal is about, when that is what it is
/// about.
///
/// Matched on the engines' own words rather than on an exit code, which
/// is 125 for every way a `run` can be refused. docker says `Bind for
/// 0.0.0.0:8080 failed: port is already allocated`; podman's rootless
/// port forwarder says `cannot listen on the TCP port: listen tcp4
/// :8080: bind: address already in use`. Both name the port in a
/// `host:port` or `:port` token, which is what is read back out — the
/// number is the whole point of the message mirage writes instead, and a
/// remedy that cannot name the port is not much better than the engine's.
///
/// A message that matches but names no port is not a match: mirage would
/// be replacing the engine's specific words with vaguer ones of its own.
fn host_port_in_use(said: &str) -> Option<String> {
    const SIGNATURES: &[&str] = &["port is already allocated", "address already in use"];
    if !SIGNATURES.iter().any(|s| said.contains(s)) {
        return None;
    }
    said.split(|c: char| c.is_whitespace() || c == ',')
        .filter_map(|token| {
            let token = token.trim_end_matches([':', '.', ';', ')', '"', '\'']);
            let (_, port) = token.rsplit_once(':')?;
            (!port.is_empty() && port.bytes().all(|b| b.is_ascii_digit())).then(|| port.to_string())
        })
        .next()
}

/// An engine's words with its "now go and read the manual" trailer
/// removed.
///
/// `docker run` ends a refusal with `See 'docker run --help'.`, which is
/// advice for someone who mistyped a flag. Kept in a message that has
/// nothing else to offer, dropped from one where mirage supplies the
/// remedy itself — the pointer would then be the only imperative
/// sentence in the error, and the wrong one.
fn without_usage_pointer(said: &str) -> String {
    said.split("; ")
        .filter(|line| !(line.starts_with("See ") && line.contains("--help")))
        .collect::<Vec<_>>()
        .join("; ")
        .trim()
        .to_string()
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
    ///
    /// One refusal is picked out of the rest by name. See
    /// [`ContainerError::HostPortInUse`].
    fn exited(&self, waited: std::time::Duration) -> ContainerError {
        self.settle_output();
        let said = self.output_tail();
        if let Some(port) = host_port_in_use(&said) {
            return ContainerError::HostPortInUse {
                name: self.name.clone(),
                port,
                said: without_usage_pointer(&said),
            };
        }
        let (code, signal) = self.exit_codes();
        ContainerError::ClientExited {
            name: self.name.clone(),
            waited,
            code,
            signal,
            output: said,
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

    /// A node container could not start because the host port it
    /// publishes is held by something else.
    ///
    /// Carved out of [`ContainerError::ClientExited`] because it is the
    /// one refusal in that family whose fix belongs to the user rather
    /// than to the image: the engine reports it as a driver or
    /// forwarder failure, ends with `See 'docker run --help'`, and never
    /// mentions `--port` — so the reader is sent to the manual for a
    /// flag mirage passed on their behalf, about a conflict with a
    /// process that has nothing to do with this session.
    #[error(
        "container `{name}` could not start: host port {port} is already in use on this \
         machine, so the container could not publish onto it. Nothing of mirage's is holding \
         it — a `--port` conflict *within* a session is refused before bring-up — so this is \
         another program: `ss -ltnp \"sport = :{port}\"` names it. Publish a different host \
         port (`--port <other>:{port}`), or stop what is on it. The engine said: {said}"
    )]
    HostPortInUse {
        /// The container that could not be started.
        name: String,
        /// The host port it was asked to publish onto.
        port: String,
        /// What the engine said, minus its pointer at `--help`.
        said: String,
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

    /// A bind mount names a container path mirage has already spoken for.
    ///
    /// Every containerised session mounts mirage's own binary, config
    /// directory, session scratch and emulator libraries under
    /// [`mirage_core::container::CONTAINER_MIRAGE_DIR`]. A user mount at
    /// or above that path is laid over them, and the engine resolves the
    /// overlap by creating mirage's destinations *inside the user's host
    /// directory* — as root, since the container writes them — while the
    /// run reports success. Refused instead, naming both paths.
    #[error(
        "--mount {spec}: the container path `{container_path}` is at or above `{reserved}`, \
         which is where mirage bind-mounts its own binary, configuration and session scratch \
         inside every node container. The two overlap, and what the container creates under \
         mirage's paths lands in `{host_path}` on the host instead — root-owned `bin`, `config`, \
         `lib` and `runtime` entries you did not make and cannot delete. Mount it at some other \
         path in the container; everything outside `{reserved}` is yours."
    )]
    ReservedMount {
        /// The mount as the provider would have been given it
        /// (`HOST:CONTAINER[:ro]`).
        spec: String,
        /// The host path the user asked to mount.
        host_path: String,
        /// The container path they asked to mount it at.
        container_path: String,
        /// The tree mirage keeps for itself.
        reserved: &'static str,
    },

    /// Published ports were asked for on a session with several nodes.
    ///
    /// Every node of a session runs the same container with the same
    /// argv, so a published port is published by all of them onto the
    /// same host port. The first node binds it and the rest cannot, which
    /// used to fail bring-up halfway with the engine's own words about a
    /// port — after node 0 was already running.
    #[error(
        "--port {ports} cannot be published from a {nodes}-node session: every node runs the \
         same container, so all {nodes} of them would publish onto the same host port and only \
         the first could bind it. Refused up front rather than halfway through bring-up. Publish \
         from a single node (`--num-nodes 1`), or drop `--port` — nodes already reach each other \
         by container name on the session's own network."
    )]
    PortsMultiNode {
        /// The published ports, as the user spelled them.
        ports: String,
        /// How many nodes the session was asked for.
        nodes: u32,
    },

    /// Two different mappings want the same host port.
    ///
    /// Repeating an identical `--port` is deduplicated silently — saying
    /// the same thing twice is not an error. Two mappings that disagree
    /// are, and the engine's "address already in use" describes the
    /// symptom of it rather than the cause.
    #[error(
        "--port {first} and --port {second} both publish host port {host_port}/{protocol}, and \
         the host can only give it to one container. Pick a different host port for one of them."
    )]
    PortConflict {
        /// The mapping that claimed the host port first.
        first: String,
        /// The mapping that wanted it as well.
        second: String,
        /// The host port both asked for.
        host_port: u16,
        /// The protocol they both asked for it on.
        protocol: String,
    },

    /// The image reference is not one an engine can be asked for.
    #[error("--image {image:?}: {problem}")]
    Image {
        /// The image reference as configured.
        image: String,
        /// What is wrong with it, and what to do.
        problem: String,
    },

    /// The configured provider is not a container engine mirage can
    /// drive.
    ///
    /// Checked before the session is built, because everything mirage
    /// does with a provider is spawning it: an unusable one is otherwise
    /// discovered as a bare `No such file or directory` from somewhere in
    /// the middle of bring-up, attached to whichever step happened to be
    /// first.
    #[error("container provider `{provider}`: {problem}")]
    Provider {
        /// The provider as configured (a name, or a path).
        provider: String,
        /// What is wrong with it, and what to do.
        problem: String,
    },

    /// The working directory an exec asked for is not in the container.
    ///
    /// The host-side check cannot answer this: a containerised workload
    /// chdirs inside the container's own filesystem, where a path that
    /// exists out here usually does not exist at all. Left to the
    /// provider it surfaces as an OCI runtime error about `chdir to cwd`
    /// and the container's exit code, which names neither the flag nor
    /// the fact that the two filesystems are different.
    #[error(
        "--workdir {workdir}: there is no such directory inside container `{container}`. The \
         path has to exist in the *container*, not on this machine — the image has its own \
         filesystem, so a directory you can see here is not one the workload can enter. Name a \
         directory the image already has, or `--mount` one there."
    )]
    Workdir {
        /// The directory that was asked for.
        workdir: String,
        /// The container it was not found in.
        container: String,
    },

    /// The working directory an exec asked for is there, and is not a
    /// directory.
    ///
    /// A separate variant because [`ContainerError::Workdir`]'s two
    /// sentences are both wrong about it: the path *is* in the image, so
    /// "there is no such directory" sends the user looking for a
    /// spelling mistake that is not there, and `--mount`ing something at
    /// an occupied path is not the remedy either. Distinguished by the
    /// probe rather than assumed, so an image that has the path as a
    /// file, a socket or a dangling symlink is told apart from one that
    /// does not have it at all.
    #[error(
        "--workdir {workdir}: inside container `{container}` that path exists but is not a \
         directory, so nothing can be started in it. Name the directory that holds it, or \
         another the image already has."
    )]
    WorkdirNotADirectory {
        /// The path that was asked for.
        workdir: String,
        /// The container it was found in.
        container: String,
    },

    /// The working directory an exec asked for is not an absolute path.
    ///
    /// The host-side check skips a containerised `--workdir` entirely —
    /// it names a path in the image, which this machine cannot answer
    /// for — and a relative one then reached the engine, which refuses
    /// it with `workdir must be an absolute path` and an OCI runtime
    /// error naming a container the user has never seen. It is also the
    /// one thing about a container path that can be settled without
    /// asking the container.
    #[error(
        "--workdir {workdir}: a containerised working directory must be an absolute path. \
         There is nothing for a relative one to be relative *to*: the host directory you are \
         standing in is not the container's, and the workload would otherwise start wherever \
         the image's own `WORKDIR` happens to be. Write the path in full, e.g. `/{workdir}`."
    )]
    WorkdirRelative {
        /// The path that was asked for, as the user spelled it.
        workdir: String,
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

/// Resolve every bind mount, refusing the ones no container should be
/// given.
///
/// The *container* side is refused for one reason, which is that mirage
/// mounts things there too: see
/// [`ContainerError::ReservedMount`]. Two shapes of *host* path are
/// trouble, and each is trouble differently on the two engines mirage
/// drives:
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

/// Resolve and check one bind mount.
fn resolve_mount(mount: &FileMount) -> Result<FileMount> {
    let refuse = |path: &std::path::Path, problem: String| ContainerError::Mount {
        spec: mount.to_volume_arg(),
        path: path.display().to_string(),
        problem,
    };

    // Checked first, and on the unresolved spec: this is the one failure
    // where the *container* path is the mistake, and resolving the host
    // side would only put a path the user did not type into the error.
    //
    // Mirage's own mounts are all strictly below the reserved directory
    // and so are never caught by this — which is what makes the check
    // safe to apply here, after the supervisor has already added them to
    // the same list.
    if mirage_core::container::covers_mirage_dir(&mount.container_path) {
        return Err(ContainerError::ReservedMount {
            spec: mount.to_volume_arg(),
            host_path: mount.host_path.clone(),
            container_path: mount.container_path.clone(),
            reserved: mirage_core::container::CONTAINER_MIRAGE_DIR,
        });
    }

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

/// The protocol a `-p` mapping means when it does not say. Both engines
/// default to TCP, so two mappings that differ only in whether they spell
/// it out are asking for the same host port.
const DEFAULT_PROTOCOL: &str = "tcp";

/// The ports every node container publishes, refusing the sets no engine
/// could honour and collapsing the ones that only look like two.
///
/// A published port is a host resource, and a session's nodes are
/// identical containers — which makes two of the shapes a user can write
/// impossible rather than merely unlucky:
///
/// * **The same port twice.** `--port 8080` on a profile that already
///   publishes 8080, or simply typed twice, reached the engine as two
///   `-p` arguments and failed the container with `address already in
///   use` — a message about a port conflict with *somebody else*, which
///   sent the user looking for a process that was not there. Saying the
///   same thing twice is not a mistake, so the duplicate is dropped.
/// * **Any port at all on more than one node.** Every node gets the same
///   argv, so all of them publish onto the same host port; node 0 binds
///   it and node 1 cannot. That used to be discovered halfway through
///   bring-up, with the first node already running.
///
/// Two *different* mappings for one host port are a real conflict and are
/// refused as one — see [`ContainerError::PortConflict`].
fn resolve_ports(ports: &[PortMapping], node_count: u32) -> Result<Vec<PortMapping>> {
    let protocol_of = |p: &PortMapping| p.protocol.clone().unwrap_or(DEFAULT_PROTOCOL.to_string());
    let mut kept: Vec<PortMapping> = Vec::with_capacity(ports.len());
    for port in ports {
        let protocol = protocol_of(port);
        if let Some(prior) = kept
            .iter()
            .find(|k| k.host_port == port.host_port && protocol_of(k) == protocol)
        {
            if prior.container_port == port.container_port {
                continue;
            }
            return Err(ContainerError::PortConflict {
                first: prior.to_publish_arg(),
                second: port.to_publish_arg(),
                host_port: port.host_port,
                protocol,
            });
        }
        kept.push(port.clone());
    }
    if node_count > 1 && !kept.is_empty() {
        return Err(ContainerError::PortsMultiNode {
            ports: kept
                .iter()
                .map(PortMapping::to_publish_arg)
                .collect::<Vec<_>>()
                .join(", "),
            nodes: node_count,
        });
    }
    Ok(kept)
}

/// Refuse an image reference no engine can be asked for.
///
/// Only the empty one, which is the reference a `--image ""` produces and
/// the only one mirage can be sure about: everything else is the
/// registry's opinion, and guessing at it here would refuse images that
/// work. Empty is worth catching precisely because it does not look like
/// a failure — it reaches the provider as a missing argument, and the
/// user watches `pulling image  (this can take a while)` with a hole in
/// the middle of it.
fn check_image(image: &str) -> Result<()> {
    if image.trim().is_empty() {
        return Err(ContainerError::Image {
            image: image.to_string(),
            problem: "a containerised session runs every node in an image, and this names none. \
                      Pass `--image <reference>` (for example `--image ubuntu:24.04`), or give \
                      the profile one with `mirage profile create … --image <reference>`."
                .to_string(),
        });
    }
    Ok(())
}

/// Refuse a provider that is not an executable behaving like a container
/// engine.
///
/// Everything mirage does to a container is a `Command` built around this
/// string, so a wrong one is not caught until something is spawned — and
/// what the user then sees is whichever step happened to be first,
/// wearing the OS's words for it: `failed to spawn `whale pull img`: No
/// such file or directory`. Asked here instead, once, before the session
/// exists.
///
/// The probe is `--version`, and it is the weakest question that
/// distinguishes an engine from an arbitrary executable: podman and
/// docker both answer it in milliseconds without touching a daemon, a
/// registry or the network, and a wrapper script around either will pass
/// it on. Mirage deliberately does not check the *name* — the provider is
/// allowed to be a path to a wrapper, and half this workspace's tests
/// drive a shell script standing in for an engine.
fn check_provider(provider: &str) -> Result<()> {
    use std::os::unix::process::ExitStatusExt as _;

    let refuse = |problem: String| ContainerError::Provider {
        provider: provider.to_string(),
        problem,
    };
    if mirage_core::container::provider_binary(provider).is_none() {
        return Err(refuse(format!(
            "{}. mirage drives `podman` or `docker`; name one of those, or the path to a \
             container engine — or name none at all and mirage will use whichever is installed",
            if provider.contains('/') {
                "there is no such file"
            } else {
                "not found on PATH"
            }
        )));
    }
    let probed = mirage_core::container::retrying_etxtbsy(|| {
        Command::new(provider)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
    });
    match probed {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(refuse(format!(
            "`{provider} --version` failed ({}), so this does not look like a container engine. \
             mirage speaks the podman/docker command line and nothing else",
            exit_phrase(&status.code(), &status.signal())
        ))),
        Err(e) => Err(refuse(format!(
            "cannot be run: {e}. It has to be an executable mirage can spawn"
        ))),
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
    /// Every node has reported itself up, and bring-up is giving them
    /// [`NODE_SETTLE`] to prove they meant it.
    ///
    /// Its own phase because it is its own failure: a caller that names
    /// the last phase it saw when reporting an error — which is what the
    /// supervisor does — otherwise attributed a node dying in this window
    /// to the phase before it, and printed
    /// `node 1/1 (mirage-s-node-0) started failed: …` about a container
    /// that had just stopped.
    Settling { total: u32 },
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
            BringUpPhase::LaunchingNode { .. }
            | BringUpPhase::NodeStarted { .. }
            | BringUpPhase::Settling { .. } => "starting",
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
            BringUpPhase::Settling { total } => {
                let containers = if *total == 1 {
                    "container"
                } else {
                    "containers"
                };
                format!("waiting for {total} node {containers} to stay up")
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
    /// docker)" policy — see
    /// [`resolve_provider`](mirage_core::container::resolve_provider) for
    /// why the environment variable sits where it does. Errors with
    /// [`ContainerError::NoProvider`] when nothing is available.
    ///
    /// The two things a containerised session cannot start without — an
    /// engine that runs, and an image to run — are checked here rather
    /// than left to the first provider invocation that trips over them.
    /// This is the earliest point that has the whole definition in hand:
    /// it runs before the derived-image build, before the pull, and
    /// before anything exists to tear down again. Everything downstream
    /// of it reports failures in the words of whichever step was
    /// unlucky, which is how `--image ""` became `pulling image  (this
    /// can take a while)` and a mistyped provider became a bare `No such
    /// file or directory`.
    ///
    /// # Errors
    ///
    /// [`ContainerError::Provider`] when the configured provider is not
    /// an executable that answers as a container engine, and
    /// [`ContainerError::Image`] when the image reference is empty.
    pub fn resolve(def: &ContainerizedDef) -> Result<Self> {
        check_image(&def.image)?;
        let provider = mirage_core::container::resolve_provider(def.provider.as_deref())
            .ok_or(ContainerError::NoProvider)?;
        check_provider(&provider)?;
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
    /// The image is labelled as mirage's, like every container and
    /// network mirage creates. A derived image is host state that
    /// survives the run that built it — deliberately, since the point of
    /// keying it by base image plus hacks is to build it once — so the
    /// only thing that can ever attribute it later is a mark on the image
    /// itself. See
    /// [`image_labels`](mirage_core::container::image_labels) for why
    /// those marks are the owner and the runtime directory but not a
    /// session.
    ///
    /// # Interruptible, at both ends
    ///
    /// A derived image is the *other* multi-minute step of bring-up, and
    /// teardown waits for a bring-up in flight before it decides what to
    /// remove. An uninterruptible build therefore did not merely ignore
    /// the switch: a Ctrl-C during one sat out the whole `apt-get`
    /// upgrade with the user's terminal held and nothing to say for
    /// itself, which is precisely the failure [`Cancel`] exists to end.
    /// The provider is polled and killed like the pull's.
    ///
    /// The check on entry answers a different question, and it is the
    /// one the caller cannot ask for itself. The hacks path reaches here
    /// because [`Self::image_present`] said the derived image was not
    /// there — an answer a *cancelled* probe also gives, being unable to
    /// give any other. Acting on it started a build for a session that
    /// was already being torn down; reading the switch again is what
    /// tells the two answers apart, exactly as
    /// [`Self::bring_up`] does before its pull.
    ///
    /// # Errors
    ///
    /// [`ContainerError::Cancelled`] if the caller flipped this engine's
    /// [`Cancel`] before or during the build, and
    /// [`ContainerError::Command`] carrying the build's own output if the
    /// provider refused it.
    pub fn build_image(&self, tag: &str, dockerfile: &str) -> Result<()> {
        use std::io::IsTerminal as _;
        self.check_cancelled(&format!("building image {tag}"))?;
        let echo = std::io::stderr().is_terminal();
        let mut args = vec!["build".to_string(), "-t".to_string(), tag.to_string()];
        for (k, v) in mirage_core::container::image_labels() {
            args.push("--label".to_string());
            args.push(format!("{k}={v}"));
        }
        // The build context, which is the Dockerfile on stdin and
        // nothing else, so it stays last.
        args.push("-".to_string());
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

        // Polled rather than waited on, so the switch is read while the
        // build runs. Safe to poll despite the two pipes: they are being
        // drained by the threads above rather than by this one, which is
        // the condition [`Self::wait_cancellable`] names.
        //
        // The drains are joined only once the build has finished of its
        // own accord, which is when their end is in sight: the provider
        // has closed its streams and each thread is one read from
        // returning. A cancelled build is exactly the case where that is
        // not true — killing the provider does not kill whatever it
        // spawned, and a `RUN apt-get` still holding the pipes would
        // keep this thread here for the whole of the build the user just
        // interrupted, which is the wait this change exists to end.
        let status = self.wait_cancellable(&mut child, &args, &format!("building image {tag}"))?;
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
    ///
    /// A question, so an engine that will not answer it is reported as
    /// "no" — including the engine this one *stopped* asking because the
    /// caller cancelled. Bring-up reads its own switch straight
    /// afterwards; see `Self::probe`.
    pub fn image_present(&self, image: &str) -> bool {
        self.probe(
            &[
                "image".to_string(),
                "inspect".to_string(),
                image.to_string(),
            ],
            &format!("looking for image {image}"),
        )
        .unwrap_or(false)
    }

    /// Whether a network named `name` already exists.
    ///
    /// Answered like [`Self::image_present`], and cancellable for the
    /// same reason.
    pub fn network_exists(&self, name: &str) -> bool {
        self.probe(
            &[
                "network".to_string(),
                "inspect".to_string(),
                name.to_string(),
            ],
            &format!("looking for network {name}"),
        )
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

    /// Refuse a working directory that does not exist inside
    /// `container`.
    ///
    /// The host-side `--workdir` check cannot answer this question: a
    /// containerised workload chdirs inside the image's own filesystem,
    /// where a directory that exists out here usually does not exist at
    /// all — and one that does not exist out here may well be in the
    /// image. Left to the provider, the answer arrives as
    /// `OCI runtime exec failed: … chdir to cwd ("/nope") … no such file
    /// or directory` and the container's exit code, which names neither
    /// the flag the user passed nor the filesystem it was wrong about.
    ///
    /// # A relative path is answered without asking
    ///
    /// Both engines refuse a relative `-w`, and they are right to: a
    /// container has no notion of the directory the caller was standing
    /// in, so there is nothing for the path to be relative to. That is
    /// settled here rather than in the container, because the container
    /// cannot say anything useful about it — a probe would resolve it
    /// against whatever the image's own `WORKDIR` is and answer a
    /// question nobody asked.
    ///
    /// # Only a positive answer counts
    ///
    /// The probe is a shell in the container, and the exit codes it is
    /// asked for are ones the image cannot produce by accident: `3`
    /// means "I ran, and there is nothing at that path", `4` means "I
    /// ran, there is something there, and it is not a directory".
    /// *Every* other outcome — an image with no `/bin/sh` (127), a
    /// container that has gone away, an engine that refused — leaves the
    /// question unanswered, and this returns `Ok` for all of them.
    /// Refusing on an unanswered question would mean mirage inventing a
    /// failure for a workload that was going to run perfectly well,
    /// which is worse than the raw engine error this exists to replace.
    ///
    /// # Errors
    ///
    /// [`ContainerError::WorkdirRelative`] for a path that is not
    /// absolute; [`ContainerError::Workdir`], naming the directory and
    /// the container, when the container positively reports there is
    /// nothing at that path; and [`ContainerError::WorkdirNotADirectory`]
    /// when it reports something that is not a directory.
    pub fn check_workdir(&self, container: &str, workdir: &str) -> Result<()> {
        /// The exit code the probe uses for "nothing at that path".
        const MISSING: i32 = 3;
        /// And for "something is there, and it is not a directory".
        const NOT_A_DIRECTORY: i32 = 4;

        if !std::path::Path::new(workdir).is_absolute() {
            return Err(ContainerError::WorkdirRelative {
                workdir: workdir.to_string(),
            });
        }

        let argv = Self::exec_argv(
            container,
            None,
            &[],
            "/bin/sh",
            &[
                "-c".to_string(),
                // `-L` as well as `-e`, for the symlink with nothing
                // behind it: `-e` follows the link and answers about the
                // target, so a path that is plainly occupied in the
                // image would otherwise be reported as one the image
                // does not have — sending the reader to look for a
                // spelling mistake while the link sits there.
                format!(
                    "test -d \"$1\" && exit 0; \
                     test -e \"$1\" && exit {NOT_A_DIRECTORY}; \
                     test -L \"$1\" && exit {NOT_A_DIRECTORY}; \
                     exit {MISSING}"
                ),
                // `$0` for the shell itself, so the path lands in `$1`
                // as data rather than being spliced into the script.
                "sh".to_string(),
                workdir.to_string(),
            ],
            false,
        );
        let probed = spawn_retrying_etxtbsy(|| {
            Command::new(&self.provider)
                .args(&argv)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
        });
        match probed {
            Ok(status) if status.code() == Some(MISSING) => Err(ContainerError::Workdir {
                workdir: workdir.to_string(),
                container: container.to_string(),
            }),
            Ok(status) if status.code() == Some(NOT_A_DIRECTORY) => {
                Err(ContainerError::WorkdirNotADirectory {
                    workdir: workdir.to_string(),
                    container: container.to_string(),
                })
            }
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::debug!(
                    provider = %self.provider,
                    container,
                    workdir,
                    "could not ask the container about its working directory: {e}"
                );
                Ok(())
            }
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
        // of a good error. The same goes for a `--port` no engine could
        // honour, which used to be found by the *second* node container
        // — after the first was already running.
        let ports = resolve_ports(&def.ports, node_count)?;
        let mounts = resolve_mounts(&def.mounts)?;

        // Nothing has been created yet, so an interrupt that arrived
        // before this point costs the caller nothing to honour.
        self.check_cancelled("bringing up the session's containers")?;

        // Pull the image unless it is already present locally; pulling a
        // large image is the slowest, most visible step, so report it.
        let present = self.image_present(&def.image);
        // An interrupted probe answers "the image is not here", which is
        // indistinguishable from the image genuinely not being here — so
        // the switch is read rather than inferred from the answer.
        // Without this a Ctrl-C during a wedged `image inspect` fell
        // through into a pull that was spawned only to be killed on its
        // first poll.
        self.check_cancelled(&format!("looking for image {}", def.image))?;
        if present {
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
        // And the same reading here, for the same reason: an interrupted
        // `network inspect` says "no such network", and acting on it
        // would have this bring-up create one on its way out.
        self.check_cancelled(&format!("looking for network {network}"))?;

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
            ports: &ports,
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
            progress(BringUpPhase::Settling { total: node_count });
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

    /// Ask the provider a yes/no question, ending it if the caller
    /// cancels.
    ///
    /// The interruptible twin of [`Self::status`], and the one every
    /// question bring-up asks *before* it creates anything goes through.
    /// `image inspect` and `network inspect` are instant against a
    /// healthy engine, which is why they were waited on the same way a
    /// `--version` probe is — but they are provider invocations like any
    /// other, and an engine whose daemon has wedged answers neither.
    /// Blocking on that one hung a `mirage run` before the pull, with no
    /// output to say what it was doing and no response to `SIGTERM`,
    /// which is precisely the failure [`Cancel`] exists to end. The
    /// hung-`pull` case was fixed first only because it is the one users
    /// hit often enough to report.
    ///
    /// The removal paths ([`Self::rm`], [`Self::network_rm`]) keep the
    /// uninterruptible [`Self::status`] deliberately. They run *during*
    /// teardown, when the switch is already flipped, and a cancellable
    /// `rm -f` would therefore kill itself on its first poll and leave
    /// behind the container it was called to remove.
    fn probe(&self, args: &[String], what: &str) -> Result<bool> {
        // No pipes, which is what makes this safe to wait on by polling:
        // see [`Self::wait_cancellable`].
        let mut child = spawn_retrying_etxtbsy(|| {
            Command::new(&self.provider)
                .args(args)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
        })
        .map_err(|source| ContainerError::Spawn {
            provider: self.provider.clone(),
            args: args.to_vec(),
            source,
        })?;
        Ok(self.wait_cancellable(&mut child, args, what)?.success())
    }

    /// Run the provider with `args` and return whether it exited zero,
    /// however long it takes. See `Self::probe` for the cancellable
    /// form and for which callers want which.
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

    /// A containerised profile pointing at `provider`, with `ports`.
    fn def_with_ports(provider: &Path, ports: Vec<PortMapping>) -> ContainerizedDef {
        ContainerizedDef {
            ports,
            ..def_with(provider, vec![])
        }
    }

    /// Bring up a one-node session on `engine` with `def`, discarding the
    /// progress reports.
    fn bring_up_one(engine: &Engine, def: &ContainerizedDef) -> Result<()> {
        bring_up_nodes(engine, def, 1).map(|_| ())
    }

    /// Bring up `nodes` nodes on `engine` with `def`.
    fn bring_up_nodes(
        engine: &Engine,
        def: &ContainerizedDef,
        nodes: u32,
    ) -> Result<(ContainerState, Vec<NodeClient>)> {
        engine.bring_up(session(), def, false, nodes, 6000, |_| vec![], |_| {})
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
        // A script rather than the name of a real engine: what this is
        // about is that the configured provider is the one used, and a
        // test that only passes on a machine with docker installed would
        // be about the machine.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::resolve(&def_with(&provider, vec![])).unwrap();
        assert_eq!(engine.provider(), provider.to_string_lossy());
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
    fn a_provider_that_hangs_before_the_pull_is_still_cancellable() {
        // The pull is the step everyone thinks of, and it was the step
        // that got fixed first. But bring-up asks the engine two
        // questions before it: is the image here, and does the network
        // exist. Both are instant against a healthy engine and neither
        // is instant against one whose daemon has wedged — and waiting
        // on them uninterruptibly hung the run with nothing printed and
        // no answer to `SIGTERM`, which is the worst version of this
        // failure rather than a lesser one.
        let dir = tempfile::tempdir().unwrap();
        let provider = scripted_provider(
            dir.path(),
            "wedged-inspect.sh",
            "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then sleep 30; fi\n\
             exit 0\n",
        );
        let cancel = Cancel::new();
        let engine = Engine::with_provider(provider.to_string_lossy().to_string())
            .with_cancel(cancel.clone());

        let started = std::time::Instant::now();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            cancel.cancel();
        });
        let err = bring_up_one(&engine, &def_with(&provider, vec![])).unwrap_err();
        let waited = started.elapsed();

        assert!(
            matches!(err, ContainerError::Cancelled { .. }),
            "a bring-up interrupted at `image inspect` must say so: {err}"
        );
        assert!(
            err.to_string().contains("looking for image img:latest"),
            "the error must name what was interrupted: {err}"
        );
        assert!(
            waited < std::time::Duration::from_secs(5),
            "the wedged inspect was waited out rather than ended: {waited:?}"
        );
    }

    #[test]
    fn a_wedged_network_inspect_does_not_become_a_network() {
        // The second of the two questions. It has to end on the switch
        // like the first, and its answer must not be believed once it
        // has: an interrupted `network inspect` reports "no such
        // network", which is exactly what sends the next line into
        // `network create`.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = scripted_provider(
            dir.path(),
            "wedged-network.sh",
            &format!(
                "echo \"$@\" >> {log}\n\
                 if [ \"$1\" = network ] && [ \"$2\" = inspect ]; then sleep 30; fi\n\
                 exit 0\n",
                log = log.display()
            ),
        );
        let cancel = Cancel::new();
        let engine = Engine::with_provider(provider.to_string_lossy().to_string())
            .with_cancel(cancel.clone());

        let started = std::time::Instant::now();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            cancel.cancel();
        });
        let err = bring_up_one(&engine, &def_with(&provider, vec![])).unwrap_err();
        let waited = started.elapsed();

        assert!(
            err.to_string().contains("looking for network mirage-s"),
            "the error must name what was interrupted: {err}"
        );
        assert!(
            waited < std::time::Duration::from_secs(5),
            "the wedged inspect was waited out rather than ended: {waited:?}"
        );
        let recorded = std::fs::read_to_string(&log).unwrap_or_default();
        assert!(
            !recorded.contains("network create"),
            "an interrupted probe was read as `no such network`:\n{recorded}"
        );
        assert!(
            !recorded.contains("run --rm"),
            "a cancelled bring-up started a container anyway:\n{recorded}"
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
                // The settle window is a phase of its own, so a node that
                // dies in it is not reported against the phase that said
                // the node had started.
                BringUpPhase::Settling { total: 2 },
            ]
        );
    }

    #[test]
    fn bring_up_refuses_a_mount_laid_over_mirages_own_directory() {
        // The user's mount and mirage's own overlap, and the engine
        // resolves the overlap by creating mirage's destinations inside
        // the user's host directory — as root, because the container
        // writes them. The run then reports success, and the user's
        // directory comes back holding `bin`, `config`, `lib` and
        // `runtime` entries they cannot delete.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());
        let host = dir.path().join("mine");
        std::fs::create_dir(&host).unwrap();

        let err = bring_up_one(
            &engine,
            &def_with(
                &provider,
                vec![mount(&format!("{}:/mnt/mirage", host.display()))],
            ),
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(
            matches!(err, ContainerError::ReservedMount { .. }),
            "a mount over mirage's own tree must be refused: {message}"
        );
        // Both paths, because the collision is between them and the user
        // knows only one of them exists.
        assert!(
            message.contains(&host.display().to_string()) && message.contains("/mnt/mirage"),
            "the error must name the mount and the directory it covers: {message}"
        );
        let recorded = std::fs::read_to_string(&log).unwrap_or_default();
        assert!(
            recorded.is_empty(),
            "the provider was asked to do something for a mount that could never work:\n{recorded}"
        );
    }

    #[test]
    fn an_ancestor_of_mirages_directory_is_refused_too() {
        // `--mount /host:/mnt` covers the reserved tree just as surely as
        // naming it does, and is the spelling a user reaches for when
        // they want "somewhere to put files".
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let err = bring_up_one(
            &engine,
            &def_with(
                &provider,
                vec![mount(&format!("{}:/mnt", dir.path().display()))],
            ),
        )
        .unwrap_err();
        assert!(matches!(err, ContainerError::ReservedMount { .. }), "{err}");
    }

    #[test]
    fn the_mounts_mirage_adds_itself_are_not_refused() {
        // Everything mirage bind-mounts lives *below* the reserved
        // directory, and the supervisor appends those to the same list
        // the user's mounts are in. A collision check that caught them
        // would refuse every containerised session there is.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());
        let scratch = dir.path().join("scratch");
        std::fs::create_dir(&scratch).unwrap();

        bring_up_one(
            &engine,
            &def_with(
                &provider,
                vec![
                    mount(&format!("{}:/mnt/mirage/runtime", scratch.display())),
                    mount(&format!("{}:/mnt/mirage/config:ro", scratch.display())),
                ],
            ),
        )
        .unwrap();
    }

    #[test]
    fn a_published_port_is_refused_on_a_multi_node_session() {
        // Every node runs the same container with the same argv, so all
        // of them publish onto the same host port: node 0 binds it and
        // node 1 cannot. That used to be found halfway through bring-up,
        // by the second container, in the engine's own words about a
        // port the user had only asked for once.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let err = bring_up_nodes(
            &engine,
            &def_with_ports(&provider, vec![port("8080:8000")]),
            2,
        )
        .expect_err("publishing one host port from two nodes cannot work");

        let message = err.to_string();
        assert!(
            matches!(err, ContainerError::PortsMultiNode { .. }),
            "{message}"
        );
        assert!(
            message.contains("8080:8000") && message.contains("2-node"),
            "the error must name the port and the node count: {message}"
        );
        // And nothing was started on the way to finding out, which is the
        // half of this that a user notices: the failure used to arrive
        // with node 0 already up.
        let recorded = std::fs::read_to_string(&log).unwrap_or_default();
        assert!(
            !recorded.contains("run --rm"),
            "a node container was started for a port mapping that could never work:\n{recorded}"
        );
    }

    #[test]
    fn a_single_node_session_still_publishes_its_ports() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        bring_up_one(&engine, &def_with_ports(&provider, vec![port("8080:8000")])).unwrap();
        let recorded = log_containing(&log, "run --rm");
        assert!(recorded.contains("-p 8080:8000"), "{recorded}");
    }

    #[test]
    fn the_same_port_asked_for_twice_is_published_once() {
        // Restating the profile's port on the command line, or simply
        // repeating `--port`, used to fail the container with `address
        // already in use` — a message about a conflict with somebody
        // else's process, sending the user to look for one that was not
        // there. Saying the same thing twice is not a mistake.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        bring_up_one(
            &engine,
            // The second spells the protocol both of them mean anyway.
            &def_with_ports(
                &provider,
                vec![port("8080:8000"), port("8080:8000"), port("8080:8000/tcp")],
            ),
        )
        .unwrap();

        let recorded = log_containing(&log, "run --rm");
        let run_line = recorded
            .lines()
            .find(|l| l.starts_with("run --rm"))
            .unwrap_or_else(|| panic!("no container was launched:\n{recorded}"));
        assert_eq!(
            run_line.matches("-p 8080").count(),
            1,
            "one host port must be published once: {run_line}"
        );
    }

    #[test]
    fn two_mappings_for_one_host_port_are_refused_by_name() {
        // Unlike a repeat, these disagree: the host can only give 8080 to
        // one container, and the engine's "address already in use"
        // describes the symptom rather than the two flags that caused it.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let err = bring_up_one(
            &engine,
            &def_with_ports(&provider, vec![port("8080:8000"), port("8080:9000")]),
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(
            matches!(err, ContainerError::PortConflict { .. }),
            "{message}"
        );
        assert!(
            message.contains("8080:8000") && message.contains("8080:9000"),
            "the error must name both mappings: {message}"
        );
        // The same host port on the other protocol is not a conflict at
        // all: TCP 53 and UDP 53 are two different resources.
        bring_up_one(
            &engine,
            &def_with_ports(&provider, vec![port("53:53/tcp"), port("53:53/udp")]),
        )
        .unwrap();
    }

    #[test]
    fn an_empty_image_is_refused_before_anything_is_asked_of_the_engine() {
        // `--image ""` reached the provider as a missing argument, and
        // the user watched `pulling image  (this can take a while)` with
        // a hole where the name should be.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let def = ContainerizedDef {
            image: String::new(),
            ..def_with(&provider, vec![])
        };

        let err = Engine::resolve(&def).unwrap_err();
        let message = err.to_string();
        assert!(matches!(err, ContainerError::Image { .. }), "{message}");
        assert!(
            message.contains("--image") && message.contains("ubuntu:24.04"),
            "the error must name the flag and show what one looks like: {message}"
        );
        assert!(
            std::fs::read_to_string(&log).unwrap_or_default().is_empty(),
            "the engine was asked about an image that does not exist"
        );
    }

    #[test]
    fn a_provider_that_is_not_a_container_engine_is_refused() {
        let dir = tempfile::tempdir().unwrap();

        // Not there at all. This used to surface as a bare `No such file
        // or directory` from whichever step spawned first.
        let missing = dir.path().join("whale");
        let err = Engine::resolve(&def_with(&missing, vec![])).unwrap_err();
        let message = err.to_string();
        assert!(matches!(err, ContainerError::Provider { .. }), "{message}");
        assert!(
            message.contains(&missing.display().to_string()) && message.contains("podman"),
            "the error must name what was configured and what mirage can drive: {message}"
        );

        // There, executable, and not an engine: `--version` is the
        // weakest question that tells the two apart, and both real
        // engines answer it without touching a daemon or the network.
        let impostor = scripted_provider(dir.path(), "impostor.sh", "exit 1\n");
        let err = Engine::resolve(&def_with(&impostor, vec![])).unwrap_err();
        let message = err.to_string();
        assert!(matches!(err, ContainerError::Provider { .. }), "{message}");
        assert!(
            message.contains("--version"),
            "the error must say what mirage asked it: {message}"
        );

        // And a script that answers is accepted, because a provider is
        // allowed to be a wrapper around an engine — which is what every
        // mock in this file is.
        let log = dir.path().join("log");
        let mock = mock_provider(dir.path(), &log);
        Engine::resolve(&def_with(&mock, vec![])).unwrap();
    }

    #[test]
    fn a_derived_image_is_labelled_as_mirages() {
        // Nothing else records that a `mirage-hack-…` image exists: it is
        // built once, keyed by base image plus hacks, and outlives every
        // session that uses it. The label on the image is the only thing
        // that can attribute it later.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = scripted_provider(
            dir.path(),
            "building-provider.sh",
            &format!(
                "echo \"$@\" >> {log}\n\
                 cat > /dev/null\n\
                 exit 0\n",
                log = log.display()
            ),
        );
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        engine
            .build_image("mirage-hack-abc:latest", "FROM img:latest\n")
            .unwrap();

        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            recorded.contains("--label mirage.owner=mirage"),
            "a derived image must be marked as mirage's:\n{recorded}"
        );
        assert!(
            recorded.contains(&format!(
                "--label mirage.runtime={}",
                mirage_core::container::owning_runtime()
            )),
            "a derived image must say which runtime directory built it:\n{recorded}"
        );
        // The Dockerfile still arrives on stdin, so the context argument
        // has to stay last.
        assert!(recorded.trim_end().ends_with(" -"), "{recorded}");
    }

    #[test]
    fn a_build_is_not_started_for_a_bring_up_that_is_already_cancelled() {
        // The hacks path asks `image_present` whether the derived image
        // is there and builds it when the answer is no — and a cancelled
        // probe answers no, being unable to answer anything else. Reading
        // the switch again is what tells "not built yet" from "stop",
        // and without it a Ctrl-C during bring-up bought the user an
        // entire `apt-get` upgrade.
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = scripted_provider(
            dir.path(),
            "building-provider.sh",
            &format!(
                "echo \"$@\" >> {log}\n\
                 cat > /dev/null\n\
                 exit 0\n",
                log = log.display()
            ),
        );
        let cancel = Cancel::new();
        cancel.cancel();
        let engine =
            Engine::with_provider(provider.to_string_lossy().to_string()).with_cancel(cancel);

        let err = engine
            .build_image("mirage-hack-abc:latest", "FROM img:latest\n")
            .unwrap_err();

        assert!(
            matches!(err, ContainerError::Cancelled { .. }),
            "a build asked for after a cancellation must say so: {err}"
        );
        assert!(
            err.to_string().contains("building image mirage-hack-abc"),
            "the error must name what was not started: {err}"
        );
        assert!(
            !Path::new(&log).exists(),
            "the provider was invoked for a cancelled build:\n{}",
            std::fs::read_to_string(&log).unwrap_or_default()
        );
    }

    #[test]
    fn a_build_stops_when_the_caller_cancels() {
        // A derived image is the other multi-minute step of bring-up,
        // and teardown waits for a bring-up in flight before it decides
        // what to remove — so a build that ignores the switch is not a
        // slow exit, it is a Ctrl-C that does nothing for as long as the
        // build lasts.
        let dir = tempfile::tempdir().unwrap();
        let provider = scripted_provider(
            dir.path(),
            "slow-build.sh",
            "case \"$1\" in\n\
             build) sleep 30; exit 0 ;;\n\
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
        let err = engine
            .build_image("mirage-hack-abc:latest", "FROM img:latest\n")
            .unwrap_err();
        let waited = started.elapsed();

        assert!(
            matches!(err, ContainerError::Cancelled { .. }),
            "a cancelled build must say so: {err}"
        );
        assert!(
            err.to_string().contains("building image mirage-hack-abc"),
            "the error must name what was interrupted: {err}"
        );
        assert!(
            waited < std::time::Duration::from_secs(5),
            "the build was waited out rather than ended: {waited:?}"
        );
    }

    #[test]
    fn a_host_port_already_taken_names_the_port_and_the_remedy() {
        // The engine's own words are about a driver failing to program
        // external connectivity, and they end by recommending `docker
        // run --help` — a manual page about flags, for a conflict with
        // another program entirely, and about a `-p` mirage passed on
        // the user's behalf.
        let dir = tempfile::tempdir().unwrap();
        let provider = scripted_provider(
            dir.path(),
            "bound-port-provider.sh",
            &[
                "case \"$1\" in",
                "run)",
                // docker's own words, on one line as it writes them.
                "echo 'docker: Error response from daemon: driver failed programming external \
                 connectivity: Bind for 0.0.0.0:8080 failed: port is already allocated.' >&2",
                "echo >&2",
                "echo \"See 'docker run --help'.\" >&2",
                "exit 125 ;;",
                "inspect) echo false; exit 0 ;;",
                "*) exit 0 ;;",
                "esac",
            ]
            .join("\n"),
        );
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let mut client = engine
            .launch_node("mirage-s-node-0", &bare_spec(), &[], 0)
            .unwrap();
        let err = engine
            .await_running(&mut client, std::time::Duration::from_secs(30))
            .unwrap_err();

        let message = err.to_string();
        assert!(
            matches!(err, ContainerError::HostPortInUse { .. }),
            "a bound host port is its own failure, not an anonymous one: {message}"
        );
        assert!(
            message.contains("8080"),
            "the error must name the port: {message}"
        );
        assert!(
            message.contains("--port"),
            "the error must name the flag that chose it: {message}"
        );
        assert!(
            !message.contains("--help"),
            "the engine's pointer at its own manual must not survive: {message}"
        );
        // The engine's own words still do, because they are evidence.
        assert!(
            message.contains("port is already allocated"),
            "the engine's words are the evidence for all of this: {message}"
        );
    }

    #[test]
    fn podmans_spelling_of_a_bound_port_is_recognised_too() {
        // Same failure, different words and a different port token:
        // podman's rootless forwarder reports the listen(2) that failed.
        let said = "Error: rootlessport cannot expose privileged port 80: listen tcp4 :8080:                     bind: address already in use";
        assert_eq!(host_port_in_use(said).as_deref(), Some("8080"));

        // And a refusal that is not about a port at all stays anonymous.
        assert_eq!(host_port_in_use("docker: invalid reference format."), None);
        // As does one that matches but names no port, where mirage would
        // only be making the engine's words vaguer.
        assert_eq!(host_port_in_use("bind: address already in use"), None);
    }

    #[test]
    fn a_relative_workdir_is_refused_without_asking_the_container() {
        // The host-side check skips a containerised `--workdir` — it
        // names a path in the image — so a relative one used to reach
        // the engine, which answers with `workdir must be an absolute
        // path` wrapped in an OCI runtime error about a container id the
        // user has never seen.
        let dir = tempfile::tempdir().unwrap();
        let engine = executing_provider(dir.path());

        let err = engine
            .check_workdir("mirage-s-node-0", "build")
            .unwrap_err();
        let message = err.to_string();
        assert!(
            matches!(err, ContainerError::WorkdirRelative { .. }),
            "{message}"
        );
        assert!(
            message.contains("--workdir build") && message.contains("absolute"),
            "the error must name the flag, the path and what is wrong with it: {message}"
        );
        assert!(
            !message.contains("no such directory"),
            "a relative path is not a missing one: {message}"
        );

        // Including the spellings that only look relative-ish, which are
        // absolute and are the container's business, not this check's.
        engine
            .check_workdir("mirage-s-node-0", &dir.path().to_string_lossy())
            .unwrap();
    }

    #[test]
    fn a_workdir_that_is_a_file_is_not_reported_as_a_missing_directory() {
        // "There is no such directory ... name a directory the image
        // already has, or `--mount` one there" is wrong twice over about
        // a path that is right there and is a file: nothing is missing,
        // and mounting over it is not the fix.
        let dir = tempfile::tempdir().unwrap();
        let engine = executing_provider(dir.path());
        let file = dir.path().join("config.json");
        std::fs::write(&file, b"{}").unwrap();

        let err = engine
            .check_workdir("mirage-s-node-0", &file.to_string_lossy())
            .unwrap_err();
        let message = err.to_string();
        assert!(
            matches!(err, ContainerError::WorkdirNotADirectory { .. }),
            "{message}"
        );
        assert!(
            message.contains("not a directory") && message.contains("mirage-s-node-0"),
            "the error must say what the path is and where: {message}"
        );
        assert!(
            !message.contains("no such directory"),
            "the path is there; only its kind is wrong: {message}"
        );
        assert!(
            !message.contains("--mount"),
            "mounting something over an occupied path is not the remedy: {message}"
        );
    }

    #[test]
    fn a_refusal_spread_over_several_lines_reads_as_one() {
        // Engines write their refusals as paragraphs — docker follows the
        // error with an empty line and `See 'docker run --help'`. The
        // ring is read back `; `-joined, so the blank lines came through
        // as `; ; ` and read as though mirage had lost something.
        let dir = tempfile::tempdir().unwrap();
        let provider = scripted_provider(
            dir.path(),
            "chatty-provider.sh",
            "case \"$1\" in\n\
             run) printf 'docker: invalid reference format.\\n\\nSee '\\''docker run --help'\\''.\\n\\n' >&2; \
             exit 125 ;;\n\
             inspect) echo false; exit 0 ;;\n\
             *) exit 0 ;;\n\
             esac\n",
        );
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let mut client = engine.launch_node("n", &bare_spec(), &[], 0).unwrap();
        let err = engine
            .await_running(&mut client, std::time::Duration::from_secs(30))
            .unwrap_err();

        let message = err.to_string();
        assert!(
            message.contains("invalid reference format")
                && message.contains("See 'docker run --help'"),
            "the engine's words must survive: {message}"
        );
        assert!(
            !message.contains("; ;") && !message.ends_with("; "),
            "the engine's blank lines became separators: {message}"
        );
    }

    /// A provider whose `exec` runs the command it is given, so a
    /// workdir probe is answered by this machine's filesystem standing in
    /// for the container's.
    fn executing_provider(dir: &Path) -> Engine {
        let provider = scripted_provider(
            dir,
            "exec-provider.sh",
            "case \"$1\" in\n\
             exec)\n\
             shift\n\
             while [ $# -gt 0 ]; do\n\
             case \"$1\" in -i|-t) shift ;; -w|-e) shift 2 ;; *) break ;; esac\n\
             done\n\
             shift\n\
             exec \"$@\" ;;\n\
             *) exit 0 ;;\n\
             esac\n",
        );
        Engine::with_provider(provider.to_string_lossy().to_string())
    }

    #[test]
    fn a_workdir_the_container_does_not_have_is_named_before_the_exec() {
        // The provider's own answer is `OCI runtime exec failed: … chdir
        // to cwd ("/nope/nope") … no such file or directory` plus the
        // container's exit code, which mentions neither the flag nor the
        // reason the path being on the host is beside the point.
        let dir = tempfile::tempdir().unwrap();
        let engine = executing_provider(dir.path());

        let err = engine
            .check_workdir("mirage-s-node-0", "/nope/nope")
            .unwrap_err();
        let message = err.to_string();
        assert!(matches!(err, ContainerError::Workdir { .. }), "{message}");
        assert!(
            message.contains("--workdir /nope/nope") && message.contains("mirage-s-node-0"),
            "the error must name the flag, the path and the container: {message}"
        );
        // And the part a user cannot guess: which filesystem was asked.
        assert!(
            message.contains("container"),
            "the error must say the path has to exist inside the container: {message}"
        );
    }

    #[test]
    fn a_workdir_the_container_has_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let engine = executing_provider(dir.path());
        engine
            .check_workdir("mirage-s-node-0", &dir.path().to_string_lossy())
            .unwrap();
    }

    #[test]
    fn a_container_that_cannot_answer_about_its_workdir_is_believed() {
        // No `/bin/sh` in the image, an engine that refused, a container
        // that has gone away: none of those is evidence that the
        // directory is missing, and refusing on them would invent a
        // failure for a workload that was going to run.
        let dir = tempfile::tempdir().unwrap();
        let shell_less = scripted_provider(
            dir.path(),
            "distroless-provider.sh",
            "case \"$1\" in\n\
             exec) echo 'exec: /bin/sh: not found' >&2; exit 127 ;;\n\
             *) exit 0 ;;\n\
             esac\n",
        );
        let engine = Engine::with_provider(shell_less.to_string_lossy().to_string());
        engine
            .check_workdir("mirage-s-node-0", "/nope/nope")
            .unwrap();

        let gone = Engine::with_provider(dir.path().join("no-such-engine").to_string_lossy());
        gone.check_workdir("mirage-s-node-0", "/nope/nope").unwrap();
    }
}
