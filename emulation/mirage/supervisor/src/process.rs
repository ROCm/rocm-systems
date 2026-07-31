//! Spawning, supervising and reliably killing workload processes.
//!
//! This module is where mirage's zombie problem was, and where it is
//! fixed. Three properties are maintained, and every one of them is a
//! response to a way the previous design leaked:
//!
//! 1. **Every child is waited on.** A spawned child is owned by exactly
//!    one task, and that task's only exit path runs through
//!    [`Child::wait`]. A child that is never waited on stays in the
//!    process table as a zombie until its parent dies, and the previous
//!    design's detached per-session host frequently exited without
//!    reaping — leaving zombies parented to a process that was itself
//!    gone.
//!
//! 2. **Every child leads its own process group.** Children are spawned
//!    with `process_group(0)`, so a workload that forks (a shell script,
//!    `torchrun`, an MPI launcher) puts its whole tree in one group that
//!    can be signalled as a unit. Signalling only the direct child would
//!    leave the grandchildren running — invisible, still holding the GPU
//!    socket, still holding ports.
//!
//!    Note this is a *safe* API. The equivalent in the old design was a
//!    `pre_exec` closure calling `setsid()`, which required `unsafe` and
//!    ran arbitrary code between fork and exec in a process where almost
//!    nothing is legal to call.
//!
//! 3. **Termination escalates and then confirms.** [`terminate`] sends
//!    `SIGTERM` to the group, waits a bounded grace period, sends
//!    `SIGKILL`, and only returns once the child has actually been
//!    reaped. `SIGKILL` cannot be caught, so the second stage always
//!    ends; the confirmation is what makes "the session is destroyed" a
//!    statement about the process table rather than about intent.
//!
//! Two backstops sit under that, for the cases where the escalation
//! above never gets to run:
//!
//! * `kill_on_drop(true)` — if a supervising task is cancelled abruptly,
//!   tokio kills the child rather than orphaning it.
//! * `PR_SET_PDEATHSIG` (via [`mirage_sys::die_with_parent`]) — if mirage
//!   itself is `SIGKILL`ed or OOM-killed, *no* code of ours runs, and the
//!   kernel is the only party left that can enforce the rule. It sends
//!   the child `SIGKILL` when its parent goes away.
//!
//! # Standard streams
//!
//! A workload gets the *caller's* streams, not ones mirage manufactured.
//! Whoever spawns a process here — `mirage run`, or a `mirage exec` in
//! another terminal — is the process the user is sitting in front of, so
//! inheriting its file descriptors puts the workload on the user's real
//! terminal.
//!
//! That is worth stating because the obvious alternative is what mirage
//! used to do, and it was worse in every direction. A long-lived daemon
//! owned every workload, so the workload's terminal could not be the
//! user's; it had to be a pseudo-terminal the daemon allocated, with
//! output shipped back over a socket and keystrokes shipped forward. The
//! costs were real: a PTY has one stream, so stdout and stderr were
//! merged and redirecting one of them stopped working; the client had to
//! put the user's terminal in raw mode and forward `SIGWINCH` by hand;
//! and `pre_exec`-based session-leader setup was the only `unsafe` in
//! the workspace.
//!
//! Inheriting deletes all of it. An interactive `bash` works because its
//! stdin *is* the terminal, not because mirage emulated one.
//!
//! [`StdioMode::Capture`] is the exception, and exists for one reason:
//! several ranks writing to one terminal are unreadable without labels.
//! Capturing puts mirage back in the middle so it can prefix each chunk
//! with the rank that produced it, at the price of stdin.

use std::collections::BTreeMap;
use std::time::Duration;

use nix::sys::signal::Signal;
use nix::unistd::Pid;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

use mirage_core::exec::StdStream;

/// How long a process gets to exit after `SIGTERM` before `SIGKILL`.
///
/// Long enough for a workload to flush output and run its own cleanup,
/// short enough that destroying a session stays interactive.
pub const TERM_GRACE: Duration = Duration::from_millis(2_000);

/// How often the escalation loop re-checks whether a signalled process
/// has exited.
const REAP_POLL: Duration = Duration::from_millis(20);

/// Size of the per-stream read buffer. Output frames are chunked to at
/// most this many bytes.
const READ_CHUNK: usize = 64 * 1024;

/// Where a workload really runs, when the process we supervise is only a
/// container provider's client.
///
/// `podman exec` (and `docker exec`) put the workload in the container's
/// own PID namespace and do **not** forward signals to it: killing the
/// client leaves the workload running, invisible to us and holding the
/// emulated device. Signalling it therefore has to go back through the
/// provider, which means knowing the pid it has *inside* the container.
///
/// That pid comes from the workload itself. The in-container command is
/// wrapped in `sh -c 'echo $$ > <pidfile>; exec "$0" "$@"'`, so the shell
/// records its own pid and then `exec`s the real program *into that same
/// pid*. The file lands in the session scratch directory, which is
/// already bind-mounted into every node container, so the supervisor
/// reads it straight off the host filesystem with no extra round trip.
#[derive(Debug, Clone)]
pub struct ContainerProc {
    /// Provider binary (`podman`, `docker`, or a path).
    pub provider: String,
    /// Name of the container the workload runs in.
    pub container: String,
    /// Host path of the file the wrapper writes its in-container pid to.
    pub pid_file: std::path::PathBuf,
}

/// How long to wait for a freshly-started workload to record its
/// in-container pid before giving up on signalling it.
///
/// Signalling can race the start of a very short exec, and the file
/// appears as soon as the wrapper's first command runs.
const PID_FILE_WAIT: Duration = Duration::from_millis(2_000);

impl ContainerProc {
    /// The workload's pid inside the container, if it has recorded one.
    ///
    /// `1` is rejected along with `0`. The file is written by a shell
    /// inside a container whose scratch directory is bind-mounted
    /// read-write, so its contents are workload-influenced; a `1` reaching
    /// [`ContainerProc::kill_argv`] would produce `kill -9 -1`, which POSIX
    /// defines as *every process the caller may signal* — inside the
    /// container that is every other rank sharing it. No legitimate
    /// wrapper can record `1` either: PID 1 in a node container is its
    /// `sleep infinity` entrypoint, never a `provider exec` child.
    fn pid(&self) -> Option<u32> {
        std::fs::read_to_string(&self.pid_file)
            .ok()?
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|pid| *pid > 1)
    }

    /// The workload's pid, waiting briefly for it to appear.
    async fn pid_soon(&self) -> Option<u32> {
        let deadline = tokio::time::Instant::now() + PID_FILE_WAIT;
        loop {
            if let Some(pid) = self.pid() {
                return Some(pid);
            }
            if tokio::time::Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(REAP_POLL).await;
        }
    }

    /// `<provider> exec <container> /bin/sh -c 'kill …'` for `pid`.
    ///
    /// Both the process group and the process itself are signalled. The
    /// wrapper is not guaranteed to be a group leader, so the negative
    /// form may fail — harmlessly, because the bare pid follows it — but
    /// when it does work it reaches a workload's own children too.
    ///
    /// `kill` is used as a *shell builtin*, via `/bin/sh -c`: many
    /// minimal images ship no `/bin/kill` binary, and
    /// `provider exec <c> kill …` would fail on those.
    fn kill_argv(&self, pid: u32, sig: Signal) -> Vec<String> {
        let n = sig as i32;
        vec![
            "exec".to_string(),
            self.container.clone(),
            "/bin/sh".to_string(),
            "-c".to_string(),
            format!("kill -{n} -{pid} 2>/dev/null; kill -{n} {pid} 2>/dev/null; exit 0"),
        ]
    }

    /// Deliver `sig` to the workload inside the container.
    ///
    /// Returns whether the provider ran successfully. Best effort: a
    /// container that has already gone is not an error, it is the
    /// outcome we wanted.
    pub async fn signal(&self, sig: Signal) -> bool {
        let Some(pid) = self.pid_soon().await else {
            tracing::debug!(
                container = %self.container,
                "no in-container pid recorded; cannot forward the signal"
            );
            return false;
        };
        let status = Command::new(&self.provider)
            .args(self.kill_argv(pid, sig))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .status()
            .await;
        matches!(status, Ok(s) if s.success())
    }

    /// Deliver `sig` without awaiting, for the synchronous backstop.
    ///
    /// Used from [`crate::exec::Exec::kill_now`], which has to work in a
    /// `Drop` or a panic handler where there is no runtime to await on.
    /// The provider is spawned and deliberately not waited for; the
    /// caller is on its way out and a zombie provider client is reaped by
    /// init moments later.
    pub fn signal_now(&self, sig: Signal) {
        let Some(pid) = self.pid() else {
            return;
        };
        let _ = std::process::Command::new(&self.provider)
            .args(self.kill_argv(pid, sig))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
}

/// How a process's standard streams are wired up.
///
/// Both modes are about *the caller's* terminal, because in this design
/// the process that spawns a workload is always the process the user is
/// sitting in front of. Mirage never allocates a pseudo-terminal: if the
/// caller is on a terminal, an inherited fd already is one, and if the
/// caller is a pipe, inheriting keeps the bytes exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdioMode {
    /// The child inherits the caller's own stdout and stderr, writing to
    /// the terminal directly with mirage not in the middle.
    ///
    /// `stdin` decides whether it also inherits the caller's input. Only
    /// one process can meaningfully read a terminal, so exactly one rank
    /// gets it; the rest read `/dev/null` and see an immediate EOF.
    Inherit {
        /// Whether this rank inherits the caller's stdin.
        stdin: bool,
    },
    /// Separate pipes for stdout and stderr, so the caller can read,
    /// label and forward the output itself. stdin is `/dev/null`.
    ///
    /// This is what `--capture-all` selects.
    Capture,
}

impl Default for StdioMode {
    fn default() -> Self {
        Self::Inherit { stdin: false }
    }
}

impl StdioMode {
    /// The mode for `rank` under a given capture setting.
    ///
    /// Rank 0 is the one that gets the terminal's input: it is the rank a
    /// user talks to, and the only one an interactive program is started
    /// on.
    #[must_use]
    pub fn for_rank(rank: u32, capture_all: bool) -> Self {
        if capture_all {
            Self::Capture
        } else {
            Self::Inherit { stdin: rank == 0 }
        }
    }

    /// Whether output is pumped through mirage rather than written to
    /// the terminal by the child itself.
    #[must_use]
    pub fn is_captured(self) -> bool {
        matches!(self, Self::Capture)
    }

    /// Whether a process on this mode holds the caller's terminal, and
    /// therefore needs to be made its foreground process group.
    #[must_use]
    pub fn owns_terminal(self) -> bool {
        matches!(self, Self::Inherit { stdin: true })
    }
}

/// A chunk of output read from a workload process.
#[derive(Debug)]
pub struct OutputChunk {
    /// Global rank of the process it came from.
    pub node: u32,
    /// Which stream it came from.
    pub stream: StdStream,
    /// The bytes, exactly as read.
    pub data: Vec<u8>,
}

/// A spawned workload process, owned by the supervisor.
#[derive(Debug)]
pub struct Spawned {
    /// The child handle. Always waited on before this struct is dropped.
    child: Child,
    /// Its pid, captured at spawn time.
    pid: u32,
    /// Tasks pumping stdout and stderr into the output channel. Empty
    /// for a child that inherited the caller's streams, which mirage
    /// never reads.
    pumps: Vec<tokio::task::JoinHandle<()>>,
    /// Set when `child` is a container provider client, so termination
    /// reaches the workload inside the container instead of stopping at
    /// the client.
    container: Option<ContainerProc>,
    /// Cached result once the process has been reaped, so `wait` and
    /// `terminate` are both idempotent and safe to call in either order.
    /// The supervisor races them against a cancellation token, so "wait
    /// returned, now terminate is asked for anyway" is a normal path, not
    /// an error.
    exit: Option<Exit>,
}

/// How a process finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exit {
    /// Conventional exit code: the process's own status, or `128 + signal`
    /// when it was killed by a signal.
    pub code: i32,
}

impl Exit {
    /// The conventional code for "command not found".
    pub const NOT_FOUND: i32 = 127;

    /// Build an [`Exit`] from a raw wait status.
    fn from_status(status: std::process::ExitStatus) -> Self {
        use std::os::unix::process::ExitStatusExt as _;
        let code = status.code().or_else(|| status.signal().map(|s| 128 + s));
        Self {
            // A status that is neither an exit code nor a signal is not
            // reachable on Unix, but `-1` is a defined answer rather than
            // a panic if the platform ever surprises us.
            code: code.unwrap_or(-1),
        }
    }
}

/// Everything needed to launch one workload process.
#[derive(Debug, Clone)]
pub struct SpawnSpec {
    /// Global rank; identifies the process in output and status.
    pub node: u32,
    /// Program to run.
    pub command: String,
    /// Its arguments.
    pub args: Vec<String>,
    /// Environment to apply on top of the inherited base.
    pub env: BTreeMap<String, String>,
    /// Working directory, if any.
    pub workdir: Option<String>,
    /// Pipes or a pseudo-terminal.
    pub stdio: StdioMode,
    /// When set — the default — the process inherits the caller's whole
    /// environment and `env` is layered on top. When clear, the
    /// environment is emptied first and only a small allowlist
    /// ([`INHERITED_ENV`]) plus `env` is passed.
    ///
    /// Inheriting is the default because mirage runs in the terminal the
    /// user typed in, and everything they exported there — an API token,
    /// a `PYTHONPATH`, a proxy, a tuning variable their framework reads —
    /// is something they meant for the workload. Dropping all of it made
    /// sense when a detached daemon ran the workload, because the daemon
    /// had inherited its environment from whichever shell happened to
    /// start it hours earlier; it makes no sense now that the workload's
    /// parent *is* the user's shell.
    ///
    /// `mirage run --clear-env-vars` asks for the strict form, for a run
    /// that must not depend on ambient state.
    ///
    /// Container execs always inherit: the value here governs the
    /// provider *client*, which needs its own environment to find its
    /// socket and configuration. What the containerised workload sees is
    /// passed explicitly with `-e` and never inherited from the host.
    pub inherit_env: bool,
    /// Set when this spec launches a container provider client rather
    /// than the workload itself, so signals can be forwarded into the
    /// container. `None` for a workload that runs directly on the host,
    /// where the process we spawn *is* the workload.
    pub container: Option<ContainerProc>,
}

/// The only environment variables a workload keeps under
/// `--clear-env-vars`.
///
/// Not an empty set, because an empty environment is not a useful one:
/// without `PATH` most commands cannot be found at all, and without
/// `HOME` a great deal of tooling writes to `/`. This is the floor a
/// process needs to behave like a process, with everything
/// workload-specific left to the emulator's injection and `--env`.
pub const INHERITED_ENV: &[&str] = &[
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL",
];

/// Variables whose value is a `:`-separated search list, where mirage's
/// entries have to coexist with the caller's rather than replace them.
///
/// `spec.env` is applied on top of an inherited environment, and for an
/// ordinary variable "on top" means "instead of", which is right. For a
/// search list it is not: an emulator backend that sets `LD_LIBRARY_PATH`
/// to its own directory silently deletes every directory the caller
/// exported, and one that sets `PYTHONPATH` deletes their imports. That
/// is precisely the loss inheriting the caller's environment was
/// introduced to stop, and it is what the CLI documentation promises will
/// not happen.
///
/// Mirage's entries go first: the interposer and the emulator's libraries
/// must win the search, and a workload that loses the interposer runs
/// unemulated and still exits 0.
const PATH_LIST_ENV: &[&str] = &["LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"];

/// Spawn one workload process.
///
/// Output is streamed to `output` as it arrives; the returned handle owns
/// the child and must be finished with [`Spawned::wait`] or
/// [`Spawned::terminate`].
///
/// # Errors
///
/// Returns the spawn error, with the common cases (`NotFound`,
/// `PermissionDenied`) already translated into an actionable message —
/// "command not found: foo" rather than "No such file or directory (os
/// error 2)", which says nothing about *which* file.
pub fn spawn(spec: &SpawnSpec, output: mpsc::Sender<OutputChunk>) -> Result<Spawned, String> {
    use std::process::Stdio;

    let (stdin, stdout, stderr) = match spec.stdio {
        StdioMode::Inherit { stdin } => (
            if stdin { Stdio::inherit() } else { Stdio::null() },
            Stdio::inherit(),
            Stdio::inherit(),
        ),
        StdioMode::Capture => (Stdio::null(), Stdio::piped(), Stdio::piped()),
    };

    let mut cmd = Command::new(&spec.command);
    cmd.args(&spec.args)
        .stdin(stdin)
        .stdout(stdout)
        .stderr(stderr)
        // Lead a new process group, always, so the whole descendant tree
        // can be signalled as one. A workload that forks — a shell
        // script, `torchrun`, an MPI launcher — would otherwise leave its
        // grandchildren running when signalled, invisible and still
        // holding the GPU socket and its ports. Safe API; the old design
        // needed a `pre_exec` closure calling `setsid()` for this.
        //
        // It also means no group mirage signals is ever *mirage's own*,
        // which is what keeps `kill(-pgid)` from reaching the shell the
        // user started mirage from.
        //
        // The cost is that the child is not in the terminal's foreground
        // process group, so reading the terminal would earn it a
        // `SIGTTIN` and stop it. The rank that holds the terminal is
        // therefore handed the terminal explicitly — see
        // `mirage_ctl::run`'s terminal handoff, which is exactly what a
        // shell does when it puts a job in the foreground.
        .process_group(0)
        // Backstop: if the owning task is dropped without going through
        // `wait`/`terminate`, do not orphan the child.
        .kill_on_drop(true);

    // Second backstop, for the case the first one cannot reach:
    // `SIGKILL`. Everything above is enforced by mirage running — a
    // signal it sends, a `Drop` it runs. `kill -9`, and the OOM killer
    // picking mirage during a large emulated job, leave no code of ours
    // to run at all, and the workload would be reparented to init still
    // holding the emulated device. Asking the kernel to do it is the only
    // way to close that, and is why `mirage_sys` exists.
    mirage_sys::die_with_parent(&mut cmd);

    if !spec.inherit_env {
        cmd.env_clear();
    }
    cmd.envs(resolved_env(spec));
    if let Some(wd) = &spec.workdir {
        cmd.current_dir(wd);
    }

    let mut child = cmd.spawn().map_err(|e| spawn_error(&spec.command, &e))?;

    // `id()` only returns `None` after the child has been waited on,
    // which cannot have happened yet.
    let pid = child.id().unwrap_or(0);

    // Put the child in its own process group from *this* side too.
    //
    // `process_group(0)` above asks the child to do it, and the child
    // does — but only once it is scheduled, somewhere between `fork` and
    // `exec`. Until then `kill(-pid)` does not reach it, and the failure
    // is silent in the worst possible way: the call does not error, it
    // signals *whatever other group happens to be numbered `pid`*. Pids
    // are handed out sequentially and every shell job is a group leader,
    // so on a busy machine that group frequently exists. The signal goes
    // somewhere else entirely and the workload sails on, which is how a
    // `terminate` can return having killed nothing.
    //
    // `setpgid` is idempotent and either end may call it, so doing it
    // here as well closes the window: by the time `spawn` returns, the
    // group exists and contains this child. `EACCES` means the child has
    // already exec'd — it beat us to it, which is the outcome we wanted —
    // and `ESRCH` means it is already gone.
    if let Ok(raw) = i32::try_from(pid)
        && raw > 0
    {
        let child_pid = Pid::from_raw(raw);
        match nix::unistd::setpgid(child_pid, child_pid) {
            Ok(()) | Err(nix::errno::Errno::EACCES | nix::errno::Errno::ESRCH) => {}
            Err(e) => tracing::debug!(pid, "could not confirm the child's process group: {e}"),
        }
    }

    // Only a captured child has pipes to pump; an inheriting one writes
    // to the terminal without mirage ever seeing the bytes.
    let mut pumps = Vec::with_capacity(2);
    if let Some(out) = child.stdout.take() {
        pumps.push(tokio::spawn(pump(
            out,
            spec.node,
            StdStream::Stdout,
            output.clone(),
        )));
    }
    if let Some(err) = child.stderr.take() {
        pumps.push(tokio::spawn(pump(err, spec.node, StdStream::Stderr, output)));
    }

    Ok(Spawned {
        child,
        pid,
        pumps,
        container: spec.container.clone(),
        exit: None,
    })
}

/// The environment a process should see, given the spec.
fn resolved_env(spec: &SpawnSpec) -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
    let mut env: Vec<(std::ffi::OsString, std::ffi::OsString)> = Vec::new();
    if !spec.inherit_env {
        for key in INHERITED_ENV {
            if let Some(value) = std::env::var_os(key) {
                env.push(((*key).into(), value));
            }
        }
    }
    // A terminal program needs `TERM` to be *something*. The caller
    // usually has one and it arrives either by inheritance or through the
    // allowlist above — but a run from cron, a CI runner or a bare `sh`
    // has none, and then `bash` gets an unset `TERM`: no prompt colours,
    // and `clear`, `less` and anything ncurses failing outright with
    // "TERM environment variable not set". Naming a default costs
    // nothing.
    //
    // Outside the `inherit_env` check on purpose. It used to sit inside
    // it, which was equivalent while every direct workload cleared its
    // environment; once inheriting became the default that left the
    // fallback applying only under `--clear-env-vars`, i.e. never on the
    // path that actually needed it.
    if std::env::var_os("TERM").is_none() {
        env.push(("TERM".into(), "xterm-256color".into()));
    }
    for (k, v) in &spec.env {
        // A search list is merged with what the caller exported rather
        // than replacing it; see [`PATH_LIST_ENV`]. Only when inheriting:
        // under `--clear-env-vars` the caller's environment is
        // deliberately not in play.
        if spec.inherit_env
            && PATH_LIST_ENV.contains(&k.as_str())
            && let Some(inherited) = std::env::var_os(k).filter(|v| !v.is_empty())
        {
            env.push((k.into(), merged_search_path(v, &inherited)));
            continue;
        }
        env.push((k.into(), v.into()));
    }
    env
}

/// One `:`-separated search list built from mirage's entries and the
/// caller's, in that order. See [`PATH_LIST_ENV`].
fn merged_search_path(injected: &str, inherited: &std::ffi::OsStr) -> std::ffi::OsString {
    let mut combined = std::ffi::OsString::from(injected);
    combined.push(":");
    combined.push(inherited);
    combined
}

/// Translate a spawn failure into something that names the problem.
fn spawn_error(command: &str, e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => format!("command not found: {command}"),
        std::io::ErrorKind::PermissionDenied => format!("permission denied: {command}"),
        std::io::ErrorKind::NotADirectory | std::io::ErrorKind::IsADirectory => {
            format!("not executable: {command}")
        }
        _ => format!("failed to spawn {command}: {e}"),
    }
}

/// Read one of a child's streams to EOF, forwarding it in chunks.
///
/// stdout and stderr are pumped by separate tasks so a workload that
/// writes a lot to one and nothing to the other cannot stall: with a
/// single task reading them in sequence, a full pipe on the unread stream
/// would block the writer forever. This is the classic pipe deadlock, and
/// it is why the two get independent tasks rather than a `select!`.
async fn pump<R>(mut reader: R, node: u32, stream: StdStream, tx: mpsc::Sender<OutputChunk>)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        match reader.read(&mut buf).await {
            // EOF: the child closed this stream.
            Ok(0) => return,
            Ok(n) => {
                let chunk = OutputChunk {
                    node,
                    stream,
                    data: buf[..n].to_vec(),
                };
                // A closed receiver means the exec is being torn down and
                // nobody is listening any more. Stop reading rather than
                // spinning on a dead channel.
                if tx.send(chunk).await.is_err() {
                    return;
                }
            }
            Err(e) => {
                tracing::debug!(node, ?stream, "output stream ended: {e}");
                return;
            }
        }
    }
}

impl Spawned {
    /// The process's pid.
    #[must_use]
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// The exit result, if the process has already been reaped.
    #[must_use]
    pub fn exit(&self) -> Option<Exit> {
        self.exit
    }

    /// Wait for the process to exit and reap it.
    ///
    /// Also awaits the output pumps, so by the time this returns every
    /// byte the process wrote has been forwarded. Without that join, a
    /// process that writes and immediately exits could have its final
    /// output dropped — the exec would be reported finished while a pump
    /// task still held unsent bytes.
    ///
    /// Takes `&mut self` rather than `self` so a caller can race it
    /// against a cancellation signal in a `select!` and still reach for
    /// [`Spawned::terminate`] afterwards.
    pub async fn wait(&mut self) -> Exit {
        if let Some(exit) = self.exit {
            // Drain anyway: an earlier call cancelled inside its own
            // drain leaves pumps behind, and they hold a clone of the
            // output sender. See [`Spawned::drain_pumps`].
            self.drain_pumps().await;
            return exit;
        }
        let status = self.child.wait().await;
        let exit = self.record(status);
        self.drain_pumps().await;
        exit
    }

    /// The container this process's workload runs in, if any.
    #[must_use]
    pub fn container(&self) -> Option<&ContainerProc> {
        self.container.as_ref()
    }

    /// Signal the process group without waiting.
    ///
    /// Used for `mirage exec signal`, where the caller asked to deliver a
    /// signal and not to block on the outcome.
    ///
    /// For a containerised exec the signal goes *into the container*
    /// first. The process group we own contains only the provider's
    /// client, which is in a different PID namespace from the workload
    /// and does not relay signals to it — so signalling the group alone
    /// delivers `mirage exec signal` to a proxy and leaves the workload
    /// untouched.
    pub async fn signal(&self, sig: Signal) {
        if self.exit.is_some() {
            // Already reaped. Signalling now would either do nothing or,
            // worse, reach an unrelated process that reused the pid.
            return;
        }
        if let Some(container) = &self.container {
            container.signal(sig).await;
            return;
        }
        signal_group(self.pid, sig);
    }

    /// Terminate the process and its whole group, then reap it.
    ///
    /// Sends `SIGTERM`, waits up to [`TERM_GRACE`], then sends `SIGKILL`,
    /// and does not return until the child has been waited on. The return
    /// value is how the process actually ended. Idempotent: terminating an
    /// already-reaped process returns its recorded exit.
    pub async fn terminate(&mut self) -> Exit {
        if let Some(exit) = self.exit {
            // As in [`Spawned::wait`]: a cancellation that landed inside
            // a drain leaves pumps to be finished off.
            self.drain_pumps().await;
            return exit;
        }

        // Containerised: reach the workload before the client. Killing
        // the provider's client leaves the workload running inside the
        // container — a different PID namespace, which our signals do not
        // cross and which the client does not relay into — so it would
        // survive teardown holding the emulated device and its ports,
        // with mirage reporting the exec as finished.
        let reached_workload = match self.container.clone() {
            Some(container) => container.signal(Signal::SIGTERM).await,
            None => false,
        };

        // The client's own group is signalled only when the workload
        // could *not* be reached through the provider — a rank whose
        // in-container pid was never recorded, or a provider that failed.
        //
        // Signalling it unconditionally is what makes the grace period
        // below a fiction for a containerised process: the `provider
        // exec` client has no reason to ignore SIGTERM, so it dies in
        // milliseconds, `self.child.wait()` returns, and the escalation
        // arm — the only thing that sends SIGKILL *into* the container —
        // is never reached. A workload that traps or ignores SIGTERM then
        // survives teardown with mirage reporting the exec as finished,
        // which is the exact outcome the container path exists to
        // prevent. Terminating the workload ends the client naturally.
        if !reached_workload {
            signal_group(self.pid, Signal::SIGTERM);
        }

        // Give it the grace period to exit on its own terms.
        let status = match tokio::time::timeout(TERM_GRACE, self.child.wait()).await {
            Ok(status) => status,
            Err(_elapsed) => {
                // It ignored SIGTERM, or is wedged. SIGKILL cannot be
                // caught or blocked, so this stage always terminates.
                tracing::debug!(
                    pid = self.pid,
                    "process did not exit within the grace period; sending SIGKILL"
                );
                if let Some(container) = self.container.clone() {
                    container.signal(Signal::SIGKILL).await;
                }
                signal_group(self.pid, Signal::SIGKILL);
                // Reap it. There is deliberately no timeout here: after
                // SIGKILL the wait is guaranteed to complete, and giving
                // up would leave exactly the zombie this path exists to
                // prevent.
                self.child.wait().await
            }
        };
        let exit = self.record(status);

        // The direct child is gone, but a descendant that changed its own
        // process group, or was reparented before we signalled, may still
        // be alive. Sweep the group once more.
        //
        // The *group*, and only the group: the child has just been reaped,
        // so its pid is free for the kernel to hand out again, and
        // `signal_group`'s single-pid fallback would then deliver SIGKILL
        // to whatever now owns that number — the hazard `Spawned::signal`
        // refuses to take. `kill(-pid)` only reaches a group that still
        // has a living member, which is exactly the case this sweep is
        // for, and it fails harmlessly with ESRCH otherwise.
        signal_process_group_only(self.pid, Signal::SIGKILL);

        self.drain_pumps().await;
        exit
    }

    /// Record and return the outcome of a wait.
    fn record(&mut self, status: std::io::Result<std::process::ExitStatus>) -> Exit {
        let exit = match status {
            Ok(status) => Exit::from_status(status),
            Err(e) => {
                tracing::warn!(pid = self.pid, "failed to reap child: {e}");
                Exit { code: -1 }
            }
        };
        self.exit = Some(exit);
        exit
    }

    /// Let the output pumps finish forwarding, then stop them.
    ///
    /// The pumps normally end by themselves the moment the child's stream
    /// ends, which happens when the last writer to that pipe closes it.
    /// That is not always the child: a forked grandchild inherits the
    /// write end, so if one outlives its parent the pipe stays open and
    /// the pump would wait on it indefinitely. The bounded wait keeps
    /// teardown finite in that case, and aborting only ever discards
    /// output from a process that already outlived the exec.
    ///
    /// The abort is load-bearing and must be explicit. *Dropping* a tokio
    /// `JoinHandle` detaches its task rather than cancelling it, so a
    /// pump left behind that way would run for the lifetime of the daemon
    /// holding a pipe fd and a clone of the exec's `OutputChunk` sender —
    /// and because that sender never drops, the channel never closes and
    /// `forward_output` never ends either.
    ///
    /// Which is also why each handle stays in `self.pumps` until it has
    /// been joined or aborted. `Vec::drain` hands ownership to the
    /// iterator, so this whole function being cancelled — the supervisor
    /// races `wait` against a cancellation token, and the drain is inside
    /// `wait` — would drop the remaining handles and detach exactly the
    /// tasks the paragraph above says must never be detached. Leaving
    /// them in place means a later `terminate` finishes the job.
    async fn drain_pumps(&mut self) {
        const DRAIN_GRACE: Duration = Duration::from_millis(250);
        let pid = self.pid;
        while let Some(pump) = self.pumps.last_mut() {
            if tokio::time::timeout(DRAIN_GRACE, &mut *pump).await.is_err() {
                pump.abort();
                tracing::debug!(
                    pid,
                    "output pump still open after the child exited; \
                     a descendant is holding the pipe"
                );
            }
            self.pumps.pop();
        }
    }
}

/// Send `sig` to the process group led by `pid`.
///
/// Children are spawned with `process_group(0)`, so the pid is its own
/// group leader and `kill(-pid)` reaches the program plus everything it
/// spawned. Falls back to signalling the single pid if the group send
/// fails, which happens when the group is already empty.
///
/// Non-positive pids are rejected. This is not defensive noise: `kill(-1,
/// sig)` signals *every process the user can signal*, and `kill(0, sig)`
/// signals the caller's own group — so a zero or negative pid reaching
/// here would turn a routine cleanup into killing the user's session.
pub fn signal_group(pid: u32, sig: Signal) {
    let Ok(raw) = i32::try_from(pid) else {
        return;
    };
    if raw <= 0 {
        return;
    }
    if !signal_process_group_only(pid, sig) {
        let _ = nix::sys::signal::kill(Pid::from_raw(raw), sig);
    }
}

/// Send `sig` to the process group led by `pid`, and never to `pid` alone.
///
/// Returns whether the group signal was delivered. Used where the child
/// has already been reaped: the group only exists while a member is
/// alive, whereas the bare pid may already have been recycled by the
/// kernel and handed to an unrelated process.
///
/// Non-positive pids are rejected for the same reason as in
/// [`signal_group`].
pub fn signal_process_group_only(pid: u32, sig: Signal) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    nix::sys::signal::kill(Pid::from_raw(-pid), sig).is_ok()
}

/// Whether a pid is still alive.
///
/// Used only by tests and diagnostics: the supervisor learns that a
/// process exited by waiting on it, not by polling.
///
/// Pid `0` is rejected rather than probed. In POSIX `kill(0, sig)`
/// addresses *the caller's own process group*, so probing it would report
/// "alive" for what is really "no such process" — and the same aliasing
/// makes a stray `0` genuinely dangerous in [`signal_group`].
#[must_use]
pub fn process_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    nix::sys::signal::kill(Pid::from_raw(pid), None).is_ok()
}

/// Wait for a pid to disappear from the process table, up to `timeout`.
///
/// Returns `true` if it is gone. Tests use this to assert that teardown
/// actually removed a process tree.
pub async fn wait_gone(pid: u32, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if !process_alive(pid) {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(REAP_POLL).await;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn spec(command: &str, args: &[&str]) -> SpawnSpec {
        SpawnSpec {
            node: 0,
            command: command.to_string(),
            args: args.iter().map(|s| (*s).to_string()).collect(),
            env: BTreeMap::new(),
            workdir: None,
            stdio: StdioMode::Capture,
            inherit_env: false,
            container: None,
        }
    }

    /// Spawn and collect everything the process writes.
    async fn run(spec: &SpawnSpec) -> (Exit, Vec<OutputChunk>) {
        let (tx, mut rx) = mpsc::channel(64);
        let mut child = spawn(spec, tx).unwrap();
        let collector = tokio::spawn(async move {
            let mut out = Vec::new();
            while let Some(chunk) = rx.recv().await {
                out.push(chunk);
            }
            out
        });
        let exit = child.wait().await;
        (exit, collector.await.unwrap())
    }

    fn text(chunks: &[OutputChunk], stream: StdStream) -> String {
        let mut buf = Vec::new();
        for c in chunks.iter().filter(|c| c.stream == stream) {
            buf.extend_from_slice(&c.data);
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    #[tokio::test]
    async fn exit_code_is_reported() {
        let (exit, _) = run(&spec("/bin/sh", &["-c", "exit 7"])).await;
        assert_eq!(exit.code, 7);
    }

    #[tokio::test]
    async fn stdout_and_stderr_stay_separate() {
        // The PTY-based predecessor merged these into one stream and made
        // the distinction unrecoverable.
        let (exit, chunks) = run(&spec(
            "/bin/sh",
            &["-c", "echo to-stdout; echo to-stderr 1>&2"],
        ))
        .await;
        assert_eq!(exit.code, 0);
        assert_eq!(text(&chunks, StdStream::Stdout).trim(), "to-stdout");
        assert_eq!(text(&chunks, StdStream::Stderr).trim(), "to-stderr");
    }

    #[tokio::test]
    async fn signal_death_is_reported_as_128_plus_signal() {
        let (exit, _) = run(&spec("/bin/sh", &["-c", "kill -TERM $$; sleep 5"])).await;
        assert_eq!(exit.code, 128 + libc::SIGTERM);
    }

    #[tokio::test]
    async fn missing_command_is_named_in_the_error() {
        let (tx, _rx) = mpsc::channel(1);
        let err = spawn(&spec("definitely-not-a-real-binary", &[]), tx).unwrap_err();
        assert!(err.contains("command not found"), "{err}");
        assert!(err.contains("definitely-not-a-real-binary"), "{err}");
    }

    #[tokio::test]
    async fn large_output_on_both_streams_does_not_deadlock() {
        // Both pipes fill well past their buffer. A single task reading
        // them in sequence would block the writer on the stream it is not
        // currently reading; independent pumps must not.
        let script = "for i in $(seq 1 2000); do \
                      echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; \
                      echo bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb 1>&2; \
                      done";
        let (exit, chunks) = tokio::time::timeout(
            Duration::from_secs(30),
            run(&spec("/bin/sh", &["-c", script])),
        )
        .await
        .expect("must not deadlock on full pipes");
        assert_eq!(exit.code, 0);
        assert_eq!(text(&chunks, StdStream::Stdout).lines().count(), 2000);
        assert_eq!(text(&chunks, StdStream::Stderr).lines().count(), 2000);
    }

    #[tokio::test]
    async fn output_written_just_before_exit_is_not_lost() {
        // The process writes and exits immediately. If `wait` returned
        // without draining the pumps, this output would race the exit and
        // sometimes vanish.
        for _ in 0..25 {
            let (exit, chunks) = run(&spec("/bin/sh", &["-c", "echo final"])).await;
            assert_eq!(exit.code, 0);
            assert_eq!(text(&chunks, StdStream::Stdout).trim(), "final");
        }
    }

    #[tokio::test]
    async fn env_is_not_inherited_unless_asked() {
        // A variable this process really exports and the allowlist does
        // not keep. Testing with a name that is simply absent — as this
        // did with `MIRAGE_TEST_LEAK` — proves nothing: the child prints
        // `unset` with or without `env_clear`, so deleting the
        // `env_clear` call left the test green. `expect` rather than a
        // skip, so a sanitised environment fails loudly instead of
        // quietly testing nothing.
        let (key, value) = std::env::vars()
            .find(|(k, v)| {
                !INHERITED_ENV.contains(&k.as_str())
                    && !v.is_empty()
                    && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                    && v.chars().all(|c| c.is_ascii_alphanumeric() || "_-./".contains(c))
            })
            .expect("this process must export at least one variable outside the allowlist");

        let script = format!("echo \"${{{key}:-unset}}\"");
        let mut s = spec("/bin/sh", &[script.as_str()]);
        s.args = vec!["-c".to_string(), script.clone()];

        s.inherit_env = false;
        let (_, chunks) = run(&s).await;
        assert_eq!(
            text(&chunks, StdStream::Stdout).trim(),
            "unset",
            "`{key}` reached a workload that asked for a cleared environment"
        );

        s.inherit_env = true;
        let (_, chunks) = run(&s).await;
        assert_eq!(
            text(&chunks, StdStream::Stdout).trim(),
            value,
            "`{key}` did not reach a workload that inherits the caller's environment"
        );

        // And an explicit value wins over both.
        s.inherit_env = false;
        s.env.insert(key, "visible".to_string());
        let (_, chunks) = run(&s).await;
        assert_eq!(text(&chunks, StdStream::Stdout).trim(), "visible");
    }

    #[test]
    fn a_search_path_keeps_the_callers_entries_behind_mirages() {
        // An emulator backend that sets `LD_LIBRARY_PATH` or `PYTHONPATH`
        // must not delete what the caller exported — that is the loss
        // inheriting the environment was introduced to stop — and
        // mirage's own entries still have to win the search.
        assert_eq!(
            merged_search_path("/mirage/lib", std::ffi::OsStr::new("/home/me/lib")),
            std::ffi::OsString::from("/mirage/lib:/home/me/lib")
        );
        for key in ["LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"] {
            assert!(
                PATH_LIST_ENV.contains(&key),
                "{key} is a search list and must be merged, not overwritten"
            );
        }
    }

    #[tokio::test]
    async fn a_search_path_the_caller_does_not_have_is_passed_through() {
        // Nothing to merge with: the value must arrive exactly as the
        // emulator built it, with no stray separator.
        let key = "PYTHONPATH";
        if std::env::var_os(key).is_some_and(|v| !v.is_empty()) {
            // The ambient value would legitimately be appended; this
            // half of the behaviour is covered by the unit test above.
            return;
        }
        let mut s = spec("/bin/sh", &["-c", "echo \"$PYTHONPATH\""]);
        s.inherit_env = true;
        s.env.insert(key.to_string(), "/mirage/py".to_string());
        let (_, chunks) = run(&s).await;
        assert_eq!(text(&chunks, StdStream::Stdout).trim(), "/mirage/py");
    }

    #[tokio::test]
    async fn workdir_is_applied() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let mut s = spec("/bin/sh", &["-c", "pwd"]);
        s.workdir = Some(canonical.display().to_string());
        let (exit, chunks) = run(&s).await;
        assert_eq!(exit.code, 0);
        assert_eq!(
            text(&chunks, StdStream::Stdout).trim(),
            canonical.display().to_string()
        );
    }

    #[tokio::test]
    async fn terminate_kills_a_process_that_ignores_sigterm() {
        // `trap '' TERM` makes SIGTERM a no-op, so only the SIGKILL
        // escalation can end this process.
        //
        // The readiness marker is load-bearing: `trap` is a builtin the
        // shell has to actually reach, and signalling before it does
        // kills the process with the default disposition — which looks
        // exactly like a passing test while proving nothing about the
        // escalation path.
        let dir = tempfile::tempdir().unwrap();
        let ready = dir.path().join("trap-installed");
        let script = format!(
            "trap '' TERM; : > {ready}; while true; do sleep 1; done",
            ready = ready.display()
        );
        let (tx, _rx) = mpsc::channel(64);
        let mut child = spawn(&spec("/bin/sh", &["-c", &script]), tx).unwrap();
        let pid = child.pid();
        await_marker(&ready).await;

        let started = tokio::time::Instant::now();
        let exit = tokio::time::timeout(TERM_GRACE * 3, child.terminate())
            .await
            .expect("terminate must complete even against a SIGTERM-proof process");
        assert_eq!(
            exit.code,
            128 + libc::SIGKILL,
            "a SIGTERM-proof process must be escalated to SIGKILL"
        );
        assert!(
            started.elapsed() >= TERM_GRACE,
            "escalation must wait out the full grace period first"
        );
        assert!(wait_gone(pid, Duration::from_secs(5)).await);
    }

    /// Wait for a process to create `path`, used to synchronise with a
    /// child that has to reach a particular point before the test acts.
    async fn await_marker(path: &std::path::Path) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while !path.exists() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "child never reached its readiness marker: {}",
                path.display()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn terminate_reaps_the_whole_process_tree() {
        // The shell spawns a grandchild and exits, leaving it running.
        // Signalling only the direct child would leave the grandchild
        // alive; the group kill must reach it.
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("grandchild.pid");
        let script = format!(
            "sh -c 'echo $$ > {marker}; while true; do sleep 1; done' & \
             while true; do sleep 1; done",
            marker = marker.display()
        );
        let (tx, _rx) = mpsc::channel(64);
        let mut child = spawn(&spec("/bin/sh", &["-c", &script]), tx).unwrap();

        // Wait for the grandchild to announce itself.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let grandchild = loop {
            if let Ok(s) = std::fs::read_to_string(&marker)
                && let Ok(pid) = s.trim().parse::<u32>()
            {
                break pid;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "grandchild never started"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert!(process_alive(grandchild));

        child.terminate().await;

        assert!(
            wait_gone(grandchild, Duration::from_secs(10)).await,
            "grandchild {grandchild} survived teardown; the process tree leaked"
        );
    }

    #[tokio::test]
    async fn wait_and_terminate_are_idempotent_in_either_order() {
        // The supervisor races `wait` against a cancellation token, so
        // "the process exited on its own and teardown asks for
        // termination anyway" is a routine path.
        let (tx, _rx) = mpsc::channel(64);
        let mut child = spawn(&spec("/bin/sh", &["-c", "exit 5"]), tx).unwrap();
        assert_eq!(child.wait().await.code, 5);
        assert_eq!(child.wait().await.code, 5);
        assert_eq!(child.terminate().await.code, 5);
        // Signalling a reaped process must be a no-op, not a signal to
        // whatever now owns that pid.
        child.signal(Signal::SIGKILL).await;
    }

    #[tokio::test]
    async fn terminating_an_already_dead_process_is_clean() {
        let (tx, _rx) = mpsc::channel(64);
        let mut child = spawn(&spec("/bin/sh", &["-c", "exit 3"]), tx).unwrap();
        // Let it exit on its own before we ask for termination.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let exit = tokio::time::timeout(Duration::from_secs(5), child.terminate())
            .await
            .expect("terminating an exited process must not hang");
        assert_eq!(exit.code, 3);
    }

    #[tokio::test]
    async fn a_captured_process_gets_no_stdin_at_all() {
        // `--capture-all` means mirage is reading the output, and there
        // is no sensible way to share one terminal's input across ranks.
        // `/dev/null` rather than an open-but-unwritten pipe is the
        // point: a workload that reads to EOF — `cat`, `sort`, a launcher
        // reading a manifest — must finish rather than block forever on
        // input that can never arrive.
        let (tx, mut rx) = mpsc::channel(64);
        let mut child = spawn(&spec("/bin/cat", &[]), tx).unwrap();

        let collector = tokio::spawn(async move {
            let mut out = Vec::new();
            while let Some(chunk) = rx.recv().await {
                out.extend_from_slice(&chunk.data);
            }
            out
        });
        let exit = tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .expect("cat must see EOF immediately, not hang on an open pipe");
        assert_eq!(exit.code, 0);
        assert!(collector.await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn signalling_a_group_reaches_the_child() {
        let (tx, _rx) = mpsc::channel(64);
        let mut child = spawn(&spec("/bin/sh", &["-c", "sleep 30"]), tx).unwrap();
        let pid = child.pid();
        child.signal(Signal::SIGINT).await;
        let exit = tokio::time::timeout(Duration::from_secs(10), child.wait())
            .await
            .expect("signalled process must exit");
        assert_eq!(exit.code, 128 + libc::SIGINT);
        assert!(wait_gone(pid, Duration::from_secs(5)).await);
    }

    #[tokio::test]
    async fn signal_group_on_a_bogus_pid_is_harmless() {
        // Must never turn into `kill(-1, SIGKILL)`, which would signal
        // every process the user owns.
        signal_group(0, Signal::SIGKILL);
        assert!(!process_alive(0));
    }
}
