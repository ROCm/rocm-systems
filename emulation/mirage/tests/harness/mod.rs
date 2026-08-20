//! Shared harness for the end-to-end CLI tests.
//!
//! Every test gets a private XDG root and therefore its own config store
//! and run directory. That isolation is what lets the suite run in
//! parallel and lets a test make absolute claims — "no run is live", "no
//! process is left running" — without another test's state making the
//! claim false.
//!
//! # There is nothing to tear down
//!
//! A `mirage run` is a foreground process, so a test owns its lifetime
//! directly: it spawns one and kills it, or waits for it to exit. There
//! is no daemon to stop on drop, no socket to unlink, and no way for a
//! test that panics mid-assertion to strand a background process — the
//! run is a child of the test binary, and [`Run`] kills it in its own
//! `Drop`.

// `dead_code` is the one that cannot be narrowed. This module is compiled
// separately into each of the five integration-test binaries, and none of
// them uses all of it — so every helper is dead code from the point of
// view of at least one binary, and the lint has no way to see the others.
// Everything else here is ordinary test-code opt-out: a panic in a test
// *is* the failure mechanism.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use tempfile::TempDir;

/// A private mirage installation: its own XDG root and config store.
pub(crate) struct Env {
    dir: TempDir,
    config: PathBuf,
    runtime: PathBuf,
    bin: PathBuf,
}

impl Env {
    /// Create an isolated environment.
    ///
    /// The XDG runtime root is kept shallow — directly under `TMPDIR`
    /// rather than inside the test's tempdir — because a run's control
    /// socket lives under it. `sun_path` is 108 bytes on Linux, and a
    /// deep tempdir path plus `mirage/run/<session>.sock` can exceed it,
    /// which presents as a baffling "invalid argument" at bind time.
    pub(crate) fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        Self {
            config: dir.path().join("config"),
            runtime: short_runtime_dir(),
            bin: PathBuf::from(env!("CARGO_BIN_EXE_mirage")),
            dir,
        }
    }

    /// The mirage binary under test.
    pub(crate) fn bin(&self) -> &Path {
        &self.bin
    }

    /// This environment's XDG runtime root.
    pub(crate) fn runtime(&self) -> &Path {
        &self.runtime
    }

    /// The directory stored profiles live in.
    pub(crate) fn profile_dir(&self) -> PathBuf {
        self.config.join("mirage/profile")
    }

    /// The directory live runs put their sockets in.
    pub(crate) fn run_socket_dir(&self) -> PathBuf {
        self.runtime.join("mirage/run")
    }

    /// Session ids of every run currently serving a socket.
    pub(crate) fn live_runs(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(self.run_socket_dir()) else {
            return Vec::new();
        };
        let mut ids: Vec<String> = entries
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "sock"))
            .filter_map(|e| Some(e.path().file_stem()?.to_str()?.to_string()))
            .collect();
        ids.sort();
        ids
    }

    /// The temp root.
    pub(crate) fn root(&self) -> &Path {
        self.dir.path()
    }

    /// The per-session scratch directory for `id`.
    pub(crate) fn session_scratch(&self, id: &str) -> PathBuf {
        self.runtime.join("mirage/session").join(id)
    }

    /// A `mirage` command wired to this environment.
    pub(crate) fn mirage(&self) -> Command {
        let mut c = Command::new(&self.bin);
        c.env("XDG_CONFIG_HOME", &self.config)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env_remove("MIRAGE_LOG")
            .env_remove("MIRAGE_CONFIG")
            .env_remove("MIRAGE_RUNTIME");
        c
    }

    /// The environment every mirage invocation in this test needs.
    ///
    /// Exposed separately from [`Env::mirage`] so a test that builds its
    /// own command still gets the same isolation.
    pub(crate) fn child_env(&self) -> Vec<(String, String)> {
        vec![
            (
                "XDG_CONFIG_HOME".to_string(),
                self.config.display().to_string(),
            ),
            (
                "XDG_RUNTIME_DIR".to_string(),
                self.runtime.display().to_string(),
            ),
        ]
    }

    /// Run a mirage command and return its output, whatever the status.
    pub(crate) fn run(&self, args: &[&str]) -> Output {
        self.mirage()
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("failed to run `mirage {}`: {e}", args.join(" ")))
    }

    /// Run a mirage command that must succeed, returning its stdout.
    pub(crate) fn ok(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert!(
            out.status.success(),
            "`mirage {}` failed with {:?}\nstdout: {}\nstderr: {}",
            args.join(" "),
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    /// Run a mirage command that must fail, returning its stderr.
    pub(crate) fn fails(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert!(
            !out.status.success(),
            "`mirage {}` unexpectedly succeeded\nstdout: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stdout)
        );
        String::from_utf8_lossy(&out.stderr).into_owned()
    }

    /// Create a profile named `name` on the test emulator backend.
    pub(crate) fn create_profile(&self, name: &str) {
        self.ok(&[
            "profile",
            "create",
            name,
            "--emulator",
            TEST_EMULATOR,
            "--no-input",
        ]);
    }

    /// Spawn a background `mirage run` and wait until it is serving.
    ///
    /// The returned [`Run`] owns the process: it is killed when the value
    /// is dropped, so a test that panics cannot strand a session.
    ///
    /// `args` go before the `--`; `argv` is the command to run.
    pub(crate) fn spawn_run(&self, args: &[&str], argv: &[&str]) -> Run {
        let mut cmd = self.mirage();
        cmd.arg("run")
            .args(args)
            .arg("--")
            .args(argv)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        // Snapshot the sockets that already exist, so `await_ready` can
        // tell *this* run's socket from one lying around. A run killed
        // with SIGKILL leaves its socket behind — that is the whole point
        // of the reclaim path — and picking it up here would hand the
        // test a dead run's session id for a live run.
        let existing = self.live_runs().into_iter().collect();
        let child = cmd.spawn().expect("spawning `mirage run`");
        Run {
            child: Some(child),
            socket_dir: self.run_socket_dir(),
            existing,
            stderr: None,
        }
    }
}

/// A background `mirage run`, killed when this value is dropped.
pub(crate) struct Run {
    child: Option<std::process::Child>,
    socket_dir: PathBuf,
    /// Session ids that already had a socket when this run was spawned.
    existing: std::collections::HashSet<String>,
    /// Live view of the run's stderr, once a test has asked for one.
    stderr: Option<(StderrWatch, std::sync::mpsc::Receiver<()>)>,
}

/// What a run has said on stderr *so far*.
///
/// `mirage run` narrates its own state transitions on stderr — bring-up
/// phases, and the "this command has finished, but N borrower(s) are
/// still using session X" line that says it has stopped running the
/// workload and started waiting. Those lines are the only external
/// evidence of a transition that has no other observable effect, and a
/// test that wants to act *at* the transition has to see them while the
/// run is still going. [`Run::wait`] drains the pipe only once the
/// process is over, which is too late.
///
/// Cloneable and cheap: it is a handle on the buffer a pump thread is
/// filling, not a copy of it.
#[derive(Clone)]
pub(crate) struct StderrWatch {
    buf: std::sync::Arc<std::sync::Mutex<String>>,
}

impl StderrWatch {
    /// Whether the run has written `needle` to stderr yet.
    pub(crate) fn contains(&self, needle: &str) -> bool {
        self.text().contains(needle)
    }

    /// Everything the run has written to stderr so far.
    pub(crate) fn text(&self) -> String {
        self.buf.lock().expect("stderr buffer").clone()
    }
}

impl Run {
    /// Wait until this run is serving its socket, and return its session
    /// id.
    ///
    /// Readiness is a completed `Describe` round-trip, not the socket
    /// file appearing. `mirage run` binds the socket *before* bring-up —
    /// so that a run in the middle of a long image pull is visible to
    /// `mirage state purge` and is not mistaken for an orphan — but only
    /// starts answering once the session is healthy. Waiting for the file
    /// would hand a test a session that has no containers and no emulator
    /// environment yet, and every assertion after it would race bring-up.
    pub(crate) fn await_ready(&mut self, timeout: Duration) -> String {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(child) = self.child.as_mut()
                && let Ok(Some(status)) = child.try_wait()
            {
                panic!("`mirage run` exited before serving its socket: {status:?}");
            }
            if let Ok(entries) = std::fs::read_dir(&self.socket_dir)
                && let Some(id) = entries
                    .flatten()
                    .filter(|e| e.path().extension().is_some_and(|x| x == "sock"))
                    .filter_map(|e| Some(e.path().file_stem()?.to_str()?.to_string()))
                    // Skip inside the iterator, not after it: a corpse
                    // socket sorts arbitrarily among the others, so
                    // rejecting only the first entry found would still
                    // return one.
                    .find(|id| !self.existing.contains(id))
                && describes_itself(&self.socket_dir.join(format!("{id}.sock")))
            {
                return id;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("`mirage run` did not start serving within {timeout:?}");
    }

    /// Start following this run's stderr as it is written.
    ///
    /// Call it before the transition you want to wait for; anything the
    /// run said earlier is still captured, because the pump reads the
    /// pipe from wherever it has got to and the kernel buffers the rest.
    /// [`Run::wait`] still returns the complete stderr afterwards.
    ///
    /// Panics if called twice, or after the run has been waited on.
    pub(crate) fn watch_stderr(&mut self) -> StderrWatch {
        use std::io::Read as _;

        assert!(
            self.stderr.is_none(),
            "a run's stderr is followed by one watcher"
        );
        let mut pipe = self
            .child
            .as_mut()
            .expect("a run is watched while it is alive")
            .stderr
            .take()
            .expect("`spawn_run` pipes stderr");

        let watch = StderrWatch {
            buf: std::sync::Arc::new(std::sync::Mutex::new(String::new())),
        };
        let buf = std::sync::Arc::clone(&watch.buf);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Byte at a time rather than by line: the interesting lines
            // are printed as the run reaches a state, and buffering one
            // back would make the test wait for the *next* line before it
            // could see this one.
            let mut byte = [0u8; 1];
            while let Ok(1) = pipe.read(&mut byte) {
                buf.lock().expect("stderr buffer").push(char::from(byte[0]));
            }
            // Send, rather than just ending: `wait` uses this to know the
            // pipe reached EOF and the buffer is complete.
            let _ = tx.send(());
        });
        self.stderr = Some((watch.clone(), rx));
        watch
    }

    /// This run's pid, while it is alive.
    pub(crate) fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(std::process::Child::id)
    }

    /// Whether the run is still going.
    ///
    /// A question with a deadline attached everywhere else in this
    /// suite — but some properties are about a run that must *not* have
    /// exited yet, and those need the instantaneous answer.
    pub(crate) fn is_running(&mut self) -> bool {
        matches!(
            self.child.as_mut().map(std::process::Child::try_wait),
            Some(Ok(None))
        )
    }

    /// Signal the run, as a user pressing Ctrl-C in its terminal would.
    pub(crate) fn signal(&self, sig: nix::sys::signal::Signal) {
        if let Some(pid) = self.pid() {
            let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), sig);
        }
    }

    /// Wait for the run to exit, returning its output.
    pub(crate) fn wait(&mut self, timeout: Duration) -> Output {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.as_mut().map(std::process::Child::try_wait) {
                Some(Ok(Some(_))) | None => break,
                Some(Ok(None)) => {}
                Some(Err(e)) => panic!("waiting on `mirage run`: {e}"),
            }
            assert!(
                Instant::now() < deadline,
                "`mirage run` did not exit within {timeout:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        let child = self.child.take().expect("a run is waited on once");

        // Collecting the output has its own deadline, because it can hang
        // for exactly the reason these tests exist to catch.
        // `wait_with_output` reads the run's stdout and stderr to EOF, and
        // a leaked descendant that inherited them holds the write end
        // open forever. Without the bound, the regression "a workload's
        // grandchild survived its run" hangs the whole test binary
        // instead of failing the assertion that was written for it.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(child.wait_with_output());
        });
        let mut out = match rx.recv_timeout(timeout) {
            Ok(out) => out.expect("collecting run output"),
            Err(_) => panic!(
                "`mirage run` exited but its output pipes are still open after {timeout:?}: \
                 something it started outlived it and inherited them"
            ),
        };

        // A watcher owns the stderr pipe, so `wait_with_output` above saw
        // none. Wait for its pump to reach EOF — bounded for the same
        // reason the stdout collection is — and hand back what it read,
        // so `wait` returns the whole stderr either way.
        if let Some((watch, done)) = self.stderr.take() {
            assert!(
                done.recv_timeout(timeout).is_ok(),
                "`mirage run` exited but its stderr pipe is still open after {timeout:?}: \
                 something it started outlived it and inherited it"
            );
            out.stderr = watch.text().into_bytes();
        }
        out
    }

    /// Stop the run and wait for it to go away.
    pub(crate) fn kill(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        // Runs on the failure path too, so one failing test cannot leave
        // a run (and its workloads and containers) behind.
        self.kill();
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        // The shallow runtime root lives outside the tempdir, so it is
        // this type's job to remove it.
        let _ = std::fs::remove_dir_all(&self.runtime);
    }
}

/// The emulator backend the end-to-end suites run their sessions on.
///
/// These tests exercise the session and process lifecycle, not emulation,
/// but a session still needs a backend that can produce a usable
/// injection — and mirage ships exactly one, the emulator it exists to
/// drive. rocjitsu interposes GPU calls the shell commands here never
/// make, so it adds no behaviour to what is under test, only the
/// requirement that its runtime library is present.
pub(crate) const TEST_EMULATOR: &str = "rocjitsu";

/// Whether the test emulator's runtime is available on this machine.
///
/// rocjitsu is a sibling project in this monorepo and mirage discovers
/// `librocjitsu.so` relative to its own binary, so a full build has it. A
/// mirage-only build does not, and these suites cannot run there.
///
/// Probed once and cached: it shells out to the binary, and asking
/// per-test would add a process spawn to every one of them.
pub(crate) fn test_emulator_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let probe = match tempfile::tempdir() {
            Ok(d) => d,
            Err(_) => return false,
        };
        // Isolated, so probing never reads or writes the developer's real
        // mirage directories and never starts a daemon.
        let output = Command::new(env!("CARGO_BIN_EXE_mirage"))
            .args(["--json", "emulators"])
            .env("XDG_CONFIG_HOME", probe.path().join("config"))
            .env("XDG_RUNTIME_DIR", probe.path().join("runtime"))
            .output();
        let Ok(output) = output else { return false };
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) else {
            return false;
        };
        json.as_array().is_some_and(|entries| {
            entries.iter().any(|e| {
                e["name"] == TEST_EMULATOR
                    && e["installed"] == true
                    && e["support"]["supported"] == true
            })
        })
    })
}

/// Environment variable that acknowledges the emulator is missing.
///
/// See [`assert_suite_can_run`].
pub(crate) const ENV_ALLOW_SKIP: &str = "MIRAGE_E2E_ALLOW_SKIP";

/// Skip the calling test when the test emulator is unavailable.
///
/// Returns `true` when the test should stop.
#[must_use]
pub(crate) fn skip_without_emulator() -> bool {
    if test_emulator_available() {
        return false;
    }
    eprintln!(
        "SKIP: the `{TEST_EMULATOR}` runtime was not found, so no session \
         can be brought up."
    );
    true
}

/// Fail unless this machine can actually run the suite.
///
/// Every session test skips when the emulator runtime is missing, and a
/// skipped Rust test still reports `ok` — so without this the whole suite
/// goes green in half a second while testing nothing, which is worse than
/// a red run because nobody investigates it. One deliberate failure says
/// what is missing and how to fix it, while the rest skip quietly.
///
/// Set `MIRAGE_E2E_ALLOW_SKIP=1` to accept the skips, for a build that
/// deliberately does not include rocjitsu.
///
/// Call this from exactly one test per suite.
pub(crate) fn assert_suite_can_run() {
    if test_emulator_available() {
        return;
    }
    if std::env::var_os(ENV_ALLOW_SKIP).is_some() {
        eprintln!("{ENV_ALLOW_SKIP} is set; accepting a suite that tests nothing.");
        return;
    }
    panic!(
        "the `{TEST_EMULATOR}` runtime was not found, so every session test \
in this suite skipped and the suite proves nothing.\n\n\
         Build the sibling `emulation/rocjitsu` project, or set ROCM_HOME to \
an install that provides librocjitsu.so.\n\n\
         If this build deliberately excludes rocjitsu, set \
{ENV_ALLOW_SKIP}=1 to accept the skips."
    );
}

/// A short, unique XDG runtime root.
///
/// `sun_path` is a fixed-size array (108 bytes on Linux), so a run socket
/// nested inside a long tempdir path silently fails to bind. Putting the
/// runtime root directly under `TMPDIR` keeps every socket well inside
/// the limit.
pub(crate) fn short_runtime_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let tmp = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(tmp).join(format!(
        "mrg-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Whether the run serving `socket` answers a `Describe`.
///
/// The real protocol, spoken by hand: a 4-byte big-endian length prefix
/// followed by JSON. Answering at all means bring-up finished, because
/// `mirage run` binds its socket before serving it and only begins to
/// serve once the session is healthy.
fn describes_itself(socket: &Path) -> bool {
    use std::io::{Read as _, Write as _};
    use std::os::unix::net::UnixStream;

    let Ok(mut stream) = UnixStream::connect(socket) else {
        return false;
    };
    // Short timeouts: this is polled in a loop, and a run that is still
    // pulling an image accepts the connection (the listener is bound) but
    // will not answer until it is ready.
    let brief = Duration::from_millis(250);
    let _ = stream.set_read_timeout(Some(brief));
    let _ = stream.set_write_timeout(Some(brief));

    // `Request::Describe` is a unit variant, so serde renders it as the
    // bare JSON string `"Describe"`.
    let body = b"\"Describe\"";
    let mut frame = (body.len() as u32).to_be_bytes().to_vec();
    frame.extend_from_slice(body);
    if stream.write_all(&frame).is_err() {
        return false;
    }

    let mut len = [0u8; 4];
    if stream.read_exact(&mut len).is_err() {
        return false;
    }
    let mut payload = vec![0u8; u32::from_be_bytes(len) as usize];
    if stream.read_exact(&mut payload).is_err() {
        return false;
    }
    // `Response::Description` on success, `Response::Error` while the
    // session cannot describe itself yet.
    String::from_utf8_lossy(&payload).contains("Description")
}

/// Whether a pid is still in the process table.
pub(crate) fn pid_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
}

/// Whether a pid is a zombie awaiting reaping.
///
/// A zombie still answers `kill(pid, 0)`, so liveness checks alone cannot
/// see one — which is precisely how the previous design's leaks stayed
/// invisible. This reads the process state from `/proc` instead.
pub(crate) fn pid_is_zombie(pid: u32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    // `comm` can contain spaces and parentheses, so state is the field
    // right after the final ')'.
    stat.rfind(')')
        .and_then(|i| stat[i + 1..].split_whitespace().next())
        .is_some_and(|state| state == "Z")
}

/// Count this user's processes whose command line contains `marker`.
///
/// Tests tag their workloads with a unique marker so this counts only
/// their own processes, never another test's or the harness's.
pub(crate) fn count_processes(marker: &str) -> usize {
    find_processes(marker).len()
}

/// Pids of this user's *workload* processes whose command line contains
/// `marker`.
///
/// mirage's own CLI processes are excluded, and that exclusion is what
/// makes every caller mean what it says. The marker is part of the
/// workload's argv, and the workload's argv is part of the argv of the
/// `mirage run` / `mirage exec` that was told to launch it — so `pgrep
/// -f` matches the CLI as well. Without the filter, "wait until the
/// workload is up" is satisfied by the CLI process the test just spawned,
/// before the workload exists at all, and an assertion that the run's
/// workload survived is satisfied by the run process itself.
///
/// Filtering on `comm` rather than on a pid list: `mirage exec` clients
/// and any re-exec of the binary have to go too, and the test does not
/// know their pids.
pub(crate) fn find_processes(marker: &str) -> Vec<u32> {
    let uid = nix::unistd::getuid().to_string();
    let Ok(output) = Command::new("pgrep")
        .args(["-u", &uid, "-f", marker])
        .output()
    else {
        return Vec::new();
    };
    let me = std::process::id();
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .filter(|pid| *pid != me && !is_mirage_process(*pid))
        .collect()
}

/// Whether `pid` is a mirage CLI process rather than a workload.
fn is_mirage_process(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/comm")).is_ok_and(|comm| comm.trim() == "mirage")
}

/// Wait for a child process to exit, returning whether it did.
///
/// Bounded on purpose, and the bound is the point. "The client never
/// returns" is one of the regressions this suite exists to catch, so a
/// test that waits on one with no deadline converts that regression into
/// a hung test binary — no failure, no name, nothing to read but a
/// stopped clock — instead of an assertion. On a timeout the child is
/// killed and reaped, so a test that fails this way still leaves the
/// machine as it found it.
pub(crate) fn wait_for_exit(child: &mut std::process::Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
        }
    }
}

/// Wait until `cond` holds, or fail with `what`.
pub(crate) fn wait_for(what: &str, timeout: Duration, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("timed out after {timeout:?} waiting for: {what}");
}

/// Assert that no process tagged `marker` is left running.
pub(crate) fn assert_no_leaks(marker: &str) {
    // Allow a brief moment: a reap is a syscall away, not instantaneous.
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && count_processes(marker) > 0 {
        std::thread::sleep(Duration::from_millis(50));
    }
    let leaked = find_processes(marker);
    assert!(
        leaked.is_empty(),
        "{} process(es) tagged {marker:?} outlived their session: {leaked:?}",
        leaked.len()
    );
}

/// A marker string unique to one test, for tagging its workloads.
pub(crate) fn marker(name: &str) -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    format!(
        "mirage-test-{name}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    )
}

/// A shell snippet that sleeps forever, tagged with `marker` so the test
/// can find it in the process table.
pub(crate) fn tagged_sleep(marker: &str) -> String {
    // The marker is in the command line (as an unused variable), which is
    // what `pgrep -f` matches on.
    format!("MARKER={marker}; while true; do sleep 1; done")
}
