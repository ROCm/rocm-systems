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

#![allow(clippy::unwrap_used, clippy::expect_used, dead_code, unreachable_pub)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use tempfile::TempDir;

/// A private mirage installation: its own XDG root and config store.
pub struct Env {
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
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        Self {
            config: dir.path().join("config"),
            runtime: short_runtime_dir(),
            bin: PathBuf::from(env!("CARGO_BIN_EXE_mirage")),
            dir,
        }
    }

    /// The mirage binary under test.
    pub fn bin(&self) -> &Path {
        &self.bin
    }

    /// This environment's XDG runtime root.
    pub fn runtime(&self) -> &Path {
        &self.runtime
    }

    /// The directory live runs put their sockets in.
    pub fn run_socket_dir(&self) -> PathBuf {
        self.runtime.join("mirage/run")
    }

    /// Session ids of every run currently serving a socket.
    pub fn live_runs(&self) -> Vec<String> {
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
    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    /// The per-session scratch directory for `id`.
    pub fn session_scratch(&self, id: &str) -> PathBuf {
        self.runtime.join("mirage/session").join(id)
    }

    /// A `mirage` command wired to this environment.
    pub fn mirage(&self) -> Command {
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
    pub fn child_env(&self) -> Vec<(String, String)> {
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
    pub fn run(&self, args: &[&str]) -> Output {
        self.mirage()
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("failed to run `mirage {}`: {e}", args.join(" ")))
    }

    /// Run a mirage command that must succeed, returning its stdout.
    pub fn ok(&self, args: &[&str]) -> String {
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
    pub fn fails(&self, args: &[&str]) -> String {
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
    pub fn create_profile(&self, name: &str) {
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
    pub fn spawn_run(&self, args: &[&str], argv: &[&str]) -> Run {
        let mut cmd = self.mirage();
        cmd.arg("run")
            .args(args)
            .arg("--")
            .args(argv)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let child = cmd.spawn().expect("spawning `mirage run`");
        Run {
            child: Some(child),
            socket_dir: self.run_socket_dir(),
        }
    }
}

/// A background `mirage run`, killed when this value is dropped.
pub struct Run {
    child: Option<std::process::Child>,
    socket_dir: PathBuf,
}

impl Run {
    /// Wait until this run is serving its socket, and return its session
    /// id.
    ///
    /// The socket appearing is the signal that the session is *ready*:
    /// `mirage run` binds it only after bring-up succeeds, so a test that
    /// gets an id here can exec into the session immediately.
    pub fn await_ready(&mut self, timeout: Duration) -> String {
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
                    .find_map(|e| Some(e.path().file_stem()?.to_str()?.to_string()))
            {
                return id;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("`mirage run` did not start serving within {timeout:?}");
    }

    /// This run's pid, while it is alive.
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(std::process::Child::id)
    }

    /// Signal the run, as a user pressing Ctrl-C in its terminal would.
    pub fn signal(&self, sig: nix::sys::signal::Signal) {
        if let Some(pid) = self.pid() {
            let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), sig);
        }
    }

    /// Wait for the run to exit, returning its output.
    pub fn wait(&mut self, timeout: Duration) -> Output {
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
        child.wait_with_output().expect("collecting run output")
    }

    /// Stop the run and wait for it to go away.
    pub fn kill(&mut self) {
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
pub const TEST_EMULATOR: &str = "rocjitsu";

/// Whether the test emulator's runtime is available on this machine.
///
/// rocjitsu is a sibling project in this monorepo and mirage discovers
/// `librocjitsu.so` relative to its own binary, so a full build has it. A
/// mirage-only build does not, and these suites cannot run there.
///
/// Probed once and cached: it shells out to the binary, and asking
/// per-test would add a process spawn to every one of them.
pub fn test_emulator_available() -> bool {
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
pub const ENV_ALLOW_SKIP: &str = "MIRAGE_E2E_ALLOW_SKIP";

/// Skip the calling test when the test emulator is unavailable.
///
/// Returns `true` when the test should stop.
#[must_use]
pub fn skip_without_emulator() -> bool {
    if test_emulator_available() {
        return false;
    }
    eprintln!(
        "SKIP: the `{TEST_EMULATOR}` runtime was not found, so no session          can be brought up."
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
pub fn assert_suite_can_run() {
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
pub fn short_runtime_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let tmp = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(tmp).join(format!(
        "mrg-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Whether a pid is still in the process table.
pub fn pid_alive(pid: u32) -> bool {
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
pub fn pid_is_zombie(pid: u32) -> bool {
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
pub fn count_processes(marker: &str) -> usize {
    let uid = nix::unistd::getuid().to_string();
    let Ok(output) = Command::new("pgrep")
        .args(["-u", &uid, "-f", marker])
        .output()
    else {
        return 0;
    };
    let me = std::process::id().to_string();
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && *l != me)
        .count()
}

/// Pids of this user's processes whose command line contains `marker`.
pub fn find_processes(marker: &str) -> Vec<u32> {
    let uid = nix::unistd::getuid().to_string();
    let Ok(output) = Command::new("pgrep")
        .args(["-u", &uid, "-f", marker])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect()
}

/// Wait until `cond` holds, or fail with `what`.
pub fn wait_for(what: &str, timeout: Duration, mut cond: impl FnMut() -> bool) {
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
pub fn assert_no_leaks(marker: &str) {
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
pub fn marker(name: &str) -> String {
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
pub fn tagged_sleep(marker: &str) -> String {
    // The marker is in the command line (as an unused variable), which is
    // what `pgrep -f` matches on.
    format!("MARKER={marker}; while true; do sleep 1; done")
}
