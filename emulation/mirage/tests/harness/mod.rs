//! Shared harness for the end-to-end CLI tests.
//!
//! Every test gets a private XDG root and therefore a private daemon,
//! socket and config store. That isolation is what lets the suite run in
//! parallel and lets a test make absolute claims — "no session exists",
//! "no process is left running" — without another test's state making the
//! claim false.
//!
//! [`Env`] tears its daemon down on drop, including when a test fails and
//! unwinds, so a failing assertion cannot leak a daemon or its workloads
//! into the rest of the run.

#![allow(clippy::unwrap_used, clippy::expect_used, dead_code, unreachable_pub)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use tempfile::TempDir;

/// A private mirage installation: its own XDG root, socket and daemon.
pub struct Env {
    dir: TempDir,
    config: PathBuf,
    runtime: PathBuf,
    state: PathBuf,
    socket: PathBuf,
    bin: PathBuf,
}

impl Env {
    /// Create an isolated environment. No daemon is started yet; the
    /// first command that needs one starts it.
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // The socket lives outside the XDG runtime dir. Unix socket paths
        // are limited to ~104 bytes, and a tempdir under a long
        // `TMPDIR` plus `runtime/mirage/mirage.sock` can exceed it — a
        // failure that presents as a baffling "invalid argument" at bind
        // time. Keeping the socket shallow avoids it.
        let socket = short_socket_path();
        Self {
            config: root.join("config"),
            runtime: root.join("runtime"),
            state: root.join("state"),
            socket,
            bin: PathBuf::from(env!("CARGO_BIN_EXE_mirage")),
            dir,
        }
    }

    /// The mirage binary under test.
    pub fn bin(&self) -> &Path {
        &self.bin
    }

    /// This environment's control socket.
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// The temp root.
    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    /// The per-session scratch directory for `id`.
    pub fn session_scratch(&self, id: &str) -> PathBuf {
        self.runtime.join("mirage/session").join(id)
    }

    /// The daemon's log, useful in assertion messages.
    pub fn daemon_log(&self) -> String {
        std::fs::read_to_string(self.runtime.join("mirage/daemon.log")).unwrap_or_default()
    }

    /// A `mirage` command wired to this environment.
    pub fn mirage(&self) -> Command {
        let mut c = Command::new(&self.bin);
        c.env("XDG_CONFIG_HOME", &self.config)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("XDG_STATE_HOME", &self.state)
            .env("MIRAGE_SOCKET", &self.socket)
            // Auto-start must re-exec *this* binary, not whatever
            // `mirage` happens to be installed on the machine.
            .env("MIRAGE_BIN", &self.bin)
            .env_remove("MIRAGE_LOG")
            .env_remove("MIRAGE_CONFIG")
            .env_remove("MIRAGE_RUNTIME")
            .env_remove("MIRAGE_STATE")
            .env_remove("MIRAGE_AUTOSTART");
        c
    }

    /// The environment every mirage invocation in this test needs.
    ///
    /// Exposed separately from [`Env::mirage`] so a test that builds its
    /// own command — one running on a pseudo-terminal, say — still gets
    /// the same isolation.
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
            (
                "XDG_STATE_HOME".to_string(),
                self.state.display().to_string(),
            ),
            ("MIRAGE_SOCKET".to_string(), self.socket.display().to_string()),
            ("MIRAGE_BIN".to_string(), self.bin.display().to_string()),
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
            "`mirage {}` failed with {:?}\nstdout: {}\nstderr: {}\ndaemon log:\n{}",
            args.join(" "),
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
            self.daemon_log()
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

    /// Start a session on `profile` and return its id.
    pub fn start_session(&self, profile: &str, id: &str) -> String {
        self.ok(&[
            "session", "start", "--profile", profile, "--id", id, "--no-input",
        ])
        .trim()
        .to_string()
    }

    /// Whether a daemon is currently serving this environment.
    pub fn daemon_running(&self) -> bool {
        let out = self.run(&["daemon", "status"]);
        out.status.success()
            && String::from_utf8_lossy(&out.stdout).starts_with("running")
    }

    /// The daemon's pid, if one is running.
    pub fn daemon_pid(&self) -> Option<u32> {
        let out = self.run(&["daemon", "status"]);
        let text = String::from_utf8_lossy(&out.stdout);
        text.lines()
            .find_map(|l| l.strip_prefix("running   pid "))
            .and_then(|p| p.trim().parse().ok())
    }

    /// Stop this environment's daemon and wait for it to go away.
    ///
    /// Runs from `Drop`, including while a failed assertion is unwinding,
    /// so it must not panic: a panic during unwinding aborts the process,
    /// which takes down every test sharing this binary and loses the
    /// original assertion message. `run` panics when it cannot spawn —
    /// exactly what an EMFILE or a fork failure under the strain suite's
    /// load produces — so spawn directly here and give up quietly.
    pub fn stop_daemon(&self) {
        let Ok(status) = self.mirage().args(["daemon", "status"]).output() else {
            return;
        };
        let pid = String::from_utf8_lossy(&status.stdout)
            .lines()
            .find_map(|l| l.strip_prefix("running   pid "))
            .and_then(|p| p.trim().parse::<u32>().ok());
        let _ = self.mirage().args(["daemon", "stop"]).output();
        let Some(pid) = pid else { return };
        // `daemon stop` is acknowledged before the daemon exits, so wait
        // for the process itself: a test asserting nothing leaked needs
        // teardown to have actually finished.
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline && pid_alive(pid) {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        // Runs on the failure path too, so one failing test cannot leave
        // a daemon (and its workloads) behind for the rest of the suite.
        self.stop_daemon();
        let _ = std::fs::remove_file(&self.socket);
        let _ = std::fs::remove_file(self.socket.with_extension("lock"));
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
            .env("XDG_STATE_HOME", probe.path().join("state"))
            .env("MIRAGE_SOCKET", probe.path().join("probe.sock"))
            .env("MIRAGE_AUTOSTART", "0")
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

/// A short, unique socket path.
///
/// `sun_path` is a fixed-size array (108 bytes on Linux), so a socket
/// nested inside a long tempdir path silently fails to bind. Tests that
/// need a private daemon put their socket directly under `TMPDIR`.
pub fn short_socket_path() -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let tmp = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(tmp).join(format!(
        "mrg-{}-{}.sock",
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
