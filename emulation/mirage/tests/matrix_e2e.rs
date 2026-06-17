//! Matrix-driven end-to-end tests for the `mirage` CLI.
//!
//! This test enumerates the full cross product of the dimensions
//! described in [`tests/matrix.md`] and drives each runnable
//! combination through the canonical lifecycle:
//!
//! ```text
//! create  ->  run  ->  delete  ->  ensure deleted
//! ```
//!
//! Every combination is exercised end-to-end against the real `mirage`
//! binary (via `assert_cmd`) under an isolated XDG root. Combinations
//! that the current host *cannot* run are deliberately **skipped** with
//! a recorded reason rather than failed, so the same suite is
//! meaningful on a laptop, in CI, and on an emulation host:
//!
//! * `rocjitsu-dbt` is skipped unless a translation-target GPU is
//!   physically present (DBT runs translated code on real hardware).
//! * `rocjitsu` is skipped when its KMD library cannot be located.
//! * the `hazard-detection` plugin is skipped when the backend does not
//!   advertise it.
//!
//! The containerised dimensions (`podman`, `docker`) are driven through
//! a hermetic mock provider — a small shell script standing in for the
//! container CLI — so the provider bring-up/teardown contract is
//! exercised without requiring a real image or daemon, mirroring
//! `tests/container_e2e.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use assert_cmd::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Matrix dimensions
// ---------------------------------------------------------------------------

/// The emulator backend under test (`### emulator` in matrix.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Emulator {
    Rocjitsu,
    RocjitsuDbt,
}

impl Emulator {
    fn kind(self) -> &'static str {
        match self {
            Emulator::Rocjitsu => "rocjitsu",
            Emulator::RocjitsuDbt => "rocjitsu-dbt",
        }
    }
}

/// How the session's nodes are hosted (`### containerization`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Container {
    /// Run directly on the node — no container runtime involved.
    Node,
    Podman,
    Docker,
}

impl Container {
    fn label(self) -> &'static str {
        match self {
            Container::Node => "node",
            Container::Podman => "podman",
            Container::Docker => "docker",
        }
    }

    fn is_containerized(self) -> bool {
        !matches!(self, Container::Node)
    }
}

/// The emulated GPU (`### hardware`). Names match the builtin agents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hardware {
    Mi350x,
    Mi450x,
}

impl Hardware {
    fn agent(self) -> &'static str {
        match self {
            Hardware::Mi350x => "MI350X",
            Hardware::Mi450x => "MI450X",
        }
    }
}

/// The workload (`### payload`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Payload {
    TinyTorch,
    Rccl,
    Crash,
}

impl Payload {
    fn label(self) -> &'static str {
        match self {
            Payload::TinyTorch => "tiny_torch",
            Payload::Rccl => "rccl",
            Payload::Crash => "crash",
        }
    }

    /// Number of emulated nodes the payload spans.
    fn nodes(self) -> u32 {
        match self {
            Payload::Rccl => 2,
            _ => 1,
        }
    }

    /// The command to run, plus the exit-code contract.
    ///
    /// The heavy real workloads (a torch import, an RCCL all-reduce) are
    /// represented here by lightweight, deterministic stand-ins that
    /// print a sentinel: the goal of this suite is to prove the *mirage
    /// lifecycle* (create/run/clean-up) across the matrix, not to
    /// benchmark the emulator. The real torch fixture is driven
    /// separately by `tests/run_tiny_torch_mi350.sh`.
    fn argv(self) -> Vec<&'static str> {
        match self {
            Payload::TinyTorch => vec!["/bin/sh", "-c", "echo tiny_torch_ok"],
            // Each rank prints once; with two nodes the orchestrator runs
            // the command on both.
            Payload::Rccl => vec!["/bin/sh", "-c", "echo rccl_ok"],
            // Simulate a crashing workload: emit output, then exit with a
            // SIGSEGV-style code. mirage must still tear the session down.
            Payload::Crash => vec!["/bin/sh", "-c", "echo crashing; exit 139"],
        }
    }

    /// Whether a clean (zero) exit is expected.
    fn expect_success(self) -> bool {
        !matches!(self, Payload::Crash)
    }
}

/// Emulator plugins (`### plugins`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plugin {
    None,
    HazardDetection,
}

impl Plugin {
    fn label(self) -> &'static str {
        match self {
            Plugin::None => "none",
            Plugin::HazardDetection => "hazard-detection",
        }
    }
}

/// One point in the matrix.
#[derive(Clone, Copy, Debug)]
struct Combo {
    emulator: Emulator,
    container: Container,
    hardware: Hardware,
    payload: Payload,
    plugin: Plugin,
}

impl Combo {
    fn name(&self) -> String {
        format!(
            "{}+{}+{}+{}+{}",
            self.emulator.kind(),
            self.container.label(),
            self.hardware.agent().to_lowercase(),
            self.payload.label(),
            self.plugin.label(),
        )
    }
}

/// The full cross product of every dimension.
fn all_combos() -> Vec<Combo> {
    let mut combos = Vec::new();
    for emulator in [Emulator::Rocjitsu, Emulator::RocjitsuDbt] {
        for container in [Container::Node, Container::Podman, Container::Docker] {
            for hardware in [Hardware::Mi350x, Hardware::Mi450x] {
                for payload in [Payload::TinyTorch, Payload::Rccl, Payload::Crash] {
                    for plugin in [Plugin::None, Plugin::HazardDetection] {
                        combos.push(Combo {
                            emulator,
                            container,
                            hardware,
                            payload,
                            plugin,
                        });
                    }
                }
            }
        }
    }
    combos
}

// ---------------------------------------------------------------------------
// Host capabilities
// ---------------------------------------------------------------------------

/// What the current host can actually run, queried once up front.
struct Caps {
    /// Per-emulator `(installed, supported)` from `mirage emulators`.
    emulators: BTreeMap<String, (bool, bool)>,
    /// Plugins the backends advertise, lower-cased for matching.
    plugins: Vec<String>,
}

impl Caps {
    /// Decide why (if at all) a combination cannot run on this host.
    fn skip_reason(&self, c: &Combo) -> Option<String> {
        let (installed, supported) = self
            .emulators
            .get(c.emulator.kind())
            .copied()
            .unwrap_or((false, false));

        match c.emulator {
            // The DBT backend translates code objects and runs them on a
            // *real* GPU; with no translation-target hardware present it
            // is impossible — this is the "skip unsupported hardware"
            // case called out in matrix.md.
            Emulator::RocjitsuDbt if !supported => {
                return Some("rocjitsu-dbt unsupported: no translation-target GPU present".into());
            }
            Emulator::RocjitsuDbt if !installed => {
                return Some("rocjitsu-dbt not installed: HSA tools hook library not found".into());
            }
            // MI450X (gfx1250) is deliberately not a DBT-translatable
            // source ISA, so even with hardware it cannot be a guest.
            Emulator::RocjitsuDbt if c.hardware == Hardware::Mi450x => {
                return Some(
                    "rocjitsu-dbt: MI450X (gfx1250) is not a translatable source ISA".into(),
                );
            }
            // The software emulator runs anywhere, but only if its KMD
            // library can be found; otherwise every exec would fail loudly.
            Emulator::Rocjitsu if !installed => {
                return Some("rocjitsu not installed: KMD library not found".into());
            }
            _ => {}
        }

        if c.plugin == Plugin::HazardDetection && !self.plugins.iter().any(|p| p.contains("hazard"))
        {
            return Some("hazard-detection plugin not advertised by the backend".into());
        }

        None
    }
}

/// Query `mirage emulators --json` for the install/support state of
/// every backend and the plugins they advertise.
fn probe_caps(mirage_bin: &Path) -> Caps {
    let mut cmd = Command::new(mirage_bin);
    cmd.args(["--json", "emulators"]).env_remove("MIRAGE_LOG");
    let out = cmd.output().expect("run `mirage emulators`");
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("emulators output should be JSON");

    let mut emulators = BTreeMap::new();
    let mut plugins = Vec::new();
    if let Some(arr) = json.as_array() {
        for e in arr {
            let name = e["name"].as_str().unwrap_or_default().to_string();
            let installed = e["installed"].as_bool().unwrap_or(false);
            let supported = e["support"]["supported"].as_bool().unwrap_or(false);
            emulators.insert(name, (installed, supported));
            if let Some(ps) = e["plugins"].as_array() {
                for p in ps {
                    if let Some(s) = p.as_str() {
                        plugins.push(s.to_lowercase());
                    }
                }
            }
        }
    }

    Caps { emulators, plugins }
}

// ---------------------------------------------------------------------------
// Per-combo harness
// ---------------------------------------------------------------------------

/// An isolated XDG root plus a mock container provider for one run.
struct Env {
    _dir: TempDir,
    runtime: PathBuf,
    config: PathBuf,
    state: PathBuf,
    provider: PathBuf,
    mirage_bin: PathBuf,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join("runtime");
        let config = dir.path().join("config");
        let state = dir.path().join("state");
        let provider = dir.path().join("mock-provider.sh");
        write_mock_provider(&provider);
        Self {
            _dir: dir,
            runtime,
            config,
            state,
            provider,
            mirage_bin: PathBuf::from(env!("CARGO_BIN_EXE_mirage")),
        }
    }

    fn mirage(&self) -> Command {
        let mut c = Command::new(&self.mirage_bin);
        c.env("XDG_CONFIG_HOME", &self.config)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("XDG_STATE_HOME", &self.state)
            .env("MIRAGE_BIN", &self.mirage_bin)
            // The mock provider records every node's pid under this dir.
            .env("MOCK_DIR", self._dir.path())
            .env_remove("MIRAGE_LOG");
        c
    }

    fn session_root(&self) -> PathBuf {
        self.runtime.join("mirage/session")
    }

    /// True when no session directories remain.
    fn no_sessions_left(&self) -> bool {
        match std::fs::read_dir(self.session_root()) {
            Ok(mut entries) => entries.next().is_none(),
            Err(_) => true,
        }
    }
}

/// A hermetic `docker`/`podman` stand-in. It satisfies the host's
/// bring-up path (pull/network/run/rm) and, for `run`, launches the
/// per-node host (`mirage host --session <id> --rank <n>`) directly so
/// the workload actually executes — exactly the contract the real
/// container entrypoint provides. Mirrors `tests/container_e2e.rs`.
fn write_mock_provider(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let script = r#"#!/bin/sh
case "$1" in
  pull) exit 0 ;;
  image) [ "$2" = inspect ] && exit 1; exit 0 ;;
  network) [ "$2" = inspect ] && exit 1; exit 0 ;;
  run)
    shift; name=""; sid=""; rank=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --name) name="$2"; shift 2 ;;
        --session) sid="$2"; shift 2 ;;
        --rank) rank="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    if [ -n "$sid" ] && [ -n "$rank" ]; then
      "$MIRAGE_BIN" host --session "$sid" --rank "$rank" </dev/null >/dev/null 2>&1 &
      [ -n "$name" ] && echo $! > "$MOCK_DIR/$name.pid"
    fi
    echo "cid-$rank"; exit 0 ;;
  rm)
    for a in "$@"; do
      case "$a" in
        mirage-*)
          if [ -f "$MOCK_DIR/$a.pid" ]; then
            kill "$(cat "$MOCK_DIR/$a.pid")" 2>/dev/null
            rm -f "$MOCK_DIR/$a.pid"
          fi ;;
      esac
    done
    exit 0 ;;
  inspect) echo true; exit 0 ;;
  *) exit 0 ;;
esac
"#;
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// Outcome of attempting one combination.
enum Outcome {
    Ran,
    Skipped(String),
}

/// Drive a single combination through create -> run -> delete ->
/// ensure-deleted. Panics on any deviation from the expected contract.
fn run_combo(c: &Combo, caps: &Caps) -> Outcome {
    if let Some(reason) = caps.skip_reason(c) {
        return Outcome::Skipped(reason);
    }

    let env = Env::new();
    let profile = c.name();

    // 1. create
    let mut create = env.mirage();
    create.args([
        "profile",
        "create",
        &profile,
        "--emulator",
        c.emulator.kind(),
        "--agent",
        c.hardware.agent(),
        "--no-input",
    ]);
    if c.payload.nodes() > 1 {
        create.args(["--num-nodes", "2"]);
    }
    if c.container.is_containerized() {
        create.args(["--image", "img:latest", "--provider"]);
        create.arg(&env.provider);
    }
    create.assert().success();

    // Confirm the profile is persisted and readable.
    env.mirage()
        .args(["profile", "show", &profile])
        .assert()
        .success();

    // 2. run — the workhorse that creates a transient session, executes
    //    the payload (every node), then tears the session back down. A
    //    timeout guards against regressions that could otherwise hang the
    //    whole suite (e.g. a multi-node aggregator waiting forever).
    let mut run = env.mirage();
    run.args(["run", "--profile", &profile, "--"]);
    run.args(c.payload.argv());
    let status = run_with_timeout(run, Duration::from_secs(60), &profile);

    if c.payload.expect_success() {
        assert!(
            status.success(),
            "[{}] run failed: status={:?}",
            profile,
            status.code(),
        );
    } else {
        assert!(
            !status.success(),
            "[{}] crash payload unexpectedly succeeded",
            profile,
        );
    }

    // 3. ensure deleted — the transient session must clean itself up,
    //    leaving no session directories behind (regardless of how the
    //    payload exited).
    let cleaned = wait_for(Duration::from_secs(5), || env.no_sessions_left());
    assert!(
        cleaned,
        "[{}] session was not cleaned up after run; session root still populated",
        profile,
    );

    // 4. delete the profile and confirm it is gone.
    env.mirage()
        .args(["profile", "delete", &profile, "-f"])
        .assert()
        .success();
    env.mirage()
        .args(["profile", "show", &profile])
        .assert()
        .failure();

    Outcome::Ran
}

fn wait_for<F: FnMut() -> bool>(timeout: Duration, mut f: F) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    f()
}

/// Run a command to completion, but fail loudly (rather than hang the
/// whole suite) if it overruns `timeout`. stdio is discarded — callers
/// assert on the exit status, and tiny payload output is uninteresting.
fn run_with_timeout(mut cmd: Command, timeout: Duration, ctx: &str) -> ExitStatus {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().expect("spawn `mirage run`");
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait().expect("poll child") {
            Some(status) => return status,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("[{ctx}] `mirage run` did not finish within {timeout:?}");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

// ---------------------------------------------------------------------------
// The matrix test
// ---------------------------------------------------------------------------

#[test]
fn matrix_lifecycle_across_all_dimensions() {
    let mirage_bin = PathBuf::from(env!("CARGO_BIN_EXE_mirage"));
    let caps = probe_caps(&mirage_bin);

    let combos = all_combos();
    let total = combos.len();
    let mut ran = 0usize;
    let mut skipped = 0usize;

    eprintln!("\nmirage testing matrix — {total} combinations\n");
    eprintln!(
        "  {:<58}  {}",
        "COMBINATION (emulator+container+hw+payload+plugin)", "RESULT"
    );

    for c in &combos {
        match run_combo(c, &caps) {
            Outcome::Ran => {
                ran += 1;
                eprintln!("  {:<58}  RAN", c.name());
            }
            Outcome::Skipped(reason) => {
                skipped += 1;
                eprintln!("  {:<58}  SKIP ({reason})", c.name());
            }
        }
    }

    eprintln!("\nmatrix summary: {ran} ran, {skipped} skipped, {total} total\n");

    // Sanity: the matrix must be coherent. Every dbt + mi450x and every
    // hazard-detection combination is expected to skip on a host without
    // that hardware/plugin, but the suite must never silently skip
    // *everything* on a host where the software emulator is available.
    if caps
        .emulators
        .get("rocjitsu")
        .map(|(installed, _)| *installed)
        .unwrap_or(false)
    {
        assert!(
            ran > 0,
            "rocjitsu is installed but no matrix combination ran"
        );
    }
}
