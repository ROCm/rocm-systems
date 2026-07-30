//! Matrix-driven end-to-end tests for the `mirage` CLI.
//!
//! This test enumerates the full cross product of the dimensions
//! described in [`tests/matrix.md`] and drives each runnable
//! combination through the canonical lifecycle:
//!
//! ```text
//! create  ->  run  ->  ensure nothing survived the run  ->  delete
//! ```
//!
//! Every combination is exercised end-to-end against the real `mirage`
//! binary under an isolated XDG root. Combinations that the current
//! machine *cannot* run are deliberately **skipped** with a recorded
//! reason rather than failed, so the same suite is meaningful on a
//! laptop, in CI, and on an emulation host:
//!
//! * `rocjitsu-dbt` is skipped unless a translation-target GPU is
//!   physically present (DBT runs translated code on real hardware).
//! * `rocjitsu` is skipped when its KMD library cannot be located.
//! * the `race` plugin is skipped when the selected backend does not
//!   advertise it.
//!
//! # The run *is* the session
//!
//! There is no daemon to start and no session to delete. `mirage run`
//! holds its session in its own process: the socket other terminals find
//! it by, the containers, the emulator and the workload all exist exactly
//! as long as that one command does. So "ensure deleted" is not a
//! separate step asking a server whether it forgot anything — it is the
//! claim that when the run process is gone, its socket, its scratch
//! directory and its containers are gone too, for every combination and
//! whether the payload exited cleanly or crashed.
//!
//! The containerised dimensions (`podman`, `docker`) are driven through
//! a hermetic mock provider — a small shell script standing in for the
//! container CLI — so the provider bring-up/teardown contract is
//! exercised without requiring a real image or container engine,
//! mirroring `tests/container_e2e.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod harness;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use harness::{Env as BaseEnv, assert_suite_can_run, skip_without_emulator};

/// How long one combination's `mirage run` gets to finish.
///
/// Generous, because it covers session bring-up (which has its own
/// 60-second budget inside mirage) as well as the payload. It exists to
/// turn a regression that would hang the whole suite — a multi-node
/// aggregator waiting forever, a container that never reports running —
/// into a failure that names the combination.
const RUN_TIMEOUT: Duration = Duration::from_secs(90);

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
    /// lifecycle* (bring up, run, clean up) across the matrix, not to
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

    /// What the payload prints on stdout, once per rank.
    ///
    /// Asserting on it is what separates "the workload ran" from "mirage
    /// exited with the status I expected". Without a sentinel the crash
    /// row would pass just as happily on a session that never came up —
    /// a failed bring-up also exits non-zero.
    fn sentinel(self) -> &'static str {
        match self {
            Payload::TinyTorch => "tiny_torch_ok",
            Payload::Rccl => "rccl_ok",
            Payload::Crash => "crashing",
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
    Race,
}

impl Plugin {
    fn label(self) -> &'static str {
        match self {
            Plugin::None => "none",
            Plugin::Race => "race",
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
                    for plugin in [Plugin::None, Plugin::Race] {
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
    /// Plugins each backend advertises, lower-cased for matching.
    plugins: BTreeMap<String, Vec<String>>,
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

        if c.plugin == Plugin::Race
            && !self
                .plugins
                .get(c.emulator.kind())
                .is_some_and(|plugins| plugins.iter().any(|plugin| plugin == "race"))
        {
            return Some(format!(
                "race plugin not advertised by {}",
                c.emulator.kind()
            ));
        }

        None
    }
}

/// Query `mirage emulators --json` for the install/support state of
/// every backend and the plugins they advertise.
fn probe_caps() -> Caps {
    // Isolated, like every other invocation in this suite. `emulators` is
    // answered in-process and starts nothing, but a test must not read or
    // write the developer's real mirage directories regardless.
    let probe = BaseEnv::new();
    let out = probe.run(&["--json", "emulators"]);
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("emulators output should be JSON");

    let mut emulators = BTreeMap::new();
    let mut plugins = BTreeMap::new();
    if let Some(arr) = json.as_array() {
        for e in arr {
            let name = e["name"].as_str().unwrap_or_default().to_string();
            let installed = e["installed"].as_bool().unwrap_or(false);
            let supported = e["support"]["supported"].as_bool().unwrap_or(false);
            let emulator_plugins = e["plugins"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|plugin| plugin.as_str().map(str::to_lowercase))
                .collect();
            plugins.insert(name.clone(), emulator_plugins);
            emulators.insert(name, (installed, supported));
        }
    }

    Caps { emulators, plugins }
}

// ---------------------------------------------------------------------------
// Per-combo harness
// ---------------------------------------------------------------------------

/// An isolated mirage installation plus a mock container provider for one
/// combination.
struct Env {
    base: BaseEnv,
    provider: PathBuf,
}

impl Env {
    /// Build an environment whose mock provider is named after `label`.
    ///
    /// The name matters: mirage decides how to pass GPU groups through to
    /// a container by looking at the provider's basename (podman takes
    /// `--group-add keep-groups`, docker does not). A single
    /// `mock-provider.sh` would make the `podman` and `docker` rows of
    /// the matrix byte-for-byte identical runs.
    fn new(label: &str) -> Self {
        let base = BaseEnv::new();
        let provider = base.root().join(format!("mock-{label}.sh"));
        write_mock_provider(&provider);
        Self { base, provider }
    }
}

/// A hermetic `docker`/`podman` stand-in. It satisfies bring-up
/// (pull/network/run/inspect) and executes `exec` invocations locally,
/// which is where workloads now arrive: a node container's own process
/// just idles. Mirrors `tests/container_e2e.rs`, which asserts on the
/// argv in detail; here the mock only has to behave.
fn write_mock_provider(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let script = r#"#!/bin/sh
case "$1" in
  pull) exit 0 ;;
  image) [ "$2" = inspect ] && exit 1; exit 0 ;;
  network)
    # `network inspect --format` is the ownership check teardown makes;
    # plain `network inspect` is the existence probe.
    if [ "$2" = inspect ]; then
      [ "$3" = "--format" ] && { printf mirage; exit 0; }
      exit 1
    fi
    exit 0 ;;
  run)
    # Deliberately does not return. A node container is no longer started
    # detached: the provider client is a child mirage owns, and killing it
    # is what stops the container. Exiting here would make every teardown
    # in this suite look successful for the wrong reason — there would be
    # nothing left to kill.
    exec sleep 300 ;;
  exec)
    shift
    envs=""
    workdir=""
    while [ $# -gt 0 ]; do
      case "$1" in
        -i|-t|-it) shift ;;
        -w) workdir="$2"; shift 2 ;;
        -e) envs="$envs $2"; shift 2 ;;
        *) break ;;
      esac
    done
    shift
    # Fail like a real provider does. `podman exec -w` on a directory
    # that does not exist inside the container aborts the exec; swallowing
    # it here would let mirage pass a *host* path as the container
    # workdir and still look correct in these tests.
    if [ -n "$workdir" ]; then
      cd "$workdir" || { echo "chdir to '$workdir': no such directory" >&2; exit 126; }
    fi
    if [ -n "$envs" ]; then
      exec env $envs "$@"
    fi
    exec "$@" ;;
  rm) exit 0 ;;
  inspect)
    # Either the ownership check (a Go template naming mirage.owner), or
    # the `-f {{.State.Running}}` liveness probe bring-up waits on before
    # it runs anything in the container.
    case "$2" in
      --format|-f)
        case "$3" in
          *mirage.owner*) printf mirage ;;
          *) echo true ;;
        esac
        exit 0 ;;
    esac
    echo true; exit 0 ;;
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

/// Drive a single combination through create -> run -> ensure nothing
/// survived -> delete. Panics on any deviation from the expected
/// contract.
fn run_combo(c: &Combo, caps: &Caps) -> Outcome {
    if let Some(reason) = caps.skip_reason(c) {
        return Outcome::Skipped(reason);
    }

    let env = Env::new(c.container.label());
    let profile = c.name();
    let provider = env.provider.to_string_lossy().into_owned();

    // 1. create
    let mut create = vec![
        "profile",
        "create",
        &profile,
        "--emulator",
        c.emulator.kind(),
        "--agent",
        c.hardware.agent(),
        "--no-input",
    ];
    if c.payload.nodes() > 1 {
        create.extend(["--num-nodes", "2"]);
    }
    if c.container.is_containerized() {
        create.extend(["--image", "img:latest", "--container-provider", &provider]);
    }
    env.base.ok(&create);

    // Confirm the profile is persisted and readable.
    env.base.ok(&["profile", "show", &profile]);

    // 2. run — the workhorse. It brings the session up in its own
    //    process, runs the payload on every node, and tears everything
    //    back down on the way out.
    let mut args = vec!["--profile", &profile];
    if c.plugin == Plugin::Race {
        args.extend(["--plugin", "race"]);
    }
    let mut run = env.base.spawn_run(&args, &c.payload.argv());
    let out = run.wait(RUN_TIMEOUT);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    if c.payload.expect_success() {
        assert!(
            out.status.success(),
            "[{profile}] run failed: status={:?}\nstdout: {stdout}\nstderr: {stderr}",
            out.status.code(),
        );
    } else {
        assert!(
            !out.status.success(),
            "[{profile}] crash payload unexpectedly succeeded\nstdout: {stdout}"
        );
    }

    // The payload's own output reaches this terminal because every rank
    // inherits it: there is no pseudo-terminal and no forwarding, the
    // workload's stdout *is* the run's stdout. Counting the sentinel also
    // proves the command ran on every node rather than only the first.
    let seen = stdout.matches(c.payload.sentinel()).count();
    assert_eq!(
        seen,
        c.payload.nodes() as usize,
        "[{profile}] expected {:?} once per node ({} node(s)), saw it {seen} time(s)\nstdout: {stdout}",
        c.payload.sentinel(),
        c.payload.nodes(),
    );

    // 3. ensure nothing survived the run. A session exists exactly while
    //    the `mirage run` that created it does, so once the process has
    //    exited there must be no socket for another terminal to find and
    //    no scratch directory left on disk — however the payload exited.
    let session = session_id(&stderr).unwrap_or_else(|| {
        panic!("[{profile}] the run never announced its session id\nstderr: {stderr}")
    });
    assert!(
        env.base.live_runs().is_empty(),
        "[{profile}] a run socket outlived the run: {:?}",
        env.base.live_runs()
    );
    let scratch = env.base.session_scratch(&session);
    assert!(
        !scratch.exists(),
        "[{profile}] session scratch survived the run: {}",
        scratch.display()
    );

    // 4. delete the profile and confirm it is gone. Profiles are the one
    //    thing a run does *not* own: they are config, and outlive it.
    env.base.ok(&["profile", "delete", &profile, "-f"]);
    env.base.fails(&["profile", "show", &profile]);

    Outcome::Ran
}

/// The session id a run announces on stderr as it starts.
///
/// This is how a user finds the session to `mirage exec` into from
/// another terminal, so it is worth insisting the line is there.
fn session_id(stderr: &str) -> Option<String> {
    stderr
        .lines()
        .find_map(|line| line.strip_prefix("mirage: session "))
        .map(|id| id.trim().to_string())
}

// ---------------------------------------------------------------------------
// The matrix test
// ---------------------------------------------------------------------------

#[test]
fn matrix_lifecycle_across_all_dimensions() {
    assert_suite_can_run();
    if skip_without_emulator() {
        return;
    }
    let caps = probe_caps();

    let combos = all_combos();
    let total = combos.len();
    let mut ran = 0usize;
    let mut skipped = 0usize;

    eprintln!("\nmirage testing matrix — {total} combinations\n");
    eprintln!(
        "  {:<58}  RESULT",
        "COMBINATION (emulator+container+hw+payload+plugin)"
    );

    for c in &combos {
        // The name goes out *before* the combination runs. A combination
        // that hangs or panics takes the whole suite with it, and a table
        // that only prints completed rows would name every combination
        // except the one that failed.
        eprint!("  {:<58}  ", c.name());
        match run_combo(c, &caps) {
            Outcome::Ran => {
                ran += 1;
                eprintln!("RAN");
            }
            Outcome::Skipped(reason) => {
                skipped += 1;
                eprintln!("SKIP ({reason})");
            }
        }
    }

    eprintln!("\nmatrix summary: {ran} ran, {skipped} skipped, {total} total\n");

    // Sanity: the matrix must be coherent. Every dbt + mi450x and every
    // race-plugin combinations are expected to skip on a host without
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
