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
//! A combination that *fails* is recorded and the matrix carries on. The
//! point of a matrix is to say which dimensions are broken, and stopping
//! at the first failure answers that question for one cell and hides it
//! for the other seventy-one; the whole table is printed either way, and
//! the test fails at the end with the list.
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
//!
//! # What a green matrix does and does not prove
//!
//! The payloads are shell commands, so no cell here executes GPU code.
//! Two of the five dimensions would therefore be free rides if nothing
//! were done about it: `MI350X` and `MI450X` run the same `echo`, and so
//! do `--plugin race` and no plugin, which means those rows could pass
//! unchanged if the flags were dropped on the floor. Each combination
//! consequently prints the emulator configuration mirage synthesised for
//! it, and [`assert_dimensions_reached_the_emulator`] insists the GPU
//! and the plugin selection are actually in it.
//!
//! That makes the two dimensions falsifiable at the layer this suite is
//! about — the CLI wiring a profile through to the backend — and no
//! further. It is **not** evidence that the emulator then models that
//! GPU correctly, or that the race detector detects anything; both need
//! a real GPU workload and belong to the emulator's own tests. The
//! containerised rows cannot check even this much, because the
//! configuration is remapped to an in-container path the mock provider
//! does not create; for those rows the hardware and plugin columns are
//! honestly just "mirage accepted the flag and the run completed".

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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

    /// The field of the synthesised emulator configuration that this
    /// agent, and only this agent, produces.
    ///
    /// `gfx_target_version` rather than the marketing name: it is a
    /// number, so it cannot be a substring of some other value in the
    /// document, and it is the field the emulated device's ISA is
    /// actually chosen by.
    fn config_marker(self) -> &'static str {
        match self {
            Hardware::Mi350x => r#""gfx_target_version":90500"#,
            Hardware::Mi450x => r#""gfx_target_version":120500"#,
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
    fn argv(self) -> Vec<String> {
        let body = match self {
            Payload::TinyTorch => "echo tiny_torch_ok",
            // Each rank prints once; with two nodes the orchestrator runs
            // the command on both.
            Payload::Rccl => "echo rccl_ok",
            // Simulate a crashing workload: emit output, then exit with a
            // SIGSEGV-style code. mirage must still tear the session down.
            Payload::Crash => "echo crashing; exit 139",
        };
        vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            format!("{DUMP_EMULATOR_CONFIG}{body}"),
        ]
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

/// The prefix every payload runs before its own body.
///
/// It prints the emulator configuration mirage synthesised for this
/// session on one line, tagged so [`config_line`] can pick it out of the
/// payload's own output. That document is where the `hardware` and
/// `plugins` dimensions land — nothing else about a `/bin/sh` payload
/// varies with either — so printing it is what stops those two columns
/// of the matrix from being decoration.
///
/// Whitespace is stripped so the whole document is one line and the
/// markers below can be written without worrying about how the JSON was
/// indented. Silent when the file is unreadable: inside a node container
/// the path is remapped to a mount the mock provider never creates, and
/// [`run_combo`] decides whether that silence is acceptable for the row.
const DUMP_EMULATOR_CONFIG: &str = concat!(
    r#"__cfg="$ROCJITSU_RUNTIME_DIR/../rj_config.json"; "#,
    r#"[ -r "$__cfg" ] && printf 'rj_config=%s\n' "$(tr -d '[:space:]' < "$__cfg")"; "#,
);

/// The tag [`DUMP_EMULATOR_CONFIG`] prints the configuration under.
const CONFIG_TAG: &str = "rj_config=";

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

    /// Check `cfg` — the emulator configuration, whitespace stripped —
    /// against this plugin selection.
    ///
    /// Both directions, deliberately. Asserting only that `race` appears
    /// when it was asked for would leave `--plugin` free to enable
    /// everything always, and the `none` rows would still pass; and a
    /// backend that quietly ignored `--plugin` would be caught by
    /// neither half on its own.
    fn assert_selected(self, cfg: &str, profile: &str) {
        // Quoted, because the bare word is a substring of `trace`.
        let race = r#""race""#;
        match self {
            Plugin::None => assert!(
                !cfg.contains("\"plugins\""),
                "[{profile}] no plugin was selected, but the emulator \
                 configuration has a plugins section:\n{cfg}"
            ),
            Plugin::Race => assert!(
                cfg.contains(race),
                "[{profile}] `--plugin race` did not reach the emulator \
                 configuration:\n{cfg}"
            ),
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

/// The cross product of the remaining dimensions for one
/// emulator/hosting slice of the matrix.
///
/// Sliced rather than enumerated whole because each cell is a real
/// `mirage run` — a bring-up, a workload and a teardown — and seventy-two
/// of them inside one `#[test]` is seventy-two of them in sequence. The
/// two outer dimensions are the ones worth cutting on: they are what a
/// cell's *cost* varies with (a containerised cell shells out to a
/// provider several times per node) and they are how a failure is
/// usually described ("docker is broken", "dbt is broken"), so a slice
/// that fails names something a person would say.
fn combos_in(emulator: Emulator, container: Container) -> Vec<Combo> {
    let mut combos = Vec::new();
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
    /// The combination ran and broke its contract, with the assertion
    /// that caught it.
    Failed(String),
}

/// Drive a single combination through create -> run -> ensure nothing
/// survived -> delete. Panics on any deviation from the expected
/// contract, which [`attempt`] turns into one row of the table rather
/// than the end of the suite.
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
    let payload = c.payload.argv();
    let payload: Vec<&str> = payload.iter().map(String::as_str).collect();
    let mut run = env.base.spawn_run(&args, &payload);
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

    // 2b. the two dimensions the payload itself cannot distinguish.
    assert_dimensions_reached_the_emulator(c, &stdout, &profile);

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

/// The emulator configuration the payload printed, if it could see it.
fn config_line(stdout: &str) -> Option<&str> {
    stdout
        .lines()
        // The line is `[rank] rj_config=…` on a labelled multi-node run,
        // so match on the tag anywhere rather than at the start.
        .find_map(|line| line.split_once(CONFIG_TAG).map(|(_, cfg)| cfg))
}

/// Assert the `hardware` and `plugins` dimensions actually changed
/// something.
///
/// These two are the matrix's weak point: the payload is a shell
/// command, so `MI350X` and `MI450X` run identical code and so do
/// `--plugin race` and no plugin. Without a check like this one, four of
/// every eight rows are the same run reported four times, and deleting
/// the code behind either flag would leave the matrix entirely green.
///
/// What is checked is the emulator configuration mirage synthesised —
/// the last artefact the CLI produces before the backend takes over, and
/// the place both dimensions land. It proves the flag was carried
/// through; it does not prove the emulator honours it, which needs GPU
/// code and belongs to the emulator's own suite.
fn assert_dimensions_reached_the_emulator(c: &Combo, stdout: &str, profile: &str) {
    let Some(cfg) = config_line(stdout) else {
        // Two rows legitimately cannot show it, and being precise about
        // which is the point: anything else is a payload that silently
        // stopped reporting, and the dimensions would go back to being
        // unfalsifiable without anyone noticing.
        assert!(
            c.container.is_containerized() || c.emulator != Emulator::Rocjitsu,
            "[{profile}] the payload could not read the emulator configuration, \
             so nothing here checks the hardware or plugin dimension\nstdout: {stdout}"
        );
        return;
    };
    assert!(
        cfg.contains(c.hardware.config_marker()),
        "[{profile}] the emulator configuration does not describe {}: expected \
         {} in it\n{cfg}",
        c.hardware.agent(),
        c.hardware.config_marker(),
    );
    c.plugin.assert_selected(cfg, profile);
}

/// Run one combination, turning whatever it does into a value.
///
/// [`run_combo`] asserts inline, which is what keeps it readable — but a
/// panic escaping into the loop would abandon every combination after it
/// and truncate the table this suite exists to print. A matrix whose
/// whole job is saying *which* dimensions are broken must not stop at the
/// first one, so each cell's failure is caught here and reported with the
/// rest at the end.
fn attempt(c: &Combo, caps: &Caps) -> Outcome {
    match panic_site::capturing(|| run_combo(c, caps)) {
        Ok(outcome) => outcome,
        Err(why) => Outcome::Failed(why),
    }
}

/// The message a caught panic carried.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panicked with a payload that is not a message".to_string()
    }
}

/// Where the last panic came from.
///
/// The unwind payload carries the assertion's message but not its
/// location, and the location is half of what makes a failed cell
/// actionable. It is only available to a panic hook, and a panic hook is
/// process-wide while the slices below are separate `#[test]`s the
/// harness runs in parallel — so the hook is installed once for the whole
/// binary and everything it remembers is thread-local. A slice that swaps
/// the hook in and out around its own combinations would be swapping it
/// out from under another slice's.
///
/// It is also only *capturing* while a combination is running. Outside
/// that window it hands the panic to the hook it displaced, so a genuine
/// failure of one of these tests is still reported the ordinary way, with
/// its message and its backtrace.
mod panic_site {
    use std::cell::{Cell, RefCell};

    thread_local! {
        /// The site of the most recent captured panic on this thread.
        static SITE: RefCell<Option<String>> = const { RefCell::new(None) };
        /// Whether this thread is inside a combination.
        static CAPTURING: Cell<bool> = const { Cell::new(false) };
    }

    /// Install the process-wide hook, once.
    fn install() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if !CAPTURING.with(Cell::get) {
                    previous(info);
                    return;
                }
                if let Some(location) = info.location() {
                    let site = format!("{}:{}", location.file(), location.line());
                    SITE.with(|slot| *slot.borrow_mut() = Some(site));
                }
            }));
        });
    }

    /// Run `body`, capturing a panic's message and the line it came from.
    ///
    /// Nothing is printed while capturing: the default hook's output
    /// would interleave with the table, and every message it swallows is
    /// reported by the caller anyway.
    pub(super) fn capturing<T>(body: impl FnOnce() -> T) -> Result<T, String> {
        install();
        SITE.with(|slot| *slot.borrow_mut() = None);
        CAPTURING.with(|c| c.set(true));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        CAPTURING.with(|c| c.set(false));
        outcome.map_err(|payload| {
            let message = super::panic_message(&*payload);
            match SITE.with(|slot| slot.borrow_mut().take()) {
                Some(site) => format!("{message} ({site})"),
                None => message,
            }
        })
    }
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
// The matrix, one slice per test
// ---------------------------------------------------------------------------
//
// The whole cross product used to be a single `#[test]`, which meant
// seventy-two real bring-ups in sequence: 94.8 seconds of a 146-second
// suite, on one core, while the rest of the machine sat idle. The
// combinations are independent — each one has its own XDG root, its own
// profile and its own mock provider — so the only thing making them
// sequential was the function they shared.
//
// One test per emulator/hosting slice, then, and the test harness
// parallelises them. The slice is also the useful unit of failure: "the
// docker rows are broken" is a sentence, where "cell 43 of 72 is broken"
// is a lookup. Each slice prints its own table and fails with its own
// list, so a red run still says which dimensions are red.

/// What the host can run, probed once for the whole binary.
///
/// [`probe_caps`] shells out to the binary, and asking it per slice would
/// add a process spawn — and a temporary XDG root — to every one of them
/// for an answer that cannot change while the suite runs.
fn caps() -> &'static Caps {
    static CAPS: std::sync::OnceLock<Caps> = std::sync::OnceLock::new();
    CAPS.get_or_init(probe_caps)
}

/// Drive every combination in one slice and fail with the whole table.
///
/// `expect_a_run` is the guard against a slice that quietly does nothing:
/// a skip is a legitimate outcome for a cell, but a slice in which
/// *every* cell skipped on a host that has the emulator is a suite
/// reporting success for work it never did.
fn run_slice(emulator: Emulator, container: Container, expect_a_run: bool) {
    if skip_without_emulator() {
        return;
    }
    let caps = caps();
    let combos = combos_in(emulator, container);
    let total = combos.len();
    let slice = format!("{}+{}", emulator.kind(), container.label());
    let mut ran = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();

    eprintln!("\n[{slice}] {total} combinations");
    for c in &combos {
        // Whole lines, and each one naming its slice. These tests run in
        // parallel, so a row printed as a prefix now and a result later
        // would be split down the middle by another slice's row under
        // `--nocapture`.
        //
        // The name still goes out *before* the combination runs: a cell
        // that hangs takes the suite with it — the run timeout bounds a
        // run, not a deadlock in the harness — and a table of completed
        // rows only would name every combination except the one that
        // wedged.
        eprintln!("[{slice}] .. {}", c.name());
        match attempt(c, caps) {
            Outcome::Ran => {
                ran += 1;
                eprintln!("[{slice}] RAN  {}", c.name());
            }
            Outcome::Skipped(reason) => {
                skipped += 1;
                eprintln!("[{slice}] SKIP {} ({reason})", c.name());
            }
            Outcome::Failed(why) => {
                // One line here, because the table is meant to be
                // scannable; the whole message is reprinted below.
                eprintln!(
                    "[{slice}] FAIL {} ({})",
                    c.name(),
                    why.lines().next().unwrap_or_default()
                );
                failures.push((c.name(), why));
            }
        }
    }

    eprintln!(
        "[{slice}] summary: {ran} ran, {} failed, {skipped} skipped, {total} total",
        failures.len()
    );

    if !failures.is_empty() {
        for (name, why) in &failures {
            eprintln!("--- {name}\n{why}\n");
        }
        panic!(
            "{} of {total} {slice} combinations failed: {}",
            failures.len(),
            failures
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    if expect_a_run {
        assert!(
            ran > 0,
            "{slice} is runnable on this host but no combination in it ran"
        );
    }
}

/// Whether a slice on this emulator must produce at least one run.
///
/// `rocjitsu` is a software emulator: if the binary reports it installed
/// there is no further excuse for a cell to skip, whatever the hosting.
/// `rocjitsu-dbt` translates onto a real GPU and every one of its cells
/// legitimately skips on a host without one, so demanding a run there
/// would fail the suite on every laptop.
fn must_run(emulator: Emulator) -> bool {
    emulator == Emulator::Rocjitsu
        && caps()
            .emulators
            .get(emulator.kind())
            .is_some_and(|(installed, _)| *installed)
}

#[test]
fn matrix_rocjitsu_on_the_node() {
    run_slice(
        Emulator::Rocjitsu,
        Container::Node,
        must_run(Emulator::Rocjitsu),
    );
}

#[test]
fn matrix_rocjitsu_under_podman() {
    run_slice(
        Emulator::Rocjitsu,
        Container::Podman,
        must_run(Emulator::Rocjitsu),
    );
}

#[test]
fn matrix_rocjitsu_under_docker() {
    run_slice(
        Emulator::Rocjitsu,
        Container::Docker,
        must_run(Emulator::Rocjitsu),
    );
}

#[test]
fn matrix_rocjitsu_dbt_on_the_node() {
    run_slice(Emulator::RocjitsuDbt, Container::Node, false);
}

#[test]
fn matrix_rocjitsu_dbt_under_podman() {
    run_slice(Emulator::RocjitsuDbt, Container::Podman, false);
}

#[test]
fn matrix_rocjitsu_dbt_under_docker() {
    run_slice(Emulator::RocjitsuDbt, Container::Docker, false);
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the matrix going green while every slice in it
    // skipped for a missing emulator runtime. See `assert_suite_can_run`.
    assert_suite_can_run();
}
