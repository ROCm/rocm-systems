//! End-to-end tests for the "a run owns its session" model.
//!
//! These drive the real [`Run`], the real control socket and the real
//! process supervisor. What they do *not* need is a GPU emulator: a stub
//! backend registers itself into the emulator registry the same way
//! `rocjitsu` and `hotswap` do, so the whole path from `Run::start` to a
//! reaped process is exercised on any machine.
//!
//! That matters because the properties under test are ownership
//! properties, not emulation ones: that a session cannot outlive the
//! process holding it, that an exec started from a description built in
//! one process behaves identically to one built in another, and that
//! nothing is left running when a run goes away.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use mirage_core::common::MaybeRef;
use mirage_core::config::OptionDef;
use mirage_core::emulator::{
    EmulatorBackend, EmulatorBackendDef, EmulatorDef, EmulatorDescription, SupportStatus,
};
use mirage_core::exec::{ExecArgs, ExecDef, ExecId, InjectionDef};
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::ProfileDef;
use mirage_core::session::{CreateSessionRequest, SessionContext, SessionHealth};
use mirage_supervisor::Run;

// ---------------------------------------------------------------------
// A stub emulator backend
// ---------------------------------------------------------------------

/// An emulator that emulates nothing.
///
/// It reports itself installed and supported and injects one marker
/// variable, which is enough for a session to reach `ready` and for a
/// test to prove the injection reached the workload's environment.
#[derive(Debug)]
struct Stub;

/// The marker the stub injects, asserted on by the tests below.
const STUB_ENV: &str = "MIRAGE_STUB_EMULATOR";

impl EmulatorBackend for Stub {
    fn description(&self) -> EmulatorDescription {
        EmulatorDescription {
            name: "stub".to_string(),
            version: "0".to_string(),
            description: "test-only emulator that emulates nothing".to_string(),
            options_schema: Vec::new(),
        }
    }

    fn boot(&self, _def: &ProfileDef) -> Result<(), String> {
        Ok(())
    }

    fn options(&self) -> Vec<OptionDef> {
        Vec::new()
    }

    fn shutdown(&self, _ctx: &SessionContext) {}

    fn validate_profile(&self, _def: &ProfileDef) -> Result<(), String> {
        Ok(())
    }

    fn installed(&self) -> bool {
        true
    }

    fn supported(&self) -> SupportStatus {
        SupportStatus::supported("stub emulator needs nothing".to_string())
    }

    fn discover_plugins(&self) -> Vec<PluginsDef> {
        Vec::new()
    }

    fn health(&self, _ctx: &SessionContext) -> SessionHealth {
        SessionHealth::phase(true, "ready", None)
    }

    fn injection_def(&self, _ctx: &SessionContext) -> mirage_core::error::Result<InjectionDef> {
        Ok(InjectionDef {
            env: BTreeMap::from([(STUB_ENV.to_string(), "1".to_string())]),
            ..Default::default()
        })
    }
}

inventory::submit! {
    EmulatorBackendDef {
        kind: "stub",
        backend: &Stub,
    }
}

// ---------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------

/// Point mirage's XDG roots at a scratch directory for this process.
///
/// Every test in this binary shares it, which is fine: they use distinct
/// session ids and the directories are per-session.
fn isolate() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    static DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    ONCE.call_once(|| {
        let dir = tempfile::tempdir().expect("scratch dir");
        mirage_core::paths::set_test_root(dir.path());
        let _ = DIR.set(dir);
    });
}

fn profile(nodes: u32) -> ProfileDef {
    ProfileDef {
        name: "stub".to_string(),
        description: None,
        emulator: EmulatorDef {
            emulator: "stub".to_string(),
            plugins: PluginsDef::default(),
            exec_mode: mirage_core::emulator::ExecMode::default(),
            options: mirage_core::common::SimpleMap::default(),
            topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
                num_nodes: nodes,
                gpus_per_node: 1,
                agent: MaybeRef::Owned(mirage_core::agent::AgentDef::default()),
            }),
        },
        containerize: None,
    }
}

async fn start(nodes: u32) -> Arc<Run> {
    isolate();
    let run = Arc::new(
        Run::start(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(profile(nodes)),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .expect("run starts"),
    );
    let health = run
        .wait_ready(Duration::from_secs(30))
        .await
        .expect("session becomes ready");
    assert!(health.healthy, "{health:?}");
    run
}

fn def_env(run: &Run, script: &str, capture_all: bool, clear_env: bool) -> ExecDef {
    let mut d = def(run, script, capture_all);
    d.clear_env = clear_env;
    d
}

fn def(run: &Run, script: &str, capture_all: bool) -> ExecDef {
    ExecDef {
        timestamp: chrono::Utc::now(),
        session: run.id().clone(),
        exec: ExecArgs {
            command: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
            env: BTreeMap::new(),
            workdir: None,
        },
        worker_exec: None,
        nproc_per_node: 1,
        capture_all,
        clear_env: false,
    }
}

/// Run `script` in `run` and return its exit code, draining output.
async fn run_to_completion(run: &Run, script: &str) -> i32 {
    let (exec, mut output) = run.exec(&def(run, script, true)).await.expect("exec starts");
    let drain = tokio::spawn(async move { while output.recv().await.is_some() {} });
    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    let _ = drain.await;
    exec.status().exit_code.expect("a finished exec has a code")
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[tokio::test]
async fn a_run_brings_a_session_up_and_runs_a_command() {
    let run = start(1).await;
    assert_eq!(run_to_completion(&run, "exit 0").await, 0);
    run.destroy().await;
}

#[tokio::test]
async fn a_workloads_exit_code_reaches_the_caller() {
    let run = start(1).await;
    assert_eq!(run_to_completion(&run, "exit 23").await, 23);
    run.destroy().await;
}

#[tokio::test]
async fn the_emulator_injection_reaches_the_workload() {
    // The property that matters most: if the injection were missing, the
    // workload would run unemulated on whatever hardware is present and
    // still exit 0. Failing loudly here is the point.
    let run = start(1).await;
    let code = run_to_completion(&run, &format!("test -n \"${STUB_ENV}\"")).await;
    assert_eq!(code, 0, "the emulator's environment must be injected");
    run.destroy().await;
}

#[tokio::test]
async fn every_node_gets_a_process_with_its_own_rank() {
    let run = start(3).await;
    let (exec, mut output) = run
        .exec(&def(&run, "test \"$RANK\" = \"$MIRAGE_RANK\"", true))
        .await
        .expect("exec starts");
    let drain = tokio::spawn(async move { while output.recv().await.is_some() {} });
    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    let _ = drain.await;

    let status = exec.status();
    assert_eq!(status.nodes.len(), 3, "one process per node");
    assert_eq!(status.exit_code, Some(0));
    run.destroy().await;
}

#[tokio::test]
async fn destroying_a_run_reaps_a_long_running_workload() {
    // The whole reason this design exists: nothing may outlive the run.
    let run = start(1).await;
    let (exec, _output) = run
        .exec(&def(&run, "sleep 600", false))
        .await
        .expect("exec starts");
    let pids = exec.live_pids();
    assert_eq!(pids.len(), 1);

    tokio::time::timeout(Duration::from_secs(30), run.destroy())
        .await
        .expect("teardown must not hang");

    assert!(
        !process_alive(pids[0]),
        "pid {} survived the run that owned it",
        pids[0]
    );
}

#[tokio::test]
async fn a_workload_that_forks_has_its_whole_tree_reaped() {
    // Signalling only the direct child would leave the grandchild
    // running, invisible and still holding whatever it had open.
    let run = start(1).await;
    let marker = std::env::temp_dir().join(format!("mirage-tree-{}", std::process::id()));
    let _ = std::fs::remove_file(&marker);
    let script = format!(
        "sh -c 'while true; do echo x >> {}; sleep 0.1; done' & sleep 600",
        marker.display()
    );
    let (exec, _output) = run.exec(&def(&run, &script, false)).await.expect("starts");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(marker.exists(), "the grandchild must actually be running");

    tokio::time::timeout(Duration::from_secs(30), run.destroy())
        .await
        .expect("teardown must not hang");
    let _ = exec;

    let size_after_teardown = std::fs::metadata(&marker).map(|m| m.len()).unwrap_or(0);
    tokio::time::sleep(Duration::from_millis(500)).await;
    let size_later = std::fs::metadata(&marker).map(|m| m.len()).unwrap_or(0);
    let _ = std::fs::remove_file(&marker);
    assert_eq!(
        size_after_teardown, size_later,
        "a grandchild was still writing after its run was destroyed"
    );
}

#[tokio::test]
async fn a_description_lets_another_process_build_the_same_processes() {
    // This is what `mirage exec` does: it never sees the `Session`, only
    // the description, and must produce the same grid from it.
    let run = start(2).await;
    let desc = run.describe().expect("a ready session describes itself");
    assert_eq!(desc.node_count, 2);
    assert_eq!(desc.env.get(STUB_ENV).map(String::as_str), Some("1"));

    let id = ExecId::new("x-1").unwrap();
    let specs =
        mirage_supervisor::build_specs(&desc, &def(&run, "exit 0", true), &id).expect("specs");
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].env.get("RANK").map(String::as_str), Some("0"));
    assert_eq!(specs[1].env.get("RANK").map(String::as_str), Some("1"));
    assert_eq!(specs[0].env.get(STUB_ENV).map(String::as_str), Some("1"));

    run.destroy().await;
}

#[tokio::test]
async fn an_exec_built_from_a_description_runs_like_one_built_by_the_run() {
    // The equivalence `mirage exec` depends on, asserted directly.
    let run = start(1).await;
    let desc = run.describe().unwrap();
    let d = def(&run, &format!("test -n \"${STUB_ENV}\""), true);

    let id = ExecId::new("x-2").unwrap();
    let specs = mirage_supervisor::build_specs(&desc, &d, &id).unwrap();
    let (exec, mut output) = mirage_supervisor::Exec::start(id, d, specs);
    let drain = tokio::spawn(async move { while output.recv().await.is_some() {} });
    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    let _ = drain.await;

    assert_eq!(
        exec.status().exit_code,
        Some(0),
        "a client-built exec must get the same environment as the run's own"
    );
    run.destroy().await;
}

#[tokio::test]
async fn destroy_is_idempotent() {
    let run = start(1).await;
    run.destroy().await;
    tokio::time::timeout(Duration::from_secs(10), run.destroy())
        .await
        .expect("a second teardown must return rather than hang");
}

#[tokio::test]
async fn an_exec_cannot_start_in_a_destroyed_session() {
    // Otherwise a process could be spawned after teardown collected the
    // list of things to kill, and would survive it.
    let run = start(1).await;
    run.destroy().await;
    let started = run.exec(&def(&run, "sleep 600", false)).await;
    assert!(
        started.is_err(),
        "a destroyed session must refuse new work, not start it"
    );
}

/// The environment variables `--clear-env-vars` keeps, mirrored from
/// `mirage_supervisor::process::INHERITED_ENV` (private there).
const KEPT_WHEN_CLEARED: &[&str] = &[
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL",
];

/// A variable this process has that a cleared workload would lose.
///
/// Chosen from the live environment rather than exported by the test:
/// `std::env::set_var` is `unsafe` in Rust 2024 and this workspace
/// forbids `unsafe`, and a process-wide mutation would race the other
/// tests in this binary regardless. The value only has to be something a
/// shell comparison can match, so anything awkward is skipped.
fn ambient_variable() -> Option<(String, String)> {
    std::env::vars().find(|(k, v)| {
        !KEPT_WHEN_CLEARED.contains(&k.as_str())
            && !k.starts_with("MIRAGE_")
            && !k.is_empty()
            && !v.is_empty()
            && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && v.chars().all(|c| c.is_ascii_alphanumeric() || "._-/:".contains(c))
    })
}

#[tokio::test]
async fn the_callers_environment_reaches_the_workload() {
    // The default, and the point of it: mirage's parent is the terminal
    // the user typed in, so a variable exported there — an API token, a
    // PYTHONPATH, a proxy, a framework tuning knob — is one they meant
    // for the workload. Dropping it silently is the failure this guards.
    let Some((key, value)) = ambient_variable() else {
        eprintln!("SKIP: this process has no ambient variable to test with");
        return;
    };
    let run = start(1).await;
    let code = run_to_completion(&run, &format!("test \"${key}\" = \"{value}\"")).await;
    assert_eq!(
        code, 0,
        "{key} was exported in the calling environment and must reach the workload"
    );
    run.destroy().await;
}

#[tokio::test]
async fn clear_env_vars_drops_the_callers_environment() {
    // The opt-out, for a run whose result must not depend on ambient
    // state: a benchmark, a reproduction, a CI job against a baseline.
    let Some((key, _)) = ambient_variable() else {
        eprintln!("SKIP: this process has no ambient variable to test with");
        return;
    };
    let run = start(1).await;
    let (exec, mut output) = run
        .exec(&def_env(&run, &format!("test -z \"${key}\""), true, true))
        .await
        .expect("exec starts");
    let drain = tokio::spawn(async move { while output.recv().await.is_some() {} });
    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    let _ = drain.await;
    assert_eq!(
        exec.status().exit_code,
        Some(0),
        "--clear-env-vars must drop {key}, which the caller exported"
    );
    run.destroy().await;
}

#[tokio::test]
async fn clearing_the_environment_keeps_what_a_process_needs() {
    // An empty environment is not a useful one: without PATH a workload
    // cannot find the commands it runs. The strict form is "almost
    // empty", not "empty".
    let run = start(1).await;
    let (exec, mut output) = run
        .exec(&def_env(&run, "test -n \"$PATH\" && test -n \"$HOME\"", true, true))
        .await
        .expect("exec starts");
    let drain = tokio::spawn(async move { while output.recv().await.is_some() {} });
    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    let _ = drain.await;
    assert_eq!(exec.status().exit_code, Some(0));
    run.destroy().await;
}

#[tokio::test]
async fn the_emulator_injection_survives_clear_env_vars() {
    // The injection is layered on explicitly rather than inherited, so
    // clearing must not reach it — a workload that lost it would run
    // unemulated on whatever hardware is present and still exit 0.
    let run = start(1).await;
    let (exec, mut output) = run
        .exec(&def_env(
            &run,
            &format!("test -n \"${STUB_ENV}\""),
            true,
            true,
        ))
        .await
        .expect("exec starts");
    let drain = tokio::spawn(async move { while output.recv().await.is_some() {} });
    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    let _ = drain.await;
    assert_eq!(exec.status().exit_code, Some(0));
    run.destroy().await;
}

/// Whether `pid` still exists. `kill(pid, 0)` is the standard probe.
fn process_alive(pid: u32) -> bool {
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(pid as i32),
        None,
    )
    .is_ok()
}

#[tokio::test]
async fn capture_all_labels_every_ranks_output_with_its_rank() {
    // The point of the flag: with three nodes writing to one terminal,
    // unlabelled output is unreadable. Assert the labelling on the real
    // chunk stream, which is exactly what the CLI's printer consumes.
    let run = start(3).await;
    let (exec, mut output) = run
        .exec(&def(&run, "echo \"line from $MIRAGE_RANK\"", true))
        .await
        .expect("exec starts");

    let mut per_rank: BTreeMap<u32, String> = BTreeMap::new();
    let collect = tokio::spawn(async move {
        while let Some(chunk) = output.recv().await {
            per_rank
                .entry(chunk.node)
                .or_default()
                .push_str(&String::from_utf8_lossy(&chunk.data));
        }
        per_rank
    });

    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    let per_rank = collect.await.unwrap();

    assert_eq!(per_rank.len(), 3, "every rank's output must be captured");
    for (rank, text) in &per_rank {
        assert!(
            text.contains(&format!("line from {rank}")),
            "rank {rank} produced {text:?}"
        );
    }
    run.destroy().await;
}

#[tokio::test]
async fn without_capture_all_mirage_never_sees_the_output() {
    // The default is that the workload writes to the terminal directly,
    // with mirage out of the way entirely. If anything arrived on this
    // channel, output would be passing through mirage when it should not
    // — which is how stdout/stderr separation and byte-exactness get
    // lost.
    let run = start(1).await;
    let (exec, mut output) = run
        .exec(&def(&run, "echo straight-to-the-terminal", false))
        .await
        .expect("exec starts");
    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    assert!(
        output.recv().await.is_none(),
        "an inheriting exec must pipe nothing through mirage"
    );
    run.destroy().await;
}
