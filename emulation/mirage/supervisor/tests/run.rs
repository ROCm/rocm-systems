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

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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

fn def_env(run: &Run, script: &str, node: Option<u32>, clear_env: bool) -> ExecDef {
    let mut d = def(run, script, node);
    d.clear_env = clear_env;
    d
}

fn def(run: &Run, script: &str, node: Option<u32>) -> ExecDef {
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
        node,
        clear_env: false,
    }
}

/// Run `script` in `run` and return its exit code, draining output.
async fn run_to_completion(run: &Run, script: &str) -> i32 {
    let (exec, mut output) = run
        .exec(&def(run, script, None))
        .await
        .expect("exec starts");
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
        .exec(&def(&run, "test \"$RANK\" = \"$MIRAGE_RANK\"", None))
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
        .exec(&def(&run, "sleep 600", None))
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

/// A process a *workload* forked, killed when the test ends however it
/// ends.
///
/// The process is nobody's child once the workload exits — that is the
/// whole point of the tests below — so nothing reaps it automatically,
/// and a failed assertion used to leave it appending to a file in `/tmp`
/// for as long as the machine stayed up. The lines after a failed
/// `assert!` never run, so cleanup has to hang off `Drop`; this is the
/// shape `mirage_core::reclaim`'s `Tagged` uses, for the same reason.
///
/// It also owns the scratch directory the process writes to, so the files
/// go with it: a struct's own `drop` runs before its fields', which is
/// the order this needs — kill first, then remove what it was writing.
struct Forked {
    dir: tempfile::TempDir,
    /// Pid of the forked process, once it has announced one. Zero before
    /// that, and no reason to signal anything.
    pid: u32,
}

impl Forked {
    /// A scratch directory for a workload that is about to fork.
    fn expected() -> Self {
        Self {
            dir: tempfile::tempdir().expect("scratch dir"),
            pid: 0,
        }
    }

    /// Where the forked process writes its own pid.
    fn pid_file(&self) -> std::path::PathBuf {
        self.dir.path().join("forked.pid")
    }

    /// The file it appends to while it is alive, so a test can watch it
    /// stop.
    fn marker(&self) -> std::path::PathBuf {
        self.dir.path().join("marker")
    }

    /// Wait for the process to announce its pid and start writing.
    async fn started(&mut self) -> u32 {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            if let Ok(text) = std::fs::read_to_string(self.pid_file())
                && let Ok(pid) = text.trim().parse::<u32>()
                && self.marker().exists()
            {
                self.pid = pid;
                return pid;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the workload's forked process never started"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Whether the pid still belongs to the process that was forked,
    /// rather than to an unrelated one the kernel handed the number to
    /// after it died.
    ///
    /// Worth checking because the expected outcome of every test here is
    /// that the process is already gone, and the cleanup is a `SIGKILL`.
    /// The scratch path is unique to this test and appears in the forked
    /// shell's command line, so `/proc` answers it exactly.
    fn is_the_forked_process(&self) -> bool {
        std::fs::read(format!("/proc/{}/cmdline", self.pid)).is_ok_and(|cmdline| {
            String::from_utf8_lossy(&cmdline).contains(&self.marker().display().to_string())
        })
    }
}

impl Drop for Forked {
    fn drop(&mut self) {
        if self.pid == 0 || !self.is_the_forked_process() {
            return;
        }
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(self.pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
}

#[tokio::test]
async fn a_workload_that_forks_has_its_whole_tree_reaped() {
    // Signalling only the direct child would leave the grandchild
    // running, invisible and still holding whatever it had open.
    let run = start(1).await;
    let mut forked = Forked::expected();
    let script = format!(
        "sh -c 'echo $$ > {pid}; while true; do echo x >> {marker}; sleep 0.1; done' & sleep 600",
        pid = forked.pid_file().display(),
        marker = forked.marker().display()
    );
    let (exec, _output) = run.exec(&def(&run, &script, None)).await.expect("starts");
    forked.started().await;

    tokio::time::timeout(Duration::from_secs(30), run.destroy())
        .await
        .expect("teardown must not hang");
    let _ = exec;

    // `expect`, not `unwrap_or(0)`. Defaulting both samples to 0 made the
    // assertion `0 == 0` — satisfied whenever the marker could not be
    // read at all, which no longer distinguishes "the grandchild was
    // reaped" from "the file went away". The file provably existed a few
    // lines above, so a read failure here is a broken test, not a pass.
    let size = |what: &str| {
        std::fs::metadata(forked.marker())
            .unwrap_or_else(|e| panic!("the marker must still be readable {what}: {e}"))
            .len()
    };
    let size_after_teardown = size("right after teardown");
    tokio::time::sleep(Duration::from_millis(500)).await;
    let size_later = size("half a second later");
    assert_eq!(
        size_after_teardown, size_later,
        "a grandchild was still writing after its run was destroyed"
    );
}

#[tokio::test]
async fn a_workload_that_exits_normally_takes_its_forks_with_it() {
    // The same promise on the path nobody was watching. The workload is
    // not cancelled and the run is not destroyed: the command simply
    // finishes, which is what almost every run does. `mirage run -- sh -c
    // "nohup sleep 900 & echo spawned"` exited 0, removed the session's
    // scratch directory, and left the `sleep` running — and `mirage
    // cleanup` then reported it as a stranded process of a session it
    // agreed no longer existed.
    //
    // The workload waits for its fork to announce itself before exiting,
    // so the process provably exists at the moment the exec ends and the
    // assertion cannot pass by racing the fork.
    let run = start(1).await;
    let mut forked = Forked::expected();
    let script = format!(
        "sh -c 'echo $$ > {pid}; while true; do echo x >> {marker}; sleep 0.1; done' & \
         while [ ! -s {pid} ]; do sleep 0.01; done; exit 0",
        pid = forked.pid_file().display(),
        marker = forked.marker().display()
    );
    let (exec, mut output) = run.exec(&def(&run, &script, None)).await.expect("starts");
    let drain = tokio::spawn(async move { while output.recv().await.is_some() {} });
    let pid = forked.started().await;

    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    let _ = drain.await;
    assert_eq!(
        exec.status().exit_code,
        Some(0),
        "the workload exited on its own, normally"
    );

    assert!(
        mirage_supervisor::process::wait_gone(pid, Duration::from_secs(10)).await,
        "pid {pid} was forked by a workload that exited normally and survived it"
    );
    run.destroy().await;
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
        mirage_supervisor::build_specs(&desc, &def(&run, "exit 0", None), &id).expect("specs");
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
    let d = def(&run, &format!("test -n \"${STUB_ENV}\""), None);

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
async fn a_concurrent_destroy_waits_for_the_one_already_running() {
    // `Run::destroy` promises that when it returns, nothing is left. That
    // has to hold for the *second* caller too, and there is always a
    // second caller: bring-up tears the session down itself when it
    // fails, publishing the terminal health first, so `mirage run` reaches
    // its own `destroy` while that teardown is still in flight. Returning
    // early there let the process exit and drop the runtime, cancelling
    // the unfinished teardown and leaking its containers and scratch
    // directory.
    //
    // The workload ignores SIGTERM, so the first teardown is still inside
    // its grace period when the second one starts.
    let run = start(1).await;
    let (_exec, output) = run
        .exec(&def(
            &run,
            "trap '' TERM; while true; do sleep 1; done",
            None,
        ))
        .await
        .expect("exec starts");
    let drain = tokio::spawn(async move {
        let mut output = output;
        while output.recv().await.is_some() {}
    });

    // Give the shell time to install its trap. Until it has, SIGTERM
    // kills it outright and the first teardown finishes instantly — which
    // would make this test pass whatever `teardown` does.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let first = tokio::spawn({
        let run = Arc::clone(&run);
        async move { run.destroy().await }
    });
    // Let the first call claim the teardown before the second arrives.
    tokio::time::sleep(Duration::from_millis(50)).await;

    tokio::time::timeout(Duration::from_secs(30), run.destroy())
        .await
        .expect("a concurrent teardown must return rather than hang");
    assert_eq!(
        run.health().state.as_deref(),
        Some("stopped"),
        "destroy returned while the teardown it was waiting on was still running"
    );

    first.await.expect("the first teardown finishes");
    let _ = drain.await;
}

// ---------------------------------------------------------------------
// Borrowers
// ---------------------------------------------------------------------
//
// `mirage exec` runs its workload in its own process, in its own
// terminal, as its own child — the run that owns the session cannot see
// it, wait on it or signal it. The lease is the only thing that stops the
// run from stopping the emulator daemon, removing the containers and
// deleting the scratch directory while that workload is mid-job.

#[tokio::test]
async fn a_session_with_no_borrowers_does_not_wait_for_any() {
    let run = start(1).await;
    assert_eq!(run.borrowers(), 0);
    tokio::time::timeout(Duration::from_secs(5), run.wait_for_borrowers())
        .await
        .expect("waiting on nobody must return immediately");
    run.destroy().await;
}

#[tokio::test]
async fn a_lease_is_counted_until_it_is_dropped() {
    let run = start(1).await;

    let first = run.attach().expect("a healthy session accepts a borrower");
    assert_eq!(run.borrowers(), 1);
    let second = run.attach().expect("several terminals may borrow at once");
    assert_eq!(run.borrowers(), 2);

    drop(second);
    assert_eq!(run.borrowers(), 1);
    drop(first);
    assert_eq!(run.borrowers(), 0);

    run.destroy().await;
}

#[tokio::test]
async fn teardown_does_not_begin_while_a_borrower_holds_a_lease() {
    // The ordering invariant `Session::teardown` documents for itself,
    // from the outside: a borrowed session stays whole until the borrower
    // lets go. Before leases existed, a `mirage run -- sleep 5` finishing
    // while another terminal was mid-job removed that job's emulator
    // socket and scratch directory underneath it.
    let run = start(1).await;
    let lease = run.attach().expect("a healthy session accepts a borrower");

    let destroying = tokio::spawn({
        let run = Arc::clone(&run);
        async move {
            run.wait_for_borrowers().await;
            run.destroy().await;
        }
    });

    // Long enough that a teardown which ignored the lease would have
    // finished: this session has no workload to wait out.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !destroying.is_finished(),
        "the session was torn down while a borrower still held it"
    );
    assert_ne!(
        run.health().state.as_deref(),
        Some("stopped"),
        "teardown began with a borrower attached"
    );

    drop(lease);
    tokio::time::timeout(Duration::from_secs(30), destroying)
        .await
        .expect("teardown must proceed once the last borrower lets go")
        .unwrap();
    assert_eq!(run.health().state.as_deref(), Some("stopped"));
}

#[tokio::test]
async fn a_session_that_is_tearing_down_refuses_new_borrowers() {
    // Same guard, and the same reason, as `start_exec`: a borrower
    // admitted now would build its process grid from a description of
    // containers that are being removed.
    let run = start(1).await;
    run.destroy().await;
    assert!(
        run.attach().is_none(),
        "a destroyed session handed out a lease on itself"
    );
}

#[tokio::test]
async fn teardown_tells_the_borrowers_it_is_not_waiting() {
    // The Ctrl-C path. The run has decided not to wait, so the borrower
    // has to be told — otherwise it discovers the session is gone by
    // having its container removed or its emulator socket deleted
    // mid-syscall.
    let run = start(1).await;
    let _lease = run.attach().unwrap();

    let told = tokio::spawn({
        let run = Arc::clone(&run);
        async move { run.wait_closing().await }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!told.is_finished(), "a healthy session is not closing");

    // Teardown with the lease still held, exactly as the interrupt path
    // does it.
    let destroying = tokio::spawn({
        let run = Arc::clone(&run);
        async move { run.destroy().await }
    });
    tokio::time::timeout(Duration::from_secs(10), told)
        .await
        .expect("a borrower must be told the session is going away")
        .unwrap();
    tokio::time::timeout(Duration::from_secs(30), destroying)
        .await
        .expect("teardown must not wait for a lease it has already disowned")
        .unwrap();
}

#[tokio::test]
async fn an_exec_cannot_start_in_a_destroyed_session() {
    // Otherwise a process could be spawned after teardown collected the
    // list of things to kill, and would survive it.
    let run = start(1).await;
    run.destroy().await;
    let started = run.exec(&def(&run, "sleep 600", None)).await;
    assert!(
        started.is_err(),
        "a destroyed session must refuse new work, not start it"
    );
}

/// The environment variables `--clear-env-vars` keeps.
///
/// Taken from the supervisor rather than copied: a list that drifts from
/// the real one makes [`ambient_variable`] pick a variable that is
/// actually preserved, and the tests below then assert the opposite of
/// the behaviour.
use mirage_supervisor::process::INHERITED_ENV as KEPT_WHEN_CLEARED;

/// A variable this process has that a cleared workload would lose.
///
/// Chosen from the live environment rather than exported by the test:
/// `std::env::set_var` is `unsafe` in Rust 2024 and this workspace
/// forbids `unsafe`, and a process-wide mutation would race the other
/// tests in this binary regardless. The value only has to be something a
/// shell comparison can match, so anything awkward is skipped.
///
/// Panics rather than returning `None` when nothing suitable is found:
/// the callers used to treat that as a reason to skip, which turned a
/// sanitised CI environment into two tests that reported success without
/// exercising either half of the environment model.
fn ambient_variable() -> (String, String) {
    std::env::vars()
        .find(|(k, v)| {
            !KEPT_WHEN_CLEARED.contains(&k.as_str())
                && !k.starts_with("MIRAGE_")
                && !k.is_empty()
                && !v.is_empty()
                && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && v.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "._-/:".contains(c))
        })
        .expect("this process must export a variable outside the allowlist to test with")
}

#[tokio::test]
async fn the_callers_environment_reaches_the_workload() {
    // The default, and the point of it: mirage's parent is the terminal
    // the user typed in, so a variable exported there — an API token, a
    // PYTHONPATH, a proxy, a framework tuning knob — is one they meant
    // for the workload. Dropping it silently is the failure this guards.
    let (key, value) = ambient_variable();
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
    let (key, _) = ambient_variable();
    let run = start(1).await;
    let (exec, mut output) = run
        .exec(&def_env(&run, &format!("test -z \"${key}\""), None, true))
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
        .exec(&def_env(
            &run,
            "test -n \"$PATH\" && test -n \"$HOME\"",
            None,
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
            None,
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

#[tokio::test]
async fn every_description_of_a_session_names_the_same_rendezvous() {
    // `describe` is called once per exec by the run itself and once per
    // `Describe` request off the control socket. If it chose a fresh
    // rendezvous port each time, `mirage run`'s ranks and a `mirage exec`
    // in another terminal would be handed different MASTER_PORTs and the
    // two halves of a distributed job would never find each other —
    // silently, as a hang rather than an error.
    let run = start(2).await;
    let first = run.describe().unwrap();
    let second = run.describe().unwrap();
    assert_eq!(
        first.head_port, second.head_port,
        "two descriptions of one session must agree on the rendezvous port"
    );
    assert_ne!(first.head_port, 0, "a usable port must actually be chosen");

    // And the specs built from them agree, which is what the workload
    // actually sees.
    let a = mirage_supervisor::build_specs(
        &first,
        &def(&run, "true", None),
        &ExecId::new("x-a").unwrap(),
    )
    .unwrap();
    let b = mirage_supervisor::build_specs(
        &second,
        &def(&run, "true", None),
        &ExecId::new("x-b").unwrap(),
    )
    .unwrap();
    assert_eq!(a[1].env.get("MASTER_PORT"), b[1].env.get("MASTER_PORT"));
    run.destroy().await;
}

#[tokio::test]
async fn naming_a_node_runs_there_and_only_there() {
    // The interactive escape hatch for a multi-node session. One process
    // means it gets the terminal, and it has to land on the node the user
    // named — a shell that silently opened on node 0 while they asked for
    // node 2 would be worse than an error.
    let run = start(4).await;
    let (exec, mut output) = run
        .exec(&def(&run, "echo \"on $MIRAGE_RANK\"", Some(2)))
        .await
        .expect("exec starts");
    let collected = tokio::spawn(async move {
        let mut seen = Vec::new();
        while let Some(chunk) = output.recv().await {
            seen.push((
                chunk.node,
                String::from_utf8_lossy(&chunk.data).into_owned(),
            ));
        }
        seen
    });
    tokio::time::timeout(Duration::from_secs(30), exec.wait_finished())
        .await
        .expect("exec finishes");
    let _ = collected.await;

    let status = exec.status();
    assert_eq!(status.nodes.len(), 1, "only the named node runs");
    assert!(
        status.nodes.contains_key(&2),
        "and it is node 2: {status:?}"
    );
    assert_eq!(status.exit_code, Some(0));
    run.destroy().await;
}

#[tokio::test]
async fn naming_a_node_that_does_not_exist_is_refused() {
    let run = start(2).await;
    let err = run
        .exec(&def(&run, "true", Some(9)))
        .await
        .expect_err("a node outside the topology must be refused");
    assert!(err.to_string().contains("no node 9"), "{err}");
    run.destroy().await;
}

/// Whether `pid` still exists. `kill(pid, 0)` is the standard probe.
fn process_alive(pid: u32) -> bool {
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None).is_ok()
}

#[tokio::test]
async fn a_multi_node_exec_labels_every_ranks_output_with_its_rank() {
    // Automatic for a multi-node job: with three nodes writing to one
    // terminal, unlabelled output is unreadable. Asserted on the real
    // chunk stream, which is exactly what the CLI's printer consumes.
    let run = start(3).await;
    let (exec, mut output) = run
        .exec(&def(&run, "echo \"line from $MIRAGE_RANK\"", None))
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
async fn a_single_process_exec_writes_straight_to_the_terminal() {
    // A one-process job writes to the terminal directly, with mirage out
    // of the way entirely. If anything arrived on this channel, output
    // would be passing through mirage when it should not — which is how
    // stdout/stderr separation and byte-exactness get lost, and how an
    // interactive shell stops being interactive.
    let run = start(1).await;
    let (exec, mut output) = run
        .exec(&def(&run, "echo straight-to-the-terminal", None))
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
