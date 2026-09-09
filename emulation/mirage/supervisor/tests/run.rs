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
use mirage_core::discovery::RuntimeLocation;
use mirage_core::emulator::{
    EmulatorBackend, EmulatorBackendDef, EmulatorDef, EmulatorDescription, RuntimeStatus,
    SupportStatus,
};
use mirage_core::exec::{ExecArgs, ExecDef, ExecId, InjectionDef};
use mirage_core::plugin::PluginsDef;
use mirage_core::profile::ProfileDef;
use mirage_core::session::{CreateSessionRequest, SessionContext, SessionHealth};
use mirage_supervisor::Run;

/// A caller whose streams are redirected, which is what a test process
/// has. Stated rather than probed so spec-building stays a function of
/// its arguments; see `mirage_supervisor::CallerStreams`.
const CAPTURED: mirage_supervisor::CallerStreams = mirage_supervisor::CallerStreams::new(false);

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

    fn runtime(&self) -> RuntimeStatus {
        // Installed by virtue of being compiled in: there is no library
        // to locate, so there is no location to report.
        RuntimeStatus::new(true, RuntimeLocation::Unknown)
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

/// A stub whose daemon refuses to start.
///
/// Identical to [`Stub`] in every other way, so a test using it isolates
/// exactly one variable: what a session does when the daemon it asked for
/// is unavailable.
#[derive(Debug)]
struct NoDaemonStub;

impl EmulatorBackend for NoDaemonStub {
    fn description(&self) -> EmulatorDescription {
        EmulatorDescription {
            name: "nodaemon".to_string(),
            version: "0".to_string(),
            description: "test-only emulator whose daemon will not start".to_string(),
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

    fn runtime(&self) -> RuntimeStatus {
        // Installed by virtue of being compiled in: there is no library
        // to locate, so there is no location to report.
        RuntimeStatus::new(true, RuntimeLocation::Unknown)
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
        Ok(InjectionDef::default())
    }

    fn start_daemon(
        &self,
        _ctx: &SessionContext,
    ) -> mirage_core::error::Result<Option<Box<dyn mirage_core::emulator::EmulatorDaemon>>> {
        Err(mirage_core::error::MirageError::other(
            "the socket was already in use".to_string(),
        ))
    }
}

inventory::submit! {
    EmulatorBackendDef {
        kind: "nodaemon",
        backend: &NoDaemonStub,
    }
}

/// A stub that knows in advance it cannot host a daemon.
///
/// The complement of [`NoDaemonStub`], and the distinction is the whole
/// point of `daemon_capability`: this one is not a daemon that fails when
/// started, it is a runtime that can be *asked* before anything is
/// created. Its `start_daemon` panics rather than erroring, so a test can
/// tell the two apart — if bring-up ever reaches it, the pre-flight did
/// not do its job and the test fails loudly instead of passing on the
/// late error that looks the same from outside.
#[derive(Debug)]
struct NoCapabilityStub;

impl EmulatorBackend for NoCapabilityStub {
    fn description(&self) -> EmulatorDescription {
        EmulatorDescription {
            name: "nocapability".to_string(),
            version: "0".to_string(),
            description: "test-only emulator that cannot host a daemon here".to_string(),
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

    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::new(true, RuntimeLocation::Unknown)
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
        Ok(InjectionDef::default())
    }

    fn daemon_capability(&self) -> mirage_core::error::Result<()> {
        Err(mirage_core::error::MirageError::other(
            "this runtime predates the daemon API".to_string(),
        ))
    }

    fn start_daemon(
        &self,
        _ctx: &SessionContext,
    ) -> mirage_core::error::Result<Option<Box<dyn mirage_core::emulator::EmulatorDaemon>>> {
        panic!("bring-up reached start_daemon despite the capability check refusing it")
    }
}

inventory::submit! {
    EmulatorBackendDef {
        kind: "nocapability",
        backend: &NoCapabilityStub,
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
    let pid = forked.started().await;

    tokio::time::timeout(Duration::from_secs(30), run.destroy())
        .await
        .expect("teardown must not hang");
    let _ = exec;

    // Asked of the process table, not inferred from a file it stopped
    // growing. Sampling the marker's length twice across a fixed sleep
    // measures the same property the sibling test three lines below
    // asserts directly with `wait_gone` — but slower, and with a
    // half-second sleep in the middle deciding how much of a still-alive
    // grandchild's writing counts as "stopped".
    assert!(
        mirage_supervisor::process::wait_gone(pid, Duration::from_secs(10)).await,
        "pid {pid} was forked by a workload and survived the run that owned it"
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
    let specs = mirage_supervisor::build_specs(&desc, &def(&run, "exit 0", None), &id, CAPTURED)
        .expect("specs");
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
    let specs = mirage_supervisor::build_specs(&desc, &d, &id, CAPTURED).unwrap();
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
    //
    // The trap has to be *installed* before the first teardown starts, or
    // SIGTERM kills the shell outright, that teardown finishes instantly,
    // and the second one has nothing to wait for — the test then passes
    // whatever `teardown` does. A sleep is not enough to say that
    // happened, and the sibling in `process.rs` already shows the answer:
    // the shell announces itself and the test waits for the marker.
    let run = start(1).await;
    let scratch = tempfile::tempdir().expect("scratch dir");
    let trapped = scratch.path().join("trap-installed");
    let (_exec, output) = run
        .exec(&def(
            &run,
            &format!(
                "trap '' TERM; : > {trapped}; while true; do sleep 1; done",
                trapped = trapped.display()
            ),
            None,
        ))
        .await
        .expect("exec starts");
    let drain = tokio::spawn(async move {
        let mut output = output;
        while output.recv().await.is_some() {}
    });

    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while !trapped.exists() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the workload never installed its SIGTERM trap, so this test \
             would have proved nothing about a concurrent teardown"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

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

    let first = run
        .attach(None, None)
        .expect("a healthy session accepts a borrower");
    assert_eq!(run.borrowers(), 1);
    let second = run
        .attach(None, None)
        .expect("several terminals may borrow at once");
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
    let lease = run
        .attach(None, None)
        .expect("a healthy session accepts a borrower");

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
        run.attach(None, None).is_none(),
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
    let _lease = run.attach(None, None).unwrap();

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

/// A process standing in for one a `mirage exec` started and then failed
/// to reap, killed when the test ends however it ends.
///
/// Tagged exactly as a real one is — [`ENV_SESSION`] and [`ENV_RUNTIME`]
/// say which session it is in, and `MIRAGE_EXEC` which invocation started
/// it — because that tag is the only thing that survives the borrower
/// dying, and finding it is what the code under test does. Spawned
/// directly rather than through an `Exec`, since the point is that it
/// belongs to no exec of the run's.
struct Borrowed(std::process::Child);

impl Borrowed {
    /// A stand-in whose exec id is a borrower's: the shape `mirage exec`
    /// builds for itself, which is by construction not one the run has
    /// ever issued.
    fn spawn(session: &mirage_core::session::SessionId) -> Self {
        Self::tagged(session, &format!("x-{}-stand-in", std::process::id()))
    }

    fn tagged(session: &mirage_core::session::SessionId, exec: &str) -> Self {
        use mirage_core::container::{ENV_RUNTIME, ENV_SESSION, owning_runtime};
        Self(
            std::process::Command::new("/bin/sh")
                .env(mirage_supervisor::spec::ENV_EXEC, exec)
                // `exec`, so there is exactly one tagged process and the
                // test is not racing a shell's own child.
                .args(["-c", "exec sleep 600"])
                .env(ENV_SESSION, session.as_str())
                .env(ENV_RUNTIME, owning_runtime())
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("a stand-in borrower's workload starts"),
        )
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }

    /// Whether the process has ended, waiting up to `timeout` for it.
    ///
    /// `try_wait`, not [`mirage_supervisor::process::wait_gone`]. In the
    /// real leak the stranded workload is reparented to init, which reaps
    /// it, so the pid genuinely leaves the table. Here the test process
    /// is its parent and reaps nothing until the guard below drops, so a
    /// killed stand-in is a *zombie* — and `kill(pid, 0)` succeeds on a
    /// zombie, which is a test that fails for a process that is provably
    /// dead.
    fn ended_within(&mut self, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if matches!(self.0.try_wait(), Ok(Some(_))) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// The same wait, for a test whose *runtime* has to keep running.
    ///
    /// Blocking the thread is fine when what will end the process has
    /// already happened — a teardown that was awaited — and fatal when it
    /// has not: these tests wait on a sweep that runs in the run's own
    /// task, on this very runtime, so a `std::thread::sleep` here would
    /// stop the thing being waited for and time out every time.
    async fn ends_within(&mut self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if matches!(self.0.try_wait(), Ok(Some(_))) {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

impl Drop for Borrowed {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn a_borrower_that_died_does_not_leave_its_workload_in_the_session() {
    // `kill -9` on a `mirage exec`. Its workload is its own child, in its
    // own process, so nothing of the run's is waiting on it — and
    // `kill -9` runs no code of the borrower's either, so nobody reaps
    // it. It is then reparented to init, still tagged with this session,
    // still holding the emulated device.
    //
    // `mirage cleanup` will not touch it, and is right not to: the
    // session is live, and reclaiming a live session's processes is
    // exactly what it must never do. Which is why this has to be the
    // run's job. The run owns the session; nothing may be left in it when
    // the session goes.
    let run = start(1).await;
    let mut stranded = Borrowed::spawn(run.id());
    let pid = stranded.pid();
    assert!(process_alive(pid), "the setup must leave a live process");

    tokio::time::timeout(Duration::from_secs(30), run.destroy())
        .await
        .expect("teardown must not hang");

    assert!(
        stranded.ended_within(Duration::from_secs(10)),
        "pid {pid} was started in this session by a borrower that died, and \
         outlived the run that owned the session"
    );
}

/// Attach to `run`'s control socket the way `mirage exec` does, and hand
/// back the connection that *is* the lease.
///
/// Over a real socket rather than through `Run::attach`, because what
/// these tests are about is the run noticing a borrower it can only see
/// as a connection: the peer credentials, the disconnect and the sweep
/// that follows are all on this path and on no other.
async fn attach_over_the_socket(
    path: &std::path::Path,
) -> tokio_util::codec::Framed<tokio::net::UnixStream, tokio_util::codec::LengthDelimitedCodec> {
    use futures::{SinkExt as _, StreamExt as _};
    use mirage_core::proto::{Request, Response, codec};

    let stream = tokio::net::UnixStream::connect(path)
        .await
        .expect("the run is serving its socket");
    let mut framed = tokio_util::codec::Framed::new(stream, codec());
    framed
        .send(
            serde_json::to_vec(&Request::Attach { exec: None })
                .unwrap()
                .into(),
        )
        .await
        .unwrap();
    let frame = framed
        .next()
        .await
        .expect("the run answers an attach")
        .unwrap();
    match serde_json::from_slice::<Response>(&frame).unwrap() {
        Response::Description(_) => framed,
        Response::Error(e) => panic!("the run refused a borrower: {e}"),
    }
}

/// A run serving its control socket, with the socket kept alive.
async fn serving(run: &Arc<Run>) -> (std::path::PathBuf, tokio::task::JoinHandle<()>) {
    let path = mirage_core::paths::run_socket_path(run.id());
    let socket = mirage_supervisor::rpc::ControlSocket::bind(&path)
        .await
        .expect("the control socket binds");
    let run = Arc::clone(run);
    let serving = tokio::spawn(async move { socket.serve(run).await });
    (path, serving)
}

#[tokio::test]
async fn a_borrower_that_disappears_has_its_workload_reaped_before_teardown() {
    // `kill -9` on a `mirage exec` *while its run carries on*. Teardown
    // already swept up after a borrower that died — but only at teardown,
    // so a run that lasts all afternoon left the borrower's workload
    // running in the session all afternoon: holding the emulated device,
    // and untouchable by `mirage cleanup`, which will not reclaim a live
    // session's processes and is right not to. `mirage exec --help`
    // promises the workload "dies with it"; this is the half of that
    // promise the run has to keep.
    let run = start(1).await;
    let (path, served) = serving(&run).await;

    let borrower = attach_over_the_socket(&path).await;
    let mut stranded = Borrowed::spawn(run.id());
    let pid = stranded.pid();
    assert!(process_alive(pid), "the setup must leave a live process");

    // The borrower goes away without stopping what it started, which is
    // what a `SIGKILL` looks like from this side of the socket.
    drop(borrower);

    assert!(
        stranded.ends_within(Duration::from_secs(30)).await,
        "pid {pid} was started in session {} by a borrower that is gone, and \
         is still running in it",
        run.id()
    );

    served.abort();
    run.destroy().await;
}

#[tokio::test]
async fn a_departing_borrower_does_not_take_the_runs_own_workload_with_it() {
    // The sweep's whole hazard. Everything in a session carries the same
    // session mark — that is what one session means — so a run that
    // reaped by that mark alone would answer a borrower's disconnect by
    // killing its own running workload, which is very much worse than
    // the leak it is fixing.
    let run = start(1).await;
    let (path, served) = serving(&run).await;

    let (exec, mut output) = run
        .exec(&def(&run, "sleep 600", None))
        .await
        .expect("the run's own workload starts");
    let drain = tokio::spawn(async move { while output.recv().await.is_some() {} });
    let ours = exec.live_pids();
    assert!(!ours.is_empty(), "the run must have a workload to protect");

    let borrower = attach_over_the_socket(&path).await;
    let stranded = Borrowed::spawn(run.id());
    drop(borrower);

    // The borrower's leftover is what goes; the run's own workload is
    // what stays. Asserted in that order, so the sweep has demonstrably
    // run by the time the survival check is made — otherwise this passes
    // for a sweep that never happened at all.
    let mut stranded = stranded;
    assert!(
        stranded.ends_within(Duration::from_secs(30)).await,
        "the borrower's leftover survived the sweep"
    );
    for pid in &ours {
        assert!(
            process_alive(*pid),
            "the run's own workload (pid {pid}) was killed by another \
             terminal's `mirage exec` exiting"
        );
    }
    assert!(
        !exec.is_ended(),
        "the run's own exec was ended underneath it"
    );

    served.abort();
    run.destroy().await;
    let _ = drain.await;
}

#[tokio::test]
async fn a_departing_borrower_does_not_take_a_second_borrowers_workload_with_it() {
    // Two terminals borrowing at once is ordinary, and the first of them
    // finishing is exactly when this sweep runs. The second borrower's
    // processes carry a foreign exec mark just as a stranded one does;
    // what separates them is that theirs still has a live borrower above
    // it, and that borrower still holds its lease.
    let run = start(1).await;

    // A stand-in for the borrower that is staying: a process holding a
    // lease, with a tagged workload of its own beneath it.
    let mut staying = std::process::Command::new("/bin/sh")
        .args(["-c", "sleep 600 & echo $!; wait"])
        .env(mirage_supervisor::spec::ENV_EXEC, "x-999999-staying")
        .env(mirage_core::container::ENV_SESSION, run.id().as_str())
        .env(
            mirage_core::container::ENV_RUNTIME,
            mirage_core::container::owning_runtime(),
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("the second borrower starts");
    let workload: u32 = {
        use std::io::BufRead as _;
        let stdout = staying.stdout.take().expect("piped");
        let mut line = String::new();
        std::io::BufReader::new(stdout)
            .read_line(&mut line)
            .unwrap();
        line.trim().parse().expect("the pid of its workload")
    };
    let held = run
        .attach(Some(staying.id()), None)
        .expect("a healthy session accepts a borrower");

    // And the one that leaves.
    let (path, served) = serving(&run).await;
    let leaving = attach_over_the_socket(&path).await;
    let mut stranded = Borrowed::spawn(run.id());
    drop(leaving);

    assert!(
        stranded.ends_within(Duration::from_secs(30)).await,
        "the departed borrower's leftover survived the sweep"
    );
    assert!(
        process_alive(workload),
        "pid {workload} belongs to a borrower that is still attached, and was \
         killed because a different borrower left"
    );
    assert!(
        process_alive(staying.id()),
        "the second borrower itself was killed"
    );

    drop(held);
    let _ = staying.kill();
    let _ = staying.wait();
    served.abort();
    run.destroy().await;
}

#[tokio::test]
async fn a_reparented_workload_of_a_live_borrower_is_spared() {
    // The same property as the test above, for the workload that has left
    // its borrower's process tree.
    //
    // Ancestry was the only thing sparing a live borrower's processes.
    // A borrower's exec id is minted client-side — the run does not start
    // these processes and never saw the id — so `inner.execs` does not
    // contain it and the "not one of ours" filter lets every borrower
    // workload through. What stopped them being killed was descending
    // from a live borrower pid, and a workload that reparents does not:
    // `setsid`, a double fork, anything daemonised. Then the next
    // borrower to disconnect *normally* sent it `SIGTERM` and, a grace
    // period later, `SIGKILL` — while its own borrower sat there holding
    // its lease, having done nothing at all.
    //
    // Now the lease carries the exec id, and the mark survives
    // reparenting because it is in the environment rather than in the
    // process tree.
    let run = start(1).await;
    let exec = ExecId::new("x-999999-reparenting").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("reparented.pid");

    // `setsid --fork` orphans the workload: setsid exits straight after
    // forking, so the sleep is reparented away from this shell while
    // keeping every environment mark it inherited. The shell itself stays
    // alive as the borrower's stand-in.
    let mut staying = std::process::Command::new("/bin/sh")
        .args([
            "-c",
            &format!(
                "setsid --fork /bin/sh -c 'echo $$ > {}; exec sleep 600'; sleep 600",
                pidfile.display()
            ),
        ])
        .env(mirage_supervisor::spec::ENV_EXEC, exec.as_str())
        .env(mirage_core::container::ENV_SESSION, run.id().as_str())
        .env(
            mirage_core::container::ENV_RUNTIME,
            mirage_core::container::owning_runtime(),
        )
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("the staying borrower starts");

    // Wait for the reparented workload to exist and say where it is.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let workload: u32 = loop {
        if let Ok(text) = std::fs::read_to_string(&pidfile)
            && let Ok(pid) = text.trim().parse::<u32>()
        {
            break pid;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the reparented workload never started"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    // The premise of the test: it really has left the borrower's tree, so
    // the ancestry check cannot save it.
    assert_ne!(
        parent_of(workload),
        Some(staying.id()),
        "the workload did not reparent, so this test is not testing anything"
    );

    let held = run
        .attach(Some(staying.id()), Some(exec.clone()))
        .expect("a healthy session accepts a borrower");

    // A different borrower comes and goes, which is what runs the sweep.
    let (path, served) = serving(&run).await;
    let leaving = attach_over_the_socket(&path).await;
    let mut stranded = Borrowed::spawn(run.id());
    drop(leaving);

    assert!(
        stranded.ends_within(Duration::from_secs(30)).await,
        "the departed borrower's leftover survived the sweep"
    );
    assert!(
        process_alive(workload),
        "pid {workload} is a live borrower's workload that had reparented, \
         and was killed because a different borrower left"
    );

    drop(held);
    let _ = nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(workload as i32),
        nix::sys::signal::Signal::SIGKILL,
    );
    let _ = staying.kill();
    let _ = staying.wait();
    served.abort();
    run.destroy().await;
}

#[tokio::test]
async fn a_borrower_the_kernel_would_not_name_is_still_protected_by_its_exec_mark() {
    // The two handles are independent protections, so losing one must not
    // withdraw the other.
    //
    // A borrower's pid comes from `SO_PEERCRED` on its connection, and it
    // is not always there to be had: the call can fail, and a pid the
    // kernel reports from a namespace this process cannot resolve is no
    // use either. The lease was registered only when the pid was known,
    // so such a borrower was recorded as nothing at all — and the exec id
    // it had supplied, which is the handle that does not need the pid,
    // protected nothing. The next borrower to disconnect normally sent
    // its live workload `SIGTERM` and then `SIGKILL`.
    //
    // Nothing here is reparented: the point is that ancestry is not
    // available *as a question* when there is no pid to ask it about, so
    // the mark has to carry the whole answer on its own.
    let run = start(1).await;
    let exec = format!("x-{}-pidless", std::process::id());
    let mut workload = Borrowed::tagged(run.id(), &exec);

    let held = run
        .attach(None, Some(ExecId::new(&exec).unwrap()))
        .expect("a healthy session accepts a borrower");

    // A different borrower comes and goes, which is what runs the sweep.
    let (path, served) = serving(&run).await;
    let leaving = attach_over_the_socket(&path).await;
    let mut stranded = Borrowed::spawn(run.id());
    drop(leaving);

    // The sweep really did run, so the assertion below is about being
    // spared rather than about nothing having happened yet.
    assert!(
        stranded.ends_within(Duration::from_secs(30)).await,
        "the departed borrower's leftover survived the sweep"
    );
    // `ends_within`, not `process_alive`: this workload is a child of the
    // test process, so a killed one is a zombie until the guard reaps it
    // — and `kill(pid, 0)` succeeds on a zombie, which would pass this
    // assertion for a process the sweep had provably just killed.
    assert!(
        !workload.ends_within(Duration::from_secs(3)).await,
        "pid {} is the workload of a live borrower whose credentials the \
         kernel would not give up, and was killed because a different \
         borrower left",
        workload.pid()
    );

    // And the mark stops protecting it the moment the lease ends, exactly
    // as it does for a borrower that had a pid.
    drop(held);
    let second = attach_over_the_socket(&path).await;
    drop(second);
    assert!(
        workload.ends_within(Duration::from_secs(30)).await,
        "a pidless borrower's lease went on sparing its workload after the \
         lease had ended"
    );

    served.abort();
    run.destroy().await;
}

#[tokio::test]
async fn a_reparented_workload_is_still_reaped_once_its_borrower_leaves() {
    // The complement, and the reason the one above is not simply "spare
    // anything with an exec mark". Sparing by mark is scoped to marks a
    // *live* lease claims; the moment that lease ends, the same process
    // is exactly the leak the sweep exists for — and a reparented one is
    // the worst case, because nothing else will ever come looking for it.
    let run = start(1).await;
    let exec = format!("x-{}-departing", std::process::id());

    // Tagged with the departing borrower's exec id, and orphaned outright
    // so it has no live ancestor at all.
    let mut workload = Borrowed::tagged(run.id(), &exec);
    let held = run
        .attach(Some(std::process::id()), Some(ExecId::new(&exec).unwrap()))
        .expect("a healthy session accepts a borrower");

    // While the lease is held, a sweep spares it.
    let (path, served) = serving(&run).await;
    let first = attach_over_the_socket(&path).await;
    drop(first);
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        process_alive(workload.pid()),
        "a live borrower's workload was reaped while its lease was held"
    );

    // The borrower goes away, and the next sweep collects it.
    drop(held);
    let second = attach_over_the_socket(&path).await;
    drop(second);
    assert!(
        workload.ends_within(Duration::from_secs(30)).await,
        "a workload whose borrower has gone was spared for carrying a mark \
         no live lease claims any more"
    );

    served.abort();
    run.destroy().await;
}

#[tokio::test]
async fn tearing_one_session_down_leaves_another_sessions_processes_alone() {
    // The other half, and the reason the sweep is filtered to one
    // session rather than reusing `mirage cleanup`'s "everything under
    // this runtime directory" scan. Two runs under one `MIRAGE_RUNTIME`
    // is the ordinary case — a second terminal — and a run tearing itself
    // down that reclaimed the other's workloads would be killing a
    // healthy job.
    let mine = start(1).await;
    let theirs = start(1).await;
    let mut stranded = Borrowed::spawn(theirs.id());

    tokio::time::timeout(Duration::from_secs(30), mine.destroy())
        .await
        .expect("teardown must not hang");

    assert!(
        !stranded.ended_within(Duration::from_millis(200)),
        "tearing down session {} killed a process belonging to session {}",
        mine.id(),
        theirs.id()
    );
    theirs.destroy().await;
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
        CAPTURED,
    )
    .unwrap();
    let b = mirage_supervisor::build_specs(
        &second,
        &def(&run, "true", None),
        &ExecId::new("x-b").unwrap(),
        CAPTURED,
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

/// The parent of `pid`, for asserting that a workload really did leave
/// the process tree a test is about.
fn parent_of(pid: u32) -> Option<u32> {
    std::fs::read_to_string(format!("/proc/{pid}/status"))
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("PPid:")?.trim().parse().ok())
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

// ---------------------------------------------------------------------
// Cancelling a bring-up that is still in flight
// ---------------------------------------------------------------------

/// A container engine that answers `--version` and then wedges.
///
/// `image inspect` is the first blocking provider call a containerised
/// bring-up makes, so hanging there puts the run in exactly the state
/// this test is about: inside a child process that no future can be
/// dropped to hurry along. `exec` so the script *is* the sleep — killing
/// the provider then kills the wait, rather than leaving a minute-long
/// orphan behind for the rest of the suite.
fn wedged_provider(dir: &std::path::Path, probed: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt as _;
    let script = dir.join("wedged-provider.sh");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\n\
             if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n\
             \t: > {probed}\n\
             \texec sleep 120\n\
             fi\n\
             exit 0\n",
            probed = probed.display()
        ),
    )
    .expect("the fake provider is written");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("the fake provider is executable");
    script
}

#[tokio::test]
async fn tearing_down_ends_a_container_bring_up_rather_than_waiting_it_out() {
    // The `Cancel` switch has thorough tests in `mirage_container` and
    // one in the supervisor for teardown flipping it — and every one of
    // them constructs the switch itself. The single line of production
    // code that hands a session's switch to the engine it is meant to
    // interrupt had none, so deleting `.with_cancel(cancel)` from
    // `Run::bring_up_containers` left the whole suite green while a
    // Ctrl-C during a ten-minute image pull went back to sitting out the
    // rest of that pull with mirage looking like it had ignored the
    // signal.
    //
    // This is the only test that can catch that, because it is the only
    // one that lets the real bring-up build the real engine.
    isolate();
    let dir = tempfile::tempdir().expect("scratch dir");
    let probed = dir.path().join("probed");
    let provider = wedged_provider(dir.path(), &probed);

    let mut p = profile(1);
    p.containerize = Some(mirage_core::profile::ContainerizedDef {
        provider: Some(provider.to_string_lossy().into_owned()),
        image: "img:latest".to_string(),
        mounts: Vec::new(),
        ports: Vec::new(),
        devices: Vec::new(),
        groups: Vec::new(),
        hacks: Vec::new(),
    });

    let run = Arc::new(
        Run::start(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(p),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .expect("run starts"),
    );

    // Wait for the bring-up to be provably *inside* the provider. Without
    // this the teardown below could beat it there, the engine would
    // refuse to start at all, and the test would pass whether or not the
    // switch was ever wired up.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while !probed.exists() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the bring-up never reached the container engine, so this test \
             would have proved nothing about cancelling one"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Generously less than the provider's own sleep, and generously more
    // than a cancellation costs: the engine polls its switch every 25ms.
    let started = std::time::Instant::now();
    tokio::time::timeout(Duration::from_secs(30), run.destroy())
        .await
        .expect("teardown must end the bring-up it waits for, not wait it out");
    let waited = started.elapsed();
    assert!(
        waited < Duration::from_secs(20),
        "teardown sat out the provider it was supposed to interrupt: {waited:?}"
    );

    // And it ended for the right reason. The timing alone would also be
    // satisfied by a provider that happened to die on its own, so the
    // fingerprint of `ContainerError::Cancelled` — which nothing but the
    // switch produces — is what says the interrupt is why.
    let reason = format!("{:?}", run.health());
    assert!(
        reason.contains("interrupted while"),
        "the bring-up did not end as a cancellation: {reason}"
    );
}

/// A daemon that will not start must fail the session, not become a
/// different kind of run.
///
/// This used to be a `warn!` and a fallback to in-process emulation. The
/// justification was that in-process might still work — which is true,
/// and is the trap: in-process cannot share emulated GPU memory between
/// processes, so a multi-GPU collective returns a plausible wrong number
/// instead of an error. The user asked for the daemon (it is the default;
/// `--in-process` is the opt-out) and silently got something that
/// computes differently.
#[tokio::test]
async fn a_daemon_that_will_not_start_fails_the_session() {
    isolate();
    let mut p = profile(1);
    p.emulator.emulator = "nodaemon".to_string();

    let run = Arc::new(
        Run::start(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(p),
            workdir: "/tmp".to_string(),
            daemon: true,
        })
        .expect("run starts"),
    );

    let health = run
        .wait_ready(Duration::from_secs(30))
        .await
        .expect("bring-up settles");
    assert!(
        !health.healthy,
        "the session reported healthy despite its daemon failing: {health:?}"
    );

    // The reason has to survive teardown, and it has to name both the
    // cause and the way forward — a bare "session failed to start" would
    // leave the user with nowhere to go.
    let reason = format!("{health:?}");
    assert!(
        reason.contains("the socket was already in use"),
        "the backend's own reason was lost: {reason}"
    );
    assert!(
        reason.contains("--in-process"),
        "the error does not say how to proceed: {reason}"
    );
}

/// A runtime that cannot host a daemon is refused before anything is
/// created, not after everything is.
///
/// The complement of the test above, and the reason
/// `EmulatorBackend::daemon_capability` exists. Both sessions fail, and
/// from outside the two failures look alike — but one fails at the *end*
/// of bring-up, and for a containerised session that is after the image
/// pull, the network and every node container, all of which are then
/// removed again, for a fact about a file that was knowable at the start.
///
/// What makes this a real assertion rather than a restatement is
/// [`NoCapabilityStub::start_daemon`]: it panics. A bring-up that reaches
/// it has done the expensive work first, and the test fails there instead
/// of passing on an error message that would have been identical.
#[tokio::test]
async fn a_runtime_that_cannot_host_a_daemon_is_refused_before_bring_up() {
    isolate();
    let mut p = profile(1);
    p.emulator.emulator = "nocapability".to_string();

    let run = Arc::new(
        Run::start(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(p),
            workdir: "/tmp".to_string(),
            daemon: true,
        })
        .expect("run starts"),
    );

    let health = run
        .wait_ready(Duration::from_secs(30))
        .await
        .expect("bring-up settles");
    assert!(
        !health.healthy,
        "a session was reported healthy on a runtime that cannot host the \
         daemon it asked for: {health:?}"
    );

    // The backend's own reason has to reach the user: it is the whole
    // value of asking early rather than late.
    let reason = format!("{health:?}");
    assert!(
        reason.contains("predates the daemon API"),
        "the capability check's reason was lost: {reason}"
    );

    run.destroy().await;
}

/// The other half of the contract: a session that wants no daemon is not
/// refused by a capability it never needed.
///
/// `--in-process` emulation needs none of the daemon API, so gating the
/// check on anything broader than "this session asked for a daemon" would
/// refuse a mode that works. [`NoCapabilityStub`] refuses the capability
/// and panics if its daemon is started, so this session must both start
/// and never ask.
#[tokio::test]
async fn an_in_process_session_is_not_refused_by_a_daemon_capability() {
    isolate();
    let mut p = profile(1);
    p.emulator.emulator = "nocapability".to_string();

    let run = Arc::new(
        Run::start(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(p),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .expect("run starts"),
    );

    let health = run
        .wait_ready(Duration::from_secs(30))
        .await
        .expect("bring-up settles");
    assert!(
        health.healthy,
        "an in-process session was refused for a daemon it never wanted: \
         {health:?}"
    );

    run.destroy().await;
}
