//! Integration tests for [`SessionManager`] driven directly, with no
//! daemon and no socket in the way.
//!
//! Everything here exercises the real engine: real processes, real
//! teardown, real reaping. What it deliberately avoids is the transport,
//! so a failure points at the lifecycle rather than at the RPC layer.

#![allow(clippy::unwrap_used, clippy::expect_used)]

// Link-only: a backend registers itself into the emulator registry via
// `inventory` at link time. Without a reference the linker drops the
// crate object, the registration with it, and every session here fails
// with "unknown emulator".
extern crate mirage_rocjitsu as _;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use mirage_core::common::MaybeRef;
use mirage_core::ctl::{CreateSessionRequest, MirageCtl, StdStream, StreamPacket};
use mirage_core::emulator::{EmulatorDef, ExecMode};
use mirage_core::error::MirageError;
use mirage_core::exec::{ExecArgs, ExecDef, ExecId, ExecRef};
use mirage_core::profile::ProfileDef;
use mirage_core::session::SessionId;
use mirage_core::topology::TopologyDef;
use mirage_supervisor::SessionManager;
use tokio_stream::StreamExt as _;

/// A manager pointed at a private config/runtime root.
///
/// `mirage_core::paths` resolves through a process-wide override, so
/// tests that touch it must serialise on the same lock. The guard is held
/// for the whole test.
struct Env {
    manager: Arc<SessionManager>,
    _dir: tempfile::TempDir,
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl Env {
    fn new() -> Self {
        let guard = mirage_core::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        mirage_core::paths::set_test_root(dir.path());
        Self {
            manager: Arc::new(SessionManager::default()),
            _dir: dir,
            _guard: guard,
        }
    }

    fn ctl(&self) -> &SessionManager {
        &self.manager
    }

    /// Create a ready session on the test emulator backend.
    async fn session(&self) -> SessionId {
        self.session_named(None).await
    }

    async fn session_named(&self, id: Option<&str>) -> SessionId {
        let def = self
            .ctl()
            .session_create(CreateSessionRequest {
                id: id.map(|i| SessionId::new(i).unwrap()),
                profile: MaybeRef::Owned(test_profile()),
                workdir: "/tmp".to_string(),
                daemon: false,
            })
            .await
            .unwrap();
        let health = self
            .ctl()
            .session_wait_ready(&def.id, Duration::from_secs(30))
            .await
            .unwrap();
        assert!(health.healthy, "session should be ready: {health:?}");
        def.id
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        // Nothing may outlive the test, including after a failed
        // assertion unwound past the explicit cleanup.
        //
        // This deliberately uses the synchronous backstop rather than
        // awaiting `shutdown_all`. `Drop` cannot await, and blocking the
        // current-thread runtime here to drive teardown deadlocks: the
        // very tasks teardown waits on are the ones that need the thread
        // we just blocked.
        self.manager.kill_all_now();
        mirage_core::paths::clear_test_root();
    }
}

/// The emulator these tests run sessions on.
///
/// They exercise the session and process lifecycle, not emulation, but a
/// session still needs a backend that can produce a usable injection —
/// and mirage ships exactly one. rocjitsu interposes GPU calls the shell
/// commands here never make, so it adds nothing to what is under test
/// beyond requiring its runtime library.
const TEST_EMULATOR: &str = "rocjitsu";

/// Whether that runtime is present. rocjitsu is a sibling project in this
/// monorepo, so a full build has it; a mirage-only build does not.
fn emulator_available() -> bool {
    mirage_core::emulator::get_emulator_backend(TEST_EMULATOR)
        .is_some_and(mirage_core::emulator::EmulatorBackend::installed)
}

/// Skip the calling test, loudly, when the emulator runtime is missing.
#[must_use]
fn skip_without_emulator() -> bool {
    if emulator_available() {
        return false;
    }
    eprintln!(
        "SKIP: the `{TEST_EMULATOR}` runtime was not found, so no session \
         can be brought up. Build the sibling `emulation/rocjitsu` project \
         (or set ROCM_HOME) and re-run."
    );
    true
}

fn test_profile() -> ProfileDef {
    ProfileDef {
        name: "test".to_string(),
        description: None,
        emulator: EmulatorDef {
            emulator: TEST_EMULATOR.to_string(),
            plugins: BTreeMap::new(),
            exec_mode: ExecMode::Functional,
            options: Default::default(),
            topology: MaybeRef::Owned(TopologyDef {
                num_nodes: 1,
                gpus_per_node: 1,
                agent: MaybeRef::Ref("MI350X".to_string()),
            }),
        },
        containerize: None,
    }
}

fn exec_def(session: &SessionId, script: &str) -> ExecDef {
    ExecDef {
        timestamp: chrono::Utc::now(),
        session: session.clone(),
        exec: ExecArgs {
            command: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
            env: BTreeMap::new(),
            workdir: None,
        },
        worker_exec: None,
        nproc_per_node: 1,
        tty: false,
        tty_rows: 0,
        tty_cols: 0,
        keep: true,
    }
}

/// Run an exec to completion, returning its exit code and stdout.
async fn run(ctl: &SessionManager, session: &SessionId, script: &str) -> (i32, String) {
    let r = ctl.session_exec(&exec_def(session, script)).await.unwrap();
    let mut stream = ctl.session_attach(&r).await.unwrap();
    let mut stdout = Vec::new();
    let mut exit = None;
    while let Some(pkt) = stream.next().await {
        match pkt {
            StreamPacket::Output {
                stream: StdStream::Stdout,
                data,
                ..
            } => {
                stdout.extend_from_slice(&data);
            }
            StreamPacket::ExecExit { exit_code } => {
                exit = Some(exit_code);
                break;
            }
            _ => {}
        }
    }
    (
        exit.expect("attach must end with an exit"),
        String::from_utf8_lossy(&stdout).into_owned(),
    )
}

// ---- session lifecycle -----------------------------------------------------

#[tokio::test]
async fn a_created_session_becomes_ready_and_is_listed() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    assert_eq!(env.ctl().session_list().await.unwrap(), vec![id.clone()]);
    let state = env.ctl().session_state(&id).await.unwrap();
    assert!(state.health.healthy);
    assert_eq!(state.def.id, id);
    assert!(state.container.is_none());
}

#[tokio::test]
async fn a_destroyed_session_is_gone_immediately() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    env.ctl().session_destroy(&id).await.unwrap();
    assert!(env.ctl().session_list().await.unwrap().is_empty());
    assert!(matches!(
        env.ctl().session_state(&id).await,
        Err(MirageError::SessionNotFound(_))
    ));
}

#[tokio::test]
async fn destroying_a_session_twice_reports_not_found() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    env.ctl().session_destroy(&id).await.unwrap();
    let err = env.ctl().session_destroy(&id).await.unwrap_err();
    assert!(err.is_not_found(), "{err:?}");
}

#[tokio::test]
async fn duplicate_session_ids_are_rejected() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.session_named(Some("dup")).await;
    let err = env
        .ctl()
        .session_create(CreateSessionRequest {
            id: Some(SessionId::new("dup").unwrap()),
            profile: MaybeRef::Owned(test_profile()),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .await
        .unwrap_err();
    assert!(matches!(err, MirageError::SessionExists(_)), "{err:?}");
}

#[tokio::test]
async fn a_session_referring_to_a_missing_profile_fails_at_creation() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    // Failing here rather than coming up and failing every exec is the
    // point: the error names the profile, at the moment it was requested.
    let err = env
        .ctl()
        .session_create(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Ref("no-such-profile".to_string()),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .await
        .unwrap_err();
    assert!(matches!(err, MirageError::ProfileNotFound(_)), "{err:?}");
    assert!(env.ctl().session_list().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_session_with_an_unknown_emulator_fails_terminally_with_a_reason() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let mut profile = test_profile();
    profile.emulator.emulator = "not-a-real-emulator".to_string();
    let def = env
        .ctl()
        .session_create(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(profile),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .await
        .unwrap();

    // Bring-up must settle, not hang: a client waiting on readiness needs
    // an answer either way.
    let health = env
        .ctl()
        .session_wait_ready(&def.id, Duration::from_secs(30))
        .await
        .unwrap();
    assert!(!health.healthy);
    assert!(health.terminal);
    assert!(
        health
            .message
            .as_deref()
            .unwrap_or_default()
            .contains("not-a-real-emulator"),
        "the reason must name the emulator: {health:?}"
    );
}

#[tokio::test]
async fn waiting_on_a_session_that_is_already_ready_returns_immediately() {
    // Regression: health is published through a `watch` channel, and
    // `watch::Sender::send` fails *and leaves the value unchanged* when
    // nobody is subscribed. A session that finished bring-up before any
    // client asked therefore stayed at its initial `starting` forever,
    // and this call timed out against a session that was ready all along.
    // It only reproduces once the daemon is warm enough for bring-up to
    // beat the client's first query.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let def = env
        .ctl()
        .session_create(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(test_profile()),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .await
        .unwrap();

    // Let bring-up finish with nobody waiting.
    for _ in 0..300 {
        if env.ctl().session_health(&def.id).await.unwrap().is_settled() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let health = tokio::time::timeout(
        Duration::from_secs(5),
        env.ctl().session_wait_ready(&def.id, Duration::from_secs(30)),
    )
    .await
    .expect("waiting on an already-ready session must return at once")
    .unwrap();
    assert!(health.healthy, "{health:?}");
}

#[tokio::test]
async fn health_is_observable_without_a_subscriber_present_at_publish_time() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    // Read health through a fresh call, i.e. with no receiver alive when
    // the ready snapshot was published.
    let health = env.ctl().session_health(&id).await.unwrap();
    assert!(health.healthy, "{health:?}");
    assert_eq!(health.state.as_deref(), Some("ready"));
}

#[tokio::test]
async fn waiting_on_a_missing_session_is_an_error_not_a_timeout() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let err = env
        .ctl()
        .session_wait_ready(&SessionId::new("ghost").unwrap(), Duration::from_secs(1))
        .await
        .unwrap_err();
    assert!(matches!(err, MirageError::SessionNotFound(_)), "{err:?}");
}

#[tokio::test]
async fn the_session_scratch_directory_is_created_and_removed_with_the_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let scratch = mirage_core::paths::session_runtime_dir(&id);
    assert!(scratch.is_dir(), "scratch must exist while the session does");
    env.ctl().session_destroy(&id).await.unwrap();
    assert!(
        !scratch.exists(),
        "scratch must not outlive the session: {}",
        scratch.display()
    );
}

// ---- execs -----------------------------------------------------------------

#[tokio::test]
async fn an_exec_runs_and_reports_its_output_and_exit_code() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let (code, out) = run(env.ctl(), &id, "echo hello; exit 3").await;
    assert_eq!(code, 3);
    assert_eq!(out.trim(), "hello");
}

#[tokio::test]
async fn exec_ids_increase_and_are_listed() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let a = env
        .ctl()
        .session_exec(&exec_def(&id, "true"))
        .await
        .unwrap();
    let b = env
        .ctl()
        .session_exec(&exec_def(&id, "true"))
        .await
        .unwrap();
    assert_eq!(a.exec.as_str(), "e-000000");
    assert_eq!(b.exec.as_str(), "e-000001");
    assert_eq!(env.ctl().exec_list(&id).await.unwrap().len(), 2);
}

#[tokio::test]
async fn stderr_is_reported_separately_from_stdout() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let r = env
        .ctl()
        .session_exec(&exec_def(&id, "echo O; echo E 1>&2"))
        .await
        .unwrap();
    let mut stream = env.ctl().session_attach(&r).await.unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    while let Some(pkt) = stream.next().await {
        match pkt {
            StreamPacket::Output { stream, data, .. } => match stream {
                StdStream::Stdout => out.extend_from_slice(&data),
                StdStream::Stderr => err.extend_from_slice(&data),
                StdStream::Stdin => {}
            },
            StreamPacket::ExecExit { .. } => break,
            StreamPacket::NodeExit { .. } => {}
        }
    }
    assert_eq!(String::from_utf8_lossy(&out).trim(), "O");
    assert_eq!(String::from_utf8_lossy(&err).trim(), "E");
}

#[tokio::test]
async fn attaching_after_an_exec_finished_still_replays_it() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let r = env
        .ctl()
        .session_exec(&exec_def(&id, "echo done; exit 5"))
        .await
        .unwrap();

    // Let it finish with nobody attached.
    for _ in 0..300 {
        if env.ctl().exec_status(&r).await.unwrap().ended {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(env.ctl().exec_status(&r).await.unwrap().ended);

    let mut stream = env.ctl().session_attach(&r).await.unwrap();
    let mut out = Vec::new();
    let mut exit = None;
    while let Some(pkt) = stream.next().await {
        match pkt {
            StreamPacket::Output { data, .. } => out.extend_from_slice(&data),
            StreamPacket::ExecExit { exit_code } => {
                exit = Some(exit_code);
                break;
            }
            StreamPacket::NodeExit { .. } => {}
        }
    }
    assert_eq!(exit, Some(5), "a late attach must still learn the exit code");
    assert!(String::from_utf8_lossy(&out).contains("done"));
}

#[tokio::test]
async fn several_clients_can_attach_to_one_exec() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let r = env
        .ctl()
        .session_exec(&exec_def(
            &id,
            "for i in 1 2 3; do echo line$i; sleep 0.05; done",
        ))
        .await
        .unwrap();

    let mut tasks = Vec::new();
    for _ in 0..4 {
        let stream = env.ctl().session_attach(&r).await.unwrap();
        tasks.push(tokio::spawn(async move {
            let mut stream = stream;
            let mut out = Vec::new();
            while let Some(pkt) = stream.next().await {
                match pkt {
                    StreamPacket::Output { data, .. } => out.extend_from_slice(&data),
                    StreamPacket::ExecExit { exit_code } => {
                        return (exit_code, String::from_utf8_lossy(&out).into_owned());
                    }
                    StreamPacket::NodeExit { .. } => {}
                }
            }
            (i32::MIN, String::from_utf8_lossy(&out).into_owned())
        }));
    }
    for task in tasks {
        let (code, out) = task.await.unwrap();
        assert_eq!(code, 0);
        for i in 1..=3 {
            assert!(out.contains(&format!("line{i}")), "{out:?}");
        }
    }
}

#[tokio::test]
async fn signalling_an_exec_terminates_it() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let r = env
        .ctl()
        .session_exec(&exec_def(&id, "sleep 300"))
        .await
        .unwrap();
    // Wait for the process to actually exist before signalling it.
    let pid = await_pid(env.ctl(), &r).await;
    env.ctl().exec_signal(&r, libc::SIGTERM).await.unwrap();

    await_ended(env.ctl(), &r).await;
    assert_eq!(
        env.ctl().exec_status(&r).await.unwrap().exit_code,
        Some(128 + libc::SIGTERM)
    );
    assert!(
        mirage_supervisor::process::wait_gone(pid, Duration::from_secs(5)).await,
        "the signalled process must be reaped"
    );
}

#[tokio::test]
async fn an_invalid_signal_is_rejected_rather_than_dropped() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let r = env
        .ctl()
        .session_exec(&exec_def(&id, "sleep 5"))
        .await
        .unwrap();
    let err = env.ctl().exec_signal(&r, 9999).await.unwrap_err();
    assert!(err.to_string().contains("invalid signal"), "{err}");
    env.ctl().exec_remove(&r).await.unwrap();
}

#[tokio::test]
async fn removing_a_running_exec_kills_its_processes() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let r = env
        .ctl()
        .session_exec(&exec_def(&id, "sleep 300"))
        .await
        .unwrap();
    let pid = await_pid(env.ctl(), &r).await;

    env.ctl().exec_remove(&r).await.unwrap();

    assert!(
        mirage_supervisor::process::wait_gone(pid, Duration::from_secs(10)).await,
        "removing an exec must not leave its process running"
    );
    assert!(matches!(
        env.ctl().exec_status(&r).await,
        Err(MirageError::ExecNotFound(_))
    ));
}

#[tokio::test]
async fn stdin_reaches_the_workload() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let mut def = exec_def(&id, "");
    def.exec.command = "/bin/cat".to_string();
    def.exec.args = vec![];
    let r = env.ctl().session_exec(&def).await.unwrap();

    let mut stream = env.ctl().session_attach(&r).await.unwrap();
    env.ctl()
        .session_stdin(&r, b"round trip\n")
        .await
        .unwrap();

    // `cat` only exits on EOF; ending the exec is what closes stdin, so
    // read the echoed line and then terminate it.
    let mut out = Vec::new();
    while let Some(pkt) = stream.next().await {
        if let StreamPacket::Output { data, .. } = pkt {
            out.extend_from_slice(&data);
            if out.ends_with(b"\n") {
                break;
            }
        }
    }
    assert_eq!(String::from_utf8_lossy(&out), "round trip\n");
    env.ctl().exec_remove(&r).await.unwrap();
}

#[tokio::test]
async fn an_exec_whose_command_does_not_exist_still_terminates() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let mut def = exec_def(&id, "");
    def.exec.command = "definitely-not-a-real-binary".to_string();
    def.exec.args = vec![];
    let r = env.ctl().session_exec(&def).await.unwrap();

    // The failure mode this guards against is an exec that never reaches
    // a terminal state, which hangs every attached client forever.
    let mut stream = tokio::time::timeout(Duration::from_secs(30), async {
        env.ctl().session_attach(&r).await.unwrap()
    })
    .await
    .unwrap();
    let mut saw_exit = None;
    let mut stderr = Vec::new();
    while let Some(pkt) = tokio::time::timeout(Duration::from_secs(30), stream.next())
        .await
        .expect("a failed exec must still produce an exit")
    {
        match pkt {
            StreamPacket::Output { data, .. } => stderr.extend_from_slice(&data),
            StreamPacket::ExecExit { exit_code } => {
                saw_exit = Some(exit_code);
                break;
            }
            StreamPacket::NodeExit { .. } => {}
        }
    }
    assert_eq!(saw_exit, Some(127));
    assert!(
        String::from_utf8_lossy(&stderr).contains("command not found"),
        "{:?}",
        String::from_utf8_lossy(&stderr)
    );
}

#[tokio::test]
async fn execs_in_a_session_run_concurrently() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let started = std::time::Instant::now();
    let defs: Vec<ExecDef> = (0..5).map(|_| exec_def(&id, "sleep 0.5")).collect();
    let refs: Vec<ExecRef> =
        futures::future::join_all(defs.iter().map(|d| env.ctl().session_exec(d)))
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect();
    for r in &refs {
        await_ended(env.ctl(), r).await;
    }
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "five 0.5s execs took {:?}; they are running in sequence",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_multi_node_topology_runs_one_process_per_node() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let mut profile = test_profile();
    profile.emulator.topology = MaybeRef::Owned(TopologyDef {
        num_nodes: 3,
        gpus_per_node: 1,
        agent: MaybeRef::Ref("MI350X".to_string()),
    });
    let def = env
        .ctl()
        .session_create(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(profile),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .await
        .unwrap();
    env.ctl()
        .session_wait_ready(&def.id, Duration::from_secs(30))
        .await
        .unwrap();

    let r = env
        .ctl()
        .session_exec(&exec_def(&def.id, "echo rank $MIRAGE_RANK"))
        .await
        .unwrap();
    await_ended(env.ctl(), &r).await;
    let status = env.ctl().exec_status(&r).await.unwrap();
    assert_eq!(status.nodes.len(), 3, "one process per node");
    assert_eq!(status.exit_code, Some(0));
}

#[tokio::test]
async fn nproc_per_node_multiplies_the_process_grid() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let mut profile = test_profile();
    profile.emulator.topology = MaybeRef::Owned(TopologyDef {
        num_nodes: 2,
        gpus_per_node: 2,
        agent: MaybeRef::Ref("MI350X".to_string()),
    });
    let def = env
        .ctl()
        .session_create(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(profile),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .await
        .unwrap();
    env.ctl()
        .session_wait_ready(&def.id, Duration::from_secs(30))
        .await
        .unwrap();

    let mut edef = exec_def(&def.id, "echo $RANK/$WORLD_SIZE/$LOCAL_RANK");
    edef.nproc_per_node = 2;
    let r = env.ctl().session_exec(&edef).await.unwrap();
    await_ended(env.ctl(), &r).await;
    assert_eq!(
        env.ctl().exec_status(&r).await.unwrap().nodes.len(),
        4,
        "2 nodes x 2 procs"
    );
}

// ---- teardown --------------------------------------------------------------

#[tokio::test]
async fn a_session_forgets_its_oldest_finished_execs() {
    // The daemon is long-lived, and each finished exec retains its output
    // for replay. Without a bound, a session running execs in a loop grows
    // forever — a leak that only shows up after hours of use.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let manager = Arc::new(SessionManager::new(mirage_supervisor::ManagerConfig {
        replay_bytes: 4096,
        max_finished_execs: 5,
    }));
    let def = manager
        .session_create(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(test_profile()),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .await
        .unwrap();
    manager
        .session_wait_ready(&def.id, Duration::from_secs(30))
        .await
        .unwrap();

    for _ in 0..20 {
        let r = manager.session_exec(&exec_def(&def.id, "true")).await.unwrap();
        await_ended(&manager, &r).await;
    }

    let remaining = manager.exec_list(&def.id).await.unwrap();
    assert!(
        remaining.len() <= 6,
        "20 finished execs left {} behind; history is unbounded",
        remaining.len()
    );
    // The most recent are the ones kept — those are what a user asks about.
    assert!(
        remaining.iter().any(|id| id.as_str() == "e-000019"),
        "the newest exec must be retained: {remaining:?}"
    );

    manager.session_destroy(&def.id).await.unwrap();
    drop(env);
}

#[tokio::test]
async fn a_running_exec_is_never_forgotten_however_old() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let manager = Arc::new(SessionManager::new(mirage_supervisor::ManagerConfig {
        replay_bytes: 4096,
        max_finished_execs: 2,
    }));
    let def = manager
        .session_create(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(test_profile()),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .await
        .unwrap();
    manager
        .session_wait_ready(&def.id, Duration::from_secs(30))
        .await
        .unwrap();

    // Start a long-running exec first, then bury it under finished ones.
    let long = manager
        .session_exec(&exec_def(&def.id, "sleep 300"))
        .await
        .unwrap();
    for _ in 0..10 {
        let r = manager.session_exec(&exec_def(&def.id, "true")).await.unwrap();
        await_ended(&manager, &r).await;
    }

    // Evicting it would leave its processes running with nothing tracking
    // them — the exact leak this design exists to prevent.
    assert!(
        manager.exec_status(&long).await.is_ok(),
        "a running exec was forgotten"
    );

    manager.session_destroy(&def.id).await.unwrap();
    drop(env);
}

#[tokio::test]
async fn destroying_a_session_kills_its_running_execs() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let tag = marker("destroy-kills");
    let script = tagged_sleep(&tag);
    let mut pids = Vec::new();
    for _ in 0..3 {
        let r = env
            .ctl()
            .session_exec(&exec_def(&id, &script))
            .await
            .unwrap();
        pids.push(await_pid(env.ctl(), &r).await);
    }

    env.ctl().session_destroy(&id).await.unwrap();

    // `session_destroy` returning means teardown finished, so this must
    // already be true rather than eventually true.
    for pid in pids {
        assert!(
            !mirage_supervisor::process::process_alive(pid),
            "pid {pid} still alive after session_destroy returned"
        );
    }
}

#[tokio::test]
async fn destroying_a_session_kills_the_whole_process_tree() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("grandchild.pid");
    let script = format!(
        "sh -c 'echo $$ > {m}; while true; do sleep 1; done' & \
         while true; do sleep 1; done",
        m = marker.display()
    );
    let r = env.ctl().session_exec(&exec_def(&id, &script)).await.unwrap();
    let _ = await_pid(env.ctl(), &r).await;

    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let grandchild = loop {
        if let Ok(s) = std::fs::read_to_string(&marker)
            && let Ok(pid) = s.trim().parse::<u32>()
        {
            break pid;
        }
        assert!(std::time::Instant::now() < deadline, "grandchild never ran");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };

    env.ctl().session_destroy(&id).await.unwrap();

    assert!(
        mirage_supervisor::process::wait_gone(grandchild, Duration::from_secs(10)).await,
        "grandchild {grandchild} outlived its session"
    );
}

#[tokio::test]
async fn a_shutdown_requested_before_anyone_waits_is_still_honoured() {
    // Regression: the shutdown signal used to be an edge-triggered
    // `Notify`, so a request arriving before the daemon started waiting
    // (or in the window between its flag check and its await) was lost,
    // and `mirage daemon stop` would hang.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.ctl().daemon_shutdown().await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), env.manager.wait_for_shutdown())
        .await
        .expect("a shutdown requested earlier must still be observed");
    assert!(env.manager.is_shutting_down());
}

#[tokio::test]
async fn many_waiters_all_observe_a_single_shutdown_request() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let waiters: Vec<_> = (0..8)
        .map(|_| {
            let manager = env.manager.clone();
            tokio::spawn(async move { manager.wait_for_shutdown().await })
        })
        .collect();
    tokio::time::sleep(Duration::from_millis(50)).await;
    env.ctl().daemon_shutdown().await.unwrap();
    for w in waiters {
        tokio::time::timeout(Duration::from_secs(5), w)
            .await
            .expect("every waiter must be woken")
            .unwrap();
    }
}

#[tokio::test]
async fn no_session_can_be_created_once_shutdown_is_requested() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.ctl().daemon_shutdown().await.unwrap();
    let err = env
        .ctl()
        .session_create(CreateSessionRequest {
            id: None,
            profile: MaybeRef::Owned(test_profile()),
            workdir: "/tmp".to_string(),
            daemon: false,
        })
        .await
        .unwrap_err();
    // Accepting one would start processes nothing is going to tear down.
    assert!(err.to_string().contains("shutting down"), "{err}");
}

#[tokio::test]
async fn shutdown_all_tears_down_every_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let mut pids = Vec::new();
    for i in 0..4 {
        let id = env.session_named(Some(&format!("s{i}"))).await;
        let r = env
            .ctl()
            .session_exec(&exec_def(&id, "sleep 300"))
            .await
            .unwrap();
        pids.push(await_pid(env.ctl(), &r).await);
    }

    env.manager.shutdown_all().await;

    assert!(env.ctl().session_list().await.unwrap().is_empty());
    for pid in pids {
        assert!(
            !mirage_supervisor::process::process_alive(pid),
            "pid {pid} survived shutdown_all"
        );
    }
}

#[tokio::test]
async fn no_exec_can_start_into_a_session_being_destroyed() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let id = env.session().await;

    let tag = marker("destroy-race");

    // Race a burst of exec submissions against a destroy. Every
    // submission must either succeed before teardown collected its list
    // (and then be killed by it) or be refused; a process must never be
    // left running.
    let ctl = env.manager.clone();
    let sid = id.clone();
    let script = tagged_sleep(&tag);
    let submitter = tokio::spawn(async move {
        let mut refs = Vec::new();
        for _ in 0..40 {
            match ctl.session_exec(&exec_def(&sid, &script)).await {
                Ok(r) => refs.push(r),
                // Refused: either the session is gone or it is tearing
                // down. Both are correct outcomes.
                Err(_) => break,
            }
        }
        refs
    });

    tokio::time::sleep(Duration::from_millis(30)).await;
    env.ctl().session_destroy(&id).await.unwrap();
    let submitted = submitter.await.unwrap();

    // Whatever got in must be dead now.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let leaked = count_tagged(&tag);
    assert_eq!(
        leaked, 0,
        "{leaked} tagged process(es) leaked from {} submissions",
        submitted.len()
    );
}

// ---- config passthrough ----------------------------------------------------

#[tokio::test]
async fn profiles_round_trip_through_the_manager() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    assert!(env.ctl().profile_list().await.unwrap().is_empty());

    let mut p = test_profile();
    p.name = "Mixed-Case".to_string();
    env.ctl().profile_put(&p).await.unwrap();

    // Names are case-insensitive and stored lowercase.
    assert_eq!(env.ctl().profile_list().await.unwrap(), vec!["mixed-case"]);
    assert_eq!(
        env.ctl().profile_get("MIXED-CASE").await.unwrap().name,
        "mixed-case"
    );

    env.ctl().profile_delete("mixed-case").await.unwrap();
    assert!(env.ctl().profile_list().await.unwrap().is_empty());
    assert!(matches!(
        env.ctl().profile_get("mixed-case").await,
        Err(MirageError::ProfileNotFound(_))
    ));
}

#[tokio::test]
async fn deleting_a_missing_profile_reports_not_found() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let err = env.ctl().profile_delete("ghost").await.unwrap_err();
    assert!(matches!(err, MirageError::ProfileNotFound(_)), "{err:?}");
}

#[tokio::test]
async fn exec_lookups_on_a_missing_session_report_the_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let r = ExecRef {
        session: SessionId::new("ghost").unwrap(),
        exec: ExecId::from_counter(0),
    };
    assert!(matches!(
        env.ctl().exec_status(&r).await,
        Err(MirageError::SessionNotFound(_))
    ));
    assert!(matches!(
        env.ctl().session_attach(&r).await,
        Err(MirageError::SessionNotFound(_))
    ));
}

// ---- helpers ---------------------------------------------------------------

/// Wait for an exec's rank 0 to report a pid.
async fn await_pid(ctl: &SessionManager, r: &ExecRef) -> u32 {
    for _ in 0..500 {
        if let Ok(status) = ctl.exec_status(r).await
            && let Some(pid) = status.nodes.values().find_map(|n| n.pid)
        {
            return pid;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("exec never reported a pid");
}

/// Wait for an exec to reach a terminal state.
async fn await_ended(ctl: &SessionManager, r: &ExecRef) {
    for _ in 0..1500 {
        match ctl.exec_status(r).await {
            Ok(status) if status.ended => return,
            // Removed underneath us counts as ended.
            Err(_) => return,
            Ok(_) => {}
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("exec never ended");
}

/// A marker string unique to one test, for tagging its workloads.
///
/// Leak checks look at the real process table, so they must match only
/// the processes the test itself started. A generic pattern like
/// `"sleep 300"` also matches a sibling test's workloads (and the
/// harness's own command line), which turns a correct run into a
/// spurious "leak".
fn marker(name: &str) -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    format!(
        "mirage-mgr-{name}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    )
}

/// A shell snippet that sleeps forever, tagged so the test can find it.
fn tagged_sleep(marker: &str) -> String {
    format!("MARKER={marker}; while true; do sleep 1; done")
}

/// Count this user's processes whose command line contains `marker`.
///
/// This is the direct check on the process table: a leak is a process
/// that is still there, and no amount of internal bookkeeping is evidence
/// against it.
fn count_tagged(marker: &str) -> usize {
    let Ok(output) = std::process::Command::new("pgrep")
        .args(["-u", &nix::unistd::getuid().to_string(), "-f", marker])
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
