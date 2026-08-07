//! End-to-end tests for the `mirage` CLI.
//!
//! Each test drives the real binary as a subprocess against a private
//! XDG root, so what is exercised is the whole stack: CLI → session
//! bring-up → supervisor → real processes. There is no daemon in that
//! list any more: `mirage run` *is* the runtime, and a session exists
//! exactly as long as the command that created it.
//!
//! That collapses most of what this suite used to check. A session
//! cannot be started, listed, shown or stopped on its own, so there are
//! no tests for it; what remains is the observable contract — a run's
//! streams, its exit code, the socket it serves while it lives, and
//! `mirage exec` borrowing that session from another terminal.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod harness;

use std::time::Duration;

use harness::{
    Env, TEST_EMULATOR, assert_no_leaks, marker, skip_without_emulator, tagged_sleep, wait_for,
};
use nix::sys::signal::Signal;

#[test]
fn paths_reports_the_overridden_directories() {
    let env = Env::new();
    let out = env.ok(&["paths"]);
    assert!(out.contains("config:"), "{out}");
    assert!(out.contains(env.root().to_str().unwrap()), "{out}");
    // Where a run publishes itself is part of the layout users need when
    // something goes wrong: it is how they see which runs are live.
    assert!(out.contains("runs:"), "{out}");
    assert!(
        out.contains(env.run_socket_dir().to_str().unwrap()),
        "{out}"
    );
    // And there is no single well-known socket to print any more. One
    // would imply a process listening on it that outlives every command,
    // which is precisely what mirage no longer has.
    assert!(
        !out.contains("socket:"),
        "paths still advertises a daemon socket: {out}"
    );
}

#[test]
fn profile_create_list_show_delete() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p1");

    let list = env.ok(&["profile", "list"]);
    assert!(list.lines().any(|l| l.trim() == "p1"), "{list}");

    let shown = env.ok(&["profile", "show", "p1"]);
    let parsed: serde_json::Value = serde_json::from_str(&shown).unwrap();
    assert_eq!(parsed["name"], "p1");
    assert_eq!(parsed["emulator"]["emulator"], TEST_EMULATOR);

    env.ok(&["profile", "delete", "p1", "--force"]);
    let list = env.ok(&["profile", "list"]);
    assert!(!list.lines().any(|l| l.trim() == "p1"), "{list}");
}

#[test]
fn topology_create_show_delete() {
    // Pure configuration: no session, no processes, and therefore no
    // emulator runtime needed to exercise it.
    let env = Env::new();
    env.ok(&[
        "topology",
        "create",
        "t1",
        "--num-nodes",
        "2",
        "--gpus-per-node",
        "4",
    ]);

    let shown: serde_json::Value =
        serde_json::from_str(&env.ok(&["topology", "show", "t1"])).unwrap();
    assert_eq!(shown["num_nodes"], 2);
    assert_eq!(shown["gpus_per_node"], 4);

    env.ok(&["topology", "delete", "t1", "--force"]);
    let list = env.ok(&["topology", "list"]);
    assert!(!list.lines().any(|l| l.trim() == "t1"), "{list}");
}

#[test]
fn the_builtin_agents_are_unpacked_on_first_use() {
    // Every invocation writes any missing builtins, so a fresh machine
    // has agents to build a profile from without being told to run
    // `mirage state builtins` first.
    let env = Env::new();
    let list = env.ok(&["agent", "list"]);
    assert!(
        list.lines().any(|l| l.trim() == "mi350x"),
        "no builtin agent was materialised: {list}"
    );
    // Names are stored case-insensitively, so a profile referring to
    // `MI350X` resolves the document listed above.
    let shown: serde_json::Value =
        serde_json::from_str(&env.ok(&["agent", "show", "MI350X"])).unwrap();
    assert!(shown.is_object(), "{shown}");
}

#[test]
fn run_streams_output_and_propagates_the_exit_code() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let out = env.run(&[
        "run",
        "--profile",
        "p",
        "--",
        "/bin/sh",
        "-c",
        "echo to-stdout; echo to-stderr 1>&2; exit 42",
    ]);
    assert_eq!(out.status.code(), Some(42));

    // stdout and stderr must arrive on the matching streams. Under the
    // previous pseudo-terminal design they were merged into one, so
    // redirecting either captured both.
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stdout.contains("to-stdout"), "stdout was: {stdout}");
    assert!(!stdout.contains("to-stderr"), "stderr leaked into stdout: {stdout}");
    assert!(stderr.contains("to-stderr"), "stderr was: {stderr}");
    // And byte-exact: a single-process job's streams *are* this
    // command's, so mirage never sees the bytes and cannot decorate them.
    assert!(
        !stdout.contains("[0]"),
        "a single-process job's output must not be labelled: {stdout}"
    );
}

#[test]
fn a_multi_node_run_labels_every_rank_without_being_asked() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    // Automatic, not a flag: with several nodes writing to one terminal
    // at once, unlabelled output says nothing about which rank produced
    // which line, so there is no version of this a user would want. The
    // shape of the job decides, the way `docker compose up` does.
    let out = env.ok(&[
        "run",
        "--profile",
        "p",
        "--num-nodes",
        "2",
        "--",
        "/bin/sh",
        "-c",
        "echo hello",
    ]);
    assert!(out.contains("[0] hello"), "{out}");
    assert!(out.contains("[1] hello"), "{out}");
}

#[test]
fn captured_output_is_complete_for_a_process_that_writes_and_exits_immediately() {
    // The narrow window: output travels pump -> channel -> printer, and
    // if the exit is observed before the last bytes are drained, the
    // command returns having printed nothing. Repeat, because it only
    // shows up when the printer is scheduled late.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    for i in 0..6 {
        let out = env.ok(&[
            "run",
            "--profile",
            "p",
            "--",
            "/bin/sh",
            "-c",
            &format!("echo quick-{i}"),
        ]);
        assert!(
            out.contains(&format!("quick-{i}")),
            "round {i}: output was lost; got {out:?}"
        );
    }
}

#[test]
fn a_run_serves_a_socket_only_while_it_is_alive() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    // The workload announces itself, so the interrupt below lands on a
    // run that is fully up rather than one still starting: mirage
    // installs its signal handler around the workload, and a signal
    // arriving before that would kill it outright — which is a real
    // behaviour, but not the one this test is about.
    let started = env.root().join("serving");
    let script = format!("touch {}; sleep 300", started.display());
    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &script]);
    let id = run.await_ready(Duration::from_secs(90));
    assert_eq!(env.live_runs(), vec![id.clone()]);
    wait_for("the workload to start", Duration::from_secs(30), || {
        started.exists()
    });

    // Ctrl-C, as a user would. The socket is the only advertisement a
    // session has, so leaving one behind would mean `mirage exec` offers
    // a session that no longer exists — and, with no argument, silently
    // picks it.
    run.signal(Signal::SIGINT);
    run.wait(Duration::from_secs(30));

    assert!(
        env.live_runs().is_empty(),
        "the socket for {id} outlived its run: {:?}",
        env.live_runs()
    );
    assert!(!env.run_socket_dir().join(format!("{id}.sock")).exists());
}

#[test]
fn interrupting_a_run_takes_its_workload_with_it() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("interrupt");
    let started = env.root().join("workload-started");

    let script = format!("touch {}; {}", started.display(), tagged_sleep(&tag));
    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &script]);
    run.await_ready(Duration::from_secs(90));
    wait_for("the workload to start", Duration::from_secs(30), || {
        started.exists()
    });

    // A workload leads its own process group, so the terminal's Ctrl-C
    // never reaches it: mirage catches the signal and forwards it. Losing
    // that forwarding leaves the workload running with nothing supervising
    // it, still holding the emulated device.
    run.signal(Signal::SIGINT);
    run.wait(Duration::from_secs(30));
    assert_no_leaks(&tag);
}

#[test]
fn a_live_run_can_be_exec_ed_into_from_another_terminal() {
    // The one thing that survives the daemon's removal: while a run is
    // up, a second terminal can start processes in its session.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", "sleep 300"]);
    let id = run.await_ready(Duration::from_secs(90));

    let out = env.run(&["exec", "--session", &id, "--", "/bin/sh", "-c", "exit 5"]);
    assert_eq!(out.status.code(), Some(5));

    // The exec's process is this command's own child, in this terminal,
    // so its output comes back here and not to the run's terminal.
    let out = env.ok(&["exec", "--session", &id, "--", "/bin/echo", "from-exec"]);
    assert!(out.contains("from-exec"), "{out}");
}

#[test]
fn exec_picks_the_only_live_run_when_no_session_is_named() {
    // One terminal running the job and another exec'ing into it is the
    // whole workflow; making the user copy a session id for it would be
    // friction with no purpose.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", "sleep 300"]);
    run.await_ready(Duration::from_secs(90));

    let out = env.ok(&["exec", "--", "/bin/echo", "guessed"]);
    assert!(out.contains("guessed"), "{out}");
}

#[test]
fn a_run_waits_for_a_borrower_before_tearing_its_session_down() {
    // The ordering invariant `Session::teardown` documents for itself,
    // observed from outside. `mirage exec` starts its workload in its own
    // process, so the run cannot see it — and teardown stops the emulator
    // daemon, runs the backend's shutdown hook and deletes the scratch
    // directory that workload is reading. Before the lease, a
    // `mirage run -- sleep 1` beside a longer exec did exactly that.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("borrower");

    // The run's own command is short; the borrowed one is not.
    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", "sleep 1"]);
    let id = run.await_ready(Duration::from_secs(90));

    let mut borrower = std::process::Command::new(env.bin())
        .args(["exec", "--session", &id, "--", "/bin/sh", "-c", &tagged_sleep(&tag)])
        .envs(env.child_env())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    wait_for("the borrowed workload to start", Duration::from_secs(30), || {
        harness::count_processes(&tag) > 0
    });

    // Well past the run's own one-second command. The session has to
    // still be whole: the run is waiting, not tearing down.
    std::thread::sleep(Duration::from_secs(4));
    assert!(
        env.session_scratch(&id).exists(),
        "the run removed its scratch directory while a borrower was using it"
    );
    assert_eq!(
        env.live_runs(),
        vec![id.clone()],
        "the run stopped serving while a borrower was still attached"
    );
    assert!(
        harness::pid_alive(run.pid().expect("the run is still up")),
        "the run exited while a borrower was still attached"
    );

    // The borrower finishing is what releases the session. Asked to stop
    // rather than `SIGKILL`ed: a killed client cannot tear its own
    // workload down, which is a real property but a different test's.
    let _ = nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(borrower.id() as i32),
        Signal::SIGTERM,
    );
    assert!(
        wait_for_exit(&mut borrower, Duration::from_secs(60)),
        "the borrower did not stop when asked"
    );
    run.wait(Duration::from_secs(60));

    assert_no_leaks(&tag);
    assert!(
        !env.session_scratch(&id).exists(),
        "the session was never torn down after its last borrower left"
    );
}

#[test]
fn interrupting_a_waiting_run_tears_down_and_tells_the_borrower() {
    // The override. Waiting is unbounded, so there has to be a way out —
    // and taking it must still tell the borrower, rather than removing
    // the session out from under it and leaving it to find out by I/O
    // error.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("borrower-interrupt");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", "sleep 1"]);
    let id = run.await_ready(Duration::from_secs(90));

    let mut borrower = std::process::Command::new(env.bin())
        .args(["exec", "--session", &id, "--", "/bin/sh", "-c", &tagged_sleep(&tag)])
        .envs(env.child_env())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    wait_for("the borrowed workload to start", Duration::from_secs(30), || {
        harness::count_processes(&tag) > 0
    });

    // Let the run reach its wait, then decline to wait any longer.
    std::thread::sleep(Duration::from_secs(3));
    run.signal(Signal::SIGINT);
    let out = run.wait(Duration::from_secs(60));
    let text = String::from_utf8_lossy(&out.stderr);
    assert!(
        text.contains("borrower"),
        "a run that waited must say what it was waiting for:\n{text}"
    );

    // The borrower is told, stops its own workload, and exits — rather
    // than being left running against a session that no longer exists.
    let done = wait_for_exit(&mut borrower, Duration::from_secs(60));
    assert!(
        done,
        "the borrower was never told its session had gone and is still running"
    );
    assert_no_leaks(&tag);
}

/// Wait for a child to exit, returning whether it did.
fn wait_for_exit(child: &mut std::process::Child, timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) if std::time::Instant::now() < deadline => {
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

#[test]
fn exec_builds_the_same_process_grid_as_the_run() {
    // `exec` does not ask the run to start anything: it fetches the
    // session description and builds the specs itself, with the same
    // `build_specs` the run uses. If the two ever diverged, a command
    // would behave differently depending on which terminal it was
    // started from.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let mut run = env.spawn_run(
        &["--profile", "p", "--num-nodes", "2"],
        &["/bin/sh", "-c", "sleep 300"],
    );
    run.await_ready(Duration::from_secs(90));

    let out = env.ok(&[
        "exec",
        "--",
        "/bin/sh",
        "-c",
        "echo rank-$MIRAGE_RANK/$WORLD_SIZE",
    ]);
    assert!(out.contains("[0] rank-0/2"), "{out}");
    assert!(out.contains("[1] rank-1/2"), "{out}");
}

#[test]
fn exec_without_a_live_run_says_so() {
    // A session only exists while its `mirage run` does, which is
    // surprising if you came from the daemon. The error has to say it
    // rather than report a missing file.
    let env = Env::new();
    let err = env.fails(&["exec", "--", "/bin/true"]);
    assert!(
        err.contains("no `mirage run` is running"),
        "an exec with nothing to attach to must explain why: {err}"
    );
}

#[test]
fn an_invalid_session_id_is_rejected() {
    let env = Env::new();
    let err = env.fails(&["exec", "--session", "../escape", "--", "/bin/true"]);
    // Asserting on the rejection itself, not on a substring that anything
    // could satisfy: the previous `err.contains("invalid") ||
    // err.contains("id")` was really just the second arm, because
    // "invalid" *ends* in "id" — and "id" appears in almost any message
    // this command can produce, including "no such session id". A test
    // that passes when path validation is deleted is not testing it.
    assert!(
        err.contains("invalid"),
        "an id that could escape the runtime directory must be rejected as invalid: {err}"
    );
    // And nothing may have been created outside the runtime root.
    assert!(
        !env.runtime().parent().is_some_and(|p| p.join("escape").exists()),
        "`../escape` must not have resolved to a path outside {}",
        env.runtime().display()
    );
}

#[test]
fn json_output_is_parseable() {
    let env = Env::new();

    let profiles: serde_json::Value =
        serde_json::from_str(&env.ok(&["--json", "profile", "list"])).unwrap();
    assert!(profiles.is_array(), "{profiles}");

    let paths: serde_json::Value = serde_json::from_str(&env.ok(&["--json", "paths"])).unwrap();
    assert_eq!(
        paths["runs"],
        env.run_socket_dir().to_string_lossy().to_string()
    );
}

#[test]
fn a_run_on_a_missing_profile_fails_clearly() {
    let env = Env::new();
    let err = env.fails(&["run", "--profile", "nope", "--", "/bin/true"]);
    assert!(err.contains("profile not found"), "{err}");
    // And no half-created session is left advertising itself.
    assert!(env.live_runs().is_empty(), "{:?}", env.live_runs());
}

#[test]
fn running_a_command_that_does_not_exist_reports_why() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let out = env.run(&["run", "--profile", "p", "--", "definitely-not-a-binary"]);

    // The important property is that it terminates at all: a rank that
    // never started and never reports an exit hangs the command forever.
    assert_eq!(out.status.code(), Some(127));
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(combined.contains("command not found"), "{combined}");
    assert!(combined.contains("definitely-not-a-binary"), "{combined}");
}

#[test]
fn env_flags_reach_the_workload() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let out = env.ok(&[
        "run",
        "--profile",
        "p",
        "--env",
        "MIRAGE_E2E_VALUE=surprise",
        "--",
        "/bin/sh",
        "-c",
        "echo $MIRAGE_E2E_VALUE",
    ]);
    assert_eq!(out.trim(), "surprise");
}

#[test]
fn a_malformed_env_flag_is_rejected() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let err = env.fails(&[
        "run",
        "--profile",
        "p",
        "--env",
        "NO_EQUALS_SIGN",
        "--",
        "/bin/true",
    ]);
    assert!(err.contains("KEY=VALUE"), "{err}");
}

#[test]
fn the_rank_environment_is_injected() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let out = env.ok(&[
        "run",
        "--profile",
        "p",
        "--",
        "/bin/sh",
        "-c",
        "echo $MIRAGE_RANK/$RANK/$WORLD_SIZE/$LOCAL_RANK",
    ]);
    assert_eq!(out.trim(), "0/0/1/0");
}

#[test]
fn a_multi_node_topology_runs_every_node() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let out = env.ok(&[
        "run",
        "--profile",
        "p",
        "--num-nodes",
        "3",
        "--",
        "/bin/sh",
        "-c",
        "echo node-$MIRAGE_RANK",
    ]);
    for rank in 0..3 {
        assert!(out.contains(&format!("node-{rank}")), "{out}");
    }
}

#[test]
fn the_rocjitsu_dropin_shape_routes_to_run() {
    // `mirage [flags] -- ./app` with no subcommand is the upstream
    // `rocjitsu` invocation, and has to keep working so mirage can be
    // dropped into an existing command line unchanged.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let out = env.ok(&["--profile", "p", "--", "/bin/echo", "dropin-ok"]);
    assert!(out.contains("dropin-ok"), "{out}");
}

#[test]
fn state_purge_removes_the_runtime_directory_but_keeps_configuration() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.ok(&["run", "--profile", "p", "--", "/bin/true"]);
    assert!(env.runtime().join("mirage").exists());

    env.ok(&["state", "purge", "--force"]);

    assert!(
        !env.runtime().join("mirage").exists(),
        "the runtime directory survived a purge"
    );
    // Configuration is deliberately left alone without `--all`.
    assert!(
        env.root().join("config/mirage").exists(),
        "purge must not remove profiles unless --all is given"
    );
}

#[test]
fn state_purge_refuses_while_a_run_is_live() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", "sleep 300"]);
    run.await_ready(Duration::from_secs(90));

    // Purge deletes the tree a live run is working inside. It also has no
    // way to stop that run: the run owns its session in its own process,
    // and killing someone else's foreground command from a cleanup
    // subcommand would be a surprise. So it declines and says who is in
    // the way.
    let err = env.fails(&["state", "purge", "--force"]);
    assert!(err.contains("still running"), "{err}");
    assert!(env.runtime().join("mirage").exists());
}

#[test]
fn stdin_reaches_the_workload() {
    use std::io::Write as _;
    use std::process::Stdio;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    // Rank 0 inherits this process's stdin directly — there is no relay
    // and no pseudo-terminal in between, which is what makes `mirage run
    // -- bash` an ordinary interactive shell.
    let mut child = env
        .mirage()
        .args(["run", "--profile", "p", "--", "/bin/cat"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(b"piped through\n").unwrap();
    // Closing our end closes the workload's stdin, which is what makes
    // `cat` exit.
    drop(stdin);

    // Bounded, because the regression this test exists to catch is
    // exactly "the workload's stdin never closed". Collecting the output
    // unbounded would then block forever, hanging the whole e2e binary
    // until the 1800s ctest timeout with nothing saying which test wedged
    // — instead of failing the assertion written for it.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });
    let out = rx
        .recv_timeout(Duration::from_secs(60))
        .expect("`cat` must see EOF when mirage's stdin closes, not block forever")
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("piped through"), "{stdout}");
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the e2e suite going green while every test in it skipped
    // for a missing emulator runtime. See `assert_suite_can_run`.
    harness::assert_suite_can_run();
}
