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

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod harness;

use std::time::Duration;

use harness::{
    Env, TEST_EMULATOR, assert_no_leaks, count_processes, marker, skip_without_emulator,
    tagged_sleep, wait_for, wait_for_exit,
};
use nix::sys::signal::Signal;

/// What a run says on stderr when its own command has finished but a
/// `mirage exec` still holds the session.
///
/// The two borrower tests below wait for this line rather than for a
/// duration: it is the only outward sign that the run has left the
/// "running the workload" state and entered the "holding the session
/// open" one, and both tests are about what happens in the second state.
const BORROWER_WAIT: &str = "borrower(s) are still using session";

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
    assert!(
        !stdout.contains("to-stderr"),
        "stderr leaked into stdout: {stdout}"
    );
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
    let waiting = run.watch_stderr();
    let id = run.await_ready(Duration::from_secs(90));

    let mut borrower = std::process::Command::new(env.bin())
        .args([
            "exec",
            "--session",
            &id,
            "--",
            "/bin/sh",
            "-c",
            &tagged_sleep(&tag),
        ])
        .envs(env.child_env())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    wait_for(
        "the borrowed workload to start",
        Duration::from_secs(30),
        || count_processes(&tag) > 0,
    );

    // Wait for the transition, not for a duration. The run announces
    // that its own command has finished and that it is holding the
    // session open anyway, and that announcement is the moment the
    // invariant below becomes meaningful. A fixed sleep guesses at it
    // from both sides: too short on a loaded machine and the assertions
    // pass because teardown has not been reached *yet*, which proves
    // nothing; longer than needed on an idle one and the suite pays for
    // it on every run.
    wait_for(
        "the run to finish its own command and start waiting for the borrower",
        Duration::from_secs(60),
        || waiting.contains(BORROWER_WAIT),
    );
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
    let waiting = run.watch_stderr();
    let id = run.await_ready(Duration::from_secs(90));

    let mut borrower = std::process::Command::new(env.bin())
        .args([
            "exec",
            "--session",
            &id,
            "--",
            "/bin/sh",
            "-c",
            &tagged_sleep(&tag),
        ])
        .envs(env.child_env())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    wait_for(
        "the borrowed workload to start",
        Duration::from_secs(30),
        || count_processes(&tag) > 0,
    );

    // The interrupt has to land on a run that has actually reached its
    // wait — that is the state whose override is under test. Timing it
    // with a sleep instead would, whenever the machine was slow enough,
    // interrupt the workload phase instead and fail on the assertion
    // below for a reason that has nothing to do with borrowers.
    wait_for(
        "the run to reach its wait for the borrower",
        Duration::from_secs(60),
        || waiting.contains(BORROWER_WAIT),
    );
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
        !env.runtime()
            .parent()
            .is_some_and(|p| p.join("escape").exists()),
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
fn cleanup_leaves_a_run_in_another_runtime_directory_alone() {
    // Two mirages on one machine, each with its own `$XDG_RUNTIME_DIR`:
    // a CI job beside an interactive session, or a test suite beside a
    // developer's. The sockets in `run/` are the whole registry of what
    // is live, so B's registry cannot mention A's session — and a
    // reclamation that goes by session name alone therefore reads A's
    // healthy workload as the wreckage of a crashed run and `SIGKILL`s
    // it. The victim's failure looks like a product bug somewhere else
    // entirely, which is what makes this worth an end-to-end test rather
    // than only a unit one.
    let a = Env::new();
    if skip_without_emulator() {
        return;
    }
    a.create_profile("p");
    let tag = marker("cross-runtime");

    let mut run = a.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(Duration::from_secs(90));
    wait_for("A's workload to start", Duration::from_secs(30), || {
        count_processes(&tag) > 0
    });

    // A second, entirely separate mirage installation.
    let b = Env::new();
    assert!(
        b.live_runs().is_empty(),
        "the second runtime directory must start empty for this to mean anything"
    );
    let out = b.ok(&["cleanup"]);
    assert!(
        !out.contains(&id),
        "cleanup named a session belonging to another runtime directory:\n{out}"
    );

    // The claim `mirage cleanup --help` makes: a session whose run still
    // answers is left completely alone. Alive, still serving, and still
    // able to start a command — the last one is the difference between
    // "the process exists" and "the session works".
    assert!(
        count_processes(&tag) > 0,
        "cleanup under another runtime directory killed a live workload"
    );
    assert_eq!(a.live_runs(), vec![id.clone()]);
    let echoed = a.ok(&["exec", "--session", &id, "--", "/bin/echo", "ALIVE"]);
    assert!(
        echoed.contains("ALIVE"),
        "the run stopped answering after another runtime's cleanup:\n{echoed}"
    );

    // `state purge` reclaims through the same path, so it inherits the
    // same scope: it may empty its own runtime directory and nobody
    // else's.
    b.ok(&["state", "purge", "--force"]);
    assert!(
        count_processes(&tag) > 0,
        "purge under another runtime directory killed a live workload"
    );
    assert_eq!(a.live_runs(), vec![id.clone()]);

    run.signal(Signal::SIGINT);
    run.wait(Duration::from_secs(90));
    assert_no_leaks(&tag);
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

// ---------------------------------------------------------------------------
// Argument handling and the config store
//
// Nothing below needs a session — these are the answers mirage gives
// before it starts anything — which is what makes them worth having: the
// whole group runs in well under a second, and every one of them covers
// a rule that is documented, was recently fixed, or both.
// ---------------------------------------------------------------------------

/// A recognised subcommand before `--` stays that subcommand.
///
/// The drop-in shape (`mirage [flags] -- ./app`, no subcommand) exists so
/// mirage can replace `rocjitsu` on an existing command line. Deciding
/// what counts as "no subcommand" from a hardcoded list of names went
/// wrong the moment one was added or aliased, and `mirage cleanup -- echo
/// hi` — a reasonable thing to type — brought up an *emulated session* to
/// run `echo` instead of cleaning anything up. It is now clap that
/// decides, and this pins the outcome at the binary rather than at the
/// function: the argv rewrite is only correct if the process it produces
/// is.
#[test]
fn a_subcommand_before_a_double_dash_is_not_treated_as_a_drop_in_run() {
    let env = Env::new();
    let out = env.run(&["cleanup", "--", "/bin/echo", "hi-from-a-session"]);

    // `cleanup` takes no positional arguments, so reaching it at all is
    // an argument error — which is the point. The alternative is not
    // "cleanup runs"; it is "a session starts".
    assert_eq!(
        out.status.code(),
        Some(2),
        "a clap rejection exits 2; got {:?}\nstdout: {}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("Usage: mirage cleanup"),
        "the argument error must come from `cleanup`, not from `run`: {combined}"
    );
    // And the command never ran, so nothing was brought up to run it.
    assert!(
        !combined.contains("hi-from-a-session"),
        "`cleanup -- ...` executed its arguments: {combined}"
    );
    assert!(env.live_runs().is_empty(), "{:?}", env.live_runs());
}

/// `mirage about` says what this build is and what it is built from.
///
/// The third-party manifest is a licence obligation, not a nicety: it is
/// generated at build time from the dependency graph, so a change to how
/// it is rendered can empty it without anything failing. Both renderings
/// are checked because they come from two different code paths.
#[test]
fn about_describes_the_build_in_text_and_in_json() {
    let env = Env::new();

    let text = env.ok(&["about"]);
    assert!(text.contains("mirage"), "{text}");
    assert!(
        text.to_lowercase().contains("copyright"),
        "the notice must carry its copyright line: {text}"
    );
    assert!(
        text.contains("third-party crate(s)"),
        "the licence manifest is missing from the text form: {text}"
    );

    let json: serde_json::Value = serde_json::from_str(&env.ok(&["--json", "about"])).unwrap();
    assert_eq!(json["name"], "mirage");
    for key in ["version", "description", "copyright", "license"] {
        assert!(
            json[key].as_str().is_some_and(|v| !v.is_empty()),
            "`about --json` has no {key}: {json}"
        );
    }
    let third_party = json["third_party"]
        .as_array()
        .unwrap_or_else(|| panic!("`third_party` must be an array: {json}"));
    assert!(
        !third_party.is_empty(),
        "mirage has dependencies, so an empty licence manifest is a bug, \
         not a build with none: {json}"
    );
    for entry in third_party {
        for key in ["name", "version", "license"] {
            assert!(
                entry[key].as_str().is_some_and(|v| !v.is_empty()),
                "a licence entry is missing its {key}: {entry}"
            );
        }
    }
}

/// mirage never destroys a configuration document it did not write.
///
/// All three refusals are one rule seen from three sides, so they are
/// asserted together: a name that already exists, a builtin mirage will
/// simply write back, and a name whose spelling would not survive being
/// stored. Each one used to be a silent overwrite, a success that
/// changed nothing, or a document saved under a name the user never
/// typed — failures a user only notices later, by which time their
/// edits are gone.
#[test]
fn mirage_refuses_to_quietly_destroy_a_document_it_did_not_write() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("keeper");

    // 1. An existing document is not overwritten by `create`.
    let err = env.fails(&[
        "profile",
        "create",
        "keeper",
        "--emulator",
        TEST_EMULATOR,
        "--no-input",
    ]);
    assert!(
        err.contains("already exists") && err.contains("will not overwrite"),
        "creating over an existing profile must refuse and say why: {err}"
    );
    assert!(
        err.contains("profile delete keeper"),
        "the refusal must say how to proceed deliberately: {err}"
    );
    // Still exactly one, and still the original.
    env.ok(&["profile", "show", "keeper"]);

    // 2. A pristine builtin is not deleted, because deleting it would be
    //    a lie: the next command writes it straight back.
    let err = env.fails(&["profile", "delete", "mi350x", "--force"]);
    assert!(
        err.contains("builtin"),
        "deleting a pristine builtin must refuse rather than report a \
         success that changes nothing: {err}"
    );
    let list = env.ok(&["profile", "list"]);
    assert!(list.lines().any(|l| l.trim() == "mi350x"), "{list}");

    // 3. A name that would be stored under a different spelling.
    let err = env.fails(&[
        "profile",
        "create",
        "MixedCase",
        "--emulator",
        TEST_EMULATOR,
        "--no-input",
    ]);
    assert!(
        err.contains("mixedcase"),
        "the refusal must name the spelling mirage would have used: {err}"
    );
    let list = env.ok(&["profile", "list"]);
    assert!(
        !list.to_lowercase().contains("mixedcase"),
        "the refused name was stored anyway: {list}"
    );
}

/// A profile's references cannot escape the configuration directory.
///
/// A profile names its agent and topology by string, and those strings
/// become file paths under the config root. `../../` in one is a path
/// traversal with a configuration file for a vector, so it is rejected
/// where it is written rather than where it is read.
#[test]
fn a_profile_reference_that_escapes_the_config_directory_is_refused() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let err = env.fails(&[
        "profile",
        "create",
        "escapee",
        "--emulator",
        TEST_EMULATOR,
        "--agent",
        "../../outside/evil",
        "--no-input",
    ]);
    assert!(
        err.contains("invalid agent name"),
        "a traversing agent reference must be rejected as invalid: {err}"
    );
    assert!(
        env.fails(&["profile", "show", "escapee"])
            .contains("not found"),
        "the profile must not have been written"
    );
    // Nothing may have been created beside the config root either.
    assert!(
        !env.root().join("outside").exists(),
        "`../../outside` resolved to a real path"
    );
}

/// Declining a delete leaves the document alone — and says so.
///
/// Every other test in this suite passes `--force`, which means the
/// prompt itself, the exit status of a decline, and the JSON shape a
/// script would branch on were all untested. A decline is not an error:
/// the user was asked and answered, so the command succeeded at what it
/// was for. Reporting it as a failure would make `|| exit 1` scripts
/// abort on a deliberate "no".
#[test]
fn a_declined_delete_changes_nothing_and_reports_that_it_did_not() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("survivor");

    // stdin is closed, so the prompt reads EOF — which is a "no", the
    // same as an empty line at the `[y/N]` default.
    let out = env
        .mirage()
        .args(["--json", "profile", "delete", "survivor"])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "declining a delete is an answer, not an error: {:?}",
        out.status.code()
    );

    // The prompt is on stderr so that stdout stays exactly one JSON
    // document, which is the whole `--json` contract.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("delete profile survivor?"),
        "the user must be asked before anything is deleted: {stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("stdout must be one JSON document ({e}): {stdout:?}"));
    assert_eq!(json["deleted"], false, "{json}");
    assert_eq!(json["name"], "survivor", "{json}");

    // And the profile is still there.
    env.ok(&["profile", "show", "survivor"]);
}

/// `mirage state builtins` writes what it says it wrote.
///
/// It is the repair command for a config directory somebody has been
/// editing, so its report is the only evidence a user gets. A line
/// naming a path that was not written would send them looking in the
/// wrong place.
#[test]
fn state_builtins_writes_every_shipped_document_and_names_where() {
    let env = Env::new();

    let text = env.ok(&["state", "builtins"]);
    for kind in ["agent", "topology", "profile"] {
        assert!(
            text.contains(kind),
            "no {kind} in the builtins report: {text}"
        );
    }

    let json: serde_json::Value =
        serde_json::from_str(&env.ok(&["--json", "state", "builtins"])).unwrap();
    let entries = json
        .as_array()
        .unwrap_or_else(|| panic!("`state builtins --json` must be an array: {json}"));
    assert!(!entries.is_empty(), "mirage ships builtins: {json}");
    for entry in entries {
        let path = entry["path"]
            .as_str()
            .unwrap_or_else(|| panic!("a builtin entry has no path: {entry}"));
        assert!(
            std::path::Path::new(path).is_file(),
            "`state builtins` named {path}, but nothing is there: {entry}"
        );
        assert!(
            entry["kind"].as_str().is_some_and(|k| !k.is_empty()),
            "{entry}"
        );
        assert!(
            entry["name"].as_str().is_some_and(|n| !n.is_empty()),
            "{entry}"
        );
    }
    // Everything the report claims exists is loadable through the store,
    // not merely present as bytes.
    env.ok(&["agent", "show", "mi350x"]);
    env.ok(&["profile", "show", "mi350x"]);
}

/// `mirage emulators` is how a user finds out what this build can do.
///
/// Only the JSON form was ever exercised (the harness and the matrix
/// suite both probe with it), so the human-readable form — the one
/// anybody actually types — could have rendered nothing at all. The
/// expectations are derived from the JSON rather than hardcoded, so this
/// says the same thing on a host with a GPU and on one without.
#[test]
fn emulators_lists_every_backend_and_marks_the_default() {
    let env = Env::new();
    let json: serde_json::Value = serde_json::from_str(&env.ok(&["--json", "emulators"])).unwrap();
    let rows = json.as_array().expect("emulators --json is an array");
    assert!(
        !rows.is_empty(),
        "a build with no backends at all cannot run anything: {json}"
    );

    let short = env.ok(&["emulators"]);
    assert!(short.contains("NAME"), "no table header: {short}");
    for row in rows {
        let name = row["name"].as_str().unwrap();
        assert!(
            short.contains(name),
            "backend {name} is not listed: {short}"
        );
    }
    if rows.iter().any(|r| r["default"] == true) {
        assert!(
            short.contains('*'),
            "one backend is the default for new profiles and the table must \
             say which: {short}"
        );
    }

    // The long form is what a user reads when a backend is *not* usable,
    // so it has to carry the reason rather than repeat the verdict.
    let long = env.ok(&["emulators", "--long"]);
    for row in rows {
        let name = row["name"].as_str().unwrap();
        assert!(long.contains(name), "{name} missing from -l: {long}");
        if row["support"]["supported"] == false {
            let reason = row["support"]["reason"].as_str().unwrap_or_default();
            assert!(
                !reason.is_empty() && long.contains(reason),
                "{name} is unsupported but `emulators -l` does not say why \
                 (expected {reason:?}):\n{long}"
            );
        }
    }
}

/// An argument mirage cannot honour is diagnosed before anything starts.
///
/// Every case here used to be accepted and then quietly ignored — an
/// unknown emulator option dropped, a `--plugin` that was never loaded,
/// a `--config` that could not be read falling back to no config at all.
/// Silently doing something other than what was asked is the worst of
/// the three possible behaviours, because the run *succeeds* and the
/// result is wrong.
///
/// Checked as a table because it is one property with many instances:
/// mirage says what is wrong with the input, names it, and creates no
/// session on the way to saying so. The expected fragments are the parts
/// of each message that identify the problem — never a whole sentence,
/// which would make this a change-detector for wording.
#[test]
fn an_unusable_argument_is_diagnosed_and_no_session_is_created() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let cases: &[(&[&str], &[&str])] = &[
        // An option the backend does not have.
        (
            &[
                "run",
                "--profile",
                "p",
                "-o",
                "notakey=1",
                "--",
                "/bin/true",
            ],
            &["notakey", "rocjitsu"],
        ),
        // An option that is not `KEY=VALUE` at all.
        (
            &["run", "--profile", "p", "-o", "bare", "--", "/bin/true"],
            &["bare", "KEY=VALUE"],
        ),
        // A plugin this backend does not ship.
        (
            &[
                "run",
                "--profile",
                "p",
                "--plugin",
                "notaplugin",
                "--",
                "/bin/true",
            ],
            &["notaplugin"],
        ),
        // A plugin name that is not a plugin name.
        (
            &[
                "run",
                "--profile",
                "p",
                "--plugin",
                "../evil",
                "--",
                "/bin/true",
            ],
            &["../evil", "invalid plugin name"],
        ),
        // A config file that is not there.
        (
            &[
                "run",
                "--profile",
                "p",
                "--config",
                "/nonexistent/emulator.json",
                "--",
                "/bin/true",
            ],
            &["--config", "/nonexistent/emulator.json"],
        ),
        // A working directory that is not there, and one that is a file.
        (
            &[
                "run",
                "--profile",
                "p",
                "--workdir",
                "/nonexistent/dir",
                "--",
                "/bin/true",
            ],
            &["--workdir", "/nonexistent/dir"],
        ),
        (
            &[
                "run",
                "--profile",
                "p",
                "--workdir",
                "/etc/hostname",
                "--",
                "/bin/true",
            ],
            &["--workdir", "not a directory"],
        ),
        // A backend that does not exist.
        (
            &[
                "run",
                "--profile",
                "p",
                "--emulator",
                "notabackend",
                "--",
                "/bin/true",
            ],
            &["notabackend"],
        ),
        // More processes than mirage will start for one exec. Refused up
        // front rather than after standing up the first few thousand.
        (
            &[
                "run",
                "--profile",
                "p",
                "--num-nodes",
                "4096",
                "--nproc-per-node",
                "64",
                "--",
                "/bin/true",
            ],
            &["4096", "processes"],
        ),
    ];

    for (args, expected) in cases {
        let err = env.fails(args);
        for fragment in *expected {
            assert!(
                err.contains(fragment),
                "`mirage {}` must explain itself and mention {fragment:?}; said: {err}",
                args.join(" ")
            );
        }
        assert!(
            env.live_runs().is_empty(),
            "`mirage {}` was rejected but left a session behind: {:?}",
            args.join(" "),
            env.live_runs()
        );
    }
}

/// Arguments the parser itself rejects exit 2, not 1.
///
/// The exit-code table is part of the CLI contract — 2 means "I did not
/// understand you", 1 means "I understood and could not" — and a script
/// that retries on one but not the other needs the distinction to hold.
/// Nothing asserted it before, so a validation moved from clap into
/// mirage's own code would silently change it.
#[test]
fn arguments_the_parser_rejects_exit_two() {
    let env = Env::new();

    let cases: &[&[&str]] = &[
        // Counts are `range(1..)`: zero is not a small grid, it is no job.
        &[
            "run",
            "--profile",
            "p",
            "--num-nodes",
            "0",
            "--",
            "/bin/true",
        ],
        &[
            "run",
            "--profile",
            "p",
            "--gpus-per-node",
            "0",
            "--",
            "/bin/true",
        ],
        &[
            "run",
            "--profile",
            "p",
            "--nproc-per-node",
            "0",
            "--",
            "/bin/true",
        ],
        // `--config` supplies the whole emulator configuration, so the
        // flags that would build one are refused rather than silently
        // losing to it.
        &[
            "run",
            "--profile",
            "p",
            "--config",
            "/etc/hostname",
            "--gpus-per-node",
            "2",
            "--",
            "/bin/true",
        ],
    ];

    for args in cases {
        let out = env.run(args);
        assert_eq!(
            out.status.code(),
            Some(2),
            "`mirage {}` must be a parse error (exit 2), not a runtime one; \
             got {:?}\nstderr: {}",
            args.join(" "),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        );
    }
    assert!(env.live_runs().is_empty(), "{:?}", env.live_runs());
}

// ---------------------------------------------------------------------------
// Session-backed behaviour with no coverage
// ---------------------------------------------------------------------------

/// Bundled verbosity does not swallow the drop-in command.
///
/// `mirage -vvv -- ./app` has to route to `run` like `mirage -- ./app`
/// does. Deciding which leading arguments are global flags by comparing
/// against a list of spellings meant `-vvv` was not one of them, so the
/// rewrite gave up and the command was never run — and the failure was
/// silent, because the argv it produced was still a valid one. The unit
/// tests cover the rewrite; this covers the process it produces.
#[test]
fn bundled_verbosity_does_not_swallow_a_drop_in_command() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let out = env.run(&[
        "-vvv",
        "--profile",
        "p",
        "--",
        "/bin/echo",
        "verbose-dropin",
    ]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("verbose-dropin"),
        "the workload did not run: {stdout}"
    );
    // And `-vvv` did what it was for, on stderr, where it cannot corrupt
    // the workload's output.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("DEBUG") || stderr.contains("INFO"),
        "`-vvv` produced no diagnostics: {stderr}"
    );
    assert!(
        !stdout.contains("INFO"),
        "log lines leaked into the workload's stdout: {stdout}"
    );
}

/// With several runs live, `mirage exec` asks which one.
///
/// Guessing would be worse than failing: the sessions are other
/// terminals' jobs, and starting a command in the wrong one is not
/// recoverable by retrying. The error has to name the candidates and
/// show the flag, because the session ids are the one thing a user
/// cannot invent.
#[test]
fn exec_asks_which_session_when_several_runs_are_live() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let mut first = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", "sleep 300"]);
    let first_id = first.await_ready(Duration::from_secs(90));
    let mut second = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", "sleep 300"]);
    let second_id = second.await_ready(Duration::from_secs(90));
    assert_ne!(first_id, second_id);

    let err = env.fails(&["exec", "--", "/bin/echo", "which-one"]);
    assert!(
        err.contains(&first_id) && err.contains(&second_id),
        "the error must name every candidate, since they cannot be guessed: {err}"
    );
    assert!(
        err.contains("--session"),
        "the error must show how to choose: {err}"
    );

    // Naming one is still unambiguous, and reaches the one that was named.
    let out = env.ok(&["exec", "--session", &second_id, "--", "/bin/echo", "chosen"]);
    assert!(out.contains("chosen"), "{out}");

    first.signal(Signal::SIGINT);
    second.signal(Signal::SIGINT);
    first.wait(Duration::from_secs(60));
    second.wait(Duration::from_secs(60));
}

/// `--nproc-per-node` gives every local process its own rank.
///
/// This is the flag that makes a single-node run look like `torchrun`,
/// and the environment is the whole interface: PyTorch reads `RANK`,
/// `LOCAL_RANK` and `WORLD_SIZE` and nothing else. Getting `WORLD_SIZE`
/// from the node count alone — the obvious mistake — gives every process
/// rank 0 of 1, and a distributed job silently degrades to N independent
/// copies of itself, each convinced it is alone.
#[test]
fn nproc_per_node_gives_every_local_process_its_own_rank() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let out = env.ok(&[
        "run",
        "--profile",
        "p",
        "--nproc-per-node",
        "3",
        "--",
        "/bin/sh",
        "-c",
        "echo grid $RANK/$LOCAL_RANK/$WORLD_SIZE",
    ]);
    for local in 0..3 {
        assert!(
            out.contains(&format!("grid {local}/{local}/3")),
            "process {local} did not get its own rank out of 3:\n{out}"
        );
    }
    // Several processes writing to one terminal are labelled, for the
    // same reason several nodes are.
    for local in 0..3 {
        assert!(out.contains(&format!("[{local}]")), "{out}");
    }
}

/// `mirage exec --node N` reaches one node of a session, not all of them.
///
/// It is how you get an interactive shell on one node of a grid — the
/// alternative being N shells sharing a terminal, which is unusable. A
/// node the session does not have is an error rather than a fallback to
/// node 0: silently landing somewhere else is how a user debugs the
/// wrong machine for an hour.
#[test]
fn exec_can_target_one_node_of_a_multi_node_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let mut run = env.spawn_run(
        &["--profile", "p", "--num-nodes", "2"],
        &["/bin/sh", "-c", "sleep 300"],
    );
    let id = run.await_ready(Duration::from_secs(90));

    let out = env.ok(&[
        "exec",
        "--session",
        &id,
        "--node",
        "1",
        "--",
        "/bin/sh",
        "-c",
        "echo landed-on-$MIRAGE_RANK",
    ]);
    assert!(
        out.contains("landed-on-1"),
        "`--node 1` did not run on node 1: {out}"
    );
    assert!(
        !out.contains("landed-on-0"),
        "`--node 1` also ran on node 0, so it is not selecting a node: {out}"
    );

    let err = env.fails(&["exec", "--session", &id, "--node", "7", "--", "/bin/true"]);
    assert!(
        err.contains('7') && err.contains('2'),
        "a node the session does not have must be refused, saying how many \
         there are: {err}"
    );

    run.signal(Signal::SIGINT);
    run.wait(Duration::from_secs(60));
}

/// The harness must take its runtime root with it, however the test ends.
///
/// The root cannot live inside the test's `TempDir`: a run's control
/// socket goes under it and `sun_path` is 108 bytes, so the harness puts
/// it directly under `TMPDIR` and owns the removal itself. That makes
/// `Env`'s `Drop` the *only* thing standing between the suite and a
/// `/tmp/mrg-*` directory per test — and the case that matters is the
/// failing test, because a suite nobody is watching leaks hardest when
/// it is red. Asserting it here rather than trusting the `Drop` to exist
/// is the difference between a fix and a fix that stays.
#[test]
fn the_harness_removes_its_runtime_root_even_when_a_test_panics() {
    let env = Env::new();
    let runtime = env.runtime().to_path_buf();
    // What a session leaves under the root: nested directories with
    // files in them, not an empty directory.
    std::fs::create_dir_all(runtime.join("mirage/run")).unwrap();
    std::fs::write(runtime.join("mirage/run/pretend.sock"), b"x").unwrap();
    assert!(runtime.exists());

    // `AssertUnwindSafe` because the point is precisely to observe what
    // unwinding does to `env`. The panic message below is expected
    // output, not a failure.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _owned = env;
        panic!("deliberate: standing in for a test that fails mid-assertion");
    }));
    assert!(outcome.is_err(), "the deliberate panic did not happen");

    assert!(
        !runtime.exists(),
        "the runtime root {} outlived the panicking test that owned it",
        runtime.display()
    );
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the e2e suite going green while every test in it skipped
    // for a missing emulator runtime. See `assert_suite_can_run`.
    harness::assert_suite_can_run();
}
