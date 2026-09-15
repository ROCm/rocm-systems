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
fn version_identifies_the_source_revision() {
    let version = Env::new().ok(&["--version"]);
    let mut lines = version.lines();
    assert_eq!(
        lines.next(),
        Some(concat!("mirage ", env!("CARGO_PKG_VERSION")))
    );
    let identity = lines
        .next()
        .expect("version output needs a RocJITsu identity");
    if identity.starts_with("rocjitsu build identity unavailable:") {
        assert_eq!(lines.next(), None, "unexpected output after: {version}");
        return;
    }

    assert!(identity.starts_with("rocjitsu "), "{version}");
    let revision = lines
        .next()
        .and_then(|line| line.strip_prefix("git revision: "))
        .unwrap_or_else(|| panic!("missing revision in: {version}"));
    assert!(
        revision == "unknown"
            || (revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit())),
        "{version}"
    );
    assert!(
        lines
            .next()
            .is_some_and(|line| line.starts_with("git commit: ")),
        "{version}"
    );
    assert_eq!(lines.next(), None, "unexpected output after: {version}");
}

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

/// A minimal but complete agent document, for the tests that need an
/// agent of their own.
///
/// Written rather than taken from the builtins because both tests below
/// delete it, and a pristine builtin refuses to be deleted — mirage would
/// write it straight back, so reporting success would be a lie.
const A_MINIMAL_AGENT: &str = r#"{"vm":{},"topology":{"root":{"name":"soc","type":"soc"}}}"#;

#[test]
fn a_topology_naming_an_agent_that_is_not_there_is_refused() {
    // The two write verbs held references to different standards.
    // `profile create` follows the chain — its emulator backend has to,
    // to answer "can you run this?" — and refuses a reference that
    // resolves to nothing. `topology create` checked only that the name
    // was a legal filename, so this exited 0 and wrote a document that
    // fails at every later command, with nothing said at the moment the
    // mistake was made.
    let env = Env::new();
    let err = env.fails(&["topology", "create", "t-ghost", "--agent", "ghostagent"]);
    assert!(err.contains("dangling agent reference"), "{err}");
    assert!(err.contains("ghostagent"), "{err}");
    assert!(err.contains("mirage agent list"), "{err}");
    // And it names the document holding the reference — which is the
    // word the user just typed, and the one they need to fix it.
    assert!(
        err.contains("the topology \"t-ghost\""),
        "the error must name the document that holds the reference: {err}"
    );
    let list = env.ok(&["topology", "list"]);
    assert!(
        !list.lines().any(|l| l.trim() == "t-ghost"),
        "a refused topology reached the disk anyway: {list}"
    );

    // The same command succeeds once the agent exists, which is what
    // makes the refusal a diagnosis rather than a ban.
    let agent = env.root().join("agent.json");
    std::fs::write(&agent, A_MINIMAL_AGENT).unwrap();
    env.ok(&["agent", "import", "ghostagent", agent.to_str().unwrap()]);
    env.ok(&["topology", "create", "t-ghost", "--agent", "ghostagent"]);
}

#[test]
fn deleting_a_document_says_which_others_it_just_broke() {
    // The delete is allowed: these files are the user's and mirage has no
    // veto over which of them exist. Saying nothing is not. A reference
    // that stops resolving surfaces later, in a command with no visible
    // connection to the delete, as an error about a name the user has
    // half forgotten typing.
    let env = Env::new();
    let agent = env.root().join("agent.json");
    std::fs::write(&agent, A_MINIMAL_AGENT).unwrap();
    env.ok(&["agent", "import", "doomed", agent.to_str().unwrap()]);
    env.ok(&["topology", "create", "t-refers", "--agent", "doomed"]);

    let out = env.run(&["agent", "delete", "doomed", "--force"]);
    assert!(out.status.success(), "the delete itself must go through");
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(
        said.contains("topology t-refers"),
        "deleting an agent must name what still refers to it: {said}"
    );
    let list = env.ok(&["agent", "list"]);
    assert!(!list.lines().any(|l| l.trim() == "doomed"), "{list}");

    // A profile naming a topology is the same breakage one level up.
    // Built by rewriting a real profile rather than by hand, so the
    // document is exactly the shape mirage writes.
    let mut profile: serde_json::Value = serde_json::from_str(&env.ok(&[
        "profile",
        "create",
        "seed",
        "--emulator",
        TEST_EMULATOR,
        "--no-input",
        "--json",
    ]))
    .unwrap();
    env.ok(&["topology", "create", "t-named"]);
    profile["name"] = serde_json::json!("by-name");
    profile["emulator"]["topology"] = serde_json::json!("t-named");
    let path = env.root().join("by-name.json");
    std::fs::write(&path, profile.to_string()).unwrap();
    env.ok(&["profile", "import", path.to_str().unwrap()]);

    let out = env.run(&["topology", "delete", "t-named", "--force"]);
    assert!(out.status.success(), "the delete itself must go through");
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(
        said.contains("profile by-name"),
        "deleting a topology must name the profile that refers to it: {said}"
    );

    // And a document nobody names is deleted without a word about
    // referrers, or the warning would be noise on the ordinary path.
    env.ok(&["topology", "create", "t-lonely"]);
    let out = env.run(&["topology", "delete", "t-lonely", "--force"]);
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(
        !said.contains("referring"),
        "an unreferenced document was reported as breaking something: {said}"
    );
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
    //
    // `--nproc-per-node`, and not the default shape, because the default
    // shape does not have the bug. One node with one process *inherits*
    // mirage's streams: the workload writes to this test's pipe itself
    // and there is no pump, no channel and no printer between them, so
    // the version of this test that ran `mirage run -- echo` asserted
    // that a pipe works. The race lives in the captured path, and the
    // captured path starts at two processes.
    //
    // Every rank asserted, not just one. The drain is per stream, so a
    // printer that returns after flushing the first source it hears from
    // loses the others and nothing else here would notice.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    const RANKS: u32 = 3;
    for i in 0..6 {
        let out = env.ok(&[
            "run",
            "--profile",
            "p",
            "--nproc-per-node",
            &RANKS.to_string(),
            "--",
            "/bin/sh",
            "-c",
            // Write and exit immediately, which is the whole point:
            // the last bytes are in flight when the process is reaped.
            &format!("echo quick-{i}-$RANK"),
        ]);
        for rank in 0..RANKS {
            assert!(
                out.contains(&format!("[{rank}] quick-{i}-{rank}")),
                "round {i}: rank {rank}'s output was lost or unlabelled; \
                 got {out:?}"
            );
        }
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

// `interrupting_a_run_takes_its_workload_with_it` used to sit here. It is
// `strain.rs::interrupting_a_run_tears_down_its_workload` line for line —
// same signal, same tagged sleep, same `assert_no_leaks` — except that
// the strain copy also asserts the scratch directory went. Two full
// bring-ups for one property, and the weaker of the two would have been
// the one to keep.

#[test]
fn an_interrupted_run_exits_with_128_plus_the_signal() {
    // The convention every shell and CI system reads a killed job by, and
    // the one `mirage run` has to speak because it reports its workload's
    // exit code as its own. Three commits this cycle depend on it — the
    // interrupt that arrives during bring-up returns it directly, the one
    // that arrives during the workload inherits it from the workload, and
    // `SIGHUP` reaches the handling at all rather than taking the default
    // action — and nothing asserted the number that comes out.
    //
    // `SIGHUP` is the one worth the extra bring-up: unhandled it is fatal
    // by default, so the failure it guards against is not "the wrong
    // code" but "no teardown at all", and its exit code is the cheapest
    // outward sign of which of the two happened.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    for (signal, expected) in [
        (Signal::SIGINT, 130),
        (Signal::SIGTERM, 143),
        (Signal::SIGHUP, 129),
    ] {
        let tag = marker("exit-code");
        let started = env.root().join(format!("started-{signal:?}"));
        let script = format!("touch {}; {}", started.display(), tagged_sleep(&tag));
        let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &script]);
        run.await_ready(Duration::from_secs(90));
        // Signalling a run whose workload has not started yet takes the
        // bring-up path, which returns the same number for a different
        // reason — so the test would pass without saying anything about
        // the path it is named for.
        wait_for("the workload to start", Duration::from_secs(30), || {
            started.exists()
        });

        run.signal(signal);
        let out = run.wait(Duration::from_secs(30));
        assert_eq!(
            out.status.code(),
            Some(expected),
            "{signal:?} must leave `mirage run` exiting 128 + {}: {}",
            signal as i32,
            String::from_utf8_lossy(&out.stderr)
        );
        assert_no_leaks(&tag);
    }
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
    // Naming the directory it searched is what turns "not found" into
    // something the user can act on: the usual cause is a profile that
    // exists under a different `MIRAGE_CONFIG` than the one in force.
    assert!(
        err.contains(&env.profile_dir().display().to_string()),
        "the error must say where mirage looked: {err}"
    );
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

/// Declining a delete leaves the document alone — and says so in a way a
/// script can read.
///
/// Every other test in this suite passes `--force`, which means the
/// prompt itself, the exit status of a decline, and the JSON shape a
/// script would branch on were all untested. Three outcomes have to be
/// distinguishable, and only two of them are "nothing happened": the
/// document is gone, the user was asked and said no, and mirage could not
/// ask at all. A decline that exited 0 and printed nothing was
/// indistinguishable from a completed delete, which is how a cleanup
/// script reports success for work it never did.
#[test]
fn a_declined_delete_changes_nothing_and_reports_that_it_did_not() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("survivor");

    // Answered, deliberately, with "no" — from a real terminal, because
    // that is now the only place mirage will ask. A pipe carrying the
    // byte `n` is not a person saying no, and treating it as one is what
    // let a cron job read its own input as a decline and carry on.
    let pty = nix::pty::openpty(None, None).unwrap();
    let child = env
        .mirage()
        .args(["--json", "profile", "delete", "survivor"])
        // The prompt needs a terminal on stdin to be offered at all; the
        // answer is written to the master below.
        .stdin(std::process::Stdio::from(pty.slave.try_clone().unwrap()))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    {
        use std::io::Write as _;
        let mut master = std::fs::File::from(pty.master);
        master.write_all(b"n\n").unwrap();
        master.flush().unwrap();
    }
    let out = child.wait_with_output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "a deliberate \"no\" needs an exit code of its own — not 0, which \
         is what a completed delete says: {:?}",
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
    assert_eq!(json["declined"], true, "{json}");
    assert_eq!(json["name"], "survivor", "{json}");

    // A prompt nobody can answer is the third outcome, and it is a
    // failure rather than a "no": a `mirage profile delete x` in a cron
    // job or a pipeline that read end-of-file and called it a decline
    // would tell the script its cleanup had run.
    let out = env
        .mirage()
        .args(["profile", "delete", "survivor"])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(1),
        "a prompt that reached end-of-file is a failure: {:?}",
        out.status.code()
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--force"),
        "the error must say how to run it without a prompt: {stderr}"
    );

    // And the profile survived both.
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
        // A value outside an enumerated set is the parser's job too, and
        // the message has to list what would have worked.
        &[
            "run",
            "--profile",
            "p",
            "--exec-mode",
            "turbo",
            "--",
            "/bin/true",
        ],
        // `exec` is parsed by the same rules, and the parse happens
        // before any session is looked for — so these are exit 2 with no
        // run live at all, and would be exit 1 if the validation had
        // moved out of clap.
        &["exec", "--node", "-1", "--", "/bin/true"],
        &["exec", "--session", "../../etc", "--", "/bin/true"],
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

/// A subcommand mirage does not have is a parse error that names the one
/// it thinks you meant.
///
/// This is the top layer of the exit-code contract, and the layer below
/// it is asserted in the same test on purpose: 2 means "I did not
/// understand you" and 1 means "I understood and could not", and the
/// distinction is only worth anything if both halves hold at once. A
/// script that retries on 1 and gives up on 2 gets the opposite of what
/// it wanted if a typo starts reporting itself as a runtime failure.
#[test]
fn an_unknown_subcommand_is_a_parse_error_that_points_at_the_real_one() {
    let env = Env::new();

    let out = env.run(&["frobnicate"]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an unknown subcommand is a parse error: {:?}",
        out.status.code()
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("unrecognized subcommand 'frobnicate'"),
        "the error must name what was typed: {err}"
    );

    // A near miss gets the suggestion, which is the whole reason clap is
    // allowed to own this layer.
    let out = env.run(&["profil", "list"]);
    assert_eq!(out.status.code(), Some(2), "{:?}", out.status.code());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("'profile'"),
        "a one-letter typo must be told what it nearly was: {err}"
    );

    // Understood, and impossible: exit 1, naming the input.
    let out = env.run(&["profile", "show", "nope"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a request mirage understood and could not satisfy exits 1, not 2: {:?}",
        out.status.code()
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("profile not found: nope"),
        "the error must name the input: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Every `--json` command prints exactly one document, and prints it
/// alone.
///
/// `--json` exists so a script can pipe mirage into `jq`, and that only
/// works if stdout is the document and nothing else — no progress line,
/// no warning, no second object. Each command renders its own JSON, so
/// this is one property with a dozen independent implementations of it,
/// and a table is the honest way to check that.
#[test]
fn every_json_command_prints_one_document_on_stdout_and_nothing_on_stderr() {
    let env = Env::new();

    let commands: &[&[&str]] = &[
        &["paths"],
        &["about"],
        &["emulators"],
        &["emulators", "--long"],
        &["profile", "list"],
        &["profile", "list", "--long"],
        &["topology", "list"],
        &["agent", "list"],
        &["state", "builtins"],
        &["cleanup", "--dry-run"],
        &["profile", "show", "mi350x"],
        &["topology", "show", "MI350X-1x8"],
        &["agent", "show", "mi350x"],
    ];

    for command in commands {
        let mut args = vec!["--json"];
        args.extend_from_slice(command);
        let out = env.run(&args);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "`mirage --json {}` failed: {:?}\n{stderr}",
            command.join(" "),
            out.status.code()
        );
        serde_json::from_str::<serde_json::Value>(&stdout).unwrap_or_else(|e| {
            panic!(
                "`mirage --json {}` did not print one JSON document ({e}): {stdout:?}",
                command.join(" ")
            )
        });
        assert!(
            stderr.is_empty(),
            "`mirage --json {}` wrote to stderr, which a script reading both \
             streams would splice into the document: {stderr:?}",
            command.join(" ")
        );
    }
}

/// A `--json` command that fails writes nothing at all to stdout.
///
/// The failure mode this rules out is the expensive one: half a document,
/// or a document describing a state that does not exist, left on stdout
/// beside an error on stderr. A consumer that checks the exit code is
/// fine either way, but one that pipes straight into `jq` sees a parse
/// error at best and a plausible wrong answer at worst.
#[test]
fn a_command_that_fails_under_json_writes_nothing_to_stdout() {
    let env = Env::new();

    let commands: &[&[&str]] = &[
        &["profile", "show", "nope"],
        &["topology", "show", "nope"],
        &["agent", "show", "nope"],
        &["profile", "delete", "nope", "--force"],
        &["run", "--profile", "nope", "--", "/bin/true"],
    ];

    for command in commands {
        let mut args = vec!["--json"];
        args.extend_from_slice(command);
        let out = env.run(&args);
        assert!(
            !out.status.success(),
            "`mirage --json {}` was supposed to fail",
            command.join(" ")
        );
        assert!(
            out.stdout.is_empty(),
            "`mirage --json {}` failed but left {:?} on stdout",
            command.join(" "),
            String::from_utf8_lossy(&out.stdout)
        );
        assert!(
            !out.stderr.is_empty(),
            "`mirage --json {}` failed silently, so nothing says why",
            command.join(" ")
        );
    }
}

/// `mirage help <cmd>` and `mirage <cmd> --help` are the same page.
///
/// Two spellings of one request, and users pick between them by habit. If
/// they ever diverge it is because one of them is being rendered by
/// something other than clap, and the one that is will be the one that
/// goes stale.
#[test]
fn help_for_a_subcommand_is_the_same_page_as_its_own_help_flag() {
    let env = Env::new();

    for command in [
        "run",
        "exec",
        "profile",
        "topology",
        "agent",
        "emulators",
        "paths",
        "about",
        "state",
        "cleanup",
    ] {
        let via_help = env.run(&["help", command]);
        let via_flag = env.run(&[command, "--help"]);
        assert_eq!(
            String::from_utf8_lossy(&via_help.stdout),
            String::from_utf8_lossy(&via_flag.stdout),
            "`mirage help {command}` and `mirage {command} --help` differ"
        );
        assert_eq!(via_help.status.code(), via_flag.status.code());
    }

    // Help is output, not a diagnostic: it goes to stdout so it can be
    // piped into a pager, and leaves stderr empty.
    let out = env.run(&["--help"]);
    assert!(out.status.success(), "{:?}", out.status.code());
    assert!(!out.stdout.is_empty(), "`--help` printed nothing");
    assert!(
        out.stderr.is_empty(),
        "`--help` wrote to stderr: {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A mistyped drop-in is a parse error, not a session.
///
/// The two ways to get the drop-in shape wrong are a flag mirage does not
/// have and a command with no `--` in front of it, and both used to end
/// somewhere expensive. `mirage --nodes 2 -- ./app` (the plural is the
/// natural typo for `--num-nodes`) brought up a whole emulated machine
/// and then reported `command not found: --nodes`, blaming the workload
/// for mirage's own argument; `mirage ./app` dead-ended on "unrecognized
/// subcommand" with nothing said about the separator that would have made
/// it work. Both answers are cheap to give before anything starts, and
/// the test insists they *are* given before anything starts.
#[test]
fn a_mistyped_drop_in_is_a_parse_error_rather_than_a_session() {
    let env = Env::new();

    let out = env.run(&["--nodes", "2", "--", "/bin/echo", "hi"]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a flag mirage does not have is a parse error: {:?}",
        out.status.code()
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("unexpected argument") && err.contains("--nodes"),
        "the error must name the flag it could not use: {err}"
    );
    assert!(
        !err.contains("command not found"),
        "mirage's own flag was handed to the workload: {err}"
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).is_empty(),
        "the workload ran despite the parse error"
    );

    // And the separator itself, when it is what is missing.
    let out = env.run(&["/bin/echo", "hi"]);
    assert_eq!(out.status.code(), Some(2), "{:?}", out.status.code());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("/bin/echo"),
        "the error must name what was typed: {err}"
    );
    assert!(
        err.contains("--"),
        "the error must point at the separator that would have worked: {err}"
    );

    // Neither of them started anything on the way to saying so.
    assert!(env.live_runs().is_empty(), "{:?}", env.live_runs());
}

/// A hostile name inside an imported document is refused exactly as one
/// typed on the command line is.
///
/// `profile create ../escape` is the obvious attack and is checked
/// elsewhere; this is the same attack arriving by the route nobody looks
/// at. The name in an imported document is chosen by whoever wrote the
/// file, not by the person running the command, and it becomes a path
/// under the config directory — so a document fetched from a colleague or
/// a repository could write outside it. The validation has to live where
/// the name is stored, not where it is typed.
#[test]
fn a_hostile_name_inside_an_imported_document_is_refused() {
    let env = Env::new();

    let document = env.root().join("pwn.json");
    std::fs::create_dir_all(env.root()).unwrap();
    std::fs::write(
        &document,
        r#"{"name":"../../PWNED","emulator":{"emulator":"rocjitsu","plugins":{},
           "exec_mode":"Functional","options":{},
           "topology":{"num_nodes":1,"gpus_per_node":1,"agent":"MI350X"}}}"#,
    )
    .unwrap();

    let err = env.fails(&["profile", "import", document.to_str().unwrap()]);
    assert!(
        err.contains("invalid profile name"),
        "a traversing name in an imported document must be rejected as \
         invalid: {err}"
    );
    // Nowhere on the way out of the config directory, either.
    for dir in [env.root(), &env.root().join("config")] {
        assert!(
            !dir.join("PWNED.json").exists(),
            "the document was written to {}",
            dir.display()
        );
    }
    let list = env.ok(&["profile", "list"]);
    assert!(
        !list.contains("PWNED"),
        "the refused name reached the store anyway: {list}"
    );
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

/// A relayed signal reaches the workload without ending the run.
///
/// `SIGUSR1` and `SIGUSR2` are application-defined, and what usually
/// sends them is a scheduler saying "checkpoint now" — Slurm's
/// `--signal=USR1@60` is the common spelling. They used to sit in the
/// same escalation ladder as Ctrl-C, which cost a job two things: during
/// bring-up any of them aborted the session, so the warning destroyed
/// what it was warning about; and after startup the *second* one reached
/// "not waiting any longer" and terminated the workload — so a job
/// checkpointing on a timer killed itself on its second checkpoint.
///
/// Two of them, deliberately. One proves forwarding; the second is the
/// one that used to be fatal.
#[test]
fn an_application_defined_signal_is_forwarded_and_does_not_end_the_run() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    // Traps `USR1`, reports it, and keeps going. The sleep is short and
    // repeated rather than one long one because a shell runs a trap
    // between commands: with `sleep 300` the handler would not run until
    // the sleep was over.
    //
    // The `touch` comes *after* both traps and is what the test waits on,
    // which is the whole synchronisation. A signal that arrives before
    // `/bin/sh` has installed its handlers is not relayed-and-ignored, it
    // is fatal by default disposition — so a barrier that lets one
    // through does not report a flaky test, it reports a signal-relay
    // regression that did not happen.
    let started = env.root().join("trapping");
    let script = format!(
        "trap 'echo CHECKPOINTED' USR1; \
         trap 'echo ROTATED' USR2; \
         touch {}; \
         i=0; while [ $i -lt 200 ]; do sleep 0.1; i=$((i+1)); done; \
         echo FINISHED",
        started.display()
    );
    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &script]);
    run.await_ready(Duration::from_secs(90));

    // The barrier this used to have was `watch.contains("session")`, which
    // matches a line `mirage run` prints before it has even bound its
    // control socket — true on the first poll, every time — leaving a bare
    // 500 ms sleep as the only thing standing between bring-up and the
    // signals. The sentinel is the idiom the rest of this file uses, and
    // it is the workload itself saying it is ready to be signalled.
    wait_for(
        "the workload to install its traps",
        Duration::from_secs(60),
        || started.exists(),
    );

    run.signal(Signal::SIGUSR1);
    run.signal(Signal::SIGUSR2);

    let out = run.wait(Duration::from_secs(120));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        stdout.contains("CHECKPOINTED"),
        "SIGUSR1 did not reach the workload:\nstdout: {stdout}\nstderr: {stderr}"
    );
    // Both of them, separately. Asserting only the first would pass on a
    // build that armed `SIGUSR1` and dropped `SIGUSR2` — and `SIGUSR2` is
    // the one this test exists for, because it is the second signal that
    // used to reach "not waiting any longer" and end the run.
    assert!(
        stdout.contains("ROTATED"),
        "SIGUSR2 did not reach the workload:\nstdout: {stdout}\nstderr: {stderr}"
    );
    // The run ran to the end of its own accord. On the old ladder the
    // second signal printed "not waiting any longer" and terminated the
    // workload, so this line never appeared.
    assert!(
        stdout.contains("FINISHED"),
        "the workload was stopped by a signal that only asked it to \
         checkpoint:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "the run must exit with the workload's own code:\n{stderr}"
    );
    assert!(
        !stderr.contains("not waiting any longer"),
        "an application-defined signal counted toward shutdown \
         escalation:\n{stderr}"
    );
}

/// `--node` and `--nproc-per-node` compose, and the count decides the
/// streams.
///
/// The pair reads as a contradiction if `--node` is described as "the
/// interactive flag": naming a node is what gets you a shell, and yet
/// `--node 1 --nproc-per-node 3` is plainly not one shell. It is not a
/// contradiction — `--node` says *where* and the process count says
/// *whether anyone gets stdin* — but nothing pinned it, so this asserts
/// the shape from outside: three processes, all on the named node, all
/// captured and labelled with the job's own rank numbers.
#[test]
fn a_named_node_takes_nproc_per_node_and_is_captured_when_it_does() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    // A job of two nodes with three slots each: ranks 0-5, and node 1's
    // own slots are 3, 4 and 5.
    let mut run = env.spawn_run(
        &[
            "--profile",
            "p",
            "--num-nodes",
            "2",
            "--nproc-per-node",
            "3",
        ],
        &["/bin/sh", "-c", "sleep 300"],
    );
    let id = run.await_ready(Duration::from_secs(90));

    let out = env.ok(&[
        "exec",
        "--session",
        &id,
        "--node",
        "1",
        "--nproc-per-node",
        "3",
        "--",
        "/bin/sh",
        "-c",
        "echo slot-$RANK/$LOCAL_RANK",
    ]);

    // Node 1's ranks, and only node 1's. A `--nproc-per-node` that
    // ignored `--node` would start six processes; one that renumbered
    // from zero would print `slot-0/0` and put two live processes on
    // rank 0 of the rendezvous.
    for (global, local) in [(3, 0), (4, 1), (5, 2)] {
        assert!(
            out.contains(&format!("slot-{global}/{local}")),
            "node 1 slot {local} should be the job's rank {global}: {out}"
        );
    }
    for absent in ["slot-0/", "slot-1/", "slot-2/"] {
        assert!(
            !out.contains(absent),
            "`--node 1` also started node 0's ranks: {out}"
        );
    }

    // And it is captured, because three processes cannot share one
    // terminal — the labels are the observable form of that.
    for label in ["[3]", "[4]", "[5]"] {
        assert!(
            out.contains(label),
            "a multi-process exec must be labelled even when it named a \
             node: {out}"
        );
    }

    run.signal(Signal::SIGINT);
    run.wait(Duration::from_secs(60));
}

/// The run's exit code is the workload's own, however the workload ended.
///
/// The bottom layer of the exit-code contract, and the one scripts lean
/// on hardest: `mirage run -- make` has to be substitutable for `make`.
/// A table, because each row is a different path through the supervisor —
/// a plain exit, a large code that must not be truncated or turned into
/// 1, and the shell's `128 + signal` convention for a workload the kernel
/// killed. Reporting 1 for all of them, which is what a supervisor that
/// forgets to plumb the status through does, would leave every one of
/// these passing except the first.
#[test]
fn the_run_exits_with_the_workloads_own_code_however_the_workload_ended() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let cases: &[(&str, i32)] = &[
        ("exit 0", 0),
        ("exit 1", 1),
        ("exit 42", 42),
        // The top of the byte, where a code that is masked wrongly comes
        // back as -1 or 0.
        ("exit 255", 255),
        // Signals, via the shell convention mirage documents.
        ("kill -TERM $$", 143),
        ("kill -SEGV $$", 139),
        ("kill -KILL $$", 137),
    ];

    for (script, expected) in cases {
        let out = env.run(&["run", "--profile", "p", "--", "/bin/sh", "-c", script]);
        assert_eq!(
            out.status.code(),
            Some(*expected),
            "`{script}` must make the run exit {expected}\nstderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert!(env.live_runs().is_empty(), "{:?}", env.live_runs());
}

/// With many ranks, the worst exit across them is the run's.
///
/// There is one exit code and several processes to derive it from, so the
/// rule has to be one nobody can be surprised by: a job is a failure if
/// any part of it failed. Taking rank 0's — the obvious implementation,
/// since rank 0 is the one a user thinks of as "the job" — silently
/// reports a distributed run as successful when a worker crashed and the
/// head process shut down cleanly, which is the exact case a CI job
/// exists to catch.
#[test]
fn the_worst_exit_across_ranks_is_the_runs_exit_code() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    // Rank 0 succeeds and a worker does not.
    let out = env.run(&[
        "run",
        "--profile",
        "p",
        "--nproc-per-node",
        "4",
        "--",
        "/bin/sh",
        "-c",
        "case $RANK in 3) exit 17;; *) exit 0;; esac",
    ]);
    assert_eq!(
        out.status.code(),
        Some(17),
        "a job with a crashed worker and a clean rank 0 must report the \
         crash\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // And when several fail, the furthest from zero wins regardless of
    // which rank it was — so the answer does not depend on scheduling.
    for script in [
        "case $RANK in 0) exit 99;; 3) exit 5;; *) exit 0;; esac",
        "case $RANK in 0) exit 5;; 3) exit 99;; *) exit 0;; esac",
    ] {
        let out = env.run(&[
            "run",
            "--profile",
            "p",
            "--nproc-per-node",
            "4",
            "--",
            "/bin/sh",
            "-c",
            script,
        ]);
        assert_eq!(
            out.status.code(),
            Some(99),
            "`{script}` must report the worst rank's exit\nstderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// A redirected single-process run is byte-exact.
///
/// The README's claim, tested with the bytes that break every design that
/// is not: a NUL, a byte that is not valid UTF-8, and enough output that
/// anything buffering it would have to flush. A single-process job's
/// stdout *is* this command's stdout — no relay, no line discipline, no
/// lossy `String` in the middle — so `mirage run -- ./app > out.bin` has
/// to produce exactly what `./app > out.bin` would.
#[test]
fn a_redirected_run_is_byte_exact_including_nul_and_high_bytes() {
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
        r#"printf 'A\000B\377C'"#,
    ]);
    assert!(out.status.success(), "{:?}", out.status.code());
    assert_eq!(
        out.stdout, b"A\0B\xffC",
        "the workload's bytes were altered on the way out: {:?}",
        out.stdout
    );

    // Volume, for the other half: a pipe that is drained late, or in
    // fixed-size reads that lose the tail, only shows up past the pipe
    // buffer.
    const BYTES: usize = 4 * 1024 * 1024;
    let out = env.run(&[
        "run",
        "--profile",
        "p",
        "--",
        "/bin/sh",
        "-c",
        &format!("yes abcdefghij | head -c {BYTES}"),
    ]);
    assert!(out.status.success(), "{:?}", out.status.code());
    assert_eq!(
        out.stdout.len(),
        BYTES,
        "{BYTES} bytes of output came back as {}",
        out.stdout.len()
    );
    let expected: Vec<u8> = b"abcdefghij\n"
        .iter()
        .copied()
        .cycle()
        .take(BYTES)
        .collect();
    assert!(
        out.stdout == expected,
        "a large stdout was reordered or corrupted"
    );
}

/// The drop-in shape hands every argument to the workload untouched.
///
/// `mirage [opts] -- ./app ...` exists so mirage can be dropped in front
/// of an existing `rocjitsu` command line, which means everything after
/// the separator belongs to the workload — including a second `--`, and
/// including spellings mirage has flags of its own for. A parser that
/// kept looking after the separator would quietly eat the workload's
/// `--json` and change what the command does rather than failing.
#[test]
fn the_drop_in_shape_hands_every_argument_to_the_workload_untouched() {
    use std::os::unix::fs::PermissionsExt as _;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let script = env.root().join("argv.sh");
    std::fs::write(
        &script,
        "#!/bin/sh\nfor a in \"$@\"; do echo \"arg=<$a>\"; done\n",
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let script = script.to_str().unwrap();

    let out = env.ok(&[
        "--profile",
        "p",
        "--",
        script,
        "a",
        "b c",
        "--",
        "-x",
        "--json",
    ]);
    for argument in ["a", "b c", "--", "-x", "--json"] {
        assert!(
            out.contains(&format!("arg=<{argument}>")),
            "the workload did not receive {argument:?}: {out}"
        );
    }

    // And a run flag before the separator is still mirage's, so the shape
    // is usable rather than all-or-nothing.
    let out = env.ok(&["--profile", "p", "--num-nodes", "1", "--", script, "kept"]);
    assert!(out.contains("arg=<kept>"), "{out}");
}

/// A workload starts in the directory it was given, or in the one the
/// user was standing in.
///
/// Every existing test drives `--workdir` through its error cases, which
/// leaves the two spellings that are supposed to *work* untested — and a
/// relative path in a workload's argv means nothing without them. The
/// default is the surprising half: a session is an emulated machine, so
/// "the directory `mirage run` was started from" is a decision rather
/// than an inevitability, and a run that quietly landed in `/` or in its
/// own scratch directory would break `mirage run -- ./configure` for
/// reasons nobody would look for in mirage.
#[test]
fn a_workload_starts_in_the_directory_it_was_given() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let here = env.root().join("here");
    let there = env.root().join("there");
    std::fs::create_dir_all(&here).unwrap();
    std::fs::create_dir_all(&there).unwrap();

    let run_in = |cwd: &std::path::Path, args: &[&str]| -> String {
        let mut argv = vec!["run", "--profile", "p"];
        argv.extend_from_slice(args);
        argv.extend(["--", "/bin/pwd"]);
        let out = env
            .mirage()
            .args(&argv)
            .current_dir(cwd)
            .output()
            .unwrap_or_else(|e| panic!("running `mirage {}`: {e}", argv.join(" ")));
        assert!(
            out.status.success(),
            "`mirage {}` failed: {}",
            argv.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    // `canonicalize`, because the tempdir may be reached through a
    // symlink (`/tmp` is one on macOS and on some Linux setups) and
    // `pwd` reports the resolved path.
    let canonical = |p: &std::path::Path| p.canonicalize().unwrap().display().to_string();

    assert_eq!(
        run_in(&here, &[]),
        canonical(&here),
        "a run must start where the user was standing"
    );
    assert_eq!(
        run_in(&here, &["--workdir", there.to_str().unwrap()]),
        canonical(&there),
        "`--workdir` must win over the caller's directory"
    );
    // Trailing slashes are what a shell's tab completion leaves behind,
    // so they cannot be the difference between a run and an error.
    let sloppy = format!("{}///", there.display());
    assert_eq!(
        run_in(&here, &["--workdir", &sloppy]),
        canonical(&there),
        "a path with trailing slashes names the same directory"
    );
}

/// `--clear-env-vars` drops the ambient environment and keeps what the
/// session needs.
///
/// Both halves matter and they pull in opposite directions. The flag is
/// for reproducibility, so a variable the user happened to have exported
/// must not reach the workload — but the emulator's own variables, the
/// rank identity and the `MIRAGE_SESSION`/`MIRAGE_RUNTIME` marker are not
/// ambient, they are the session, and dropping the marker would make the
/// run's workloads invisible to `mirage cleanup` after a `SIGKILL`.
#[test]
fn clear_env_vars_drops_the_ambient_environment_but_keeps_what_the_session_needs() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let ambient = "MIRAGE_E2E_AMBIENT";
    let dump = |clear: bool| -> String {
        let mut cmd = env.mirage();
        cmd.args(["run", "--profile", "p"]);
        if clear {
            cmd.arg("--clear-env-vars");
        }
        let out = cmd
            .args(["--", "/usr/bin/env"])
            .env(ambient, "from-the-callers-shell")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "dumping the workload's environment failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    let inherited = dump(false);
    assert!(
        inherited.contains(&format!("{ambient}=from-the-callers-shell")),
        "without the flag the workload sees what the caller exported: {inherited}"
    );

    let cleared = dump(true);
    assert!(
        !cleared.contains(ambient),
        "`--clear-env-vars` let an ambient variable through: {cleared}"
    );
    for kept in [
        "PATH=",
        "HOME=",
        "MIRAGE_SESSION=",
        "MIRAGE_RUNTIME=",
        "RANK=",
    ] {
        assert!(
            cleared.lines().any(|line| line.starts_with(kept)),
            "`--clear-env-vars` dropped {kept}, which is the session rather \
             than the caller's shell:\n{cleared}"
        );
    }

    // An explicit `--env` is a request, not ambient, so it survives too.
    let out = env.ok(&[
        "run",
        "--profile",
        "p",
        "--clear-env-vars",
        "--env",
        "FOO=bar",
        "--",
        "/bin/sh",
        "-c",
        "echo [$FOO]",
    ]);
    assert!(out.contains("[bar]"), "{out}");
}

/// `--env` is one key and one value, the last spelling wins, and the
/// names mirage owns are not for sale.
///
/// The parse is a single `split_once('=')`, which is only obviously right
/// once the awkward cases are written down: a value that is empty, and a
/// value that itself contains `=`. The last case is the one with teeth —
/// letting `--env RANK=99` through would give one process a rank that
/// disagrees with every other process's idea of the job, and the failure
/// would surface inside the user's collective, not here.
#[test]
fn an_env_flag_is_one_pair_and_the_names_mirage_owns_are_refused() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let echo = |args: &[&str]| -> String {
        let mut argv = vec!["run", "--profile", "p"];
        argv.extend_from_slice(args);
        argv.extend(["--", "/bin/sh", "-c", "echo [$K]"]);
        env.ok(&argv).trim().to_string()
    };
    assert_eq!(echo(&["--env", "K="]), "[]", "an empty value is a value");
    assert_eq!(
        echo(&["--env", "K=a=b=c"]),
        "[a=b=c]",
        "only the first `=` separates the key from the value"
    );
    assert_eq!(
        echo(&["--env", "K=first", "--env", "K=second"]),
        "[second]",
        "a repeated key takes its last spelling, as `env` itself does"
    );

    // A pair with no key at all is refused, and says which input it
    // could not use.
    let err = env.fails(&[
        "run",
        "--profile",
        "p",
        "--env",
        "=value",
        "--",
        "/bin/true",
    ]);
    assert!(
        err.contains("key is empty") && err.contains("=value"),
        "an empty key must be refused, naming the input: {err}"
    );

    // A name mirage sets itself is ignored rather than obeyed — and said
    // out loud, because silently discarding what the user asked for is
    // the failure this is protecting against, not a fix for it.
    let out = env
        .mirage()
        .args([
            "run",
            "--profile",
            "p",
            "--env",
            "RANK=99",
            "--",
            "/bin/sh",
            "-c",
            "echo [$RANK]",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "{:?}", out.status.code());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "[0]",
        "`--env RANK=…` overwrote the rank mirage assigned"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("RANK") && stderr.contains("ignored"),
        "an ignored `--env` must say so: {stderr}"
    );
}

/// The rank grid is the product of the node count and the processes per
/// node.
///
/// `WORLD_SIZE = num_nodes * nproc_per_node`, `RANK` unique and dense
/// over it, `LOCAL_RANK` restarting at zero on every node — the formula
/// `torch.distributed` assumes and the README documents. The two flags
/// are covered separately elsewhere; only their product distinguishes the
/// right formula from the several plausible wrong ones (`RANK` restarting
/// per node, `WORLD_SIZE` taken from the node count, `LOCAL_RANK` running
/// to `WORLD_SIZE`), and every one of those wrong ones is invisible until
/// both flags are used at once.
#[test]
fn the_rank_grid_is_the_product_of_nodes_and_processes_per_node() {
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
        "2",
        "--nproc-per-node",
        "3",
        "--",
        "/bin/sh",
        "-c",
        "echo grid $RANK $LOCAL_RANK $WORLD_SIZE $MIRAGE_RANK",
    ]);

    let mut rows: Vec<(u32, u32, u32, u32)> = out
        .lines()
        .filter_map(|line| line.split_once("grid ").map(|(_, rest)| rest))
        .map(|rest| {
            let f: Vec<u32> = rest
                .split_whitespace()
                .filter_map(|v| v.parse().ok())
                .collect();
            assert_eq!(f.len(), 4, "a rank printed {rest:?}");
            (f[0], f[1], f[2], f[3])
        })
        .collect();
    rows.sort_unstable();

    assert_eq!(
        rows.len(),
        6,
        "2 nodes of 3 processes is 6 processes, not {}:\n{out}",
        rows.len()
    );
    for (index, (rank, local, world, node)) in rows.iter().enumerate() {
        let index = u32::try_from(index).unwrap();
        assert_eq!(
            *rank, index,
            "global ranks must be dense and unique:\n{out}"
        );
        assert_eq!(
            *local,
            index % 3,
            "LOCAL_RANK must restart at zero on every node:\n{out}"
        );
        assert_eq!(*world, 6, "WORLD_SIZE is nodes * processes:\n{out}");
        assert_eq!(
            *node,
            index / 3,
            "each node hosts 3 consecutive ranks:\n{out}"
        );
    }

    // The `torchrun` spelling of the same flag, which `run --help`
    // advertises as an alias. An alias nobody exercises is an alias that
    // stops existing the next time the flag is touched, and the failure
    // is silent: an underscore mirage does not know becomes an
    // unrecognised argument, or worse, part of the workload's argv.
    let out = env.ok(&[
        "run",
        "--profile",
        "p",
        "--nproc_per_node",
        "2",
        "--",
        "/bin/sh",
        "-c",
        "echo aliased $RANK/$WORLD_SIZE",
    ]);
    for rank in 0..2 {
        assert!(
            out.contains(&format!("aliased {rank}/2")),
            "`--nproc_per_node` did not build the same grid as \
             `--nproc-per-node`:\n{out}"
        );
    }
}

/// Every process of a node sees one environment apart from its own rank.
///
/// A distributed job agrees on its shape or it deadlocks, so the
/// variables that describe the job — `WORLD_SIZE`, the rendezvous, the
/// session — have to be identical everywhere, and the ones that describe
/// the *process* have to be the only difference. Asserting it as a
/// difference rather than a list of names is what makes it a real check:
/// any future variable that leaks a per-process value into what should be
/// job-wide state fails here without anyone having to think of it in
/// advance.
#[test]
fn every_process_on_a_node_shares_one_environment_apart_from_its_own_rank() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    // `--clear-env-vars`, so what is compared is the environment mirage
    // built rather than whatever the developer's shell had in it.
    let dir = env.root().join("envdump");
    std::fs::create_dir_all(&dir).unwrap();
    env.ok(&[
        "run",
        "--profile",
        "p",
        "--num-nodes",
        "2",
        "--nproc-per-node",
        "2",
        "--clear-env-vars",
        "--",
        "/bin/sh",
        "-c",
        &format!("/usr/bin/env > {}/$RANK", dir.display()),
    ]);

    let of = |rank: u32| -> std::collections::BTreeMap<String, String> {
        let text = std::fs::read_to_string(dir.join(rank.to_string()))
            .unwrap_or_else(|e| panic!("rank {rank} wrote no environment: {e}"));
        text.lines()
            .filter_map(|line| line.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };
    let ranks: Vec<_> = (0..4).map(of).collect();

    // Two processes on one node: only their own identity may differ. A
    // key missing from one side counts as a difference, which is why the
    // comparison is over the union of both.
    let mut differing: Vec<&str> = ranks[0]
        .keys()
        .chain(ranks[1].keys())
        .filter(|key| ranks[0].get(*key) != ranks[1].get(*key))
        .map(String::as_str)
        .collect();
    differing.sort_unstable();
    differing.dedup();
    assert_eq!(
        differing,
        ["LOCAL_RANK", "RANK"],
        "two processes on the same node must differ in nothing but their \
         own rank"
    );

    // Across the whole job: the shape and the rendezvous are one answer.
    for key in ["WORLD_SIZE", "MASTER_PORT", "MIRAGE_SESSION"] {
        let values: std::collections::BTreeSet<_> =
            ranks.iter().map(|r| r.get(key).cloned()).collect();
        assert_eq!(
            values.len(),
            1,
            "every rank must agree on {key}, but they reported {values:?}"
        );
    }
    // And the host identity is per node, because that is what it names.
    assert_eq!(ranks[0].get("NCCL_HOSTID"), ranks[1].get("NCCL_HOSTID"));
    assert_ne!(
        ranks[0].get("NCCL_HOSTID"),
        ranks[2].get("NCCL_HOSTID"),
        "two different nodes reported the same host id"
    );
}

/// An exec joins the running job rather than standing up a new one.
///
/// The claim `mirage exec --help` makes is that the process "still
/// believes it is that node — same rank variables, same `WORLD_SIZE`,
/// same rendezvous as its neighbours". A job of several processes per
/// node is what makes the claim falsifiable: an exec that derived the
/// world from its own shape rather than the session's would report a
/// `WORLD_SIZE` of 2 for a job of 6, and number its ranks against it —
/// while still pointing them at the run's `MASTER_PORT`, so the ranks
/// collide on a live rendezvous instead of failing to find one. The port
/// is checked in the same test for that reason: neither half is a
/// problem on its own, and together they are a mis-formed collective.
#[test]
fn an_exec_joins_the_running_job_rather_than_standing_up_a_new_one() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    // The run's own rank 0 records its rendezvous in a file rather than
    // on stdout: the run is still going, so its stdout cannot be
    // collected until it exits, and the comparison has to happen while
    // the session is up.
    let report = "echo id=$MIRAGE_SESSION rank=$RANK world=$WORLD_SIZE port=$MASTER_PORT";
    let recorded = env.root().join("rendezvous");
    let mut run = env.spawn_run(
        &[
            "--profile",
            "p",
            "--num-nodes",
            "2",
            "--nproc-per-node",
            "3",
        ],
        &[
            "/bin/sh",
            "-c",
            &format!(
                "if [ \"$RANK\" = 0 ]; then {report} > {}; fi; sleep 300",
                recorded.display()
            ),
        ],
    );
    let id = run.await_ready(Duration::from_secs(90));
    wait_for(
        "the run to publish its rendezvous",
        Duration::from_secs(60),
        || recorded.exists(),
    );

    let field = |text: &str, key: &str| -> String {
        text.split_once(&format!("{key}="))
            .map(|(_, rest)| {
                rest.split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .to_string()
            })
            .unwrap_or_else(|| panic!("no {key}= in {text:?}"))
    };
    let announced = std::fs::read_to_string(&recorded).unwrap();
    let run_port = field(&announced, "port");
    assert_eq!(field(&announced, "world"), "6", "{announced}");

    let joined = env.ok(&[
        "exec",
        "--session",
        &id,
        "--node",
        "1",
        "--",
        "/bin/sh",
        "-c",
        report,
    ]);
    assert_eq!(
        field(&joined, "id"),
        id,
        "the exec did not report the session it joined: {joined}"
    );
    assert_eq!(
        field(&joined, "world"),
        "6",
        "the exec reported the size of its own invocation rather than the \
         job's: {joined}"
    );
    assert_eq!(
        field(&joined, "port"),
        run_port,
        "the exec picked its own rendezvous port instead of the session's \
         ({run_port}): {joined}"
    );
    // Node 1 of a 2x3 job hosts ranks 3, 4 and 5. Landing on one of the
    // ranks node *0* already has would put two live processes on one
    // rank of one rendezvous.
    let rank: u32 = field(&joined, "rank").parse().unwrap();
    assert!(
        (3..6).contains(&rank),
        "`--node 1` must take a rank that node 1 hosts (3..5), not {rank}: {joined}"
    );

    run.signal(Signal::SIGINT);
    run.wait(Duration::from_secs(60));
}

/// An exec that cannot start its command fails the way a run does.
///
/// `mirage exec` is a second front door to the same machinery, and the
/// two have separate argument handling — so every diagnostic worth having
/// on `run` can be missing here without a single `run` test noticing. A
/// user in the second terminal is usually there *because* something is
/// already wrong, which is the worst moment to be handed a bare errno or
/// an exit code with no message.
#[test]
fn an_exec_that_cannot_start_its_command_fails_the_way_a_run_does() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", "sleep 300"]);
    let id = run.await_ready(Duration::from_secs(90));

    // A command that is not there: the shell's 127, and the name, so the
    // user can see the typo.
    let out = env.run(&["exec", "--session", &id, "--", "/no/such/binary"]);
    assert_eq!(out.status.code(), Some(127), "{:?}", out.status.code());
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        said.contains("command not found") && said.contains("/no/such/binary"),
        "the exec must say what it could not run: {said}"
    );

    // A working directory that is not there: a mirage-level failure, so
    // exit 1, naming the flag and the path.
    let err = env.fails(&[
        "exec",
        "--session",
        &id,
        "--workdir",
        "/no/such/dir",
        "--",
        "/bin/pwd",
    ]);
    assert!(
        err.contains("--workdir") && err.contains("/no/such/dir"),
        "the exec must name the flag and the directory: {err}"
    );

    // A session id nothing is serving: exit 1, naming the id and saying
    // why an id can stop working, which is the surprising part.
    let err = env.fails(&["exec", "--session", "s-never-existed-0", "--", "/bin/true"]);
    assert!(
        err.contains("s-never-existed-0") && err.contains("alive"),
        "an id with no run behind it must be explained, not just refused: {err}"
    );

    // The run is untouched by any of it.
    assert_eq!(env.live_runs(), vec![id.clone()]);
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
