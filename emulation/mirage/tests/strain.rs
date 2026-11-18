//! Strain tests: rapidly create, use and kill runs, then assert that
//! nothing survived.
//!
//! # What these are actually testing
//!
//! Cleanup bugs are almost never visible in a single clean run. They show
//! up under churn, at the boundaries — a Ctrl-C that arrives while a
//! process is still forking, an exec client that dies mid-command, a run
//! that is killed outright with work in flight. These tests manufacture
//! those boundaries deliberately and then check the one thing that cannot
//! be faked: the operating system's process table.
//!
//! Every assertion here is external. Not "the supervisor believes the
//! session is gone" but "no process with this marker exists" and "no
//! zombie with this parent exists". Internal bookkeeping agreeing with
//! itself is exactly the failure mode the previous design had — it
//! confidently reported sessions as stopped while their process trees
//! kept running.
//!
//! # What ownership means now
//!
//! There is no daemon to outlive anything. A session exists exactly while
//! the `mirage run` that created it is alive, so every claim in this file
//! is anchored to one pid: kill that process, or ask it to stop, and
//! nothing it started may still be there afterwards. `mirage exec`
//! borrows the session but owns its own processes in its own terminal, so
//! it gets the same treatment from the other side — an exec dying must
//! take its workload with it and leave the run untouched.
//!
//! # Zombies specifically
//!
//! A zombie answers `kill(pid, 0)` just like a live process, so a
//! liveness check alone cannot see one. These tests read `/proc/<pid>/stat`
//! and assert a live run has no children in state `Z`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod harness;

use std::collections::HashSet;
use std::time::{Duration, Instant};

use nix::sys::signal::Signal;

use harness::{
    Env, assert_no_leaks, count_processes, find_processes, marker, skip_without_emulator,
    tagged_sleep, wait_for,
};

/// How long a session gets to come up before a test gives up on it.
///
/// Generous on purpose: bring-up loads an emulator runtime, and a machine
/// running the whole suite in parallel is a slow machine.
const READY: Duration = Duration::from_secs(60);

/// How long a run gets to finish tearing down after it is asked to stop.
///
/// Teardown escalates `SIGTERM` to `SIGKILL` after a grace period, so a
/// stubborn workload legitimately takes a couple of seconds.
const TEARDOWN: Duration = Duration::from_secs(60);

/// Children of `pid` that are zombies.
fn zombie_children(pid: u32) -> Vec<u32> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut zombies = Vec::new();
    for entry in entries.flatten() {
        let Ok(child) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        if !harness::pid_is_zombie(child) {
            continue;
        }
        let Ok(status) = std::fs::read_to_string(format!("/proc/{child}/status")) else {
            continue;
        };
        let ppid = status.lines().find_map(|l| {
            l.strip_prefix("PPid:")
                .and_then(|v| v.trim().parse::<u32>().ok())
        });
        if ppid == Some(pid) {
            zombies.push(child);
        }
    }
    zombies
}

/// Assert a live `mirage run` is not accumulating unreaped children.
///
/// This is the direct test of the bug that motivated the rewrite. A
/// process that is waited on leaves the table immediately; one that is
/// not lingers as a zombie until its parent dies. The run is a
/// foreground process a user may leave up all day, so "until its parent
/// dies" is not a bound worth having.
fn assert_no_zombies(pid: u32) {
    // Reaping is a syscall, not instantaneous; allow it to happen.
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && !zombie_children(pid).is_empty() {
        std::thread::sleep(Duration::from_millis(50));
    }
    let zombies = zombie_children(pid);
    assert!(
        zombies.is_empty(),
        "the run (pid {pid}) has {} unreaped child process(es): {zombies:?}. \
         Every spawned child must be waited on.",
        zombies.len()
    );
}

/// How many file descriptors `pid` holds open.
fn fd_count(pid: u32) -> usize {
    std::fs::read_dir(format!("/proc/{pid}/fd"))
        .map(Iterator::count)
        .unwrap_or(0)
}

/// Send `sig` to `pid`, tolerating a process that has already exited.
fn signal(pid: u32, sig: Signal) {
    let Ok(pid) = i32::try_from(pid) else {
        return;
    };
    let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), sig);
}

/// `SIGKILL` anything still tagged `tag`.
///
/// Used only by the tests that deliberately provoke a leak: whatever the
/// verdict is, the machine must not be left with a `sleep` loop on it for
/// the next test — or the next developer — to wonder about.
fn kill_survivors(tag: &str) {
    for pid in find_processes(tag) {
        signal(pid, Signal::SIGKILL);
    }
}

/// Wait until a workload tagged `tag` is actually running.
fn wait_for_workload(tag: &str) {
    wait_for("the workload to start", Duration::from_secs(30), || {
        count_processes(tag) > 0
    });
}

/// The directory per-session scratch directories live in.
fn scratch_root(env: &Env) -> std::path::PathBuf {
    env.runtime().join("mirage/session")
}

/// Names of the session scratch directories still on disk.
fn leftover_scratch(env: &Env) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(scratch_root(env)) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()
}

/// Rapidly create and destroy runs.
#[test]
fn rapid_run_churn_leaves_nothing_behind() {
    const ROUNDS: usize = 40;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    for _ in 0..ROUNDS {
        env.ok(&["run", "--profile", "p", "--", "/bin/true"]);
    }

    // A run that has exited must leave nothing behind claiming its
    // session is live. `mirage exec` picks its target from the sockets in
    // this directory, so a corpse here is a session a user can still try
    // to enter — and be told, confusingly, that several runs are up.
    let live = env.live_runs();
    assert!(
        live.is_empty(),
        "run sockets survived {ROUNDS} rounds of run-and-exit: {live:?}"
    );

    // Scratch directories must not accumulate either: one per session,
    // forever, is a slow-motion disk leak.
    let leftover = leftover_scratch(&env);
    assert!(
        leftover.is_empty(),
        "{} session scratch director(ies) outlived their runs: {leftover:?}",
        leftover.len()
    );
}

/// Ctrl-C in the run's terminal must take the workload with it.
#[test]
fn interrupting_a_run_tears_down_its_workload() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("sigint");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    run.await_ready(READY);
    wait_for_workload(&tag);

    // The interrupt reaches `mirage run` alone: every child leads its own
    // process group, so the terminal's foreground group is the run by
    // itself. Forwarding the signal on — and then still running teardown
    // rather than exiting abruptly — is the whole reason Ctrl-C leaves a
    // clean machine instead of an orphaned process tree.
    run.signal(Signal::SIGINT);
    run.wait(TEARDOWN);

    assert_no_leaks(&tag);
    let leftover = leftover_scratch(&env);
    assert!(
        leftover.is_empty(),
        "an interrupted run left its scratch directory behind: {leftover:?}"
    );
}

/// A `mirage run` killed outright strands its workload — and
/// `mirage cleanup` reclaims it.
///
/// `SIGKILL` is the one case the run cannot handle: no signal handler
/// runs, no teardown, no `Drop`, no cancelled task. Mirage used to close
/// it with `PR_SET_PDEATHSIG`, asking the kernel to kill each workload
/// when its parent died. That was the workspace's only `unsafe`, and it
/// could not be applied to node containers — they are launched from a
/// `spawn_blocking` thread, where pdeathsig tracks a pool thread that may
/// retire while the run is perfectly healthy — so the guarantee was never
/// whole.
///
/// The trade is now explicit: the leak is allowed, and made recoverable.
/// Both halves are asserted here, and the first one deliberately. A test
/// that only checked the cleanup would still pass if something quietly
/// started killing workloads again, and the whole point of removing the
/// `unsafe` was to stop paying for a guarantee mirage was not making.
#[test]
fn a_killed_run_strands_its_workload_and_cleanup_reclaims_it() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("sigkill");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);
    wait_for_workload(&tag);

    run.kill();

    // The accepted half. Give the kernel a moment first: if something
    // *were* killing the workload it would happen within a few
    // milliseconds, and asserting instantly would pass either way.
    std::thread::sleep(Duration::from_millis(500));
    let stranded = find_processes(&tag);
    assert!(
        !stranded.is_empty(),
        "the workload did not outlive its `SIGKILL`ed run. That is a better \
         outcome than this test expects — but it means something is enforcing \
         a guarantee mirage no longer documents, and `mirage cleanup` is no \
         longer the only thing standing between a `kill -9` and a leak."
    );

    // The recovered half. Nothing on disk records these processes; the
    // only evidence is `MIRAGE_SESSION` in each one's environment, which
    // is what `cleanup` scans for.
    let out = env.ok(&["cleanup"]);
    assert!(
        out.contains("stranded process"),
        "cleanup did not report reclaiming anything:\n{out}"
    );

    wait_for("the stranded workload to be reclaimed", TEARDOWN, || {
        count_processes(&tag) == 0
    });
    kill_survivors(&tag);

    // And the scratch directory the killed run could not remove.
    let leftover = leftover_scratch(&env);
    assert!(
        !leftover.contains(&id),
        "session {id}'s scratch directory survived cleanup: {leftover:?}"
    );
}

/// `mirage cleanup` must be safe to run while other runs are healthy.
///
/// The command exists for the machine-wide mess a crash leaves, so it
/// scans every process and every mirage-labelled container. That is only
/// acceptable if a live session is untouchable: reclaiming one would kill
/// somebody else's running job from what is advertised as a tidy-up.
#[test]
fn cleanup_leaves_a_live_run_completely_alone() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("cleanup-live");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);
    wait_for_workload(&tag);

    let out = env.ok(&["cleanup"]);
    assert!(
        !out.contains(&id),
        "cleanup named a live session {id}:\n{out}"
    );

    // Still running, still serving, still able to be exec'd into.
    assert!(count_processes(&tag) > 0, "cleanup killed a live workload");
    assert_eq!(env.live_runs(), vec![id.clone()]);
    assert!(
        env.session_scratch(&id).exists(),
        "cleanup removed a live session's scratch directory"
    );

    run.signal(Signal::SIGINT);
    run.wait(TEARDOWN);
    assert_no_leaks(&tag);
}

/// `--dry-run` must name what a real run would remove, and remove nothing.
#[test]
fn cleanup_dry_run_reports_without_acting() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("cleanup-dry");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    run.await_ready(READY);
    wait_for_workload(&tag);
    run.kill();
    std::thread::sleep(Duration::from_millis(500));

    let preview = env.ok(&["cleanup", "--dry-run"]);
    assert!(
        preview.contains("would kill"),
        "a dry run must say what it would do:\n{preview}"
    );
    assert!(
        count_processes(&tag) > 0,
        "a dry run killed the process it was only supposed to report"
    );

    // And the real thing removes exactly what the preview named.
    //
    // The pids are read *before* the cleanup, not after. Reading them
    // afterwards is how this assertion used to be written, and it could
    // not fail: cleanup had just killed everything tagged, so the list
    // was empty in the ordinary case and the loop body — the only check
    // of the property this test is named for — never ran once. The
    // non-emptiness assertion below keeps it that way on purpose, so a
    // change that made the workload die early turns into a failure here
    // rather than back into a loop over nothing.
    let stranded = find_processes(&tag);
    assert!(
        !stranded.is_empty(),
        "the dry run left nothing for the real cleanup to reclaim, so there \
         is nothing to check the preview against"
    );
    let done = env.ok(&["cleanup"]);
    for pid in stranded {
        assert!(
            preview.contains(&pid.to_string()),
            "the real cleanup reclaimed process {pid}, which the dry run did \
             not name — a preview that omits what will be killed is worse \
             than no preview:\npreview:\n{preview}\ncleanup:\n{done}"
        );
    }
    wait_for("the stranded workload to be reclaimed", TEARDOWN, || {
        count_processes(&tag) == 0
    });
    kill_survivors(&tag);
}

/// The recovery report is machine-readable too, and says the same thing.
///
/// `cleanup` is the command a wrapper script runs before a CI job, and
/// such a script needs to branch on what was found rather than grep a
/// table meant for a person. Every other list command was checked for a
/// parseable `--json`; this one never was, and it is the only one whose
/// document describes work that has already happened — a dry run that
/// silently reported an empty document would read, to a script, exactly
/// like a machine with nothing wrong with it.
#[test]
fn the_cleanup_report_is_parseable_and_names_what_it_found() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("cleanup-json");

    // A clean machine first, so the empty document is pinned as well:
    // the arrays have to be there and empty, not absent.
    let empty: serde_json::Value =
        serde_json::from_str(&env.ok(&["--json", "cleanup", "--dry-run"])).unwrap();
    for key in ["processes", "sessions", "resources", "scratch"] {
        assert_eq!(
            empty[key].as_array().map(Vec::len),
            Some(0),
            "`cleanup --dry-run --json` on a clean runtime must report an \
             empty {key} array: {empty}"
        );
    }

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);
    wait_for_workload(&tag);
    run.kill();
    std::thread::sleep(Duration::from_millis(500));

    let stranded = find_processes(&tag);
    assert!(
        !stranded.is_empty(),
        "the workload did not outlive its `SIGKILL`ed run, so there is \
         nothing for the report to describe"
    );

    let out = env.run(&["--json", "cleanup", "--dry-run"]);
    assert!(
        out.stderr.is_empty(),
        "the report must be the whole of stdout and nothing else: {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    let preview: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("`cleanup --json` must be one JSON document");
    assert_eq!(preview["dry_run"], true, "{preview}");
    let reported: Vec<u64> = preview["processes"]
        .as_array()
        .unwrap_or_else(|| panic!("no processes array: {preview}"))
        .iter()
        .filter_map(|entry| entry["pid"].as_u64())
        .collect();
    for pid in &stranded {
        assert!(
            reported.contains(&u64::from(*pid)),
            "the report does not mention stranded process {pid}: {preview}"
        );
    }
    // The session and its scratch directory are the other two things the
    // killed run could not take with it, and a report that named the
    // processes alone would send a script looking for a clean machine
    // that still has a directory on it.
    assert!(
        preview["sessions"]
            .as_array()
            .is_some_and(|s| s.iter().any(|entry| entry == &serde_json::json!(id))),
        "the report does not name the dead session {id}: {preview}"
    );
    assert!(
        preview["scratch"].as_array().is_some_and(|s| s
            .iter()
            .any(|entry| { entry.as_str().is_some_and(|path| path.contains(&id)) })),
        "the killed run's scratch directory is still there, so the report \
         must name it: {preview}"
    );
    // A dry run is a preview: the pids it named are still running.
    assert!(
        count_processes(&tag) > 0,
        "`--dry-run --json` reclaimed what it was only asked to describe"
    );

    let done = env.ok(&["cleanup"]);
    assert!(
        done.contains(&id) || done.contains("stranded process"),
        "{done}"
    );
    wait_for("the stranded workload to be reclaimed", TEARDOWN, || {
        count_processes(&tag) == 0
    });
    kill_survivors(&tag);
}

/// Cleanup on a clean machine must say so rather than inventing work.
#[test]
fn cleanup_with_nothing_to_do_says_nothing_to_do() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.ok(&["run", "--profile", "p", "--", "/bin/true"]);

    let out = env.ok(&["cleanup"]);
    assert!(
        out.contains("nothing to clean up"),
        "a run that exited cleanly must leave cleanup with nothing to find:\n{out}"
    );
}

/// A workload that ignores `SIGTERM` must still be killed.
#[test]
fn a_sigterm_proof_workload_is_still_killed() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("sigterm-proof");

    // The workload reports both states this test depends on, because
    // neither is visible from outside otherwise: that its traps are
    // armed, and that it has received a `SIGTERM` and declined to die of
    // it. Handling the signal is the same refusal as ignoring it — the
    // shell runs the handler and carries on looping — and it makes the
    // refusal observable.
    let armed = env.root().join("traps-armed");
    let refused = env.root().join("term-refused");
    let script = format!(
        "trap 'touch {}' TERM; trap '' INT; touch {}; MARKER={tag}; \
         while true; do sleep 1; done",
        refused.display(),
        armed.display()
    );
    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &script]);
    run.await_ready(READY);
    wait_for_workload(&tag);
    // The traps themselves, not a pause long enough that they are
    // probably up: `pgrep` finds the workload from its argv, which
    // contains the marker before the shell has run a single line, so
    // "the workload started" says nothing about whether it can survive a
    // signal yet. Interrupting too early kills it outright and the test
    // passes without ever reaching the escalation it is named for.
    wait_for("the workload to arm its signal traps", READY, || {
        armed.exists()
    });

    // The first signal is forwarded to the workload, which refuses it.
    // The second says the user is not waiting any longer, and teardown
    // escalates: `SIGTERM` to the group, a bounded grace period, then
    // `SIGKILL`, which cannot be caught. Without the escalation the run
    // would hang here forever waiting for a process that will never
    // agree to exit.
    run.signal(Signal::SIGTERM);
    // Wait for the refusal rather than for an interval. Only one signal
    // per kind is buffered pending delivery, so a second `SIGTERM` that
    // arrives before the run has taken the first is coalesced into it —
    // and the run then waits forever for an escalation this test
    // believes it sent, which presents as a timeout in `run.wait` far
    // from the line that caused it.
    wait_for(
        "the workload to receive and refuse the first SIGTERM",
        READY,
        || refused.exists(),
    );
    run.signal(Signal::SIGTERM);
    run.wait(TEARDOWN);

    assert_no_leaks(&tag);
}

/// A workload's grandchildren must be cleaned up along with it.
#[test]
fn a_workload_that_forks_has_its_whole_tree_reaped() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let parent_tag = marker("tree-parent");
    let child_tag = marker("tree-child");

    let script = format!(
        "sh -c 'MARKER={child_tag}; while true; do sleep 1; done' & \
         MARKER={parent_tag}; while true; do sleep 1; done"
    );
    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &script]);
    run.await_ready(READY);
    wait_for("the whole process tree to start", READY, || {
        count_processes(&parent_tag) > 0 && count_processes(&child_tag) > 0
    });

    // `SIGTERM` rather than `SIGINT`: a shell without job control leaves
    // a background command in the shell's own process group — so a group
    // signal does reach it — but it also sets that command to ignore
    // `SIGINT`, so an interrupt alone would leave exactly the grandchild
    // this test is about. `SIGTERM` reaches both, which is the point:
    // signalling only the direct child would leave the grandchild
    // running, invisible, and still holding whatever the workload held.
    run.signal(Signal::SIGTERM);
    run.wait(TEARDOWN);

    assert_no_leaks(&parent_tag);
    assert_no_leaks(&child_tag);
}

/// Serving execs must not cost the run anything it never gives back.
///
/// Every `mirage exec` opens a connection to the run's socket and asks
/// one question. Leaking the accepted socket — or the task handling it —
/// eventually takes the run down with `EMFILE`, and the symptom appears
/// far from the cause: a session that has been up for hours suddenly
/// refuses to describe itself.
#[test]
fn serving_many_execs_costs_the_run_no_descriptors_or_zombies() {
    const EXECS: usize = 25;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("exec-churn");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);
    let pid = run.pid().expect("a live run has a pid");

    // Warm up first, so one-off allocations are not counted as growth.
    env.ok(&["exec", "-s", &id, "--", "/bin/sh", "-c", "echo warm"]);
    let before = fd_count(pid);
    assert!(before > 0, "could not read the run's fd table");

    for i in 0..EXECS {
        let script = format!("echo hi-{i}");
        let out = env.ok(&["exec", "-s", &id, "--", "/bin/sh", "-c", &script]);
        assert!(
            out.contains(&format!("hi-{i}")),
            "exec {i} lost its output; got {out:?}"
        );
    }

    // Slack for the runtime's own bookkeeping, but not for per-exec
    // growth: 25 execs leaking even one descriptor each would show.
    let after = fd_count(pid);
    assert!(
        after <= before + 8,
        "the run's fd count grew from {before} to {after} over {EXECS} execs; \
         something is not being closed"
    );
    assert_no_zombies(pid);

    run.signal(Signal::SIGTERM);
    run.wait(TEARDOWN);
    assert_no_leaks(&tag);
}

/// Many execs at once against one run.
#[test]
fn many_concurrent_execs_against_one_run_all_finish() {
    const WORKERS: usize = 8;
    const ROUNDS: usize = 4;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("wide");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);
    let pid = run.pid().expect("a live run has a pid");

    std::thread::scope(|scope| {
        for worker in 0..WORKERS {
            let env = &env;
            let id = id.clone();
            scope.spawn(move || {
                for round in 0..ROUNDS {
                    let code = i32::try_from(worker % 8).unwrap();
                    let script = format!("echo w{worker}-{round}; exit {code}");
                    let out = env.run(&["exec", "-s", &id, "--", "/bin/sh", "-c", &script]);
                    // Every exec must reach a terminal state and report
                    // its own code. One that never does would hang the
                    // terminal it was typed into, forever.
                    assert_eq!(
                        out.status.code(),
                        Some(code),
                        "exec w{worker}-{round} reported the wrong exit code"
                    );
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    assert!(
                        stdout.contains(&format!("w{worker}-{round}")),
                        "exec w{worker}-{round} lost its output; got {stdout:?}"
                    );
                }
            });
        }
    });

    assert_no_zombies(pid);
    run.signal(Signal::SIGTERM);
    run.wait(TEARDOWN);
    assert_no_leaks(&tag);
}

/// Output must survive churn: a burst of captured execs, none of them
/// missing a line.
#[test]
fn output_is_not_lost_under_churn() {
    const EXECS: usize = 30;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("noisy");

    let mut run = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);

    // `--capture-all` is the path that can lose output: the bytes are
    // piped through mirage, buffered until they form whole lines, and
    // printed with a `[rank]` label. A pump task that is cancelled a
    // moment early, or a partial line that is never flushed, silently
    // drops what the workload wrote.
    let mut missing: HashSet<String> = (0..EXECS).map(|i| format!("payload-{i}")).collect();
    for i in 0..EXECS {
        let payload = format!("payload-{i}");
        let script = format!("echo {payload}");
        let out = env.ok(&["exec", "-s", &id, "--", "/bin/sh", "-c", &script]);
        if out.contains(&payload) {
            missing.remove(&payload);
        }
    }
    assert!(
        missing.is_empty(),
        "{} exec(s) lost their output under churn: {missing:?}",
        missing.len()
    );

    run.signal(Signal::SIGTERM);
    run.wait(TEARDOWN);
    assert_no_leaks(&tag);
}

/// Killing an exec must not disturb the run that hosts it.
///
/// The two are separate processes owning separate work, and that
/// separation is what lets a user try something in a second terminal
/// without risking the job in the first. An exec that took the session
/// down with it — or one whose own workload survived it — would break the
/// model in opposite directions, so both are asserted here.
#[test]
fn stopping_an_exec_leaves_its_run_alone_and_its_own_workload_dead() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let run_tag = marker("host-run");
    let exec_tag = marker("guest-exec");

    let mut run = env.spawn_run(
        &["--profile", "p"],
        &["/bin/sh", "-c", &tagged_sleep(&run_tag)],
    );
    let id = run.await_ready(READY);
    wait_for_workload(&run_tag);

    for _ in 0..5 {
        let mut client = env
            .mirage()
            .args([
                "exec",
                "-s",
                &id,
                "--",
                "/bin/sh",
                "-c",
                &tagged_sleep(&exec_tag),
            ])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        wait_for_workload(&exec_tag);

        // Bounded: an exec that stops answering `SIGTERM` is one of the
        // regressions this test exists to catch, and waiting on it
        // forever would hang the strain binary rather than fail here.
        signal(client.id(), Signal::SIGTERM);
        assert!(
            harness::wait_for_exit(&mut client, TEARDOWN),
            "the exec client did not exit within {TEARDOWN:?} of being asked to stop"
        );

        // The exec owned its processes, so they go with it.
        assert_no_leaks(&exec_tag);
        // The run did not, so the session — and the job in the other
        // terminal — carries on.
        assert!(
            count_processes(&run_tag) > 0,
            "stopping an exec killed the run's own workload"
        );
    }

    run.signal(Signal::SIGTERM);
    run.wait(TEARDOWN);
    assert_no_leaks(&run_tag);
}

/// A socket left behind by a killed run must not wedge the next one.
///
/// The kernel does not unlink a socket file when its owner is `SIGKILL`ed,
/// so the file's existence proves nothing about whether a session is
/// live. The next run has to be able to start regardless, and an exec
/// aimed at the corpse has to fail immediately with an explanation rather
/// than hang waiting for an answer that is never coming. Refusing to
/// start until the user deletes files by hand is the failure this
/// prevents.
#[test]
fn a_socket_left_by_a_killed_run_does_not_wedge_the_next_run() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("corpse");

    let mut dead = env.spawn_run(&["--profile", "p"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let dead_id = dead.await_ready(READY);
    wait_for_workload(&tag);
    dead.kill();

    // Exactly what a `SIGKILL`ed run leaves: a socket file with nothing
    // behind it.
    let socket = env.run_socket_dir().join(format!("{dead_id}.sock"));
    assert!(
        socket.exists(),
        "this test is only meaningful while the corpse socket is still there"
    );

    let err = env.fails(&["exec", "-s", &dead_id, "--", "/bin/true"]);
    assert!(
        err.contains(&dead_id) || err.contains("mirage run"),
        "an exec into a dead session must say so; got: {err}"
    );

    // And a fresh run is entirely unaffected by the corpse.
    let live_tag = marker("successor");
    let mut live = env.spawn_run(
        &["--profile", "p"],
        &["/bin/sh", "-c", &tagged_sleep(&live_tag)],
    );
    let live_id = live.await_ready(READY);
    assert_ne!(live_id, dead_id);
    let out = env.ok(&["exec", "-s", &live_id, "--", "/bin/sh", "-c", "echo alive"]);
    assert!(out.contains("alive"), "the new run must be usable: {out:?}");

    live.signal(Signal::SIGTERM);
    live.wait(TEARDOWN);
    assert_no_leaks(&live_tag);

    // The killed run's workload is not this test's claim — see
    // `a_run_killed_outright_leaves_no_workload_behind` — but it must not
    // be left running for the next test either.
    kill_survivors(&tag);
}

/// The full mixed workload: everything at once, for a while.
#[test]
fn mixed_churn_under_load_leaves_a_clean_process_table() {
    const WORKERS: usize = 6;
    const ROUNDS: usize = 4;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let host_tag = marker("mixed-host");
    let guest_tag = marker("mixed-guest");

    let mut host = env.spawn_run(
        &["--profile", "p"],
        &["/bin/sh", "-c", &tagged_sleep(&host_tag)],
    );
    let id = host.await_ready(READY);
    let pid = host.pid().expect("a live run has a pid");

    std::thread::scope(|scope| {
        for _ in 0..WORKERS {
            let env = &env;
            let id = id.clone();
            let guest_tag = guest_tag.clone();
            scope.spawn(move || {
                for _ in 0..ROUNDS {
                    // Under contention an individual command may
                    // legitimately fail. What must hold is the invariant
                    // afterwards, not that every command succeeded, so
                    // failures are tolerated here and the process table
                    // is checked at the end.

                    // A short exec that exits on its own.
                    let _ = env.run(&["exec", "-s", &id, "--", "/bin/sh", "-c", "echo quick"]);

                    // One that fails to spawn at all.
                    let _ = env.run(&["exec", "-s", &id, "--", "no-such-binary-anywhere"]);

                    // A whole second run, brought up and torn down while
                    // the first one is being exec'd into.
                    let _ = env.run(&["run", "--profile", "p", "--", "/bin/sh", "-c", "echo up"]);

                    // A long-lived exec that has to be stopped.
                    let sleep = tagged_sleep(&guest_tag);
                    if let Ok(mut client) = env
                        .mirage()
                        .args(["exec", "-s", &id, "--", "/bin/sh", "-c", &sleep])
                        .stdin(std::process::Stdio::null())
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .spawn()
                    {
                        // Deliberately no wait for it to settle: stopping
                        // an exec while its processes are still forking
                        // is the race that matters, and sleeping first
                        // would tune the test away from it.
                        signal(client.id(), Signal::SIGTERM);
                        let _ = client.wait();
                    }
                }
            });
        }
    });

    assert_no_leaks(&guest_tag);
    assert_no_zombies(pid);

    host.signal(Signal::SIGTERM);
    host.wait(TEARDOWN);

    assert_no_leaks(&host_tag);
    let survivors = find_processes(&guest_tag);
    assert!(
        survivors.is_empty(),
        "{} process(es) survived mixed churn plus the run that hosted them: {survivors:?}",
        survivors.len()
    );

    // Only the runs are gone; so is everything they wrote down.
    let live = env.live_runs();
    assert!(live.is_empty(), "runs survived mixed churn: {live:?}");
    let leftover = leftover_scratch(&env);
    assert!(
        leftover.is_empty(),
        "session scratch directories survived mixed churn: {leftover:?}"
    );
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the strain suite going green while every test in it skipped
    // for a missing emulator runtime. See `assert_suite_can_run`.
    harness::assert_suite_can_run();
}
