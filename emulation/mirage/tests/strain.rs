//! Strain tests: rapidly create, use, kill and delete sessions and
//! execs, then assert that nothing survived.
//!
//! # What these are actually testing
//!
//! Cleanup bugs are almost never visible in a single clean run. They show
//! up under churn, at the boundaries — a destroy that races a spawn, a
//! kill that arrives while a process is still forking, a client that
//! disconnects mid-attach, a daemon that dies with work in flight. These
//! tests manufacture those boundaries deliberately and then check the one
//! thing that cannot be faked: the operating system's process table.
//!
//! Every assertion here is external. Not "the supervisor believes the
//! session is gone" but "no process with this marker exists" and "no
//! zombie with this parent exists". Internal bookkeeping agreeing with
//! itself is exactly the failure mode the previous design had — it
//! confidently reported sessions as stopped while their process trees
//! kept running.
//!
//! # Zombies specifically
//!
//! A zombie answers `kill(pid, 0)` just like a live process, so a
//! liveness check alone cannot see one. These tests read `/proc/<pid>/stat`
//! and assert the daemon has no children in state `Z`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod harness;

use std::collections::HashSet;
use std::time::{Duration, Instant};

use harness::{
    Env, assert_no_leaks, count_processes, find_processes, marker, pid_alive, pid_is_zombie,
    skip_without_emulator, tagged_sleep, wait_for,
};

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
        if !pid_is_zombie(child) {
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

/// Assert the daemon is not accumulating unreaped children.
///
/// This is the direct test of the bug that motivated the rewrite. A
/// process that is waited on leaves the table immediately; one that is
/// not lingers as a zombie until its parent dies.
fn assert_no_zombies(env: &Env) {
    let Some(pid) = env.daemon_pid() else {
        return;
    };
    // Reaping is a syscall, not instantaneous; allow it to happen.
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && !zombie_children(pid).is_empty() {
        std::thread::sleep(Duration::from_millis(50));
    }
    let zombies = zombie_children(pid);
    assert!(
        zombies.is_empty(),
        "the daemon (pid {pid}) has {} unreaped child process(es): {zombies:?}. \
         Every spawned child must be waited on.",
        zombies.len()
    );
}

/// Rapidly create and destroy sessions.
#[test]
fn rapid_session_create_and_destroy_leaves_nothing() {
    const ROUNDS: usize = 40;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    for round in 0..ROUNDS {
        let id = format!("churn-{round}");
        env.ok(&[
            "session", "start", "--profile", "p", "--id", &id, "--no-input",
        ]);
        env.ok(&["session", "stop", &id]);
    }

    let list = env.ok(&["session", "list"]);
    assert!(
        !list.contains("churn-"),
        "sessions survived {ROUNDS} create/destroy rounds: {list}"
    );
    assert_no_zombies(&env);

    // Scratch directories must not accumulate: one per session, forever,
    // is a slow-motion disk leak.
    let scratch_root = env.root().join("runtime/mirage/session");
    if scratch_root.exists() {
        let leftover: Vec<_> = std::fs::read_dir(&scratch_root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert!(
            leftover.is_empty(),
            "{} session scratch director(ies) outlived their sessions: {leftover:?}",
            leftover.len()
        );
    }
}

/// Create sessions with running workloads and destroy them immediately,
/// giving the workload no time to settle.
#[test]
fn destroying_sessions_with_live_workloads_leaves_no_processes() {
    const ROUNDS: usize = 25;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("live-destroy");

    for round in 0..ROUNDS {
        let id = format!("live-{round}");
        env.ok(&[
            "session", "start", "--profile", "p", "--id", &id, "--no-input",
        ]);
        env.ok(&[
            "exec",
            "start",
            &id,
            "--detach",
            "--",
            "/bin/sh",
            "-c",
            &tagged_sleep(&tag),
        ]);
        // Deliberately no wait. Destroying a session while its workload is
        // still forking is the race that matters, and sleeping first would
        // tune the test away from it.
        env.ok(&["session", "stop", &id]);
    }

    assert_no_leaks(&tag);
    assert_no_zombies(&env);
}

/// Hammer one session with execs that are killed as fast as they start.
#[test]
fn rapid_exec_churn_within_one_session_leaves_nothing() {
    const ROUNDS: usize = 60;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "hammer");
    let tag = marker("exec-churn");

    for _ in 0..ROUNDS {
        let exec = env
            .ok(&[
                "exec",
                "start",
                "hammer",
                "--detach",
                "--keep",
                "--",
                "/bin/sh",
                "-c",
                &tagged_sleep(&tag),
            ])
            .trim()
            .to_string();
        env.ok(&["exec", "remove", "hammer", &exec]);
    }

    assert_no_leaks(&tag);
    assert!(
        env.ok(&["exec", "list", "hammer"])
            .lines()
            .filter(|l| l.starts_with("e-"))
            .count()
            == 0,
        "removed execs must not remain listed"
    );
    assert_no_zombies(&env);
    env.ok(&["session", "stop", "hammer"]);
}

/// Interleave many sessions concurrently rather than in sequence.
#[test]
fn concurrent_session_churn_leaves_nothing() {
    const WORKERS: usize = 8;
    const ROUNDS: usize = 6;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("concurrent");

    std::thread::scope(|scope| {
        for worker in 0..WORKERS {
            let env = &env;
            let tag = tag.clone();
            scope.spawn(move || {
                for round in 0..ROUNDS {
                    let id = format!("c{worker}-{round}");
                    // Under contention an individual command may
                    // legitimately fail (a session id that clashes, a
                    // daemon busy tearing something down). What must hold
                    // is the invariant afterwards, not that every command
                    // succeeded, so failures are tolerated here and the
                    // process table is checked at the end.
                    let started = env
                        .run(&[
                            "session", "start", "--profile", "p", "--id", &id, "--no-input",
                        ])
                        .status
                        .success();
                    if !started {
                        continue;
                    }
                    let _ = env.run(&[
                        "exec",
                        "start",
                        &id,
                        "--detach",
                        "--",
                        "/bin/sh",
                        "-c",
                        &tagged_sleep(&tag),
                    ]);
                    let _ = env.run(&["session", "stop", &id]);
                }
            });
        }
    });

    assert_no_leaks(&tag);
    let list = env.ok(&["session", "list"]);
    assert!(
        !list.contains("c0-") && !list.contains("c7-"),
        "sessions survived concurrent churn: {list}"
    );
    assert_no_zombies(&env);
}

/// Kill the client mid-attach. Detaching must not disturb the workload,
/// and destroying the session afterwards must still clean it up.
#[test]
fn killing_an_attached_client_does_not_orphan_the_workload() {
    use std::process::Stdio;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "detach");
    let tag = marker("client-kill");

    let exec = env
        .ok(&[
            "exec",
            "start",
            "detach",
            "--detach",
            "--keep",
            "--",
            "/bin/sh",
            "-c",
            &tagged_sleep(&tag),
        ])
        .trim()
        .to_string();
    wait_for("the workload to start", Duration::from_secs(15), || {
        count_processes(&tag) > 0
    });

    for _ in 0..5 {
        let mut client = env
            .mirage()
            .args(["attach", "detach", &exec])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        std::thread::sleep(Duration::from_millis(150));
        client.kill().unwrap();
        client.wait().unwrap();

        // A client going away is not a reason to stop the workload: other
        // clients may be attached, and the workload is not the CLI's to
        // own.
        assert!(
            count_processes(&tag) > 0,
            "killing an attached client killed the workload"
        );
    }

    env.ok(&["session", "stop", "detach"]);
    assert_no_leaks(&tag);
    assert_no_zombies(&env);
}

/// A workload that ignores SIGTERM must still be cleaned up.
#[test]
fn sigterm_proof_workloads_are_still_killed() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "stubborn");
    let tag = marker("sigterm-proof");

    env.ok(&[
        "exec",
        "start",
        "stubborn",
        "--detach",
        "--",
        "/bin/sh",
        "-c",
        &format!("trap '' TERM INT; MARKER={tag}; while true; do sleep 1; done"),
    ]);
    wait_for("the stubborn workload to start", Duration::from_secs(15), || {
        count_processes(&tag) > 0
    });
    // Give the shell time to install its traps, so this really exercises
    // the SIGKILL escalation rather than a lucky early SIGTERM.
    std::thread::sleep(Duration::from_millis(500));

    env.ok(&["session", "stop", "stubborn"]);
    assert_no_leaks(&tag);
    assert_no_zombies(&env);
}

/// A workload's grandchildren must be cleaned up along with it.
#[test]
fn descendant_processes_are_cleaned_up_with_the_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "tree");
    let parent_tag = marker("tree-parent");
    let child_tag = marker("tree-child");

    env.ok(&[
        "exec",
        "start",
        "tree",
        "--detach",
        "--",
        "/bin/sh",
        "-c",
        &format!(
            "sh -c 'MARKER={child_tag}; while true; do sleep 1; done' & \
             MARKER={parent_tag}; while true; do sleep 1; done"
        ),
    ]);
    wait_for("the whole process tree to start", Duration::from_secs(15), || {
        count_processes(&parent_tag) > 0 && count_processes(&child_tag) > 0
    });

    env.ok(&["session", "stop", "tree"]);

    // Signalling only the direct child would leave the grandchild running
    // — invisible, and still holding whatever the workload held.
    assert_no_leaks(&parent_tag);
    assert_no_leaks(&child_tag);
    assert_no_zombies(&env);
}

/// Stopping the daemon must stop everything it owns.
#[test]
fn stopping_the_daemon_tears_down_every_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("daemon-stop");

    for i in 0..4 {
        let id = format!("d{i}");
        env.ok(&[
            "session", "start", "--profile", "p", "--id", &id, "--no-input",
        ]);
        env.ok(&[
            "exec",
            "start",
            &id,
            "--detach",
            "--",
            "/bin/sh",
            "-c",
            &tagged_sleep(&tag),
        ]);
    }
    wait_for("workloads to start", Duration::from_secs(15), || {
        count_processes(&tag) >= 4
    });

    env.stop_daemon();

    // The daemon owns these processes; its exit must take them with it,
    // rather than orphaning them to init where nothing will ever clean
    // them up.
    assert_no_leaks(&tag);
}

/// A `SIGKILL`ed daemon cannot clean up, but must not wedge the system:
/// the next daemon has to be able to take over.
#[test]
fn a_killed_daemon_does_not_prevent_a_new_one_from_starting() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "before");

    let pid = env.daemon_pid().expect("a daemon must be running");
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(i32::try_from(pid).unwrap()),
        nix::sys::signal::Signal::SIGKILL,
    )
    .unwrap();
    wait_for("the daemon to die", Duration::from_secs(15), || {
        !pid_alive(pid)
    });

    // The socket file is still on disk with nothing behind it. Startup
    // must recognise it as stale (nobody holds the lock) and reclaim it,
    // rather than refusing forever and leaving the user to delete files
    // by hand.
    let list = env.ok(&["session", "list"]);
    assert!(
        !list.contains("before"),
        "sessions must not survive the daemon that owned them: {list}"
    );
    let new_pid = env.daemon_pid().expect("a fresh daemon must have started");
    assert_ne!(new_pid, pid);

    // And the new daemon is fully functional.
    env.start_session("p", "after");
    assert!(env.ok(&["session", "list"]).contains("after"));
}

/// A daemon whose socket is deleted must not linger holding workloads.
///
/// This is not hypothetical: it happened repeatedly while developing
/// these tests. A test binary was killed, its tempdir was cleaned up, and
/// the daemon inside it kept running for hours — unreachable by any
/// client, `daemon stop` included, while still owning live processes.
#[test]
fn a_daemon_whose_socket_disappears_shuts_itself_down() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "orphaned");
    let tag = marker("orphan");
    env.ok(&[
        "exec",
        "start",
        "orphaned",
        "--detach",
        "--",
        "/bin/sh",
        "-c",
        &tagged_sleep(&tag),
    ]);
    wait_for("the workload to start", Duration::from_secs(15), || {
        count_processes(&tag) > 0
    });
    let pid = env.daemon_pid().expect("a daemon must be running");

    // Simulate the runtime directory being removed out from under it.
    std::fs::remove_file(env.socket()).expect("remove the socket");

    // The daemon must notice and tear everything down. The watchdog polls
    // on a ten-second cadence, so allow a few cycles.
    wait_for("the daemon to notice it is unreachable", Duration::from_secs(90), || {
        !pid_alive(pid)
    });
    assert_no_leaks(&tag);
}

/// Many execs at once in one session.
#[test]
fn many_concurrent_execs_all_complete_and_are_reaped() {
    const EXECS: usize = 50;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "wide");

    let mut ids = Vec::with_capacity(EXECS);
    for i in 0..EXECS {
        ids.push(
            env.ok(&[
                "exec",
                "start",
                "wide",
                "--detach",
                "--keep",
                "--",
                "/bin/sh",
                "-c",
                &format!("echo out-{i}; exit {}", i % 8),
            ])
            .trim()
            .to_string(),
        );
    }

    // Every exec must reach a terminal state. One that never does would
    // hang any client attached to it, forever.
    for (i, id) in ids.iter().enumerate() {
        wait_for(&format!("exec {id} to finish"), Duration::from_secs(60), || {
            serde_json::from_str::<serde_json::Value>(&env.ok(&["exec", "show", "wide", id]))
                .map(|s| s["ended"] == true)
                .unwrap_or(false)
        });
        let status: serde_json::Value =
            serde_json::from_str(&env.ok(&["exec", "show", "wide", id])).unwrap();
        assert_eq!(
            status["exit_code"],
            i % 8,
            "exec {id} reported the wrong exit code"
        );
    }

    assert_no_zombies(&env);
    env.ok(&["session", "stop", "wide"]);
}

/// Output must survive churn: a burst of execs each producing output,
/// all of it readable afterwards.
#[test]
fn output_is_not_lost_under_churn() {
    const EXECS: usize = 30;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "noisy");

    let mut expected = HashSet::new();
    let mut ids = Vec::new();
    for i in 0..EXECS {
        expected.insert(format!("payload-{i}"));
        ids.push((
            i,
            env.ok(&[
                "exec",
                "start",
                "noisy",
                "--detach",
                "--keep",
                "--",
                "/bin/sh",
                "-c",
                &format!("echo payload-{i}"),
            ])
            .trim()
            .to_string(),
        ));
    }

    for (i, id) in &ids {
        wait_for(&format!("exec {id} to finish"), Duration::from_secs(60), || {
            serde_json::from_str::<serde_json::Value>(&env.ok(&["exec", "show", "noisy", id]))
                .map(|s| s["ended"] == true)
                .unwrap_or(false)
        });
        let logs = env.ok(&["logs", "noisy", id]);
        assert!(
            logs.contains(&format!("payload-{i}")),
            "exec {id} lost its output; got {logs:?}"
        );
    }

    env.ok(&["session", "stop", "noisy"]);
}

/// The daemon must not accumulate file descriptors across churn.
///
/// Every session opens pipes, a scratch directory and possibly a socket.
/// Leaking any of them eventually takes the daemon down with EMFILE, and
/// the symptom appears far from the cause.
#[test]
fn the_daemon_does_not_leak_file_descriptors() {
    const ROUNDS: usize = 25;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    // Warm up so one-off allocations are not counted as growth.
    env.start_session("p", "warmup");
    env.ok(&["run", "--profile", "p", "--", "/bin/true"]);
    env.ok(&["session", "stop", "warmup"]);

    let pid = env.daemon_pid().expect("a daemon must be running");
    let count_fds = || {
        std::fs::read_dir(format!("/proc/{pid}/fd"))
            .map(|d| d.count())
            .unwrap_or(0)
    };
    let before = count_fds();
    assert!(before > 0, "could not read the daemon's fd table");

    for round in 0..ROUNDS {
        let id = format!("fd-{round}");
        env.ok(&[
            "session", "start", "--profile", "p", "--id", &id, "--no-input",
        ]);
        env.ok(&[
            "exec", "start", &id, "--", "/bin/sh", "-c", "echo hi",
        ]);
        env.ok(&["session", "stop", &id]);
    }

    // Allow slack for the runtime's own bookkeeping, but not for
    // per-round growth: 25 rounds leaking even one fd each would show.
    let after = count_fds();
    assert!(
        after <= before + 16,
        "the daemon's fd count grew from {before} to {after} over {ROUNDS} rounds; \
         something is not being closed"
    );
}

/// Sessions destroyed while bring-up is still in flight must not leak.
#[test]
fn destroying_a_session_during_bring_up_is_clean() {
    const ROUNDS: usize = 20;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

    for round in 0..ROUNDS {
        let id = format!("early-{round}");
        // `--no-wait` returns as soon as the session is registered, so
        // the destroy lands while bring-up is still running.
        env.ok(&[
            "session", "start", "--profile", "p", "--id", &id, "--no-wait", "--no-input",
        ]);
        let _ = env.run(&["session", "stop", &id]);
    }

    let list = env.ok(&["session", "list"]);
    assert!(
        !list.contains("early-"),
        "sessions survived being destroyed mid-bring-up: {list}"
    );
    assert_no_zombies(&env);
}

/// The full mixed workload: everything at once, for a while.
#[test]
fn mixed_churn_under_load_leaves_a_clean_process_table() {
    const WORKERS: usize = 6;
    const ROUNDS: usize = 5;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let tag = marker("mixed");

    std::thread::scope(|scope| {
        for worker in 0..WORKERS {
            let env = &env;
            let tag = tag.clone();
            scope.spawn(move || {
                for round in 0..ROUNDS {
                    let id = format!("m{worker}-{round}");
                    if !env
                        .run(&[
                            "session", "start", "--profile", "p", "--id", &id, "--no-input",
                        ])
                        .status
                        .success()
                    {
                        continue;
                    }

                    // A short-lived exec that exits on its own.
                    let _ = env.run(&["exec", "start", &id, "--", "/bin/sh", "-c", "echo quick"]);

                    // A long-lived one that must be killed.
                    let _ = env.run(&[
                        "exec",
                        "start",
                        &id,
                        "--detach",
                        "--keep",
                        "--",
                        "/bin/sh",
                        "-c",
                        &tagged_sleep(&tag),
                    ]);

                    // One that fails to spawn at all.
                    let _ = env.run(&["exec", "start", &id, "--", "no-such-binary-anywhere"]);

                    // Half the workers destroy their sessions; the other
                    // half leave them for the daemon shutdown to collect.
                    // Both paths must end with nothing running.
                    if worker % 2 == 0 {
                        let _ = env.run(&["session", "stop", &id]);
                    }
                }
            });
        }
    });

    // Whatever is left is the daemon's responsibility on the way out.
    env.stop_daemon();
    assert_no_leaks(&tag);

    let survivors = find_processes(&tag);
    assert!(
        survivors.is_empty(),
        "{} workload process(es) survived mixed churn plus a daemon shutdown: {survivors:?}",
        survivors.len()
    );
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the strain suite going green while every test in it skipped
    // for a missing emulator runtime. See `assert_suite_can_run`.
    harness::assert_suite_can_run();
}
