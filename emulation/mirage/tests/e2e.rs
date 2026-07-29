//! End-to-end tests for the `mirage` CLI.
//!
//! Each test drives the real binary as a subprocess against a private
//! XDG root and a private daemon, so what is exercised is the whole
//! stack: CLI → Unix socket → daemon → supervisor → real processes.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod harness;

use std::time::Duration;

use harness::{
    Env, TEST_EMULATOR, assert_no_leaks, marker, skip_without_emulator, tagged_sleep, wait_for,
};

#[test]
fn paths_reports_the_overridden_directories_and_the_socket() {
    let env = Env::new();
    let out = env.ok(&["paths"]);
    assert!(out.contains("config:"), "{out}");
    assert!(out.contains(env.root().to_str().unwrap()), "{out}");
    // The socket is part of the layout now, and users need to be able to
    // find it when something goes wrong.
    assert!(out.contains("socket:"), "{out}");
    assert!(out.contains(env.socket().to_str().unwrap()), "{out}");
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
}

#[test]
fn run_cleans_up_its_transient_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.ok(&["run", "--profile", "p", "--", "/bin/true"]);

    // `mirage run` creates a session, uses it and destroys it. Leaving it
    // behind would accumulate one session per invocation.
    let list = env.ok(&["session", "list"]);
    assert!(
        !list.contains("ready"),
        "a transient session outlived `mirage run`: {list}"
    );
}

#[test]
fn run_keeps_the_session_when_asked() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.ok(&[
        "run",
        "--profile",
        "p",
        "--keep-session",
        "--",
        "/bin/true",
    ]);
    let list = env.ok(&["session", "list"]);
    assert!(list.contains("ready"), "{list}");
}

#[test]
fn a_session_started_in_one_invocation_is_usable_from_another() {
    // The reason mirage has a daemon at all: sessions outlive the command
    // that created them.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let id = env.start_session("p", "cross");
    assert_eq!(id, "cross");

    let out = env.run(&["exec", "start", "cross", "--", "/bin/sh", "-c", "exit 5"]);
    assert_eq!(out.status.code(), Some(5));

    let state: serde_json::Value =
        serde_json::from_str(&env.ok(&["session", "show", "cross"])).unwrap();
    assert_eq!(state["def"]["id"], "cross");
    assert_eq!(state["health"]["healthy"], true);

    env.ok(&["session", "stop", "cross"]);
    let list = env.ok(&["session", "list"]);
    assert!(!list.contains("cross"), "{list}");
}

#[test]
fn a_stopped_session_stays_gone() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "gone");
    env.ok(&["session", "stop", "gone"]);

    // Nothing may resurrect it. The previous design could: a heartbeat
    // write racing the destroy re-created the session directory, and the
    // "stopped" session reappeared in listings.
    std::thread::sleep(Duration::from_millis(500));
    let list = env.ok(&["session", "list"]);
    assert!(!list.contains("gone"), "{list}");
    let err = env.fails(&["session", "show", "gone"]);
    assert!(err.contains("not found"), "{err}");
}

#[test]
fn destroying_a_session_kills_its_running_execs() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "kill-me");
    let tag = marker("destroy");

    env.ok(&[
        "exec",
        "start",
        "kill-me",
        "--detach",
        "--",
        "/bin/sh",
        "-c",
        &tagged_sleep(&tag),
    ]);
    wait_for("the workload to start", Duration::from_secs(15), || {
        harness::count_processes(&tag) > 0
    });

    env.ok(&["session", "stop", "kill-me"]);

    // `session stop` returns only once teardown has finished, so the
    // process must already be gone rather than eventually gone.
    assert_no_leaks(&tag);
}

#[test]
fn attaching_to_a_long_running_exec_and_signalling_it_ends_it() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "sig");
    let tag = marker("signal");

    let exec = env
        .ok(&[
            "exec",
            "start",
            "sig",
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
        harness::count_processes(&tag) > 0
    });

    env.ok(&["exec", "signal", "sig", &exec, "TERM"]);

    wait_for("the exec to end", Duration::from_secs(15), || {
        serde_json::from_str::<serde_json::Value>(&env.ok(&["exec", "show", "sig", &exec]))
            .map(|s| s["ended"] == true)
            .unwrap_or(false)
    });
    let status: serde_json::Value =
        serde_json::from_str(&env.ok(&["exec", "show", "sig", &exec])).unwrap();
    assert_eq!(status["exit_code"], 128 + libc::SIGTERM);
    assert_no_leaks(&tag);
}

#[test]
fn a_removed_exec_takes_its_processes_with_it() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "rm");
    let tag = marker("remove");

    let exec = env
        .ok(&[
            "exec",
            "start",
            "rm",
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
        harness::count_processes(&tag) > 0
    });

    env.ok(&["exec", "remove", "rm", &exec]);
    assert_no_leaks(&tag);
}

#[test]
fn logs_replays_a_finished_exec() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "logs");

    let exec = env
        .ok(&[
            "exec",
            "start",
            "logs",
            "--detach",
            "--keep",
            "--",
            "/bin/sh",
            "-c",
            "echo recorded-output",
        ])
        .trim()
        .to_string();

    wait_for("the exec to finish", Duration::from_secs(15), || {
        serde_json::from_str::<serde_json::Value>(&env.ok(&["exec", "show", "logs", &exec]))
            .map(|s| s["ended"] == true)
            .unwrap_or(false)
    });

    let logs = env.ok(&["logs", "logs", &exec]);
    assert!(logs.contains("recorded-output"), "{logs:?}");
}

#[test]
fn logs_are_complete_for_a_process_that_writes_and_exits_immediately() {
    // The narrow window: output travels pump -> channel -> hub, and if
    // the exit packet is published before the last bytes arrive, a client
    // that stops reading at the exit sees nothing. Repeat, because it
    // only shows up when the forwarder is scheduled late.
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "fast");

    for i in 0..15 {
        let exec = env
            .ok(&[
                "exec",
                "start",
                "fast",
                "--detach",
                "--keep",
                "--",
                "/bin/sh",
                "-c",
                &format!("echo quick-{i}"),
            ])
            .trim()
            .to_string();
        wait_for("the exec to finish", Duration::from_secs(30), || {
            serde_json::from_str::<serde_json::Value>(&env.ok(&["exec", "show", "fast", &exec]))
                .map(|s| s["ended"] == true)
                .unwrap_or(false)
        });
        let logs = env.ok(&["logs", "fast", &exec]);
        assert!(
            logs.contains(&format!("quick-{i}")),
            "round {i}: output was lost; got {logs:?}"
        );
    }

    env.ok(&["session", "stop", "fast"]);
}

#[test]
fn exec_ids_are_listed_in_order() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "many");
    for _ in 0..3 {
        env.ok(&[
            "exec", "start", "many", "--detach", "--keep", "--", "/bin/true",
        ]);
    }
    let list = env.ok(&["exec", "list", "many"]);
    assert!(list.contains("e-000000"), "{list}");
    assert!(list.contains("e-000001"), "{list}");
    assert!(list.contains("e-000002"), "{list}");
}

#[test]
fn json_output_is_parseable() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "js");

    let sessions: serde_json::Value =
        serde_json::from_str(&env.ok(&["--json", "session", "list"])).unwrap();
    assert!(sessions.is_array(), "{sessions}");
    assert_eq!(sessions[0]["def"]["id"], "js");

    let profiles: serde_json::Value =
        serde_json::from_str(&env.ok(&["--json", "profile", "list"])).unwrap();
    assert!(profiles.is_array());
}

#[test]
fn an_invalid_session_id_is_rejected() {
    let env = Env::new();
    let err = env.fails(&["session", "show", "../escape"]);
    assert!(
        err.contains("invalid") || err.contains("id"),
        "an id that could escape the runtime directory must be rejected: {err}"
    );
}

#[test]
fn a_duplicate_session_id_is_rejected() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "dup");
    let err = env.fails(&[
        "session", "start", "--profile", "p", "--id", "dup", "--no-input",
    ]);
    assert!(err.contains("already exists"), "{err}");
}

#[test]
fn a_session_on_a_missing_profile_fails_clearly() {
    let env = Env::new();
    let err = env.fails(&[
        "session", "start", "--profile", "nope", "--id", "s", "--no-input",
    ]);
    assert!(err.contains("profile not found"), "{err}");
    // And no half-created session is left behind.
    let list = env.ok(&["session", "list"]);
    assert!(!list.contains("\ns "), "{list}");
}

#[test]
fn running_a_command_that_does_not_exist_reports_why() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    let out = env.run(&["run", "--profile", "p", "--", "definitely-not-a-binary"]);

    // The important property is that it terminates at all: an exec with
    // no terminal state hangs the client forever.
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
fn state_purge_stops_everything_and_removes_the_runtime_directory() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");
    env.start_session("p", "purged");
    let tag = marker("purge");
    env.ok(&[
        "exec",
        "start",
        "purged",
        "--detach",
        "--",
        "/bin/sh",
        "-c",
        &tagged_sleep(&tag),
    ]);
    wait_for("the workload to start", Duration::from_secs(15), || {
        harness::count_processes(&tag) > 0
    });

    env.ok(&["state", "purge", "--force"]);

    // Purge must stop the daemon before deleting the runtime tree, or it
    // races the daemon's own teardown inside the directory it is removing.
    assert_no_leaks(&tag);
    assert!(
        !env.root().join("runtime/mirage").exists(),
        "the runtime directory survived a purge"
    );
    // Configuration is deliberately left alone without `--all`.
    assert!(
        env.root().join("config/mirage").exists(),
        "purge must not remove profiles unless --all is given"
    );
}

#[test]
fn stdin_is_forwarded_to_the_workload() {
    use std::io::Write as _;
    use std::process::Stdio;

    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_profile("p");

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
    // Closing our stdin ends the forwarder, which closes the workload's
    // stdin, which is what makes `cat` exit.
    drop(stdin);

    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("piped through"), "{stdout}");
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the e2e suite going green while every test in it skipped
    // for a missing emulator runtime. See `assert_suite_can_run`.
    harness::assert_suite_can_run();
}
