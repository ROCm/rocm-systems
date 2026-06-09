//! End-to-end tests for the `mirage` CLI.
//!
//! Each test:
//!
//! * Creates a fresh tempdir and points `XDG_CONFIG_HOME`,
//!   `XDG_RUNTIME_DIR`, and `XDG_STATE_HOME` at it.
//! * Drives the unified `mirage` binary as a subprocess via
//!   `assert_cmd`. The CLI re-execs itself with the `host` subcommand
//!   to spawn per-session hosts \u2014 no separate host binary needed.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::str;
use tempfile::TempDir;

struct Env {
    _dir: TempDir,
    config: PathBuf,
    runtime: PathBuf,
    mirage_bin: PathBuf,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config");
        let runtime = dir.path().join("runtime");
        let mirage_bin = PathBuf::from(env!("CARGO_BIN_EXE_mirage"));
        Self {
            _dir: dir,
            config,
            runtime,
            mirage_bin,
        }
    }

    fn mirage(&self) -> Command {
        let mut c = Command::new(&self.mirage_bin);
        c.env("XDG_CONFIG_HOME", &self.config)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("XDG_STATE_HOME", self._dir.path().join("state"))
            // ensure the CLI re-execs the same binary when spawning
            // hosts (covers the case where it lives in a non-standard
            // location during testing).
            .env("MIRAGE_BIN", &self.mirage_bin)
            // make sure user env doesn't leak through.
            .env_remove("MIRAGE_LOG");
        c
    }
}

#[test]
fn paths_prints_overridden_dirs() {
    let env = Env::new();
    env.mirage()
        .arg("paths")
        .assert()
        .success()
        .stdout(str::contains(env.config.display().to_string()))
        .stdout(str::contains(env.runtime.display().to_string()));
}

#[test]
fn profile_create_list_show_delete() {
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "p1", "--description", "the test"])
        .assert()
        .success();
    env.mirage()
        .args(["profile", "list"])
        .assert()
        .success()
        .stdout(str::contains("p1"));
    env.mirage()
        .args(["profile", "list", "-l"])
        .assert()
        .success()
        .stdout(str::contains("the test"));
    env.mirage()
        .args(["profile", "show", "p1"])
        .assert()
        .success()
        .stdout(str::contains("\"name\": \"p1\""));
    env.mirage()
        .args(["profile", "delete", "p1", "-f"])
        .assert()
        .success();
    env.mirage()
        .args(["profile", "show", "p1"])
        .assert()
        .failure();
}

#[test]
fn run_command_streams_stdout_stderr_and_exit_code() {
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();
    let out = env
        .mirage()
        .args([
            "run",
            "--profile",
            "p",
            "--",
            "/bin/sh",
            "-c",
            "echo hello; echo oops 1>&2; exit 7",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The exec runs under a PTY, so stdout and stderr are merged onto the
    // terminal (the stdout stream), exactly as in a real terminal.
    let combined = format!("{stdout}{stderr}");
    assert!(combined.contains("hello"), "output was: {combined:?}");
    assert!(combined.contains("oops"), "output was: {combined:?}");
    assert_eq!(out.status.code(), Some(7));
}

#[test]
fn run_cleans_up_transient_session() {
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();
    env.mirage()
        .args(["run", "--profile", "p", "--", "/bin/true"])
        .assert()
        .success();
    let out = env.mirage().args(["session", "list"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("(no sessions)"),
        "expected no sessions, got stderr={stderr:?} stdout={:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn session_start_detach_exec_then_stop() {
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();
    env.mirage()
        .args(["session", "start", "--profile", "p", "--id", "s1"])
        .assert()
        .success();
    // submit an exec without attaching
    env.mirage()
        .args([
            "exec",
            "start",
            "s1",
            "--keep",
            "--detach",
            "--",
            "/bin/echo",
            "hi",
        ])
        .assert()
        .success()
        .stdout(str::contains("e-000000"));
    // wait for it to finish
    for _ in 0..100 {
        let out = env
            .mirage()
            .args(["exec", "show", "s1", "e-000000"])
            .output()
            .unwrap();
        if String::from_utf8_lossy(&out.stdout).contains("\"ended\": true") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    // logs (non-follow) should contain "hi"
    let out = env
        .mirage()
        .args(["logs", "s1", "e-000000"])
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("hi"),
        "stdout={:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    // stop the session
    env.mirage()
        .args(["session", "stop", "s1", "-f"])
        .assert()
        .success();
    // verify the session dir is gone
    assert!(!env.runtime.join("mirage/session/s1").exists());
}

#[test]
fn attach_to_long_running_exec_then_signal() {
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();
    env.mirage()
        .args(["session", "start", "--profile", "p", "--id", "s2"])
        .assert()
        .success();
    env.mirage()
        .args([
            "exec", "start", "s2", "--keep", "--detach", "--", "/bin/sh", "-c", "sleep 30",
        ])
        .assert()
        .success();
    // wait for it to start
    for _ in 0..100 {
        let out = env
            .mirage()
            .args(["exec", "show", "s2", "e-000000"])
            .output()
            .unwrap();
        if String::from_utf8_lossy(&out.stdout).contains("\"started\": true") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    // signal it
    env.mirage()
        .args(["exec", "signal", "s2", "e-000000", "KILL"])
        .assert()
        .success();
    // it should end soon
    let mut ended = false;
    for _ in 0..100 {
        let out = env
            .mirage()
            .args(["exec", "show", "s2", "e-000000"])
            .output()
            .unwrap();
        if String::from_utf8_lossy(&out.stdout).contains("\"ended\": true") {
            ended = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(ended, "exec did not end after SIGKILL");
    env.mirage()
        .args(["session", "stop", "s2", "-f"])
        .assert()
        .success();
}

#[test]
fn json_output_is_parseable() {
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();
    let out = env
        .mirage()
        .args(["--json", "profile", "list"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v, serde_json::json!(["p"]));
}

#[test]
fn invalid_session_id_is_rejected() {
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();
    env.mirage()
        .args(["session", "start", "--profile", "p", "--id", "../oops"])
        .assert()
        .failure();
}

#[test]
fn duplicate_session_id_fails() {
    let env = Env::new();
    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();
    env.mirage()
        .args(["session", "start", "--profile", "p", "--id", "dup"])
        .assert()
        .success();
    env.mirage()
        .args(["session", "start", "--profile", "p", "--id", "dup"])
        .assert()
        .failure();
    env.mirage()
        .args(["session", "stop", "dup", "-f"])
        .assert()
        .success();
}
