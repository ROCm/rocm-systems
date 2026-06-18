//! End-to-end tests for the file-driven exec signal pathway and the
//! `keep` flag.
//!
//! These spin up a real `mirage host` subprocess (via the
//! `session start` CLI) and then poke at the on-disk state directly
//! to assert that:
//!
//! * `mirage exec signal` writes a `signal` file under the exec
//!   directory and the host consumes it,
//! * the host removes the signal file after applying it,
//! * `keep=false` execs have their directory removed by the host on
//!   exit, regardless of whether they exited naturally or were killed,
//! * `keep=true` execs persist after death and after the kill,
//! * destroying a session with a containerised runtime record invokes
//!   the provider with `rm -f <container>` for each node and
//!   `network rm <network>` when `container.json` is present (and
//!   skips the provider entirely when absent).

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use assert_cmd::prelude::*;
use tempfile::TempDir;

struct Env {
    _dir: TempDir,
    runtime: PathBuf,
    config: PathBuf,
    state: PathBuf,
    mirage_bin: PathBuf,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join("runtime");
        let config = dir.path().join("config");
        let state = dir.path().join("state");
        let mirage_bin = PathBuf::from(env!("CARGO_BIN_EXE_mirage"));
        Self {
            _dir: dir,
            runtime,
            config,
            state,
            mirage_bin,
        }
    }

    fn mirage(&self) -> Command {
        let mut c = Command::new(&self.mirage_bin);
        c.env("XDG_CONFIG_HOME", &self.config)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("XDG_STATE_HOME", &self.state)
            .env("MIRAGE_BIN", &self.mirage_bin)
            .env_remove("MIRAGE_LOG");
        c
    }

    fn session_dir(&self, id: &str) -> PathBuf {
        self.runtime.join("mirage/session").join(id)
    }

    fn exec_dir(&self, sid: &str, eid: &str) -> PathBuf {
        self.session_dir(sid).join("exec").join(eid)
    }

    fn start_profile_and_session(&self, sid: &str) {
        self.mirage()
            .args(["profile", "create", "p"])
            .assert()
            .success();
        self.mirage()
            .args(["session", "start", "--profile", "p", "--id", sid])
            .assert()
            .success();
    }

    fn submit_long_running(&self, sid: &str, keep: bool) -> String {
        let mut args = vec!["exec", "start", sid];
        if keep {
            args.push("--keep");
        }
        args.extend(["--detach", "--", "/bin/sh", "-c", "sleep 30"]);
        let out = self.mirage().args(&args).output().unwrap();
        assert!(out.status.success(), "exec start failed: {:?}", out);
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn wait_started(&self, sid: &str, eid: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let s = self
                .mirage()
                .args(["exec", "show", sid, eid])
                .output()
                .unwrap();
            if String::from_utf8_lossy(&s.stdout).contains("\"started\": true") {
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        panic!("exec {eid} never started");
    }
}

fn wait_for<F: FnMut() -> bool>(timeout: Duration, mut f: F) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    false
}

#[test]
fn signal_request_is_written_then_consumed_by_host() {
    let env = Env::new();
    env.start_profile_and_session("s-sigfile");
    let eid = env.submit_long_running("s-sigfile", true);
    env.wait_started("s-sigfile", &eid);

    let exec_dir = env.exec_dir("s-sigfile", &eid);
    let signal_path = exec_dir.join("signal");
    assert!(!signal_path.exists(), "signal file should start absent");

    env.mirage()
        .args(["exec", "signal", "s-sigfile", &eid, "KILL"])
        .assert()
        .success();

    // The host's poll cadence is ~50ms; give it generous slack but
    // expect the file to disappear (i.e. the signal was consumed).
    let consumed = wait_for(Duration::from_secs(3), || !signal_path.exists());
    assert!(consumed, "host never consumed the signal file");

    // And the exec should end shortly after.
    let ended = wait_for(Duration::from_secs(3), || {
        let s = env
            .mirage()
            .args(["exec", "show", "s-sigfile", &eid])
            .output()
            .unwrap();
        String::from_utf8_lossy(&s.stdout).contains("\"ended\": true")
    });
    assert!(ended, "exec did not end after SIGKILL");

    env.mirage()
        .args(["session", "stop", "s-sigfile", "-f"])
        .assert()
        .success();
}

#[test]
fn keep_true_exec_persists_after_signal_kill() {
    let env = Env::new();
    env.start_profile_and_session("s-keep");
    let eid = env.submit_long_running("s-keep", /*keep=*/ true);
    env.wait_started("s-keep", &eid);

    env.mirage()
        .args(["exec", "signal", "s-keep", &eid, "KILL"])
        .assert()
        .success();

    // Wait for it to end on disk.
    let ended = wait_for(Duration::from_secs(3), || {
        let s = env
            .mirage()
            .args(["exec", "show", "s-keep", &eid])
            .output()
            .unwrap();
        String::from_utf8_lossy(&s.stdout).contains("\"ended\": true")
    });
    assert!(ended);

    // keep=true: the directory must still be there.
    let exec_dir = env.exec_dir("s-keep", &eid);
    assert!(
        exec_dir.exists(),
        "keep=true exec directory was unexpectedly removed"
    );
    // And exec_list should still see it.
    let list = env
        .mirage()
        .args(["exec", "list", "s-keep"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&list.stdout).contains(&eid));

    env.mirage()
        .args(["session", "stop", "s-keep", "-f"])
        .assert()
        .success();
}

#[test]
fn keep_false_exec_is_removed_after_signal_kill() {
    let env = Env::new();
    env.start_profile_and_session("s-nokeep");
    let eid = env.submit_long_running("s-nokeep", /*keep=*/ false);
    env.wait_started("s-nokeep", &eid);

    env.mirage()
        .args(["exec", "signal", "s-nokeep", &eid, "KILL"])
        .assert()
        .success();

    // The host should remove the exec dir once it exits.
    let exec_dir = env.exec_dir("s-nokeep", &eid);
    let gone = wait_for(Duration::from_secs(5), || !exec_dir.exists());
    assert!(gone, "keep=false exec dir was never removed: {exec_dir:?}");

    env.mirage()
        .args(["session", "stop", "s-nokeep", "-f"])
        .assert()
        .success();
}

#[test]
fn keep_false_exec_is_removed_after_natural_exit() {
    // Sanity check that keep=false cleanup also applies to non-killed
    // execs.
    let env = Env::new();
    env.start_profile_and_session("s-natural");
    let out = env
        .mirage()
        .args(["exec", "start", "s-natural", "--detach", "--", "/bin/true"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let eid = String::from_utf8_lossy(&out.stdout).trim().to_string();

    let exec_dir = env.exec_dir("s-natural", &eid);
    let gone = wait_for(Duration::from_secs(5), || !exec_dir.exists());
    assert!(gone, "keep=false exec dir was not removed after exit");

    env.mirage()
        .args(["session", "stop", "s-natural", "-f"])
        .assert()
        .success();
}

#[test]
fn signal_invalid_number_is_rejected() {
    let env = Env::new();
    env.start_profile_and_session("s-bad");
    let eid = env.submit_long_running("s-bad", true);
    env.wait_started("s-bad", &eid);

    let r = env
        .mirage()
        .args(["exec", "signal", "s-bad", &eid, "9999"])
        .output()
        .unwrap();
    assert!(!r.status.success(), "expected failure for bogus signal");
    let exec_dir = env.exec_dir("s-bad", &eid);
    assert!(
        !exec_dir.join("signal").exists(),
        "no signal file should have been written for an invalid signal"
    );

    env.mirage()
        .args(["session", "stop", "s-bad", "-f"])
        .assert()
        .success();
}

#[test]
fn session_destroy_invokes_container_provider_when_state_present() {
    let env = Env::new();
    let provider_log = env._dir.path().join("provider.log");
    let provider = env._dir.path().join("mock-provider.sh");
    std::fs::write(
        &provider,
        format!("#!/bin/sh\necho \"$@\" >> {}\n", provider_log.display()),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();

    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();

    // Create the session without a host, then write the container
    // runtime record the host would normally persist. `session destroy`
    // reads `container.json` and tears every node + the network
    // down via the recorded provider.
    env.mirage()
        .args([
            "session",
            "start",
            "--profile",
            "p",
            "--id",
            "s-cont",
            "--no-host",
        ])
        .assert()
        .success();
    let session_dir = env.session_dir("s-cont");
    let state = serde_json::json!({
        "provider": provider.to_string_lossy(),
        "image": "ignored:latest",
        "network": "mirage-s-cont",
        "head_port": 6000,
        "nodes": [
            { "rank": 0, "name": "mirage-s-cont-node-0" },
            { "rank": 1, "name": "mirage-s-cont-node-1" },
        ],
    });
    std::fs::write(
        session_dir.join("container.json"),
        serde_json::to_vec_pretty(&state).unwrap(),
    )
    .unwrap();
    std::fs::create_dir_all(session_dir.join("node/0")).unwrap();
    std::fs::create_dir_all(session_dir.join("node/1")).unwrap();
    std::fs::write(session_dir.join("node/0/cid"), "cid-0").unwrap();
    std::fs::write(session_dir.join("node/1/cid"), "cid-1").unwrap();

    env.mirage()
        .args(["session", "stop", "s-cont", "-f"])
        .assert()
        .success();

    let recorded = std::fs::read_to_string(&provider_log).unwrap_or_default();
    assert!(
        recorded.contains("rm -f mirage-s-cont-node-0"),
        "provider never asked to remove node 0; log: {recorded:?}"
    );
    assert!(
        recorded.contains("rm -f mirage-s-cont-node-1"),
        "provider never asked to remove node 1; log: {recorded:?}"
    );
    assert!(
        recorded.contains("network rm mirage-s-cont"),
        "provider never asked to remove the network; log: {recorded:?}"
    );
    // The whole session directory (including the container dir and cid
    // files) must be gone after destroy.
    assert!(
        !env.session_dir("s-cont").exists(),
        "session dir should be removed"
    );
}

#[test]
fn session_destroy_skips_provider_when_no_container_state() {
    let env = Env::new();
    let provider_log = env._dir.path().join("provider.log");
    let provider = env._dir.path().join("mock-provider.sh");
    std::fs::write(
        &provider,
        format!("#!/bin/sh\necho \"$@\" >> {}\n", provider_log.display()),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();

    env.mirage()
        .args(["profile", "create", "p"])
        .assert()
        .success();
    env.mirage()
        .args([
            "session",
            "start",
            "--profile",
            "p",
            "--id",
            "s-no-cid",
            "--no-host",
        ])
        .assert()
        .success();
    // No container.json is written → the provider must not run.

    env.mirage()
        .args(["session", "stop", "s-no-cid", "-f"])
        .assert()
        .success();
    assert!(
        !provider_log.exists(),
        "provider must not be invoked without container state"
    );
}
