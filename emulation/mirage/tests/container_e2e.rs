//! End-to-end tests for the containerised workflow using a mock
//! container provider (a small shell script standing in for
//! `docker`/`podman`). No real container runtime is required.
//!
//! These assert the full containerised lifecycle:
//!
//! * `mirage profile create --image ... --provider <mock>` records the
//!   containerisation on the profile;
//! * starting a containerised session brings up the per-node container
//!   and network (the host pulls the image, creates the network, and
//!   runs the node container) and persists `container.json` plus a
//!   per-node `node/<rank>/cid` file;
//! * `mirage run` against a containerised profile actually executes the
//!   workload *inside* the container (via the mock provider's `exec`,
//!   which runs the command) and the MIRAGE_* environment is injected;
//! * destroying the session removes every node container and the network
//!   via the provider and deletes the whole session directory, so a
//!   containerised session cleans up after itself.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::TempDir;

struct Env {
    _dir: TempDir,
    runtime: PathBuf,
    config: PathBuf,
    state: PathBuf,
    mirage_bin: PathBuf,
    provider: PathBuf,
    provider_log: PathBuf,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join("runtime");
        let config = dir.path().join("config");
        let state = dir.path().join("state");
        let provider = dir.path().join("mock-provider.sh");
        let provider_log = dir.path().join("provider.log");
        write_mock_provider(&provider, &provider_log);
        let mirage_bin = PathBuf::from(env!("CARGO_BIN_EXE_mirage"));
        Self {
            _dir: dir,
            runtime,
            config,
            state,
            mirage_bin,
            provider,
            provider_log,
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

    fn provider_log(&self) -> String {
        std::fs::read_to_string(&self.provider_log).unwrap_or_default()
    }

    fn create_containerized_profile(&self, name: &str) {
        self.mirage()
            .args([
                "profile",
                "create",
                name,
                "--image",
                "img:latest",
                "--provider",
                &self.provider.to_string_lossy(),
            ])
            .assert()
            .success();
    }
}

/// A mock `docker`/`podman` that logs every invocation and behaves just
/// enough for the host's bring-up and exec paths:
///   * `pull` / `network create|rm` / `rm` succeed silently,
///   * `network inspect` fails so `ensure_network` takes the create path,
///   * `run -d ...` prints a fake container id,
///   * `exec [-i] [-w DIR] [-e K=V ...] <container> CMD ARGS...` strips
///     the flags and the container name and runs CMD locally, so the
///     workload actually executes.
fn write_mock_provider(path: &Path, log: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let script = r#"#!/bin/sh
echo "$@" >> __LOG__
case "$1" in
  pull) exit 0 ;;
  network)
    case "$2" in
      inspect) exit 1 ;;
      *) exit 0 ;;
    esac ;;
  run) echo cid-12345 ; exit 0 ;;
  rm) exit 0 ;;
  inspect) echo true ; exit 0 ;;
  exec)
    shift
    while [ $# -gt 0 ]; do
      case "$1" in
        -i) shift ;;
        -w) shift 2 ;;
        -e) shift 2 ;;
        *) break ;;
      esac
    done
    shift
    exec "$@" ;;
  *) exit 0 ;;
esac
"#
    .replace("__LOG__", &log.display().to_string());
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
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
fn profile_create_records_containerization() {
    let env = Env::new();
    env.create_containerized_profile("cp");
    let out = env
        .mirage()
        .args(["profile", "show", "cp"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["containerize"]["image"], "img:latest");
    assert_eq!(
        json["containerize"]["provider"],
        env.provider.to_string_lossy().to_string()
    );
}

#[test]
fn containerized_run_executes_in_container_and_cleans_up() {
    let env = Env::new();
    env.create_containerized_profile("cp");

    // `mirage run` creates a transient containerised session, runs the
    // command inside the (mock) container, then destroys the session.
    env.mirage()
        .args(["run", "--profile", "cp", "--", "/bin/echo", "hello-mirage"])
        .assert()
        .success()
        .stdout(predicate::str::contains("hello-mirage"));

    let log = env.provider_log();
    // Bring-up sequence.
    assert!(log.contains("pull img:latest"), "missing pull; log:\n{log}");
    assert!(
        log.contains("network create mirage-"),
        "missing network create; log:\n{log}"
    );
    assert!(
        log.contains("run -d --name mirage-"),
        "missing container run; log:\n{log}"
    );
    // Workload executed via exec, with the always-present rank env.
    assert!(
        log.contains("-e MIRAGE_RANK=0"),
        "missing MIRAGE_RANK injection; log:\n{log}"
    );
    assert!(
        log.contains("/bin/echo hello-mirage"),
        "command not exec'd in container; log:\n{log}"
    );
    // Cleanup on session destroy.
    assert!(
        log.contains("rm -f mirage-"),
        "container not removed; log:\n{log}"
    );
    assert!(
        log.contains("network rm mirage-"),
        "network not removed; log:\n{log}"
    );
}

#[test]
fn containerized_session_writes_cid_then_cleans_up_on_stop() {
    let env = Env::new();
    env.create_containerized_profile("cp");

    env.mirage()
        .args(["session", "start", "--profile", "cp", "--id", "s-box"])
        .assert()
        .success();

    // The host should have persisted the runtime record and the per-node
    // container id.
    let container_json = env.session_dir("s-box").join("container.json");
    assert!(
        wait_for(Duration::from_secs(5), || container_json.exists()),
        "container.json was never written"
    );
    let cid_path = env.session_dir("s-box").join("node/0/cid");
    assert!(cid_path.exists(), "per-node cid file missing");
    let cid = std::fs::read_to_string(&cid_path).unwrap();
    assert_eq!(cid.trim(), "cid-12345");

    // Stopping the session must remove the container + network and delete
    // the whole session directory.
    env.mirage()
        .args(["session", "stop", "s-box", "-f"])
        .assert()
        .success();

    assert!(
        wait_for(Duration::from_secs(5), || !env
            .session_dir("s-box")
            .exists()),
        "session dir should be removed after stop"
    );
    let log = env.provider_log();
    assert!(
        log.contains("rm -f mirage-s-box-node-0"),
        "container not removed; log:\n{log}"
    );
    assert!(
        log.contains("network rm mirage-s-box"),
        "network not removed; log:\n{log}"
    );
}
