//! End-to-end tests for the containerised workflow using a mock
//! container provider (a small shell script standing in for
//! `docker`/`podman`). No real container runtime is required.
//!
//! These assert the full containerised lifecycle:
//!
//! * `mirage profile create --image ... --container-provider <mock>` records the
//!   containerisation on the profile;
//! * starting a containerised session brings up the per-node container
//!   and network (the host pulls the image, creates the network, and
//!   runs the node container) and persists `container.json` plus a
//!   per-node `node/<rank>/cid` file;
//! * `mirage run` against a containerised profile actually executes the
//!   workload via the per-node host that is the container's foreground
//!   process (`mirage host --session <id> --rank <n>`), and the MIRAGE_*
//!   environment is injected on the node container;
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
                "--container-provider",
                &self.provider.to_string_lossy(),
            ])
            .assert()
            .success();
    }
}

/// A mock `docker`/`podman` that logs every invocation and behaves just
/// enough for the host's bring-up path:
///   * `pull` / `network create|rm` succeed silently,
///   * `network inspect` fails so `ensure_network` takes the create path,
///   * `run -d ... <image> mirage host --session <id> --rank <n>` launches
///     the per-node host directly on the host (there is no real container
///     here) so the workload actually executes, records its pid, and
///     prints a fake container id,
///   * `rm -f <name>` stops that per-node host (mirroring how removing a
///     real container kills its in-container host).
///
/// The per-node host is the container's foreground process in the real
/// architecture; running it on the host lets it resolve the session under
/// the test's XDG dirs and run the exec, so an attached `mirage run`
/// observes the workload's output and exit instead of blocking forever.
fn write_mock_provider(path: &Path, log: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let piddir = log.parent().unwrap();
    let script = r#"#!/bin/sh
echo "$@" >> __LOG__
case "$1" in
  pull) exit 0 ;;
  image)
    case "$2" in
      inspect) exit 1 ;;
      *) exit 0 ;;
    esac ;;
  network)
    case "$2" in
      inspect) exit 1 ;;
      *) exit 0 ;;
    esac ;;
  run)
    # Mirror the real container entrypoint (`mirage host --session <id>
    # --rank <n>`) by launching that per-node host directly on the host.
    # Detach its stdio so the cid we print on stdout — which the engine
    # captures as the container id — stays clean, and record its pid so
    # `rm` can stop it on teardown.
    shift
    name=""; sid=""; rank=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --name) name="$2"; shift 2 ;;
        --session) sid="$2"; shift 2 ;;
        --rank) rank="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    if [ -n "$sid" ] && [ -n "$rank" ]; then
      "$MIRAGE_BIN" host --session "$sid" --rank "$rank" </dev/null >/dev/null 2>&1 &
      [ -n "$name" ] && echo $! > "__PIDDIR__/$name.pid"
    fi
    echo cid-12345
    exit 0 ;;
  rm)
    for a in "$@"; do
      case "$a" in
        mirage-*)
          if [ -f "__PIDDIR__/$a.pid" ]; then
            kill "$(cat "__PIDDIR__/$a.pid")" 2>/dev/null
            rm -f "__PIDDIR__/$a.pid"
          fi ;;
      esac
    done
    exit 0 ;;
  inspect) echo true ; exit 0 ;;
  *) exit 0 ;;
esac
"#
    .replace("__LOG__", &log.display().to_string())
    .replace("__PIDDIR__", &piddir.display().to_string());
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
    // Each node container's foreground process is the per-node host
    // (`mirage host --session <id> --rank <n>`); that is what actually
    // runs the workload inside the container.
    assert!(
        log.contains("host --session"),
        "node container entrypoint is not the per-node host; log:\n{log}"
    );
    // The always-present rank env is injected on the node container.
    assert!(
        log.contains("-e MIRAGE_RANK=0"),
        "missing MIRAGE_RANK injection; log:\n{log}"
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
