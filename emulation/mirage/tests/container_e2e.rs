//! End-to-end tests for the containerised workflow, using a mock
//! container provider (a small shell script standing in for
//! `docker`/`podman`). No real container runtime is required.
//!
//! These assert the full containerised lifecycle:
//!
//! * `mirage profile create --image … --container-provider <mock>`
//!   records the containerisation on the profile;
//! * starting a containerised session pulls the image, creates the
//!   per-session network and launches one container per node;
//! * an exec runs *inside* the node container, via the provider's `exec`;
//! * destroying the session removes every container and the network.
//!
//! # What changed, and why the mock is simpler now
//!
//! A node container used to have `mirage host --session <id> --rank <n>`
//! as its entrypoint: a second mirage process, inside the container,
//! polling a bind-mounted directory for work. The mock had to launch that
//! host for the workload to run at all.
//!
//! Now the container's foreground process just idles, and the supervisor
//! runs workloads through `provider exec` from outside. So the mock only
//! has to do what a container engine does — which is also a far better
//! test, because it exercises the argv mirage actually builds rather than
//! a mirage process the mock started itself.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod harness;

use std::path::{Path, PathBuf};
use std::time::Duration;

use harness::{
    Env as BaseEnv, TEST_EMULATOR, assert_no_leaks, marker, skip_without_emulator, tagged_sleep,
    wait_for,
};

struct Env {
    base: BaseEnv,
    provider: PathBuf,
    provider_log: PathBuf,
}

impl Env {
    fn new() -> Self {
        let base = BaseEnv::new();
        let provider = base.root().join("mock-provider.sh");
        let provider_log = base.root().join("provider.log");
        write_mock_provider(&provider, &provider_log);
        Self {
            base,
            provider,
            provider_log,
        }
    }

    fn provider_log(&self) -> String {
        std::fs::read_to_string(&self.provider_log).unwrap_or_default()
    }

    fn create_containerized_profile(&self, name: &str) {
        self.base.ok(&[
            "profile",
            "create",
            name,
            "--emulator",
            TEST_EMULATOR,
            "--no-input",
            "--image",
            "img:latest",
            "--container-provider",
            &self.provider.to_string_lossy(),
        ]);
    }
}

/// A mock `docker`/`podman` that logs every invocation and behaves just
/// enough for bring-up, exec and teardown:
///
/// * `pull`, `network create|rm`, `rm` succeed silently;
/// * `network inspect` and `image inspect` fail, so mirage takes the
///   create/pull paths;
/// * `run -d …` prints a fake container id and records that the container
///   is "running";
/// * `exec [-i] [-w dir] [-e K=V …] <container> <cmd> [args…]` runs the
///   command locally with that environment, which is what a real engine
///   would do inside the container.
fn write_mock_provider(path: &Path, log: &Path) {
    use std::os::unix::fs::PermissionsExt;
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
    echo cid-12345
    exit 0 ;;
  exec)
    # Consume the flags mirage builds, collecting `-e K=V` into the
    # environment, then run the command. This is the mock's real job:
    # it proves the argv mirage produces is one an engine can execute.
    shift
    envs=""
    workdir=""
    while [ $# -gt 0 ]; do
      case "$1" in
        -i|-t|-it) shift ;;
        -w) workdir="$2"; shift 2 ;;
        -e) envs="$envs $2"; shift 2 ;;
        *) break ;;
      esac
    done
    # The next argument is the container name; the rest is the command.
    shift
    # Fail like a real provider does. `podman exec -w` on a directory
    # that does not exist inside the container aborts the exec; swallowing
    # it here would let mirage pass a *host* path as the container
    # workdir and still look correct in these tests.
    if [ -n "$workdir" ]; then
      cd "$workdir" || { echo "chdir to '$workdir': no such directory" >&2; exit 126; }
    fi
    if [ -n "$envs" ]; then
      exec env $envs "$@"
    fi
    exec "$@" ;;
  rm) exit 0 ;;
  inspect) echo true ; exit 0 ;;
  *) exit 0 ;;
esac
"#
    .replace("__LOG__", &log.display().to_string());
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn profile_create_records_containerization() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let json: serde_json::Value =
        serde_json::from_str(&env.base.ok(&["profile", "show", "cp"])).unwrap();
    assert_eq!(json["containerize"]["image"], "img:latest");
    assert_eq!(
        json["containerize"]["provider"],
        env.provider.to_string_lossy().to_string()
    );
}

#[test]
fn a_containerised_run_brings_up_executes_and_cleans_up() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    let out = env.base.ok(&[
        "run",
        "--profile",
        "cp",
        "--",
        "/bin/echo",
        "hello-from-container",
    ]);
    assert!(out.contains("hello-from-container"), "{out}");

    let log = env.provider_log();
    assert!(log.contains("pull img:latest"), "missing pull:\n{log}");
    assert!(
        log.contains("network create mirage-"),
        "missing network create:\n{log}"
    );
    assert!(
        log.contains("run -d --name mirage-"),
        "missing container run:\n{log}"
    );
    // The workload reaches the container through the provider's `exec`,
    // not through a second mirage process inside it.
    assert!(
        log.contains("exec -i"),
        "the workload was not run via `provider exec`:\n{log}"
    );
    // And with no `-w`, because none was asked for. The session's working
    // directory is the *host* path the client was in; passing it here
    // makes the provider chdir to a directory that does not exist inside
    // the container and fail the exec outright. Only an explicit
    // `--workdir` may become `-w`.
    //
    // The mock cannot catch this by executing, since it runs on the host
    // where that path does happen to exist — so the argv is asserted
    // directly.
    let exec_line = log
        .lines()
        .find(|l| l.starts_with("exec -i"))
        .unwrap_or_else(|| panic!("no provider exec was recorded:\n{log}"));
    assert!(
        !exec_line.contains(" -w "),
        "a containerised exec was given a host working directory:\n{exec_line}"
    );
    // Teardown removes everything bring-up created.
    assert!(log.contains("rm -f mirage-"), "container not removed:\n{log}");
    assert!(
        log.contains("network rm mirage-"),
        "network not removed:\n{log}"
    );
}

#[test]
fn the_node_container_entrypoint_just_idles() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    env.base.start_session("cp", "idle");

    let log = env.provider_log();
    // With the per-session host gone there is nothing for the container's
    // own process to do; running mirage in there again would be a second
    // supervisor with no one to supervise.
    assert!(
        log.contains("--entrypoint sleep"),
        "expected an idling entrypoint:\n{log}"
    );
    assert!(
        !log.contains("host --session"),
        "no mirage host may run inside the container any more:\n{log}"
    );

    env.base.ok(&["session", "stop", "idle"]);
}

#[test]
fn the_container_carries_the_in_container_mirage_directories() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    env.base.start_session("cp", "envs");

    let log = env.provider_log();
    // The in-container mirage directories point at their mounts, so an
    // emulator runtime inside the container resolves the same assets the
    // supervisor wrote outside it.
    assert!(
        log.contains("-e MIRAGE_RUNTIME=/mnt/mirage/runtime"),
        "missing MIRAGE_RUNTIME:\n{log}"
    );
    assert!(
        log.contains("-e MIRAGE_CONFIG=/mnt/mirage/config"),
        "missing MIRAGE_CONFIG:\n{log}"
    );
    assert!(
        log.contains(":/mnt/mirage/runtime"),
        "the session scratch directory is not mounted:\n{log}"
    );

    env.base.ok(&["session", "stop", "envs"]);
}

#[test]
fn the_rank_environment_is_injected_at_exec_time() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    // Ranks belong to an exec, not to a container: the same container
    // serves every exec in its session, and with `--nproc-per-node` one
    // container hosts several ranks at once. Injecting them at launch —
    // which the old design had to, because the container's entrypoint was
    // the process that ran the workload — would bake in one rank per
    // container and be wrong for both cases.
    let out = env.base.ok(&[
        "run",
        "--profile",
        "cp",
        "--",
        "/bin/sh",
        "-c",
        "echo rank=$MIRAGE_RANK world=$WORLD_SIZE local=$LOCAL_RANK",
    ]);
    assert!(out.contains("rank=0 world=1 local=0"), "{out}");

    let log = env.provider_log();
    let exec_line = log
        .lines()
        .find(|l| l.starts_with("exec "))
        .unwrap_or_else(|| panic!("no exec invocation in:\n{log}"));
    assert!(
        exec_line.contains("-e MIRAGE_RANK=0"),
        "the exec must carry the rank env:\n{exec_line}"
    );
    assert!(
        exec_line.contains("-e WORLD_SIZE=1"),
        "the exec must carry the world size:\n{exec_line}"
    );
}

#[test]
fn container_state_is_reported_and_removed_with_the_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    env.base.start_session("cp", "s-box");

    // Container state is held in memory and reported through the control
    // plane. It used to be a `container.json` on disk, which outlived the
    // process that wrote it and left teardown guessing.
    let state: serde_json::Value =
        serde_json::from_str(&env.base.ok(&["session", "show", "s-box"])).unwrap();
    assert_eq!(state["container"]["image"], "img:latest");
    assert_eq!(state["container"]["nodes"][0]["name"], "mirage-s-box-node-0");
    assert!(
        !env.base.session_scratch("s-box").join("container.json").exists(),
        "container state must not be written to disk"
    );

    env.base.ok(&["session", "stop", "s-box"]);

    let log = env.provider_log();
    assert!(
        log.contains("rm -f mirage-s-box-node-0"),
        "container not removed:\n{log}"
    );
    assert!(
        log.contains("network rm mirage-s-box"),
        "network not removed:\n{log}"
    );
    // And the scratch directory goes with it.
    assert!(
        !env.base.session_scratch("s-box").exists(),
        "session scratch outlived the session"
    );
}

#[test]
fn a_multi_node_containerised_session_launches_one_container_per_node() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    let out = env.base.ok(&[
        "run",
        "--profile",
        "cp",
        "--num-nodes",
        "3",
        "--",
        "/bin/sh",
        "-c",
        "echo rank-$MIRAGE_RANK",
    ]);
    for rank in 0..3 {
        assert!(out.contains(&format!("rank-{rank}")), "{out}");
    }

    let log = env.provider_log();
    for rank in 0..3 {
        assert!(
            log.contains("--name mirage-") && log.contains(&format!("-node-{rank}")),
            "node {rank} container was not launched:\n{log}"
        );
    }
}

#[test]
fn destroying_a_containerised_session_kills_the_exec_process() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    env.base.start_session("cp", "cbox");
    let tag = marker("container-exec");

    env.base.ok(&[
        "exec",
        "start",
        "cbox",
        "--detach",
        "--",
        "/bin/sh",
        "-c",
        &tagged_sleep(&tag),
    ]);
    wait_for("the containerised workload to start", Duration::from_secs(15), || {
        harness::count_processes(&tag) > 0
    });

    env.base.ok(&["session", "stop", "cbox"]);

    // The provider process mirage spawned is the exec's process group
    // leader, so killing that group must reach the workload the mock ran
    // underneath it.
    assert_no_leaks(&tag);
}

#[test]
fn a_provider_that_cannot_be_found_fails_the_session_with_a_reason() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.base.ok(&[
        "profile",
        "create",
        "bad",
        "--emulator",
        TEST_EMULATOR,
        "--no-input",
        "--image",
        "img:latest",
        "--container-provider",
        "/nonexistent/provider-binary",
    ]);

    let err = env.base.fails(&[
        "session", "start", "--profile", "bad", "--id", "nope", "--no-input",
    ]);
    // A session that cannot come up must say why and must not linger.
    assert!(
        err.contains("failed") || err.contains("provider") || err.contains("No such file"),
        "{err}"
    );
    let list = env.base.ok(&["session", "list"]);
    assert!(!list.contains("nope"), "{list}");
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the container suite going green while every test in it skipped
    // for a missing emulator runtime. See `assert_suite_can_run`.
    harness::assert_suite_can_run();
}
