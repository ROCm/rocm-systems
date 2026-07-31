//! End-to-end tests for the containerised workflow, using a mock
//! container provider (a small shell script standing in for
//! `docker`/`podman`). No real container runtime is required.
//!
//! These assert the full containerised lifecycle:
//!
//! * `mirage profile create --image … --container-provider <mock>`
//!   records the containerisation on the profile;
//! * a `mirage run` on that profile pulls the image, creates the
//!   per-session network and launches one container per node;
//! * the workload runs *inside* the node container, via the provider's
//!   `exec`, and so does a `mirage exec` from another terminal;
//! * when the run ends — cleanly or not — every container and the
//!   network go with it.
//!
//! # Why the mock stays in the foreground
//!
//! A node container used to be launched with `run -d`: the provider
//! client exited immediately, the container was detached, and its
//! lifetime was whatever remembered to remove it later. A session
//! outlived the command that created it, so that was the only option.
//!
//! It is not any more. `mirage run` owns its session, so a container is
//! launched with `run --rm` and *no* `-d`: the provider client is a child
//! process mirage holds for as long as the container should live, and
//! `--rm` deletes the container the moment that client goes away —
//! however it went away. The mock reproduces exactly that shape (it
//! blocks until its parent dies, then removes its own record), because a
//! mock that returned immediately would let a regression back to `-d`
//! pass every test in this file.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod harness;

use std::path::{Path, PathBuf};
use std::time::Duration;

use harness::{
    Env as BaseEnv, TEST_EMULATOR, assert_no_leaks, marker, skip_without_emulator, tagged_sleep,
    wait_for,
};
use nix::sys::signal::Signal;

/// How long a containerised session gets to come up. Bring-up shells out
/// to the provider several times per node, so it is slower than a plain
/// one.
const READY: Duration = Duration::from_secs(60);

struct Env {
    base: BaseEnv,
    provider: PathBuf,
    provider_log: PathBuf,
    containers: PathBuf,
}

impl Env {
    fn new() -> Self {
        let base = BaseEnv::new();
        let provider = base.root().join("mock-provider.sh");
        let provider_log = base.root().join("provider.log");
        let containers = base.root().join("containers");
        std::fs::create_dir_all(&containers).unwrap();
        write_mock_provider(&provider, &provider_log);
        Self {
            base,
            provider,
            provider_log,
            containers,
        }
    }

    fn provider_log(&self) -> String {
        std::fs::read_to_string(&self.provider_log).unwrap_or_default()
    }

    /// Names of the containers the mock engine currently holds.
    ///
    /// The mock creates one of these when a `run` client starts and
    /// deletes it when the container is removed — either by an explicit
    /// `rm -f` or, for `--rm`, when the client that owned it dies. So
    /// "this list is empty" is the mock's answer to "did mirage leak a
    /// container?".
    fn live_containers(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(&self.containers) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .flatten()
            .filter_map(|e| Some(e.file_name().to_str()?.to_string()))
            .collect();
        names.sort();
        names
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
/// * `run …` records the container and then *stays in the foreground*,
///   like a non-detached client, until the mirage that spawned it dies;
/// * `exec [-i] [-w dir] [-e K=V …] <container> <cmd> [args…]` runs the
///   command locally with that environment, which is what a real engine
///   would do inside the container.
fn write_mock_provider(path: &Path, log: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let script = r#"#!/bin/sh
echo "$@" >> __LOG__
STATE=__STATE__
case "$1" in
  pull) exit 0 ;;
  image)
    case "$2" in
      inspect) exit 1 ;;
      *) exit 0 ;;
    esac ;;
  network)
    case "$2" in
      # `network inspect --format` is the ownership check teardown makes
      # before removing anything; without a label it would (correctly)
      # refuse. Plain `network inspect` is the existence probe.
      inspect)
        if [ "$3" = "--format" ]; then printf mirage; exit 0; fi
        exit 1 ;;
      *) exit 0 ;;
    esac ;;
  run)
    # Parse the parts of the argv an engine actually acts on: the
    # container's name, whether it is removed when it stops, and the host
    # directory bind-mounted at the session scratch path. That last one
    # lets `exec` below emulate the mount; without it the wrapper mirage
    # wraps every workload in writes its pid to a path that does not
    # exist on this host, and the in-container signalling this mock
    # exists to exercise is silently untested.
    name=""
    autoremove=0
    while [ $# -gt 0 ]; do
      case "$1" in
        --rm) autoremove=1 ;;
        --name) name="$2"; shift ;;
        *:/mnt/mirage/runtime|*:/mnt/mirage/runtime:*)
          echo "${1%%:/mnt/mirage/runtime*}" > "$STATE/runtime-mount" ;;
      esac
      shift
    done
    if [ -z "$name" ]; then echo "run: no --name given" >&2; exit 1; fi
    : > "$STATE/containers/$name"
    # Not detached: this client *is* the container's lifetime. A real
    # engine's foreground client exits when the container stops and stops
    # the container when it is killed; here the container lasts exactly
    # as long as the mirage that asked for it.
    owner=$PPID
    while kill -0 "$owner" 2>/dev/null; do sleep 0.2; done
    # The client is gone. `--rm` is what removes the container now — and
    # a run that was SIGKILLed never gets to ask for anything else.
    if [ "$autoremove" = 1 ]; then rm -f "$STATE/containers/$name"; fi
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
    target="$1"
    shift
    # An engine cannot exec into a container that is not there, and
    # neither can this. A mirage that execs into a session whose
    # containers it already removed must fail, not run the workload on
    # the host.
    if [ ! -f "$STATE/containers/$target" ]; then
      echo "no such container: $target" >&2; exit 1
    fi
    # Fail like a real provider does. `podman exec -w` on a directory
    # that does not exist inside the container aborts the exec; swallowing
    # it here would let mirage pass a *host* path as the container
    # workdir and still look correct in these tests.
    if [ -n "$workdir" ]; then
      cd "$workdir" || { echo "chdir to '$workdir': no such directory" >&2; exit 126; }
    fi
    # Emulate the bind mount: rewrite the in-container scratch path back
    # to the host directory it is mounted from, so the pid-recording
    # wrapper mirage generates actually lands where mirage reads it.
    host_runtime=""
    [ -f "$STATE/runtime-mount" ] && host_runtime=$(cat "$STATE/runtime-mount")
    if [ -n "$host_runtime" ]; then
      rewritten=""
      for a in "$@"; do
        case "$a" in
          *"/mnt/mirage/runtime"*)
            a=$(printf '%s' "$a" | sed "s#/mnt/mirage/runtime#$host_runtime#g") ;;
        esac
        rewritten="$rewritten$a$(printf '\001')"
      done
      # Split back on the sentinel so arguments keep their spaces.
      OIFS=$IFS; IFS=$(printf '\001')
      # shellcheck disable=SC2086
      set -- $rewritten
      IFS=$OIFS
    fi
    if [ -n "$envs" ]; then
      exec env $envs "$@"
    fi
    exec "$@" ;;
  rm)
    # `rm -f <name>`: teardown's belt to `--rm`'s braces.
    rm -f "$STATE/containers/$3"
    exit 0 ;;
  inspect)
    # `inspect --format` is the ownership check (a Go template naming
    # mirage.owner); `inspect -f` is the running-state probe bring-up
    # polls before it lets the first exec near the container.
    if [ "$2" = "--format" ]; then
      case "$3" in
        *mirage.owner*) printf mirage ;;
        *) echo true ;;
      esac
      exit 0
    fi
    if [ "$2" = "-f" ]; then
      if [ -f "$STATE/containers/$4" ]; then echo true; else echo false; fi
      exit 0
    fi
    echo true ; exit 0 ;;
  *) exit 0 ;;
esac
"#
    .replace("__LOG__", &log.display().to_string())
    .replace(
        "__STATE__",
        &log.parent().unwrap_or(Path::new("/tmp")).display().to_string(),
    );
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// Kill anything tagged `marker` that is still running.
///
/// Only for the tests that `SIGKILL` a run on purpose: such a run cannot
/// tear its workload down — that is the failure being reproduced — so
/// what it stranded is this test's to remove rather than the machine's to
/// keep.
fn sweep(tag: &str) {
    for pid in harness::find_processes(tag) {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            Signal::SIGKILL,
        );
    }
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
        log.contains("network create --label mirage.owner=mirage"),
        "missing network create:\n{log}"
    );
    // Ownership is a label, not a name. Teardown and `state purge` both
    // check it before removing anything, so a container mirage did not
    // create is never destroyed by a name collision.
    assert!(
        log.contains("--label mirage.session="),
        "resources must record the session they belong to:\n{log}"
    );
    assert!(
        log.contains("run --rm --name mirage-"),
        "missing container run:\n{log}"
    );
    assert!(
        log.contains("--label mirage.owner=mirage"),
        "node containers must be labelled as mirage's:\n{log}"
    );
    // The workload reaches the container through the provider's `exec`,
    // not through a second mirage process inside it.
    assert!(
        log.contains("exec -i"),
        "the workload was not run via `provider exec`:\n{log}"
    );
    // And with no `-w`, because none was asked for. The session's working
    // directory is the *host* path the caller was in; passing it here
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
    assert!(
        env.live_containers().is_empty(),
        "containers outlived the run: {:?}",
        env.live_containers()
    );
}

#[test]
fn node_containers_are_launched_with_rm_and_are_not_detached() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("container-argv");

    let mut run = env
        .base
        .spawn_run(&["--profile", "cp"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);

    // These two flags are the whole ownership model, and neither is
    // observable from inside a healthy test: they only matter when the
    // run dies badly.
    //
    // `-d` would detach the container from the client, so the container
    // would survive the `mirage run` that made it — the leak the daemon
    // era lived with, back when a session had to outlive its command.
    // `--rm` is what deletes the container once the client is gone,
    // including when mirage was `SIGKILL`ed and never ran a line of
    // teardown code.
    let log = env.provider_log();
    let run_line = log
        .lines()
        .find(|l| l.starts_with("run "))
        .unwrap_or_else(|| panic!("no container was launched:\n{log}"));
    assert!(
        run_line.starts_with(&format!("run --rm --name mirage-{id}-node-0")),
        "a node container must be created with --rm: {run_line}"
    );
    assert!(
        !run_line
            .split_whitespace()
            .any(|a| a == "-d" || a == "--detach"),
        "a node container must not be detached: {run_line}"
    );
    assert_eq!(env.live_containers(), vec![format!("mirage-{id}-node-0")]);

    run.signal(Signal::SIGINT);
    run.wait(READY);
    assert_no_leaks(&tag);
}

#[test]
fn a_container_does_not_outlive_the_run_that_created_it() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("container-orphan");

    let mut run = env
        .base
        .spawn_run(&["--profile", "cp"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);
    let container = format!("mirage-{id}-node-0");
    assert_eq!(env.live_containers(), vec![container.clone()]);

    // `SIGKILL`, deliberately: no teardown runs, no `rm -f` is sent, no
    // Drop fires. Everything that removes the container here has to have
    // been decided at launch time — the client is mirage's child so it
    // dies with it, and `--rm` collects the container behind it.
    //
    // Under the old detached design this is exactly where a container was
    // stranded: `run -d` had returned, nothing held it, and the only
    // record that it existed had just been killed.
    run.kill();

    wait_for(
        "the container to be removed with its run",
        Duration::from_secs(30),
        || env.live_containers().is_empty(),
    );

    sweep(&tag);
}

#[test]
fn the_node_container_entrypoint_just_idles() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("container-idle");

    let mut run = env
        .base
        .spawn_run(&["--profile", "cp"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    run.await_ready(READY);

    let log = env.provider_log();
    // There is nothing for the container's own process to do: workloads
    // arrive from outside through `provider exec`. Running mirage in
    // there would be a second supervisor with no one to supervise.
    assert!(
        log.contains("--entrypoint sleep"),
        "expected an idling entrypoint:\n{log}"
    );
    assert!(
        !log.contains("host --session"),
        "no mirage host may run inside the container any more:\n{log}"
    );

    run.signal(Signal::SIGINT);
    run.wait(READY);
    assert_no_leaks(&tag);
}

#[test]
fn the_container_carries_the_in_container_mirage_directories() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("container-envs");

    let mut run = env
        .base
        .spawn_run(&["--profile", "cp"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    run.await_ready(READY);

    let log = env.provider_log();
    // The in-container mirage directories point at their mounts, so an
    // emulator runtime inside the container resolves the same assets the
    // run wrote outside it.
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

    run.signal(Signal::SIGINT);
    run.wait(READY);
    assert_no_leaks(&tag);
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
        .find(|l| l.starts_with("exec -i"))
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
fn container_state_is_never_written_to_disk_and_dies_with_the_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("container-state");

    let mut run = env
        .base
        .spawn_run(&["--profile", "cp"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);

    // Which containers a session has is held in the run's own memory. It
    // used to be a `container.json` on disk, which outlived the process
    // that wrote it and left teardown guessing whether the containers it
    // named were still there.
    assert!(
        !env.base.session_scratch(&id).join("container.json").exists(),
        "container state must not be written to disk"
    );

    run.signal(Signal::SIGINT);
    run.wait(READY);

    let log = env.provider_log();
    assert!(
        log.contains(&format!("rm -f mirage-{id}-node-0")),
        "container not removed:\n{log}"
    );
    assert!(
        log.contains(&format!("network rm mirage-{id}")),
        "network not removed:\n{log}"
    );
    // And the scratch directory goes with it.
    assert!(
        !env.base.session_scratch(&id).exists(),
        "session scratch outlived the session"
    );
    assert_no_leaks(&tag);
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
    // Every rank writes to this terminal, so every rank's line has to be
    // here — `--capture-all` is what labels them, and without it the
    // three nodes would interleave with nothing saying which wrote what.
    for rank in 0..3 {
        assert!(out.contains(&format!("rank-{rank}")), "{out}");
        assert!(out.contains(&format!("[{rank}]")), "{out}");
    }

    let log = env.provider_log();
    for rank in 0..3 {
        assert!(
            log.lines()
                .any(|l| l.starts_with("run --rm --name mirage-") && l.contains(&format!("-node-{rank} "))),
            "node {rank} container was not launched:\n{log}"
        );
    }
    assert!(
        env.live_containers().is_empty(),
        "containers outlived the run: {:?}",
        env.live_containers()
    );
}

#[test]
fn an_exec_runs_inside_the_containers_of_a_live_run() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("container-exec");

    let mut run = env
        .base
        .spawn_run(&["--profile", "cp"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    let id = run.await_ready(READY);

    // Naming the session is optional while exactly one run is live,
    // which is the shape this is used in: one terminal running the job,
    // another one exec'ing into it.
    let out = env.base.ok(&[
        "exec",
        "--",
        "/bin/sh",
        "-c",
        "echo exec-in-container rank=$MIRAGE_RANK world=$WORLD_SIZE",
    ]);
    assert!(out.contains("exec-in-container rank=0 world=1"), "{out}");

    // And naming it explicitly works too, which is what a second
    // terminal has to do once several runs are up.
    let out = env
        .base
        .ok(&["exec", "-s", &id, "--", "/bin/echo", "named-session"]);
    assert!(out.contains("named-session"), "{out}");

    let log = env.provider_log();
    let container = format!("mirage-{id}-node-0");
    let execs: Vec<&str> = log.lines().filter(|l| l.starts_with("exec -i")).collect();
    // The run's own workload plus the two execs above.
    assert!(
        execs.len() >= 3,
        "an exec did not go through the provider:\n{log}"
    );
    // An exec borrows the run's session, so it must land in that
    // session's container. Building the process grid client-side is only
    // safe because both sides build it from the same description; a
    // client that guessed would exec into the wrong container, or onto
    // the host.
    assert!(
        execs.iter().all(|l| l.contains(&container)),
        "an exec was not addressed to {container}:\n{log}"
    );
    // No `-t`, ever. Mirage allocates no pseudo-terminal: the provider
    // client inherits the caller's real streams, so asking the engine for
    // a tty would merge stderr into stdout and break redirection for a
    // containerised exec only.
    assert!(
        execs
            .iter()
            .all(|l| !l.split_whitespace().any(|a| a == "-t")),
        "a containerised exec asked the provider for a tty:\n{log}"
    );

    run.signal(Signal::SIGINT);
    run.wait(READY);
    assert_no_leaks(&tag);
}

#[test]
fn ending_a_containerised_run_kills_the_workload_inside_the_container() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("container-signal");

    let mut run = env
        .base
        .spawn_run(&["--profile", "cp"], &["/bin/sh", "-c", &tagged_sleep(&tag)]);
    run.await_ready(READY);
    wait_for(
        "the containerised workload to start",
        Duration::from_secs(15),
        || harness::count_processes(&tag) > 0,
    );

    // Ctrl-C in the run's terminal, which is the way a containerised job
    // normally ends.
    run.signal(Signal::SIGINT);
    run.wait(READY);

    assert_no_leaks(&tag);

    // And the workload must have been signalled *through the provider*,
    // not just locally. `podman exec` puts the workload in the
    // container's own PID namespace and does not relay signals into it,
    // so a host-side group kill reaches the client and leaves the real
    // process running — invisible to mirage and still holding the
    // emulated device. This mock shares a namespace with us, so only the
    // recorded argv can tell the two apart.
    let log = env.provider_log();
    let signalled = log
        .lines()
        .find(|l| l.starts_with("exec ") && l.contains("kill -"))
        .unwrap_or_else(|| {
            panic!(
                "the run never asked the provider to signal inside the \
                 container, so a real containerised workload would have \
                 survived it:\n{log}"
            )
        });
    // The pid the wrapper recorded is the workload's own, inside the
    // container: `sh -c 'echo $$ > …; exec "$0" "$@"'` writes its pid and
    // then `exec`s the workload into that same pid, so it names the
    // process that has to die and not a wrapper that already exited.
    assert!(
        signalled.contains(&format!("kill -{} ", libc::SIGINT))
            || signalled.contains(&format!("kill -{} ", libc::SIGTERM)),
        "the forwarded signal must be the one the workload was sent: {signalled}"
    );
}

#[test]
fn a_provider_that_cannot_be_found_fails_the_run_with_a_reason() {
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

    let err = env
        .base
        .fails(&["run", "--profile", "bad", "--", "/bin/true"]);
    // A session that cannot come up must say why, and the run that owns
    // it must not stay around pretending to serve it.
    assert!(
        err.contains("failed") || err.contains("provider") || err.contains("No such file"),
        "{err}"
    );
    assert!(
        env.base.live_runs().is_empty(),
        "a run whose session never came up is still serving: {:?}",
        env.base.live_runs()
    );
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the container suite going green while every test in it skipped
    // for a missing emulator runtime. See `assert_suite_can_run`.
    harness::assert_suite_can_run();
}
