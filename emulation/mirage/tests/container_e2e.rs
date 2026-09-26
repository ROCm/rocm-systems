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
//!   network go with it;
//! * and the configurations that could never work — a mount laid over
//!   mirage's own directory in the container, a published port on a
//!   multi-node session, an empty image, a provider that is not an
//!   engine — are refused before anything is created, in mirage's words
//!   rather than the engine's.
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
//!
//! The mock enforces `--rm` on itself, though, which is the one thing it
//! cannot be trusted about: it is playing both sides of the contract. So
//! one test here drives the machine's real podman or docker end to end —
//! `a_real_engine_creates_and_removes_the_container_with_the_run` — and
//! skips when neither is installed with the image it needs.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod harness;

use std::path::{Path, PathBuf};
use std::time::Duration;

use harness::{
    Env as BaseEnv, TEST_EMULATOR, assert_no_leaks, count_processes, marker, skip_without_emulator,
    tagged_sleep, wait_for,
};
use nix::sys::signal::Signal;

/// How long a containerised session gets to come up. Bring-up shells out
/// to the provider several times per node, so it is slower than a plain
/// one.
const READY: Duration = Duration::from_secs(60);

/// A container `--workdir` the mock provider takes five seconds to deny.
///
/// See the `exec)` case of the mock for why the workdir probe is the
/// only place this window can be opened.
const STALL_WORKDIR: &str = "/mirage-test-stalls-then-missing";

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

    /// Make the mock take five seconds over each removal verb, so a
    /// teardown lasts long enough for a test to dial into it.
    fn arm_teardown_stall(&self) {
        std::fs::write(self.base.root().join("stall-teardown"), "").unwrap();
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

    /// The runtime directory a mirage run in this environment resolves,
    /// and therefore stamps on everything it creates.
    fn own_runtime(&self) -> String {
        self.base.runtime().join("mirage").display().to_string()
    }

    /// Give a resource the `mirage.runtime` label the mock reads back on
    /// inspect, as its creation would have.
    fn label_runtime(&self, name: &str, runtime: &str) {
        let labels = self.base.root().join("labels");
        std::fs::create_dir_all(&labels).unwrap();
        std::fs::write(labels.join(name), runtime).unwrap();
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
mkdir -p "$STATE/containers" "$STATE/networks" "$STATE/labels"

# The session a resource belongs to, recovered from its name the way a
# real engine recovers it from the `mirage.session` label. Containers are
# `mirage-<session>-node-<rank>`, networks are `mirage-<session>`.
session_of() {
  s=${1#mirage-}
  printf '%s' "${s%-node-*}"
}

# The runtime directory a resource records, stored when it was created
# and read back on inspect. Unlike the session this cannot be recovered
# from the name, and it is the difference between a resource this mirage
# may reclaim and one belonging to a mirage elsewhere on the machine — so
# the mock keeps it the way an engine keeps a label.
runtime_of() {
  if [ -f "$STATE/labels/$1" ]; then cat "$STATE/labels/$1"; else printf '<no value>'; fi
}

# A deliberate stall on the removal verbs, armed by a test dropping a
# marker file. Teardown is the one long stretch of a run that this mock
# can otherwise make instantaneous, and
# `a_run_answers_its_socket_while_it_is_tearing_down` needs it to last
# long enough to dial into.
stall_teardown_if_asked() {
  [ -f "$STATE/stall-teardown" ] && sleep 5
  return 0
}

# Record `--label mirage.runtime=<dir>` from an argv, if it carries one.
record_runtime() {
  name=$1; shift
  prev=""
  for a in "$@"; do
    if [ "$prev" = "--label" ]; then
      case "$a" in
        mirage.runtime=*) printf '%s' "${a#mirage.runtime=}" > "$STATE/labels/$name" ;;
      esac
    fi
    prev=$a
  done
}

case "$1" in
  pull)
    # A real pull is chatty on both streams: layer progress on stderr,
    # and the digest of what was pulled on stdout. Mirage shows a user
    # both — and must put neither on *its* stdout, which belongs to the
    # workload. See `provider_chatter_stays_off_a_redirected_stdout`.
    echo 'Trying to pull img:latest...' >&2
    echo 'sha256:1111feed2222beef-pulled-digest'
    exit 0 ;;
  image)
    case "$2" in
      inspect) exit 1 ;;
      *) exit 0 ;;
    esac ;;
  build)
    # A derived image (a profile hack). The Dockerfile arrives on stdin
    # and has to be read: a real engine consumes it, and a mock that
    # exited without reading would fail mirage's write with EPIPE and
    # test a code path no engine produces.
    cat > /dev/null
    exit 0 ;;
  # Listing verbs, used by orphan reclamation after a run died without
  # tearing down. A real engine filters on the mirage.owner label; the
  # mock creates nothing that is not mirage's, so everything it holds is
  # a candidate.
  ps)
    ls "$STATE/containers" 2>/dev/null
    exit 0 ;;
  network)
    case "$2" in
      ls)
        ls "$STATE/networks" 2>/dev/null
        exit 0 ;;
      create)
        # The name is the last argument, after any --label pairs.
        for a in "$@"; do last=$a; done
        : > "$STATE/networks/$last"
        record_runtime "$last" "$@"
        exit 0 ;;
      rm)
        stall_teardown_if_asked
        rm -f "$STATE/networks/$3"
        rm -f "$STATE/labels/$3"
        exit 0 ;;
      # `network inspect --format` reads one label: the ownership check
      # teardown makes before removing anything, or the session a
      # reclaimed network belonged to. Plain `network inspect` is the
      # existence probe.
      inspect)
        if [ "$3" = "--format" ]; then
          case "$4" in
            *mirage.session*) session_of "$5" ;;
            *mirage.runtime*) runtime_of "$5" ;;
            *) printf mirage ;;
          esac
          exit 0
        fi
        [ -f "$STATE/networks/$3" ] && exit 0
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
    runtime_label=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --rm) autoremove=1 ;;
        --name) name="$2"; shift ;;
        --label)
          case "$2" in mirage.runtime=*) runtime_label=${2#mirage.runtime=} ;; esac
          shift ;;
        *:/mnt/mirage/runtime|*:/mnt/mirage/runtime:*)
          echo "${1%%:/mnt/mirage/runtime*}" > "$STATE/runtime-mount" ;;
      esac
      shift
    done
    if [ -z "$name" ]; then echo "run: no --name given" >&2; exit 1; fi
    : > "$STATE/containers/$name"
    if [ -n "$runtime_label" ]; then
      printf '%s' "$runtime_label" > "$STATE/labels/$name"
    fi
    # Not detached: this client *is* the container's lifetime. A real
    # engine's foreground client exits when the container stops and stops
    # the container when it is killed; here the container lasts exactly
    # as long as the mirage that asked for it.
    #
    # What this proves, and what it does not. It proves mirage asked for
    # `--rm` and no `-d`, that it holds the client for the session's
    # whole life, and that nothing in mirage's own teardown depends on
    # the client having been asked politely. It does *not* prove the
    # engine keeps its half of that bargain, because the two lines below
    # are the mock keeping it for itself: a real client is reparented to
    # init when mirage is SIGKILLed and goes on holding its container
    # (which is the leak `mirage cleanup` exists to reclaim, and why the
    # reclaim test below stands its orphan up by hand rather than by
    # killing a run). For the engine's half, see
    # `a_real_engine_creates_and_removes_the_container_with_the_run`.
    owner=$PPID
    while kill -0 "$owner" 2>/dev/null; do sleep 0.2; done
    # The client is gone. `--rm` is what removes the container now — and
    # a run that was SIGKILLed never gets to ask for anything else.
    if [ "$autoremove" = 1 ]; then rm -f "$STATE/containers/$name"; fi
    exit 0 ;;
  exec)
    # A deliberate stall, and the only thing in this mock that is not
    # imitating an engine. The container `--workdir` probe is the one
    # step of a run that talks to the provider after the session is
    # healthy, which makes it the only place a test can hold a run open
    # *between* "this session is up and borrowable" and "this run's own
    # command has failed". A `--workdir` naming this path takes five
    # seconds to answer "no such directory"; every other path answers at
    # once. See `a_failing_runs_own_command_still_waits_for_a_borrower`.
    case " $* " in
      *__STALL_PATH__*) sleep 5 ;;
    esac
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
    stall_teardown_if_asked
    rm -f "$STATE/containers/$3" "$STATE/labels/$3"
    exit 0 ;;
  inspect)
    # `inspect --format` reads one label: the ownership check (a Go
    # template naming mirage.owner), or the session an orphan belonged to.
    # `inspect -f` is the running-state probe bring-up polls before it
    # lets the first exec near the container.
    if [ "$2" = "--format" ]; then
      case "$3" in
        *mirage.session*) session_of "$4" ;;
        *mirage.runtime*) runtime_of "$4" ;;
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
    .replace("__STALL_PATH__", STALL_WORKDIR)
    .replace(
        "__STATE__",
        &log.parent()
            .unwrap_or(Path::new("/tmp"))
            .display()
            .to_string(),
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
        let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), Signal::SIGKILL);
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
    assert!(
        log.contains("rm -f mirage-"),
        "container not removed:\n{log}"
    );
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

    let mut run = env.base.spawn_run(
        &["--profile", "cp"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
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

    let mut run = env.base.spawn_run(
        &["--profile", "cp"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
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

    let mut run = env.base.spawn_run(
        &["--profile", "cp"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
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

    let mut run = env.base.spawn_run(
        &["--profile", "cp"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
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

    let mut run = env.base.spawn_run(
        &["--profile", "cp"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
    let id = run.await_ready(READY);

    // Which containers a session has is held in the run's own memory. It
    // used to be a `container.json` on disk, which outlived the process
    // that wrote it and left teardown guessing whether the containers it
    // named were still there.
    assert!(
        !env.base
            .session_scratch(&id)
            .join("container.json")
            .exists(),
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
            log.lines().any(|l| l.starts_with("run --rm --name mirage-")
                && l.contains(&format!("-node-{rank} "))),
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

    let mut run = env.base.spawn_run(
        &["--profile", "cp"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
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

    let mut run = env.base.spawn_run(
        &["--profile", "cp"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
    run.await_ready(READY);
    wait_for(
        "the containerised workload to start",
        Duration::from_secs(15),
        || count_processes(&tag) > 0,
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
    //
    // The reason has to name the binary that could not be found. The
    // previous three-way disjunction added nothing over the non-zero exit
    // `fails` already asserted: "provider" is a substring of the
    // `--container-provider` flag the error echoes, and "failed" appears
    // in every bring-up error there is — so a regression to a bare
    // "session bring-up failed" would still have passed a test whose name
    // promises a reason.
    assert!(
        err.contains("/nonexistent/provider-binary"),
        "the error must name the provider that could not be found: {err}"
    );
    assert!(
        env.base.live_runs().is_empty(),
        "a run whose session never came up is still serving: {:?}",
        env.base.live_runs()
    );
}

#[test]
fn cleanup_reclaims_containers_no_live_run_accounts_for() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }

    // A container and a network with nothing that knows they exist: what
    // a `SIGKILL`ed run leaves when its provider client is reparented to
    // init rather than dying with it. Nothing on disk records them — the
    // only surviving evidence is the labels on the resources themselves,
    // and that is what cleanup goes looking for.
    //
    // Stood up directly rather than by killing a run, because the mock's
    // client emulates `--rm` faithfully and removes its own container on
    // the way out. A real `podman run` client does not: it is reparented
    // and keeps the container alive, which is the leak being reclaimed
    // here.
    let orphan = "mirage-deadsession-node-0";
    let network = "mirage-deadsession";
    std::fs::write(env.containers.join(orphan), "").unwrap();
    let networks = env.base.root().join("networks");
    std::fs::create_dir_all(&networks).unwrap();
    std::fs::write(networks.join(network), "").unwrap();
    // Including the runtime directory they were created under, without
    // which they are unattributable and cleanup leaves them alone.
    env.label_runtime(orphan, &env.own_runtime());
    env.label_runtime(network, &env.own_runtime());

    let out = env
        .base
        .mirage()
        .arg("cleanup")
        // No profile is involved, so this is the only way to point
        // cleanup at the mock rather than at a real engine on PATH.
        .env("MIRAGE_CONTAINER_PROVIDER", &env.provider)
        .output()
        .unwrap();
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "cleanup failed:\n{text}");
    assert!(
        text.contains(orphan),
        "the container was not reclaimed:\n{text}"
    );
    assert!(
        text.contains(network),
        "the network was not reclaimed:\n{text}"
    );

    assert!(
        env.live_containers().is_empty(),
        "the orphaned container survived cleanup: {:?}",
        env.live_containers()
    );
    assert!(
        !networks.join(network).exists(),
        "the orphaned network survived cleanup"
    );
}

#[test]
fn cleanup_spares_a_container_created_by_another_runtime_directory() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }

    // Two mirages sharing one container engine. The other one's session
    // is not in *this* one's registry of live runs — it cannot be, the
    // registry is the sockets under this runtime directory — so every
    // other test cleanup applies would call this container an orphan.
    // The `mirage.runtime` label is the only thing that says otherwise.
    let theirs = "mirage-theirsession-node-0";
    std::fs::write(env.containers.join(theirs), "").unwrap();
    env.label_runtime(theirs, "/some/other/runtime");

    let out = env
        .base
        .mirage()
        .arg("cleanup")
        .env("MIRAGE_CONTAINER_PROVIDER", &env.provider)
        .output()
        .unwrap();
    assert!(out.status.success());
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(
        !text.contains(theirs),
        "cleanup named another runtime directory's container:\n{text}"
    );
    assert_eq!(
        env.live_containers(),
        vec![theirs.to_string()],
        "cleanup removed a container belonging to another mirage"
    );
}

#[test]
fn cleanup_spares_the_containers_of_a_live_run() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("cleanup-live-container");

    let mut run = env.base.spawn_run(
        &["--profile", "cp"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
    let id = run.await_ready(READY);
    let container = format!("mirage-{id}-node-0");
    assert_eq!(env.live_containers(), vec![container.clone()]);

    let out = env
        .base
        .mirage()
        .arg("cleanup")
        .env("MIRAGE_CONTAINER_PROVIDER", &env.provider)
        .output()
        .unwrap();
    assert!(out.status.success());

    // The safety property that makes this command safe to run at any
    // time: a session whose run still answers is not an orphan, however
    // much its containers look like one from the outside.
    assert_eq!(
        env.live_containers(),
        vec![container],
        "cleanup removed a live session's container"
    );

    run.signal(Signal::SIGINT);
    run.wait(READY);
    assert_no_leaks(&tag);
}

/// A run whose own command fails still waits for a borrower.
///
/// The borrower wait used to live only on the path where everything went
/// right. Once a session is healthy it is borrowable, but the two steps
/// between readiness and the workload starting — building the exec
/// definition and starting it — returned with `?` straight past the wait
/// and into `Run::destroy`. So a run whose own `--workdir` was wrong tore
/// the session down under a borrower that had done nothing wrong, while
/// an identical run whose workdir was fine waited for it. Same session,
/// same borrower, opposite treatment, decided by something the borrower
/// has no part in.
///
/// Containerised because the workdir probe is the only step after
/// readiness that asks the provider anything, and therefore the only
/// place the window can be held open long enough to attach to; see
/// [`STALL_WORKDIR`].
#[test]
fn a_failing_runs_own_command_still_waits_for_a_borrower() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("failing-run-borrower");

    // The session comes up, and then this run's own command fails: the
    // probe stalls five seconds and answers "no such directory".
    let mut run = env.base.spawn_run(
        &["--profile", "cp", "--workdir", STALL_WORKDIR],
        &["/bin/true"],
    );
    let watch = run.watch_stderr();
    let id = run.await_ready(READY);

    // Inside the stall: borrow the session the way another terminal
    // would, and hold it. That this can be done at all is the other half
    // of the fix — the run keeps serving its socket across the probe.
    let mut borrower = std::process::Command::new(env.base.bin())
        .args([
            "exec",
            "--session",
            &id,
            "--",
            "/bin/sh",
            "-c",
            &tagged_sleep(&tag),
        ])
        .envs(env.base.child_env())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    wait_for("the borrower to start its workload", READY, || {
        count_processes(&tag) > 0
    });

    // The property. On the unfixed code the run is already gone by now,
    // having reported the workdir error and destroyed the session; this
    // line is never printed because `wait_for_borrowers` was never
    // reached.
    wait_for("the failing run to wait for its borrower", READY, || {
        watch.contains("borrower(s) are still using session")
    });
    assert!(
        run.is_running(),
        "a run that failed its own command exited while a borrower held \
         the session:\n{}",
        watch.text()
    );

    // And the interrupt escape survives the move: the user still decides
    // when they have waited long enough, and the error that ended the run
    // is still the error they are told about.
    run.signal(Signal::SIGINT);
    let out = String::from_utf8_lossy(&run.wait(READY).stderr).into_owned();
    assert!(
        out.contains(STALL_WORKDIR),
        "the run must still report why its own command failed, rather \
         than losing it to the borrower wait:\n{out}"
    );

    let _ = borrower.kill();
    let _ = borrower.wait();
    assert_no_leaks(&tag);
}

/// A run answers its socket while it is tearing down.
///
/// Teardown is the second-longest thing a run does — a container removal
/// per node, then a network — and for the whole of it nothing was calling
/// `accept`. The socket file was still on disk and the listener was still
/// bound, so `mirage exec` from another terminal connected happily, sat
/// in the kernel backlog, and thirty seconds later was told the run "is
/// either still starting up, or shutting down". It was shutting down, and
/// the session had known that since before the client dialled.
///
/// The property is about *latency*, not about the wording: every attempt
/// has to be answered promptly, whatever the answer is. So the loop times
/// each one and bounds it well under `DESCRIBE_TIMEOUT`, and separately
/// insists that at least one attempt landed inside the teardown window —
/// otherwise a test that never opened the window would pass by never
/// having tried.
#[test]
fn a_run_answers_its_socket_while_it_is_tearing_down() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");
    let tag = marker("teardown-answers");

    let mut run = env.base.spawn_run(
        &["--profile", "cp"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
    let id = run.await_ready(READY);
    wait_for("the workload to start", READY, || count_processes(&tag) > 0);

    // Armed after bring-up, so only the removals are slowed.
    env.arm_teardown_stall();
    run.signal(Signal::SIGINT);

    // `DESCRIBE_TIMEOUT` is thirty seconds; anything approaching it is
    // the hang this test exists for. Ten is far above a real round trip
    // and far below the bug.
    const PROMPT: Duration = Duration::from_secs(10);
    let mut saw_teardown = false;
    let mut slowest = Duration::ZERO;
    while run.is_running() {
        let started = std::time::Instant::now();
        let out = env.base.run(&["exec", "--session", &id, "--", "/bin/true"]);
        let elapsed = started.elapsed();
        slowest = slowest.max(elapsed);
        let text = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(
            elapsed < PROMPT,
            "an exec against a run that is tearing down took {elapsed:?}; \
             the run stopped answering its socket. It said: {text}"
        );
        if text.contains("shutting down") {
            saw_teardown = true;
        }
    }
    assert!(
        saw_teardown,
        "no attempt landed while the session was tearing down, so this \
         proved nothing; slowest attempt was {slowest:?}"
    );

    run.wait(READY);
    assert_no_leaks(&tag);
}

#[test]
fn provider_chatter_stays_off_a_redirected_stdout() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    // `mirage run --profile cp -- ./app > out.txt`, typed at a prompt:
    // stdout is a file, stderr is still the terminal. Both halves matter.
    // Without a terminal on stderr mirage captures the provider instead
    // of showing it, which is the branch every other test in this file
    // exercises and the one that was never broken.
    let pty = nix::pty::openpty(None, None).unwrap();
    let out_path = env.base.root().join("redirected.out");
    let out = std::fs::File::create(&out_path).unwrap();

    let mut child = env
        .base
        .mirage()
        .args([
            "run",
            "--profile",
            "cp",
            "--",
            "/bin/echo",
            "hello-from-container",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(out))
        // Moved in, so the parent's copy of the slave is closed once the
        // child has it and the master below sees an end of file when the
        // run exits.
        .stderr(std::process::Stdio::from(pty.slave))
        .spawn()
        .unwrap();

    // Drained on a thread: a pty has a small buffer, and a pull that
    // filled it while nobody was reading would block the run rather than
    // fail the test.
    let terminal = std::thread::spawn(move || {
        use std::io::Read as _;
        let mut master = std::fs::File::from(pty.master);
        let mut seen = Vec::new();
        // The read ends in `EIO` on Linux once the last slave is closed,
        // which is this loop's end of file rather than a failure.
        let _ = master.read_to_end(&mut seen);
        String::from_utf8_lossy(&seen).into_owned()
    });

    // Bounded, and killed rather than left behind if the bound is hit:
    // this run is not owned by the harness's `Run`, so nothing else
    // would stop it.
    let deadline = std::time::Instant::now() + READY;
    let status = loop {
        match child.try_wait().unwrap() {
            Some(status) => break status,
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("the run did not finish within {READY:?}");
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    };
    let terminal = terminal.join().unwrap();
    assert!(status.success(), "the run failed:\n{terminal}");

    // The whole promise, in one assertion: what the workload wrote, and
    // nothing else. The provider's pull writes a digest to *its* stdout,
    // and mirage inherited that descriptor — so a redirected run's file
    // held `sha256:…` above the workload's own output.
    let redirected = std::fs::read_to_string(&out_path).unwrap();
    assert_eq!(
        redirected, "hello-from-container\n",
        "something other than the workload wrote to the run's stdout \
         (the terminal saw:\n{terminal})"
    );
    // And it is not simply gone: a user watching the pull still sees it,
    // on the stream mirage talks about itself on.
    assert!(
        terminal.contains("pulled-digest"),
        "the provider's output was discarded rather than sent to stderr:\n{terminal}"
    );
}

#[test]
fn a_mount_whose_host_path_is_missing_is_refused_before_anything_is_created() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    // The engines disagree about this: docker creates the path as a
    // root-owned directory on the host and starts the container, podman
    // refuses with `statfs …: no such file or directory`. One `--mount`
    // must not mean two things, so mirage decides it before either is
    // asked.
    let missing = env.base.root().join("not-here");
    let err = env.base.fails(&[
        "run",
        "--profile",
        "cp",
        "--mount",
        &format!("{}:/data", missing.display()),
        "--",
        "/bin/true",
    ]);
    assert!(
        err.contains(&missing.display().to_string()) && err.contains("does not exist"),
        "the error must name the host path that is missing: {err}"
    );
    assert!(
        !missing.exists(),
        "mirage created the host path it refused to mount"
    );
    let log = env.provider_log();
    assert!(
        !log.contains("run --rm"),
        "a container was created for a mount that could never work:\n{log}"
    );
    assert!(
        env.live_containers().is_empty(),
        "containers outlived a failed bring-up: {:?}",
        env.live_containers()
    );
}

#[test]
fn a_mount_laid_over_mirages_own_directory_is_refused() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    // Mirage bind-mounts its own binary, config and session scratch under
    // `/mnt/mirage` in every node container. A user mount at or above
    // that path overlaps them, and the engine resolves the overlap by
    // creating mirage's destinations inside the user's host directory —
    // as root, because the container writes them. The run reported
    // success, and the user's directory came back holding `bin`,
    // `config`, `lib` and `runtime` entries they could not delete.
    let mine = env.base.root().join("mine");
    std::fs::create_dir_all(&mine).unwrap();
    let err = env.base.fails(&[
        "run",
        "--profile",
        "cp",
        "--mount",
        &format!("{}:/mnt/mirage", mine.display()),
        "--",
        "/bin/true",
    ]);
    assert!(
        err.contains(&mine.display().to_string()) && err.contains("/mnt/mirage"),
        "the refusal must name both the mount and the directory it covers: {err}"
    );
    assert!(
        std::fs::read_dir(&mine).unwrap().next().is_none(),
        "something was created in the user's mount directory: {:?}",
        std::fs::read_dir(&mine).unwrap().flatten().count()
    );
    let log = env.provider_log();
    assert!(
        !log.contains("run --rm"),
        "a container was created for a mount that could never work:\n{log}"
    );

    // And a mount *inside* one of mirage's own, which is the same
    // collision in the direction that reads as harmless. `/mnt/mirage` is
    // where the session scratch is mounted, and `rj_config.json` in it is
    // what tells the emulator's interposer which device to simulate.
    // Overlaying that one file does not fail: the interposer finds no
    // config, declines to interpose, and the workload runs on the real
    // device at full speed, producing results that look exactly like a
    // successful emulated run.
    //
    // The at-or-above check cannot catch this. It has to answer "no" for
    // every path below the reserved directory, because mirage's own
    // mounts are all down there — so this was refused by nothing until
    // the user's mounts were checked against mirage's concrete
    // destinations before the two lists were combined.
    let config = env.base.root().join("my-config.json");
    std::fs::write(&config, "{}").unwrap();
    let err = env.base.fails(&[
        "run",
        "--profile",
        "cp",
        "--mount",
        &format!("{}:/mnt/mirage/runtime/rj_config.json", config.display()),
        "--",
        "/bin/true",
    ]);
    assert!(
        err.contains("/mnt/mirage/runtime"),
        "the refusal must name the mirage mount the user's lands inside: {err}"
    );
    let log = env.provider_log();
    assert!(
        !log.contains("run --rm"),
        "a container was created for a mount that would disable emulation:\n{log}"
    );

    // A mount that merely shares a textual prefix with the reserved
    // directory is the user's to make, and a check that compared strings
    // rather than path components would refuse it.
    let out = env.base.ok(&[
        "run",
        "--profile",
        "cp",
        "--mount",
        &format!("{}:/mnt/mirage-of-my-own", mine.display()),
        "--",
        "/bin/echo",
        "mounted-elsewhere",
    ]);
    assert!(out.contains("mounted-elsewhere"), "{out}");
}

#[test]
fn a_published_port_is_refused_on_a_multi_node_session() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    // Every node runs the same container with the same argv, so all of
    // them publish onto the same host port: the first binds it and the
    // rest cannot. This used to be discovered by the *second* container,
    // in the engine's own words ("Bind for 0.0.0.0:18099 failed: port is
    // already allocated"), with node 1 already running.
    let err = env.base.fails(&[
        "run",
        "--profile",
        "cp",
        "--port",
        "18099:8000",
        "--num-nodes",
        "2",
        "--",
        "/bin/true",
    ]);
    assert!(
        err.contains("18099:8000") && err.contains("2-node"),
        "the refusal must name the port and the node count: {err}"
    );
    assert!(
        err.contains("--num-nodes 1"),
        "the refusal must say what to do instead: {err}"
    );
    let log = env.provider_log();
    assert!(
        !log.contains("run --rm"),
        "a node container was started before the impossible port was noticed:\n{log}"
    );
    assert!(
        env.live_containers().is_empty(),
        "containers outlived a refused bring-up: {:?}",
        env.live_containers()
    );
}

#[test]
fn one_port_asked_for_twice_is_published_once() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    // Restating the profile's port on the command line, or repeating
    // `--port`, reached the engine as two `-p` arguments and failed the
    // container with `address already in use` — a message about somebody
    // else's process, which is not what happened.
    let out = env.base.ok(&[
        "run",
        "--profile",
        "cp",
        "--port",
        "18098:8000",
        "--port",
        "18098:8000",
        "--",
        "/bin/echo",
        "ports-deduplicated",
    ]);
    assert!(out.contains("ports-deduplicated"), "{out}");

    let log = env.provider_log();
    let run_line = log
        .lines()
        .find(|l| l.starts_with("run --rm"))
        .unwrap_or_else(|| panic!("no container was launched:\n{log}"));
    assert_eq!(
        run_line.matches("-p 18098").count(),
        1,
        "one host port must be published once: {run_line}"
    );
}

#[test]
fn a_session_with_no_image_says_so_instead_of_pulling_nothing() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    // `--image ""` was handed to the engine as a missing argument, and
    // the user watched `pulling image  (this can take a while)` with a
    // hole where the name should be.
    let err = env
        .base
        .fails(&["run", "--profile", "cp", "--image", "", "--", "/bin/true"]);
    assert!(
        err.contains("--image") && err.contains("ubuntu:24.04"),
        "the refusal must name the flag and show what one looks like: {err}"
    );
    assert!(
        !env.provider_log().contains("pull"),
        "the engine was asked to pull an image that was never named:\n{}",
        env.provider_log()
    );
}

#[test]
fn a_provider_that_is_not_a_container_engine_is_refused_by_name() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }

    // Executable, and not an engine. Without a check this became a bare
    // `No such file or directory` — or, here, whatever the first provider
    // invocation of bring-up happened to make of it.
    let impostor = env.base.root().join("impostor.sh");
    std::fs::write(&impostor, "#!/bin/sh\nexit 1\n").unwrap();
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&impostor, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    env.base.ok(&[
        "profile",
        "create",
        "impostor",
        "--emulator",
        TEST_EMULATOR,
        "--no-input",
        "--image",
        "img:latest",
        "--container-provider",
        &impostor.to_string_lossy(),
    ]);

    let err = env
        .base
        .fails(&["run", "--profile", "impostor", "--", "/bin/true"]);
    assert!(
        err.contains(&impostor.to_string_lossy().to_string()) && err.contains("--version"),
        "the refusal must name the provider and what mirage asked it: {err}"
    );
}

#[test]
fn the_provider_environment_variable_loses_to_the_profile_and_says_so() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    // The documented order is: what the session names (its profile, or
    // `--container-provider`) beats `MIRAGE_CONTAINER_PROVIDER`, which
    // beats autodetection. The bug was not the order — mirage cannot tell
    // a profile-pinned provider from a flag-given one by the time it
    // resolves either — it was the silence: an exported variable that did
    // nothing at all, with the run carrying on as though it had never
    // been set.
    let out = env
        .base
        .mirage()
        .args(["run", "--profile", "cp", "--", "/bin/echo", "profile-won"])
        .env("MIRAGE_CONTAINER_PROVIDER", "/nonexistent/other-engine")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "the profile's provider must still be the one used:\n{stderr}"
    );
    assert!(stdout.contains("profile-won"), "{stdout}");
    assert!(
        stderr.contains("MIRAGE_CONTAINER_PROVIDER=/nonexistent/other-engine")
            && stderr.contains(&env.provider.to_string_lossy().to_string()),
        "an ignored provider variable must say so, naming both: {stderr}"
    );
}

#[test]
fn a_derived_hack_image_is_labelled_like_everything_else_mirage_creates() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    env.create_containerized_profile("cp");

    // A `mirage-hack-…` image is host state that outlives the session
    // that built it, and nothing on disk records that it exists. Without
    // a label on the image itself there is nothing to attribute it to
    // later — no command can find it, and the user cannot tell it from
    // any other stray image.
    let out = env.base.ok(&[
        "run",
        "--profile",
        "cp",
        "--hack",
        "update-gcc-via-ppa",
        "--",
        "/bin/echo",
        "hacked",
    ]);
    assert!(out.contains("hacked"), "{out}");

    let log = env.provider_log();
    let build = log
        .lines()
        .find(|l| l.starts_with("build "))
        .unwrap_or_else(|| panic!("no derived image was built:\n{log}"));
    assert!(
        build.contains("--label mirage.owner=mirage"),
        "a derived image must be marked as mirage's: {build}"
    );
    // The runtime directory too, matched by key rather than by value: it
    // is recorded canonicalised, and the temporary root this test runs
    // under is not necessarily spelled the way it was handed to us.
    assert!(
        build.contains("--label mirage.runtime="),
        "a derived image must say which runtime directory built it: {build}"
    );
}

/// A stand-in engine whose answers about *images* a test dictates.
///
/// The shared mock above is built for a session's lifecycle and has no
/// images at all, which leaves the image half of `mirage cleanup` — the
/// only host state mirage deliberately lets outlive a session — resting
/// on two decisions that no test could reach: whether an image is in use,
/// and whether it is mirage's. Both are answered by the engine, so both
/// need an engine that can be told what to say.
///
/// `ps` is the interesting one: cleanup asks it twice with different
/// arguments. `ps --all --filter … --format {{.ID}}` lists mirage's
/// containers, and `ps --all --format {{.ID}}\t{{.Image}}` asks what
/// every container is running — so the script tells them apart by the
/// presence of `--filter` and only applies `ps_all` to the second.
fn write_image_provider(
    path: &Path,
    image_id: &str,
    image_tag: &str,
    owner: &str,
    runtime: &str,
    ps_all: &str,
) {
    use std::os::unix::fs::PermissionsExt;
    let script = r#"#!/bin/sh
case "$1" in
  ps)
    for a in "$@"; do
      [ "$a" = "--filter" ] && exit 0
    done
    __PS_ALL__
    ;;
  images)
    printf '%s\t%s\n' '__IMAGE_ID__' '__IMAGE_TAG__'
    exit 0 ;;
  image)
    # `image inspect --format '<owner>\t<runtime>' <id>`.
    printf '%s\t%s\n' '__OWNER__' '__RUNTIME__'
    exit 0 ;;
  *) exit 0 ;;
esac
"#
    .replace("__PS_ALL__", ps_all)
    .replace("__IMAGE_ID__", image_id)
    .replace("__IMAGE_TAG__", image_tag)
    .replace("__OWNER__", owner)
    .replace("__RUNTIME__", runtime);
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// The images `mirage cleanup --dry-run --json` says it would remove.
fn images_cleanup_would_remove(env: &Env, provider: &Path) -> Vec<String> {
    let out = env
        .base
        .mirage()
        .args(["cleanup", "--dry-run", "--json"])
        .env("MIRAGE_CONTAINER_PROVIDER", provider)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cleanup failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!("{e}: {}", String::from_utf8_lossy(&out.stdout));
    });
    report["resources"]
        .as_array()
        .expect("a cleanup report lists resources")
        .iter()
        .filter(|r| r["kind"] == "image")
        .map(|r| r["id"].as_str().unwrap_or_default().to_string())
        .collect()
}

#[test]
fn an_image_is_left_alone_when_the_engine_cannot_say_what_is_using_it() {
    // "No answer" is not "nothing". Cleanup asks the engine what every
    // container was created from, and reading a failed question as an
    // empty answer makes every image on the host look unused — including
    // the one a live session's containers are running right now, which
    // cleanup would then remove out from under it.
    //
    // The pair is the test. A single "no image was reclaimed" proves
    // nothing, because an image can fail to be reclaimed for a dozen
    // reasons; the control run differs in exactly one thing, whether `ps`
    // answers at all.
    let env = Env::new();
    let id = "abcdef012345";
    let tag = "mirage-hack-demo:latest";

    let mute = env.base.root().join("mute-engine.sh");
    write_image_provider(&mute, id, tag, "mirage", &env.own_runtime(), "exit 1");
    assert!(
        images_cleanup_would_remove(&env, &mute).is_empty(),
        "an image was reclaimed on an engine that could not be asked what \
         is using it"
    );

    let answers = env.base.root().join("answering-engine.sh");
    write_image_provider(&answers, id, tag, "mirage", &env.own_runtime(), "exit 0");
    assert_eq!(
        images_cleanup_would_remove(&env, &answers),
        vec![tag.to_string()],
        "the same image must be reclaimed once the engine answers, or the \
         assertion above proved nothing"
    );
}

#[test]
fn an_image_is_reclaimed_only_when_it_says_mirage_built_it() {
    // The engine is asked for images carrying `mirage.owner=mirage` and
    // then asked again, per image, what that label actually says. The
    // re-check is not redundant: the filter is the engine's to honour,
    // the answer arrives as a Go template that renders a missing key as
    // text rather than failing, and `mirage-hack-…` images left by a
    // mirage older than the labels look exactly like ours by name. Only
    // a positive `mirage.owner` makes an image mirage's to delete.
    let env = Env::new();
    let id = "abcdef012345";
    let tag = "mirage-hack-demo:latest";

    let theirs = env.base.root().join("someone-elses.sh");
    write_image_provider(&theirs, id, tag, "not-mirage", &env.own_runtime(), "exit 0");
    assert!(
        images_cleanup_would_remove(&env, &theirs).is_empty(),
        "an image the engine says is not mirage's was reclaimed anyway"
    );

    // An image whose owner label is absent altogether: the Go template
    // renders `<no value>`, which is a string and would compare equal to
    // nothing at all if it were not recognised as "no label".
    let unlabelled = env.base.root().join("unlabelled.sh");
    write_image_provider(
        &unlabelled,
        id,
        tag,
        "<no value>",
        &env.own_runtime(),
        "exit 0",
    );
    assert!(
        images_cleanup_would_remove(&env, &unlabelled).is_empty(),
        "an image with no owner label was reclaimed"
    );

    let ours = env.base.root().join("ours.sh");
    write_image_provider(&ours, id, tag, "mirage", &env.own_runtime(), "exit 0");
    assert_eq!(
        images_cleanup_would_remove(&env, &ours),
        vec![tag.to_string()],
        "an image mirage built under this runtime directory must be \
         reclaimable, or the two assertions above proved nothing"
    );
}

/// A real container engine on this machine that already has the image
/// these tests need, preferring podman exactly as mirage does.
///
/// Both halves are required. An engine without the image would make the
/// test pull one over the network, which is neither this test's subject
/// nor a thing a test suite should do to a machine.
fn real_engine_with_image() -> Option<&'static str> {
    ["podman", "docker"].into_iter().find(|engine| {
        std::process::Command::new(engine)
            .args(["image", "inspect", REAL_IMAGE])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// The image the real-engine test runs its node in.
///
/// Anything with a shell and a glibc: the emulator's `LD_PRELOAD` is
/// applied to the container's own entrypoint, and a musl image (alpine,
/// busybox) cannot load it — which is a real failure mode, but not this
/// test's.
const REAL_IMAGE: &str = "ubuntu:24.04";

/// A real container, force-removed when this value is dropped.
///
/// A test that fails between creating one and asserting it is gone must
/// not leave it on the machine — and unlike the mock's, a real client is
/// reparented when the test kills its `mirage run` and goes on holding
/// the container.
struct RealContainer {
    engine: &'static str,
    name: String,
}

impl RealContainer {
    fn exists(&self) -> bool {
        std::process::Command::new(self.engine)
            .args(["container", "inspect", &self.name])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }
}

impl Drop for RealContainer {
    fn drop(&mut self) {
        let _ = std::process::Command::new(self.engine)
            .args(["rm", "-f", &self.name])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

#[test]
fn a_real_engine_creates_and_removes_the_container_with_the_run() {
    let env = Env::new();
    if skip_without_emulator() {
        return;
    }
    let Some(engine) = real_engine_with_image() else {
        eprintln!(
            "SKIP: neither podman nor docker has {REAL_IMAGE} locally, so the real-engine \
             container lifecycle cannot be exercised here."
        );
        return;
    };

    // The mock elsewhere in this file honours `--rm` by removing its own
    // record, which means the crate's headline claim — a container never
    // outlives the run that made it — is asserted against a script
    // playing both sides. This is the same claim against a real engine:
    // mirage asks for the container, a real `podman`/`docker` creates it,
    // and a Ctrl-C in the run's terminal has to be enough to remove it.
    env.base.ok(&[
        "profile",
        "create",
        "real",
        "--emulator",
        TEST_EMULATOR,
        "--no-input",
        "--image",
        REAL_IMAGE,
        "--container-provider",
        engine,
    ]);
    let tag = marker("real-container");

    let mut run = env.base.spawn_run(
        &["--profile", "real"],
        &["/bin/sh", "-c", &tagged_sleep(&tag)],
    );
    let id = run.await_ready(READY);
    let container = RealContainer {
        engine,
        name: format!("mirage-{id}-node-0"),
    };
    assert!(
        container.exists(),
        "the session reported ready without a container: {}",
        container.name
    );

    run.signal(Signal::SIGINT);
    run.wait(READY);

    wait_for(
        "the real container to be removed with its run",
        Duration::from_secs(60),
        || !container.exists(),
    );
    assert_no_leaks(&tag);
}

#[test]
fn the_suite_can_actually_run() {
    // Guards against the container suite going green while every test in it skipped
    // for a missing emulator runtime. See `assert_suite_can_run`.
    harness::assert_suite_can_run();
}
