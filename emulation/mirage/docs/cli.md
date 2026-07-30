# mirage CLI reference

`mirage` is organized into a small set of verbs, each managing one kind of
resource. Every command supports `--help`, and most support `--json` for
machine-readable output.

```text
mirage <command> [options]
mirage <command> --help        # per-command help
```

There is one executable and nothing behind it. The configuration verbs
(`profile`, `topology`, `agent`, `emulators`, `state`, `paths`) are pure
filesystem work answered in this process. The execution verbs are `run`,
which *owns* a session for as long as it runs, and `exec`, which
*borrows* one from a live `mirage run`. No command starts anything in the
background, and nothing outlives the process you typed it into.

## Global options

| Flag          | Description                                                       |
| ------------- | ---------------------------------------------------------------- |
| `--json`      | Emit JSON output where applicable.                               |
| `-v`, `-vv`   | Increase logging verbosity (`-v` = info, `-vv` = debug).         |
| `--help`      | Print help (top level or per command).                           |
| `--version`   | Print the mirage version.                                        |

Global flags may appear before or after the subcommand, e.g.
`mirage --json profile list`.

## Command summary

| Command            | Purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| `mirage run`       | Bring up a session, run a command in it, tear it down.         |
| `mirage exec`      | Run a command in the session a live `mirage run` owns.         |
| `mirage emulators` | List emulator backends and their install / support status.    |
| `mirage profile`   | Manage profiles (reusable emulator presets).                   |
| `mirage topology`  | Manage topologies (rack/node/GPU layouts).                     |
| `mirage agent`     | Manage agents (hardware GPU definitions).                      |
| `mirage state`     | Manage mirage's on-disk state (builtins, purge).              |
| `mirage paths`     | Print where mirage stores its state.                          |
| `mirage about`     | Show version, copyright, and third-party licenses.            |

## `mirage run`

```text
mirage run [--profile NAME] [--emulator NAME]
           [--num-nodes N] [--gpus-per-node N] [--nproc-per-node N]
           [--workdir DIR] [--env KEY=VALUE]...
           [--image IMAGE] [--mount SPEC]... [--port SPEC]...
           [--container-provider PROV] [--hack HACK]...
           [--exec-mode functional|clocked] [-o|--option KEY=VALUE]...
           [--plugin NAME]... [--config PATH]
           [--daemon | --in-process] [--capture-all]
           -- <cmd> [args...]
```

```sh
mirage run --profile mi350x -- pytest -x my_tests/
```

`run` is the runtime. It builds a session from the profile (plus any
overrides below), waits up to 60 seconds for it to become healthy, starts
your command in it, and tears the whole thing down when the command
exits — every workload process reaped, every container removed, the
scratch directory deleted. The session lives in *this* process's memory,
so it exists exactly as long as this command does. Ctrl-C is part of that
contract: the first interrupt is forwarded to the workload so it can
clean up, the second stops waiting for it, and either way teardown runs.

The session id is printed on stderr as soon as it is created:

```sh
$ mirage run --profile mi350x -- ./app
mirage: session s-20260730-191636-1f4a-0
```

Time spent pulling or building a container image is not counted against
the readiness timeout — a first-time image pull can take minutes without
being mistaken for a session that will not come up. A session that fails
to come up is destroyed and the reason reported, rather than left behind
for you to clean up.

### Selecting what to emulate

* `--profile NAME` picks the profile; it defaults to the `mi350x`
  builtin. Every other flag in this group overrides a field of it for
  this run only, leaving the stored profile untouched.
* `--emulator NAME` overrides the profile's backend (see
  `mirage emulators` for what this build has).
* `--num-nodes N` and `--gpus-per-node N` override the profile
  topology's counts.
* `--exec-mode functional|clocked` selects functional emulation
  (correct results, no timing model — the default) or clocked emulation.
* `-o`/`--option KEY=VALUE` sets an emulator option directly. Repeatable.
  Values that look like integers or booleans are stored as such.
* `--plugin NAME` enables an execution plugin (e.g. `race`, `logging`)
  with its schema defaults. Repeatable, and merged with whatever the
  profile already enables.
* `--config PATH` hands the backend an explicit emulator config file
  instead of synthesising one from the profile (this is the upstream
  `rocjitsu --config`). The path is made absolute, so it resolves
  regardless of the workload's working directory.
* `--daemon` runs the emulator out-of-process. This is the default; the
  flag exists for explicitness and for the rocjitsu drop-in alias.
  `--in-process` selects the opposite. In-process mode cannot share GPU
  memory between processes, so multi-GPU RCCL collectives need the
  daemon.

### Containers

* `--image IMAGE` runs every node inside a container built from that
  image, enabling containerisation for a profile that does not have it.
* `--mount HOST[:CONTAINER[:ro|rw]]` adds a bind mount to every node
  container. Repeatable.
* `--port HOST_PORT[:CONTAINER_PORT][/tcp|/udp]` publishes a container
  port on the host, like docker's `-p`. Repeatable.
* `--container-provider podman|docker|PATH` chooses the engine;
  autodetected (podman, then docker) when omitted.
* `--hack HACK` builds a derivative image from the base before launching.
  The only value today is `update-gcc-via-ppa`, which updates
  `libstdc++6`/`libgcc-s1` from the `ubuntu-toolchain-r/test` PPA and
  fixes `GLIBCXX_*`/`GCC_*` "version not found" errors from binaries built
  against a newer toolchain than the image ships.

`--mount`, `--port`, `--container-provider` and `--hack` all require a
containerised profile or `--image`.

Containers are started with `--rm` and are **not** detached: the provider
client is a child process mirage owns, so killing it stops the container
and `--rm` removes it. There is no container that can outlive the `mirage
run` that started it, and nothing to garbage-collect afterwards.

### The command, its environment and its terminal

Everything after `--` is passed verbatim to the workload.

* `--workdir DIR` is the working directory for the command; it defaults
  to the directory you ran `mirage run` in.
* `--env KEY=VALUE` injects extra environment variables. Repeatable.
  `LD_PRELOAD` is merged with the emulator's interposer rather than
  clobbering it.

The workload **inherits mirage's own stdin, stdout and stderr**. There is
no pseudo-terminal anywhere in mirage: if your terminal is a terminal, an
inherited descriptor already is one, and if it is a pipe, inheriting keeps
the bytes exact. That single fact is what makes this work:

```sh
mirage run --profile mi350x -- bash        # a real shell: prompt, echo,
                                           # line editing, Ctrl-C, job control
mirage run --profile mi350x -- python3     # a working REPL
```

and what makes this work too:

```sh
mirage run --profile mi350x -- ./app > out.txt 2> err.txt   # streams stay split
```

stdout and stderr are never merged, output is never re-encoded, and window
resizes need no handling because the terminal is simply the one you were
already using.

### Many processes: `--nproc-per-node`

`--nproc-per-node N` (alias `--nproc_per_node`) launches N workload
processes per node, like `torchrun --nproc-per-node`. Each process gets a
distinct `LOCAL_RANK` (`0..N`) and a distinct global `RANK`, and the job's
`WORLD_SIZE` becomes `num_nodes * nproc_per_node`, so `torch.distributed`
runs without a separate launcher. `MASTER_ADDR`/`MASTER_PORT` are set on
every rank as well, aliasing mirage's own `MIRAGE_HEAD_ADDR`/
`MIRAGE_HEAD_PORT`. Give each node at least N GPUs (`--gpus-per-node`) so
every process can pin its own device.

Only one process can meaningfully read a terminal, so rank 0 inherits
stdin and every other rank reads `/dev/null` and sees an immediate EOF.
Rank 0 also stays in mirage's process group for that reason — a
background process group reading the controlling terminal is stopped with
`SIGTTIN`.

### `--capture-all`

By default, every rank writes to your terminal directly and mirage is not
in the middle. That is what keeps `bash` interactive and redirection
byte-exact, but with several nodes writing at once the lines interleave
with nothing to say which rank wrote what.

`--capture-all` pipes every rank's stdout and stderr through mirage,
which prints them line by line, prefixed with the rank that produced
them:

```sh
$ mirage run --profile mi350x --num-nodes 2 -- sh -c 'echo hello from $MIRAGE_RANK'
mirage: session s-20260730-191636-1f4a-0
[0] hello from 0
[1] hello from 1
```

Details worth knowing:

* Labelling is line-oriented. Each rank's stream is buffered until it
  holds a complete line, so a line split across three reads is still
  printed once, with one prefix. A trailing partial line — a prompt, a
  progress bar — is flushed when the stream ends rather than dropped.
* stdout and stderr stay separate all the way out, so redirecting one of
  them still works under `--capture-all`.
* **No rank gets stdin.** Capturing costs you interactivity; that is the
  trade. Do not use it for `bash`.

## `mirage exec`

```text
mirage exec [-s|--session ID] [--nproc-per-node N] [--capture-all]
            [--env KEY=VALUE]... [--workdir DIR]
            -- <cmd> [args...]
```

Run a command inside the session a live `mirage run` owns — the usual
shape is one terminal running the job and another one exec'ing into it:

```sh
# terminal 1
$ mirage run --profile mi350x -- ./train.py
mirage: session s-20260730-191636-1f4a-0

# terminal 2
$ mirage exec -- rocm-smi
$ mirage exec -- bash            # an interactive shell, in *this* window
```

`exec` asks the run process exactly one question over that run's Unix
socket (`Request::Describe`, answered with a `SessionDescription`) and
then builds the process grid locally with the same
`mirage_supervisor::build_specs` the run itself uses — so a command
behaves identically whichever way it was started. The processes are
spawned **by this command, as its own children, in its own terminal**.
That is the whole reason the client spawns rather than the run: a child
inherits the standard streams of whoever forked it, so a process started
by the run process would talk to the run's terminal, not to yours.
Spawning here is what makes `mirage exec -- bash` an interactive shell in
the window you typed it in, with no pseudo-terminal, no output forwarding
and no stdin relay.

* `-s`/`--session ID` names the session. It is optional and usually
  omitted: with exactly one run live, mirage picks it. With several, the
  error lists the candidates rather than guessing. It is a flag rather
  than a positional because everything after `--` belongs to the command:
  with both positional, `mirage exec -- bash` could equally mean "session
  `bash`".
* `--nproc-per-node N`, `--capture-all`, `--env KEY=VALUE` and
  `--workdir DIR` mean exactly what they mean on `run`, including the
  `[<rank>] ` prefixes and the loss of stdin under `--capture-all`.
* The exec's processes die with this command, and this command exits with
  the workload's exit code. Ctrl-C is forwarded, then escalates, exactly
  as in `run`.

If no run is serving the session, `exec` says so:

```text
error: no `mirage run` is serving session s-20260730-191636-1f4a-0 (…).
A session exists only while the `mirage run` that created it is alive.
```

## `mirage emulators`

List the emulator backends compiled into this build, whether each one's
runtime is installed, and whether this machine's hardware supports it.

```text
mirage emulators [-l|--long]
```

* The default emulator for new profiles is marked with `*`.
* `-l` shows a detailed block per backend, including the support reason.
* `--json` emits the full backend descriptors.

```sh
$ mirage emulators
NAME          INSTALLED  SUPPORTED  DESCRIPTION
rocjitsu      no         yes        ROCm just-in-time GPU emulator (cycle-accurate or functional)
...
* = default emulator for new profiles
```

## `mirage profile`

Profiles are reusable emulator presets stored in
`$XDG_CONFIG_HOME/mirage/profile/<name>.json`.

```text
mirage profile list [-l|--long]
mirage profile show <name>
mirage profile create [<name>] [--emulator NAME] [--agent NAME]
                      [--num-nodes N] [--gpus-per-node N]
                      [--description TEXT]
                      [--image IMAGE] [--mount SPEC]... [--port SPEC]...
                      [--container-provider PROV]
                      [--no-input]
mirage profile import <file>          # use '-' for stdin
mirage profile delete <name> [-f|--force]
```

* `--emulator` defaults to the first installed backend (`rocjitsu` on a
  normal install). `--agent` defaults to `MI350X`.
* `--image` containerises the profile (every node runs inside a container
  built from that image). `--mount HOST[:CONTAINER[:ro|rw]]`, `--port`
  and `--container-provider` (`podman`, `docker`, or a path) require
  `--image`.
* On a terminal, `profile create` interactively prompts for any field not
  passed as a flag. Pass `--no-input` (or pipe/redirect stdin) to use defaults
  and stay non-interactive.
* Profiles are validated against their emulator at creation time, so an
  unusable setup is reported immediately.

## `mirage topology`

Topologies describe a rack/node/GPU layout and reference an agent. Builtin
topologies are written on first run.

```text
mirage topology list
mirage topology show <name>
mirage topology create <name> [--agent NAME] [--num-nodes N] [--gpus-per-node N]
mirage topology import <name> <file>   # use '-' for stdin
mirage topology delete <name> [-f|--force]
```

## `mirage agent`

Agents are hardware GPU definitions (e.g. `MI300X`, `MI350X`, `MI450X`).
Builtin agents are written on first run.

```text
mirage agent list
mirage agent show <name>
mirage agent import <name> <file>      # use '-' for stdin
mirage agent delete <name> [-f|--force]
```

## `mirage state`

```text
mirage state builtins                 # (re)write builtin agents/topologies/profiles
mirage state purge [-f|--force] [--all]
```

* mirage writes any *missing* builtins on every run. `state builtins`
  additionally **overwrites** existing builtins (useful after upgrading).
* `state purge` removes the runtime directory. The config directory
  (profiles, topologies, agents) is left alone unless `--all` is given.
* `purge` refuses to run while any `mirage run` is live, and tells you so.
  Killing someone else's foreground command from a state-cleanup
  subcommand would be a surprise; stop it with Ctrl-C in its own terminal
  instead. Each run cleans up after itself when it exits.
* What `purge` *is* for is the wreckage of a run that could not clean up
  — one that was `SIGKILL`ed, OOM-killed, or lost with the machine. It
  reclaims containers and networks carrying mirage's label that no live
  run accounts for. Resources without that label are never candidates, so
  this is safe on a shared engine.

## `mirage paths`

Print the resolved mirage directories. Useful in scripts and for verifying
that environment overrides took effect.

```sh
$ mirage paths
config:   /home/me/.config/mirage
runtime:  /run/user/1000/mirage
profiles: /home/me/.config/mirage/profile
sessions: /run/user/1000/mirage/session
runs:     /run/user/1000/mirage/run
```

`runs` is where each live `mirage run` serves its socket, as
`<runs>/<session>.sock`. One socket per run, named after its session,
rather than one well-known socket for a daemon: the socket *is* the
registration, so connecting to it either reaches the owner or fails — and
failing is how a stale entry, left by a run that was killed, is
recognised.

## `mirage about`

Print the mirage version, copyright, and the third-party crates (with
licenses) the binary is built from.

## Drop-in `rocjitsu` mode

For compatibility with the upstream `rocjitsu` CLI, a bare invocation with a
`--` application separator and no recognised subcommand is routed to
`mirage run`:

```sh
mirage --config cfg.json -- ./app           # == mirage run --config cfg.json -- ./app
mirage --attach --config cfg.json -- ./app  # --attach maps to --daemon
```

Invocations that name a subcommand, or that have no `--` separator (so
`--help`/`--version` keep working), are left untouched.

## Environment variables

| Variable                     | Purpose                                                          |
| ---------------------------- | ---------------------------------------------------------------- |
| `MIRAGE_LOG`                 | Tracing-subscriber filter, e.g. `debug` or `mirage_supervisor=debug`. Takes precedence over `-v`/`-vv`. |
| `MIRAGE_CONTAINER_PROVIDER`  | Default container provider when none is given (`podman`/`docker`/path). |
| `MIRAGE_CONFIG`              | Override the config dir (else `$XDG_CONFIG_HOME/mirage`).         |
| `MIRAGE_RUNTIME`             | Override the runtime dir — run sockets and emulator scratch (else `$XDG_RUNTIME_DIR/mirage`). |
| `XDG_CONFIG_HOME` / `XDG_RUNTIME_DIR` | Standard XDG base directories used when the `MIRAGE_*` overrides are unset. |

mirage sets these *in* every workload process, on every rank:
`MIRAGE_RANK` (the node's rank, 0 = head), `MIRAGE_HEAD_ADDR` and
`MIRAGE_HEAD_PORT`, plus their `torch.distributed` aliases `RANK`,
`LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`.

rocjitsu discovery additionally honours `ROCM_HOME` and the ROCm SDK
install root reported by `rocm-sdk path --root` (see
[`building.md`](building.md)).

## Exit codes

| Code | Meaning                                                              |
| ---- | ------------------------------------------------------------------- |
| 0    | Success.                                                            |
| 1    | A mirage-level error (bad arguments, resource not found, a session that failed to come up, no run serving the named session, …). |
| 2    | Argument parse error (clap).                                        |
| N    | For `run` and `exec`: the workload's exit code, masked to a byte — which preserves the shell's `128 + signal` convention for a signal-killed workload. |

## JSON output

`--json` makes list/show commands emit a parseable JSON document:

```sh
$ mirage --json profile list
[
  "mi300x",
  "mi350x",
  "mi450x"
]
$ mirage --json paths
{
  "config": "/home/me/.config/mirage",
  "runtime": "/run/user/1000/mirage",
  "profiles": "/home/me/.config/mirage/profile",
  "sessions": "/run/user/1000/mirage/session",
  "runs": "/run/user/1000/mirage/run"
}
```

`run` and `exec` have no JSON form: their output is the workload's own.
