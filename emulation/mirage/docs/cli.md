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
| `mirage cleanup`   | Reclaim what a run that died abruptly left behind.            |
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
           [--daemon | --in-process] [--clear-env-vars]
           -- <cmd> [args...]
```

```sh
mirage run --profile mi350x -- pytest -x my_tests/
```

`run` is the runtime. It builds a session from the profile (plus any
overrides below), waits for it to become healthy, starts your command in
it, and tears the whole thing down when the command exits — every workload
process reaped, every container removed, the scratch directory deleted.
The session lives in *this* process's memory, so it exists exactly as long
as this command does. Ctrl-C is part of that contract: the first interrupt
is forwarded to the workload so it can clean up, the second stops waiting
for it, and either way teardown runs.

The session id is printed on stderr as soon as it is created:

```sh
$ mirage run --profile mi350x -- ./app
mirage: session s-20260730-191636-1f4a-0
```

Bring-up says what it is doing. Each phase it enters is printed on stderr,
prefixed like the session line — `mirage: pulling image <ref> (this can
take a while)…`, `mirage: creating session network <name>`, `mirage:
starting node 2/4 (<name>)` — so a slow bring-up is visibly working rather
than apparently hung. Only changed messages are printed, and they go to
stderr so they never land in the middle of a workload's piped output.

The 60-second readiness budget bounds the *gap between phases*, not the
total. Bring-up is a sequence of steps whose count depends on the session
— node containers are launched one at a time — so a single budget for the
whole of it is a budget a large session exceeds by being large: a
four-node session at twenty seconds a node is healthy, and charging it
against one deadline tore it down. The clock is restarted on every phase
change, making it what it always claimed to be, a detector for bring-up
that has stopped moving. Pulling and building an image suspend it
outright, because a pull reports itself once and then goes quiet for
however long the registry takes, so restarting alone would still expire
mid-download. A first-time pull can therefore take minutes without being
mistaken for a session that will not come up.

A session that fails to come up is destroyed and the reason reported,
rather than left behind for you to clean up.

### Selecting what to emulate

* `--profile NAME` picks the profile; it defaults to the `mi350x`
  builtin. Every other flag in this group overrides a field of it for
  this run only, leaving the stored profile untouched.
* `--emulator NAME` overrides the profile's backend (see
  `mirage emulators` for what this build has).
* `--num-nodes N` and `--gpus-per-node N` override the profile
  topology's counts. Both reject `0`, which is not a smaller job but a
  job with nothing to run it on. `--gpus-per-node` also has an upper
  bound on rocjitsu — 64 — because each emulated GPU is a whole software
  device, with its own KFD node, memory image and queues, built at
  bring-up and torn down at exit; a count far above that produces a
  session that never finishes starting and cannot be stopped promptly.
  The widest physical AMD node is 8. Spread more GPUs over more nodes
  with `--num-nodes`.
* `--exec-mode functional|clocked` selects functional emulation
  (correct results, no timing model — the default) or clocked emulation.
* `-o`/`--option KEY=VALUE` sets an emulator option directly. Repeatable.
  Values that look like integers or booleans are stored as such. Keys are
  checked against the chosen backend's own schema and an unknown one is
  refused, naming what the backend does accept — or saying that it
  accepts none and pointing at the agent and topology instead, which is
  the case for `rocjitsu`. A typo'd option is otherwise indistinguishable
  from one the emulator ignored. `mirage emulators -l` lists the options
  each backend takes.
* `--plugin NAME` enables an execution plugin (e.g. `race`, `logging`)
  with its schema defaults. Repeatable, and merged with whatever the
  profile already enables. An unknown name is refused and the available
  ones listed, on the same reasoning as `-o`.
* `--config PATH` hands the backend an explicit emulator config file
  instead of synthesising one from the profile (this is the upstream
  `rocjitsu --config`). The path is made absolute, so it resolves
  regardless of the workload's working directory, and a file that does
  not exist is refused by name rather than handed to the backend.

  Because the file goes to the backend verbatim, the flags that would
  have gone into a synthesised config cannot also be honoured:
  `--gpus-per-node`, `--exec-mode`, `-o`/`--option` and `--plugin` are
  **refused** alongside `--config` rather than silently ignored. Put them
  in the config file instead. `--num-nodes` is not among them — it shapes
  the process grid mirage builds, not the emulator's configuration, so
  the two compose.
* `--daemon` runs the emulator out-of-process. This is the default; the
  flag exists for explicitness and for the rocjitsu drop-in alias.
  `--in-process` selects the opposite. In-process mode cannot share GPU
  memory between processes, so multi-GPU RCCL collectives need the
  daemon.

### Containers

* `--image IMAGE` runs every node inside a container built from that
  image, enabling containerisation for a profile that does not have it.
* `--mount HOST[:CONTAINER[:ro|rw]]` adds a bind mount to every node
  container. Repeatable. The host path is resolved to an absolute one
  before the run, so a relative path means what it meant in the shell you
  typed it in, and a host path that does not exist is refused by name.
  Mirage will not create it for you: the two engines disagree about what
  that means — docker silently makes a root-owned directory on your host,
  podman refuses to start the container — and neither is what you asked
  for. Creating it yourself makes the two behave identically.
* `--port HOST_PORT[:CONTAINER_PORT][/tcp|/udp]` publishes a container
  port on the host, like docker's `-p`. Repeatable. Port `0` is rejected:
  to an engine it means "pick one", and a published port nobody named is
  a port nothing can connect to.
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
  These beat both the emulator's own variables and anything you exported,
  which is the point of passing one. Two exceptions, described in full
  under [The workload's environment](#the-workloads-environment): the
  job's identity (`MIRAGE_*`, `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
  `MASTER_ADDR`, `MASTER_PORT`, `NCCL_HOSTID`) is set by mirage last and
  a `--env` naming one of those is accepted, ignored and warned about;
  and `LD_PRELOAD` is merged with the emulator's interposer rather than
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

### Output, and who gets the terminal

The shape of the job decides this. There is no flag, because a flag could
only ever ask for the behaviour that already applies.

**A single-process job** gets your terminal whole: its stdin, stdout and
stderr *are* yours. That is what makes `mirage run -- bash` an ordinary
interactive shell — prompt, echo, line editing, job control — and what
keeps redirection byte-exact with stdout and stderr distinct. Mirage is
not in the middle.

**A multi-process job** — more than one node, or `--nproc-per-node`
above one — has every rank's stdout and stderr piped through mirage,
which prints them a line at a time prefixed with the rank that wrote
them. This is what `docker compose up` does with a multi-container
project, and for the same reason:

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
  A stream that never sends a newline is flushed at 1 MiB rather than
  buffered without bound.
* stdout and stderr stay separate all the way out, so redirecting one of
  them still works.
* **No rank gets stdin.** One terminal cannot be shared between readers,
  and quietly giving it to rank 0 would send keystrokes somewhere you
  cannot see. For a terminal on a multi-node job, use
  `mirage exec --node N` — see below.

### The workload's environment

A workload inherits the environment of the terminal you ran mirage from.
Mirage's parent *is* your shell, so anything you exported there — an API
token, a `PYTHONPATH`, an `HTTP_PROXY`, a framework tuning variable — is
something you meant for the workload, and it arrives unchanged.

Three things are layered on top of it, in this order:

1. the emulator's own variables and its `LD_PRELOAD` interposer,
2. anything you passed with `--env KEY=VALUE`,
3. mirage's rank variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`,
   `MASTER_ADDR`, `MASTER_PORT`, `MIRAGE_RANK`, …).

Mirage's own go last, so a workload cannot break its own rendezvous by
exporting `RANK` — and neither can you, with `--env`. A rank that
disagrees with the grid does not fail; it deadlocks its own collectives,
waiting for a peer that was never told to expect it. So `--env RANK=…`,
and the rest of the identity set (`MIRAGE_*`, `LOCAL_RANK`, `WORLD_SIZE`,
`MASTER_ADDR`, `MASTER_PORT`, `NCCL_HOSTID`), is accepted, ignored, and
warned about on stderr naming the variable. Accepted rather than rejected
because a script that sets one is usually doing so harmlessly and should
not stop; warned rather than silent because it did not take effect.

Search lists are the other exception to "last wins":
`LD_PRELOAD`, `LD_LIBRARY_PATH` and `PYTHONPATH` are *prepended* to yours
rather than replacing them, because both sets of entries have to be
found — the emulator's interposer and libraries have to win the search,
and your imports and libraries have to survive it.

### `--clear-env-vars`

Starts the workload with an almost-empty environment instead:

```sh
mirage run --profile mi350x --clear-env-vars -- ./benchmark
```

Only what a process needs in order to be a process survives — `PATH`,
`HOME`, `USER`, `LANG`, `LC_ALL`, `TERM`, `TMPDIR`, `SHELL` — plus the
three layers above, which are passed explicitly and are therefore
unaffected. In particular the emulator's injection always survives; a
workload that lost it would run unemulated on whatever hardware is
actually present, and still exit 0.

Reach for it when a result must not depend on ambient state: a benchmark,
a reproduction, a CI job compared against a recorded baseline.

It has no effect on a containerised session. A container never inherits
the host's environment in the first place — the workload sees exactly
what mirage passes it with `-e`, and nothing else.

## `mirage exec`

```text
mirage exec [-s|--session ID] [-n|--node N] [--nproc-per-node N]
            [--clear-env-vars] [--env KEY=VALUE]... [--workdir DIR]
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
socket (`Request::Attach`, answered with a `SessionDescription`) and then
builds the process grid locally with the same
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

  The candidates are runs that *answer*, not socket files. A `SIGKILL`ed
  run leaves its socket behind — the expected leak `mirage cleanup` exists
  for — so a `kill -9` does not break auto-selection and the dead session
  is never offered: each socket is connected to, and one nothing answers
  on is removed on the way past. Naming a session explicitly skips that
  list entirely, so `--session <id>` on a session that has died says so,
  rather than saying it never existed.
* `-n`/`--node N` runs on that node **only**, instead of on every node
  in the session. This is how you get an interactive shell on a
  multi-node job: naming one node makes the exec a single-process job,
  and a single-process job gets the terminal.

  ```sh
  $ mirage exec --node 2 -- bash
  ```

  The process still believes it is that node — it gets rank 2's
  variables and the session's `WORLD_SIZE`, and points at the same
  rendezvous — so it is a shell *inside* the job rather than beside it.
  Naming a node the session does not have is an error, not a silent
  fallback to node 0.
* `--nproc-per-node N`, `--env KEY=VALUE`, `--clear-env-vars` and
  `--workdir DIR` mean exactly what they mean on `run`, including which
  jobs get the terminal and which get `[<rank>] ` prefixes.
* The exec's processes die with this command, and this command exits with
  the workload's exit code. Ctrl-C is forwarded, then escalates, exactly
  as in `run`.

### Borrowing keeps the session alive

That socket connection stays open for as long as the exec runs, and it is
a *lease*: while it is held, the run that owns the session will not tear
it down. A run whose own command finishes first waits, and says so:

```text
mirage: this command has finished, but 1 `mirage exec` borrower(s) are
mirage: still using session s-20260730-191636-1f4a-0
mirage: waiting for them; Ctrl-C to tear the session down anyway
```

This is not politeness. `exec` runs its workload in its own process, so
the run cannot see it, wait on it, or signal it — and teardown stops the
emulator daemon, removes the node containers and deletes the scratch
directory holding the emulator's config and socket. Without the lease, a
`mirage run -- sleep 5` beside a longer `mirage exec` pulled all of that
out from under a live job.

The wait is unbounded, because no timeout mirage could pick would be
right for somebody else's job. Ctrl-C is how you say you have waited
enough; the borrowers are then told the session is going away — their
connection is closed — and stop their own workloads rather than
discovering it as an I/O error.

The lease is the connection rather than a message, so it is released
however the borrower ends, including by crashing.

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

* The default emulator for new profiles is marked with `*` on its name.
* `-l` shows a detailed block per backend: the description, the support
  reason, the options and plugins the backend accepts, and — the useful
  part when something is missing — where its runtime library was found,
  or every path that was searched for it and the environment variables
  that would point at it.
* `--json` emits the full backend descriptors.

```sh
$ mirage emulators
NAME          INSTALLED  SUPPORTED  DESCRIPTION
rocjitsu*     yes        yes        ROCm just-in-time GPU emulator (cycle-accurate or functional)
rocjitsu-dbt  yes        no         rocjitsu dynamic binary translation: run a GPU's code objects on a different physical GPU by translating them at load time (e.g. gfx1250 on gfx950)

* = default emulator for new profiles
```

That is a default build; `hotswap` appears too in a build that enables its
feature (see [`building.md`](building.md)).

**Installed** is about this machine's files, **supported** about its
hardware, and they are independent. `rocjitsu` emulates in software and is
supported anywhere — which is the whole point, and why no physical GPU is
needed. `rocjitsu-dbt` translates onto a real GPU instead, so it is
installed here but unsupported, and says why. `-l` gives the reason in
each case:

```sh
$ mirage emulators -l
rocjitsu (default)
  ROCm just-in-time GPU emulator (cycle-accurate or functional)
  installed: yes
  runtime:   /path/to/rocjitsu/build/librocjitsu.so
  supported: yes  (software emulator; no special hardware required)
  options:   (none)
  plugins:   logging, race
```

For a backend that is *not* installed, `runtime:` becomes `not found`,
followed by a `searched:` list and a `set:` list naming the environment
variables that would resolve it. That is the first thing to read when a
run fails to find an emulator — see [`building.md`](building.md).

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
* Neither `create` nor `import` will overwrite a profile that is already
  there. These files are the only copy, and a profile someone tuned is
  not recoverable once it has been written over — so a name that is taken
  is an error naming the path, and `mirage profile delete <name>` is how
  you say you meant it. An untouched builtin is the one exception: mirage
  wrote it, so replacing it destroys nothing.

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

`agent import` follows the same no-overwrite rule as `profile`: an agent
that is already there and is not an untouched builtin has to be deleted
before it can be replaced.

## Configuration documents reject what they do not understand

Profiles, agents and topologies are parsed strictly. A key the schema does
not know is a parse error naming it —

```text
error: unknown field `emulaotr`, expected one of `name`, `description`, `emulator`, `containerize` at line 1 column 193
```

— rather than a field quietly dropped on the way in. The two outcomes are
easy to confuse and are not remotely equivalent: a profile with a typo'd
key used to load, report success, and then not be what the file plainly
said it was, which surfaces much later as an emulation that ran with the
wrong settings. Failing at parse time makes the file and the behaviour the
same thing again. It applies to every document mirage reads — `import` on
all three kinds, and the profiles read by `run` and `exec`.

## `mirage state`

```text
mirage state builtins                 # (re)write builtin agents/topologies/profiles
mirage state purge [-f|--force] [--all]
```

mirage writes any *missing* builtin agent, topology and profile on every
command, so the shipped set is always there. `state builtins` additionally
refreshes the ones that exist, which is what you want after upgrading —
but it refreshes only the ones you have not touched. A builtin whose file
differs from the shipped version is *your* document that happens to share
a name, and rewriting it would discard your edits with no way back, so
those are refused by name and everything else is still refreshed. Take the
shipped version by deleting the file and running the command again.

Deleting follows from the same fact, in both directions:

* Deleting an **untouched** builtin is refused. Mirage writes any missing
  builtin back on the next command, so the file would reappear
  immediately — reporting success for a delete that does not stay done
  was the defect, not the rewriting. Edit the file instead.
* Deleting an **edited** builtin is allowed, and is the documented way to
  reset one: it really does remove your version, and the shipped document
  takes its place. That is the round trip — edit to customise, delete to
  revert.

`state purge` is the blunt tool:

* It does everything `mirage cleanup` does, below, and then removes the
  runtime directory. The config directory (profiles, topologies, agents)
  is left alone unless `--all` is given.
* It refuses while any `mirage run` is live, and tells you which.
  Killing someone else's foreground command from a state-cleanup
  subcommand would be a surprise; stop it with Ctrl-C in its own terminal
  instead. Each run cleans up after itself when it exits. Liveness here
  means a socket that answers, not a socket file, so a `SIGKILL`ed run
  does not block the purge that exists to clean up after it. The check is
  taken again immediately before the destructive step, because a run
  started in between owns a socket and a scratch directory under the very
  directory about to be deleted.
* It reports what it did, and its **exit code is the verdict**: non-zero
  if anything it set out to remove is still there, whether that was a
  directory or a container. Under `--json` the whole thing is exactly one
  document — `purged` is the verdict, `removed` the directories, `failures`
  the reasons, and the reclamation nested under `reclaimed` — so it can be
  piped straight into `jq`. Declining the confirmation prompt is also a
  document (`"declined": true`), because a script has to be able to read
  that answer rather than infer it from empty output.

## `mirage cleanup`

```text
mirage cleanup [--dry-run]
```

Reclaims what a run that died abruptly left behind. In normal use there
is nothing to do: a run owns its session and tears it down on every exit
path. `kill -9`, the OOM killer and a machine losing power run no code of
mirage's at all, and then three things survive it:

* its node containers and per-session network,
* its workload processes, reparented to init,
* its session scratch directory.

All three are found by a mark on the thing itself, because the session's
own record of them died with the run: containers and networks carry the
`mirage.owner`, `mirage.session` and `mirage.runtime` labels, and every
workload carries `MIRAGE_SESSION` and `MIRAGE_RUNTIME` in its
environment. Anything without those marks is never a candidate, so this
is safe on an engine shared with other work.

Safe to run at any time. A session whose `mirage run` still answers on
its socket is not an orphan and is skipped entirely — unlike `state
purge`, which refuses outright while any run is live. Use `--dry-run` to
see what would be removed without removing it.

Everything found is named individually, and each line says what kind of
thing it is, which session it came from, and — for containers and
networks — which engine is holding it, because that is the engine you
would have to type the id at yourself:

```sh
$ mirage cleanup --dry-run
would kill stranded process 41207 (session s-20260730-191636-1f4a-0)
would remove container mirage-s-20260730-191636-1f4a-0-node-0 of session s-20260730-191636-1f4a-0 (podman)
would remove network mirage-s-20260730-191636-1f4a-0 of session s-20260730-191636-1f4a-0 (podman)
would remove scratch directory /run/user/1000/mirage/session/s-20260730-191636-1f4a-0
```

The verbs become `killed` and `removed` without `--dry-run`; the list is
otherwise the same, and deliberately so — what is reported and what is
removed come from one scan, or a `--dry-run` is not a preview of
anything. Removal follows teardown's order: processes stop before the
containers they run in, and the scratch directory every node container
bind-mounts goes last. With nothing to do it prints `nothing to clean up`.

Anything mirage meant to remove and could not is reported on **stderr**,
after the list so a script reading the list does not have to filter it
out, and the exit code is non-zero. Zero would say the machine is clean.

Every container engine installed here is consulted, not just the first one
autodetection finds. Detection prefers podman, so on a host with both, a
session created with `--container-provider docker` left containers this
command never even asked docker about — and then reported success, which
is the one answer that stops you looking. Setting
`MIRAGE_CONTAINER_PROVIDER` names one engine deliberately and is then the
only one consulted.

### Scope: one runtime directory

Cleanup reclaims only what its own runtime directory created. That is
not a limitation but the safety property: the run sockets under
`MIRAGE_RUNTIME` are the entire registry of what is live there, so a
cleanup has no way to see the live sessions of a mirage running under a
different one. Without the `mirage.runtime` mark it would read those
healthy jobs as a crashed run's leftovers and `SIGKILL` them — which is
what makes two concurrent mirages (a CI job beside an interactive
session, a test suite beside your own work) safe on one machine.

A resource that records no runtime directory at all cannot be attributed
to anybody, and is skipped for the same reason. The one thing this
leaves behind is work started by a mirage older than the mark: it is
never reclaimed by this one and has to be removed by hand (`kill`, or
`podman rm -f`). That is the deliberate trade — an orphan you can see and
remove, rather than a running job destroyed by a tidy-up.

One consequence of tagging by environment is worth knowing: the tag is
inherited, so anything started from an interactive `mirage exec -- bash`
belongs to that session as far as cleanup is concerned. That is the
intended reading — it is part of the session's process tree — but it does
mean a build kicked off inside one is reclaimed with it.

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
| `MIRAGE_RUNTIME`             | Override the runtime dir — run sockets and emulator scratch (else `$XDG_RUNTIME_DIR/mirage`). Also stamped on every workload; see below. |
| `XDG_CONFIG_HOME` / `XDG_RUNTIME_DIR` | Standard XDG base directories used when the `MIRAGE_*` overrides are unset. |

mirage sets these *in* every workload process, on every rank:
`MIRAGE_RANK` (the node's rank, 0 = head), `MIRAGE_HEAD_ADDR` and
`MIRAGE_HEAD_PORT`, plus their `torch.distributed` aliases `RANK`,
`LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`, and
`NCCL_HOSTID` so RCCL tells the ranks apart. A set of `NCCL_*` tuning
variables is layered in with the emulator's own, pinning collectives to
the loopback path an emulated machine actually has; those are ordinary
emulator variables and `--env` overrides them, unlike the identity set
above.

It also sets `MIRAGE_SESSION` and `MIRAGE_RUNTIME`, which together say
which run a process belongs to. Both are resolved by mirage and replace
whatever the caller exported, so a nested `mirage` inside a session sees
the state directory of the run it is inside — and `mirage cleanup` can
tell that run's processes from those of a mirage using a different
runtime directory. In a containerised session both name in-container
mounts rather than host paths, which do not exist in there:
`MIRAGE_RUNTIME` is the session scratch at `/mnt/mirage/runtime` and
`MIRAGE_CONFIG` the config directory at `/mnt/mirage/config`. Host-side,
such a session is still attributed correctly — the container carries the
`mirage.runtime` label and the engine client process carries the host
path.

Finding an emulator's runtime library has its own set: `ROCJITSU_LIB` and
`ROCJITSU_HOOKS_LIB` name a library file outright, `LD_LIBRARY_PATH`,
`ROCM_HOME` and `ROCM_PATH` are searched, and the ROCm SDK install root
reported by `rocm-sdk path --root` is consulted after them. `mirage
emulators -l` prints which one was used, or every path that was tried; see
[`building.md`](building.md) for the full order.

## Exit codes

| Code | Meaning                                                              |
| ---- | ------------------------------------------------------------------- |
| 0    | Success.                                                            |
| 1    | A mirage-level error (bad arguments, resource not found, a session that failed to come up, no run serving the named session, …). For `cleanup` and `state purge`, also: something they set out to remove is still there. |
| 2    | Argument parse error (clap), including a flag combination that is refused outright — `--config` with `--gpus-per-node`, a count of `0`. |
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
  "profiles": "/home/me/.config/mirage/profile",
  "runs": "/run/user/1000/mirage/run",
  "runtime": "/run/user/1000/mirage",
  "sessions": "/run/user/1000/mirage/session"
}
```

`run` and `exec` have no JSON form: their output is the workload's own.
