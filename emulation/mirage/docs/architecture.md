# Architecture

mirage is a single Cargo workspace that builds one unified `mirage`
executable: the CLI, the session runtime, and a set of pluggable emulator
backends, all in one binary. There is no second role to dispatch to.
Every subcommand — `run`, `exec`, `profile`, `topology`, `agent`,
`emulators`, `state`, `paths`, `about` — happens in the process you
invoked, and `mirage [--config c.json] -- ./app` is a `rocjitsu`-shaped
alias for `run`.

The one distinction worth drawing is what a command *owns*:

* **Configuration commands** — `profile`, `topology`, `agent`, `state`,
  `paths`, `emulators`. Pure filesystem work against the config store. No
  session, no processes, nothing running.
* **Execution commands** — `run` and `exec`. These own processes.
  `mirage run` owns a whole session; `mirage exec` owns the processes it
  starts inside somebody else's.

The word "daemon" survives in mirage only as the *emulator's* daemon
(`--daemon` / `--in-process`, the out-of-process rocjitsu runtime). Mirage
itself has none.

## Crates

| Crate               | Role                                                                 |
| ------------------- | -------------------------------------------------------------------- |
| `mirage` (root)     | The unified binary: parses argv, owns a Tokio runtime, dispatches.   |
| `mirage_ctl`        | The command line: the `CtlCmd` subcommand tree, `dispatch`, and `run`/`exec` themselves. |
| `mirage_core`       | Shared types (`ProfileDef`, `SessionDef`, `ExecDef`, …), XDG path resolution for configuration, atomic file I/O and the config store built on it, the `run`↔`exec` wire protocol, and the emulator/registry traits. |
| `mirage_supervisor` | The engine a `mirage run` is built out of: `Run`, `Session`, `Exec`, the shared spec builder, process spawn/supervise/reap, and the socket a run serves. |
| `mirage_container`  | Container provider abstraction (podman/docker) for containerised sessions. |
| `mirage_builtin`    | Embedded builtin agents, topologies, and profiles, plus their unpackers. |
| `mirage_rocjitsu`   | The `rocjitsu` (and `rocjitsu-dbt`) backend.                         |
| `mirage_hotswap`    | The `hotswap` load-time ISA-rewriting backend.                       |
| `rocjitsu_sys`      | FFI bindings to `librocjitsu.so`, plus safe RAII wrappers over them.  |
| `mirage_sys`        | The post-fork syscalls that cannot be expressed safely: today `PR_SET_PDEATHSIG`, so a workload dies with a `SIGKILL`ed supervisor. |

Emulator backends are **link-only** dependencies: each registers itself
into the emulator registry via [`inventory`] at link time. The binary
never names a backend directly, so enabling or disabling a backend's Cargo
feature simply adds or removes it from `mirage emulators`. Those features
are the only optional things left in the build: there is no `daemon`,
`webui` or `http` feature, no `MIRAGE_ENABLE_DAEMON` or
`MIRAGE_ENABLE_WEBUI` cmake option, and no Node.js toolchain.

[`inventory`]: https://docs.rs/inventory

## Data flow

```text
 terminal A                                terminal B
┌────────────────────────────────────┐    ┌────────────────────────────────┐
│ mirage run -- ./app                │    │ mirage exec -- bash            │
│  • parses argv (clap)              │    │  • finds the sole live run,    │
│  • owns the Tokio runtime          │    │    or takes -s <session>       │
│  • Run::start → the session lives  │    │                                │
│    in *this* process               │    │                                │
│  • binds the run socket first,     │◄───┤  Request::Describe             │
│    answers once session is healthy │───►│  SessionDescription (JSON)     │
│  • build_specs → spawn → wait      │    │  • build_specs, locally        │
│  • destroy() on every exit path    │    │  • spawn → wait                │
└──────────────┬─────────────────────┘    └──────────────┬─────────────────┘
               │ tokio::process                          │ tokio::process
               │ own process group, always waited on     │
               ▼                                         ▼
  workload processes on A's terminal        workload processes on B's
  (or `provider exec` into a node             terminal, same grid, same
   container)                                 shared builder
```

The socket is one connection, one request, one response, then closed. It
lives at `$XDG_RUNTIME_DIR/mirage/run/<session>.sock` and is unlinked when
the run exits.

```text
┌──────────────────────────────────────────────────────────────────┐
│ Filesystem: configuration only                                   │
│  profiles, agents, topologies  (config dir)                      │
│  per-session emulator scratch  (runtime dir)                     │
│  one socket per live run       (runtime dir)                     │
└──────────────────────────────────────────────────────────────────┘
```

Configuration is read from disk by every command. Session state is never
on disk. See [`state-layout.md`](state-layout.md) for what remains, and
`mirage paths` for where it is on your machine.

## Why the run owns its session

A session exists exactly while the `mirage run` that created it is alive.
It cannot be started on its own, listed, attached to, or left behind —
there is no `mirage session` subcommand tree, because there is nothing for
it to name that outlives its creator.

That is the third answer to the same question, and the first two are the
argument for it.

**First there were files.** A detached `mirage host` per session, with the
session directory as the channel: an exec was a `def.json` the host polled
for, output was a file to tail, stdin was a FIFO, completion was a
`status.json`. The appeal was inspectability — `ls`, `cat`, `jq`. The cost
was that a file cannot answer the question a control plane needs answered:
*is the thing that wrote this still alive?* The host was detached, so it
was reparented to init; nothing owned it and nothing reaped it, so its
children became zombies. Liveness had to be inferred from a heartbeat
file's timestamp through a `ready` → `stalled` → `dead` ladder that was an
elaborate way of not knowing. Teardown signalled pids read from files and
then deleted the directory, so a signal that did not land became
invisible: the state was gone but the process was not.

**Then there was a daemon.** One long-lived supervisor owning every
session, reached over a control socket at a well-known path, auto-started
on demand. It fixed the leaks — one owner, one reaper — and bought a
different set of problems, all of them consequences of outliving the
commands it served:

* Sessions lived in a map, so they had to be looked up by id and could be
  missing; creation raced shutdown; a shutdown flag had to be re-checked
  under a write lock.
* Nothing ever exited, so everything needed a cap: finished execs were
  evicted, output replay was bounded, memory was a budget rather than a
  consequence of scope.
* The daemon's terminal was not the user's terminal, which is where the
  pseudo-terminals, the output forwarding and the stdin relay came from
  (see below).
* The auto-start path was a whole second protocol: spawn a daemon,
  hand it a log directive, wait for its socket to appear, retry.

**Now the command is the owner.** `mirage run` brings the session up in
its own address space, serves a socket so other terminals can find it,
runs the workload, and tears everything down on the way out — including on
the error paths, because a bring-up that half-succeeded still has
containers to remove. What falls out:

* **Liveness is a fact.** "Is this session alive?" is "is that process
  alive?". No heartbeat, no staleness ladder, no map lookup that can miss.
* **Teardown is closed-loop.** `destroy` terminates every process group,
  waits for every child, stops the emulator daemon, removes the containers
  and the network, and returns only when all of that has happened.
* **Nothing needs a cap.** A process that exits frees its memory by
  exiting.
* **There is no session-manager concurrency.** There is one session, it is
  right here, and it cannot outlive the value holding it.

The cost is real and deliberate: you cannot close the terminal. A job you
want to survive your shell is a job for `nohup`, `tmux` or your batch
scheduler — tools that already solve detachment, and solve it for
everything, not just for mirage.

The remaining failure mode is a run that is `SIGKILL`ed: it takes its
record of every container with it. `--rm` covers the common case; `mirage
state purge` reclaims what an interrupted engine left behind, matching on
mirage's own label so a shared engine is safe. It refuses while any run is
live, because killing somebody else's foreground command from a cleanup
subcommand would be a surprise.

## Why `exec` spawns locally

`mirage exec` runs a command inside a session another process owns, and it
starts that command *itself*, as its own child, in its own terminal. It
does not ask the run process to do it.

The reason is terminals. A child inherits the standard streams of whoever
forked it. A process spawned by the run process would talk to the run's
terminal — the one already showing the run's workload — not to the
terminal the user typed `mirage exec` into. Having the client spawn is the
whole reason `mirage exec -- bash` is an interactive shell *in the window
you ran it from*, with no pseudo-terminal, no output forwarding and no
stdin relay.

That leaves the run process with exactly one thing to do for an exec:
describe the session. The protocol is one request (`Request::Describe`)
and one response (`SessionDescription`) — which containers exist, what
environment the emulator needs, where the rendezvous is. The description
is *data*, not a handle, so the client needs no further round trips.
Everything else — spawning, signalling, reaping, printing — belongs to the
process that owns the terminal it is happening in. The previous protocol
carried attach streams, stdin frames, terminal resizes, exec lifecycle and
a daemon handshake; none of it survives the change of ownership.

Both sides then build their process grid with the same
`mirage_supervisor::build_specs`, from the same description. That is not
tidiness. If `run` and `exec` built specs separately they would drift — a
slightly different `LD_PRELOAD`, a different rendezvous, the wrong workdir
inside a container — and the symptom would be a workload that runs
correctly one way and mysteriously fails the other. Sharing the builder
makes that class of bug unrepresentable.

`--session` is a flag rather than a positional because everything after
`--` belongs to the command: with both positional, `mirage exec -- bash`
could equally mean "session `bash`". It can usually be omitted entirely —
with exactly one run live, mirage picks it; when the guess would be
ambiguous the error lists the candidates rather than choosing one.

## Why there is no pseudo-terminal

A workload gets the *caller's* streams, not ones mirage manufactured.
Whoever spawns it — `mirage run`, or a `mirage exec` in another terminal —
is the process the user is sitting in front of, so inheriting its file
descriptors puts the workload on the user's real terminal.

The obvious alternative is what mirage used to do, and it was worse in
every direction. A daemon owned every workload, so the workload's terminal
could not be the user's; it had to be a pty the daemon allocated, with
output shipped back over a socket and keystrokes shipped forward. The
costs were not theoretical:

* A pty has one stream, so stdout and stderr were merged. Redirecting one
  of them stopped working.
* The client had to put the user's terminal into raw mode and forward
  `SIGWINCH` by hand.
* `pre_exec`-based session-leader setup (`setsid` + `TIOCSCTTY`) was the
  only `unsafe` in the workspace outside the FFI layer.

Inheriting deletes all of it. An interactive `bash` works because its
stdin *is* the terminal, not because mirage emulated one. `mirage attach`,
`mirage logs` and the `--tty` flag are gone with the machinery that needed
them: there is no stream to attach to and no buffer to tail, because the
bytes never pass through mirage at all.

The one exception is a job with more than one process, and it exists for
one reason: several ranks writing to one terminal are unreadable without
labels. Mirage goes back in the middle so it can prefix every line with
the rank that produced it — the bargain `docker compose up` makes. The
price is stdin, which no rank gets.

Which side of that a job lands on is decided by its shape, not by a flag.
There was a `--capture-all` flag; it was removed once the rule became
automatic, because it could then only ask for the behaviour that already
applied.

* **One process** — the interactive case — gets the terminal whole.
* **More than one** is captured and labelled, and nobody gets stdin. One
  terminal cannot be shared between readers, and quietly handing it to
  rank 0 sends keystrokes somewhere the user cannot see.

That leaves a real gap: there is no way to be interactive *with* a
multi-node job. `mirage exec --node N` fills it. Naming one node makes
the exec a single-process job, so it takes the first branch and gets the
terminal, while still receiving that node's rank variables and the
session's `WORLD_SIZE` — a shell inside the job rather than beside it.

A process that owns the terminal is also made the terminal's foreground
process group, because a background group that reads its controlling
terminal is stopped with `SIGTTIN`. That handoff is taken *only* for a
single-process job: the tty driver delivers Ctrl-C to the foreground
group alone, so doing it on a grid would kill one rank and leave mirage
unable to hear the interrupt at all.

Capturing is line-oriented, not chunk-oriented. A chunk is whatever one
`read` returned, so a single line can arrive in three pieces and three
lines in one; prefixing per chunk produces labels mid-line and lines with
no label. Each rank's stream is buffered until it holds a complete line,
and the tail — a prompt, a progress bar — is flushed when the stream ends
so nothing is silently dropped. stdout and stderr stay separate all the
way out, so redirecting one of them still works.

## Why containers are foreground and `--rm`

A containerised session launches one container per node with `podman run`
/ `docker run`, **not detached** and with `--rm`. The provider client is a
child process mirage owns.

Detaching a container is the same mistake as detaching a session, one
level down: it creates a thing that exists with nothing responsible for
it. In the foreground, the container's lifetime is bounded by the `mirage
run` that asked for it — killing the client stops the container — and
`--rm` closes the other half by deleting it the moment it stops, however
it stops. Between the two, a run that crashed, was `SIGKILL`ed, or simply
had its terminal closed leaves nothing behind.

Two consequences are worth knowing:

* A detached `run -d` returned only once the container existed, so the
  next `exec` was guaranteed a target. A foreground client returns
  immediately and the container comes up behind it, so bring-up
  re-establishes that guarantee explicitly by polling for "running" —
  otherwise the first exec races bring-up and fails with "no such
  container".
* The client's own streams go to `/dev/null`. The container's foreground
  process is an idle placeholder (`sleep infinity`); workloads arrive
  later via `provider exec`, so it has nothing to say, and letting it
  write to mirage's terminal would interleave provider chatter with the
  output the user asked for.

`provider exec` passes `-i` and deliberately **no** `-t`. Mirage allocates
no pseudo-terminal anywhere, and asking the provider for one would merge
stderr into stdout and break redirection — the same trade rejected above.

Signalling needs one extra step in a container. `podman exec` puts the
workload in the container's PID namespace and does not forward signals to
it, so killing the client would leave the workload running, invisible and
still holding the emulated device. The in-container command is therefore
wrapped in `sh -c 'echo $$ > <pidfile>; exec "$0" "$@"'`: the shell records
its own pid and then `exec`s the real program into that same pid. The file
lands in the session scratch directory, already bind-mounted into every
node container, so the supervisor reads it straight off the host
filesystem and signals back through the provider.

## Concurrency model

Every mirage process runs a single Tokio runtime created in `main`.

* **A session** is one value, owned by the `Run` that created it. Its
  mutable interior is a `std::sync::Mutex` rather than an async one:
  every critical section is a field update or a small map operation and
  none of them awaits, so an async lock would only add suspension points
  to operations that never block — and would make the synchronous kill
  backstop impossible.
* **Health** is a `tokio::sync::watch` channel, so waiters are woken
  rather than polling. It is published with `send_replace`, not `send`:
  `send` fails and *leaves the value unchanged* when no receiver exists,
  which would strand a session at `starting` if it became ready before
  anyone asked. Exec completion is published the same way.
* **Bring-up is backgrounded, teardown is not.** `Run::start` returns as
  soon as the session exists so the caller can watch progress through
  health rather than block for however long an image pull takes.
  `destroy` returns only once every process is reaped, every container
  removed and the scratch directory deleted, because a caller that has
  been told a session is gone must be able to rely on it.
* **An exec** owns its process grid through one supervising task per
  process. Each races `Child::wait` against cancellation; both arms end
  with the child reaped. Children are also spawned with
  `kill_on_drop(true)` as a backstop, so an abruptly cancelled supervising
  task kills its child rather than orphaning it.
* **Termination escalates and then confirms.** `SIGTERM` to the process
  group, a bounded grace period, `SIGKILL`, and a return only once the
  child has actually been reaped. `SIGKILL` cannot be caught, so the
  second stage always ends; the confirmation is what makes "the session is
  destroyed" a statement about the process table rather than about intent.
* **Ctrl-C reaches mirage, not the workload.** Children lead their own
  process groups, so the terminal's foreground group is the mirage process
  alone. It forwards the signal deliberately and then falls through to the
  normal wait — which is what gives a workload a chance to clean up, and
  what makes teardown run instead of being skipped by an abrupt exit. A
  second interrupt means the user is no longer waiting. `SIGTERM` is
  handled identically, because a CI runner cancelling a job sends that
  one.
* **The run socket is served with `select!`** against the workload rather
  than from a spawned task, so it stops when the workload does and there
  is no second thing to cancel. It is bound *before* bring-up starts, so
  a run is visible to `mirage state purge` from its first instant rather
  than only once its image has finished pulling — purge would otherwise
  reclaim the containers of a run it could not see. What waits for health
  is the *answer*, not the bind: `Session::describe` refuses until the
  session is healthy, because a client handed a description with no
  containers and no emulator environment in it would happily start a
  workload straight onto the real host.
* **Stale sockets are tested, not locked.** A socket file outlives a
  `SIGKILL`ed process, so its existence proves nothing. Binding simply
  tries to connect to whatever is already at the path: if something
  answers, a run owns this id and we refuse; if nothing does, the file is
  a corpse and is removed. The test is direct, needs no second file, and
  cannot be fooled by a stale lock.
* **Blocking work** — container providers, emulator daemon startup and
  shutdown — runs on `spawn_blocking`, never inline on the runtime.

## Safety

Every crate is `#![forbid(unsafe_code)]` through the workspace lint table,
with exactly two exceptions, each of which carries an equivalent lint
table of its own with `unsafe` permitted:

* `rocjitsu_sys`, the FFI layer to `librocjitsu.so`. Safe RAII wrappers
  over the C API (e.g. the emulator daemon handle) live in that crate
  too, so the `unsafe` and the invariants justifying it sit in the same
  file and are reviewed together.
* `mirage_sys`, which holds the post-fork operations that cannot be
  expressed safely — today just `PR_SET_PDEATHSIG`, the kernel-side
  backstop that kills a workload whose supervisor was `SIGKILL`ed.

Mirage previously hand-rolled `unsafe` in three places outside the FFI
layer. Each is now either unnecessary or delegated to a crate built for
it:

| Was | Now |
| --- | --- |
| `pre_exec` calling `setsid` to put a child in its own process group | the safe `Command::process_group(0)` |
| `pre_exec` calling `setsid` + `TIOCSCTTY` to attach a pty | deleted with the pty: children inherit the caller's real terminal |
| hand-written `termios` and `TIOCGWINSZ` in the attach path | deleted with `attach`: nothing puts the terminal in raw mode or forwards `SIGWINCH` |

The last two are the more interesting entries. They were not made safe;
the feature that required them was removed, and the `unsafe` went with it.
The rule is enforced, not aspirational: `forbid` cannot be relaxed by a
later `allow`, so an `unsafe` block anywhere outside those two crates
fails the build.

## Adding an emulator backend

1. Create a crate implementing `mirage_core::emulator::EmulatorBackend`.
   Its methods receive a `SessionContext` carrying the session's resolved
   profile and a scratch directory to materialise runtime assets in.
2. Register it into the registry with an `inventory::submit!` in its
   `lib.rs`.
3. Add it as an optional, feature-gated dependency of the root `mirage`
   crate and reference it with `extern crate … as _` so the linker keeps
   the registration.

No code in the binary needs to name the backend; it appears automatically
in `mirage emulators` and becomes selectable via `--emulator`.

## Testing strategy

* **Unit tests** live in each crate's `*::tests` modules and cover id
  validation, path resolution, atomic file I/O, option parsing, profile
  overrides, the wire protocol's round-tripping, the container argv
  builders, spec building, labelled output framing, and process
  spawn/kill/reap semantics.
* **Supervisor integration tests** (`supervisor/tests/run.rs`) drive the
  real `Run`, the real control socket and the real process supervisor
  against a stub emulator backend that registers itself the same way
  `rocjitsu` does — so the whole path from `Run::start` to a reaped
  process is exercised on any machine, GPU or not. The properties under
  test are ownership properties, not emulation ones: that a session cannot
  outlive the process holding it, and that an exec built from a
  description in one process behaves identically to one built in another.
* **End-to-end tests** under `tests/` drive the actual binary through the
  CLI, each under a private XDG root so they are independent and
  parallel-safe. What they assert is the observable contract: a run's
  streams and exit code, the socket it serves while it lives, `mirage
  exec` borrowing that session from another terminal, and the containerised
  lifecycle against a mock provider script (`container_e2e.rs`) that needs
  no real container runtime. `matrix_e2e.rs` walks the cross product in
  `tests/matrix.md`.
* **Strain tests** (`tests/strain.rs`) manufacture the boundaries where
  cleanup bugs actually live — a Ctrl-C that arrives while a process is
  still forking, an exec client that dies mid-command, a run killed
  outright with work in flight — and then check the one thing that cannot
  be faked: the operating system's process table. Every assertion is
  external. Not "the supervisor believes the session is gone" but "no
  process with this marker exists" and "no zombie with this parent
  exists". Internal bookkeeping agreeing with itself is exactly the
  failure mode the old designs had.
