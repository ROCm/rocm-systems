# Architecture

mirage is a single Cargo workspace that builds one unified `mirage`
executable. The binary is both the CLI and the supervisor daemon, plus a
set of pluggable emulator backends. Which role runs is decided by the
subcommand:

* `mirage <ctl-command>` — every user-facing verb (`profile`, `topology`,
  `agent`, `emulators`, `session`, `exec`, `run`, `attach`, `logs`,
  `state`, `paths`, `about`). These reach their sessions by talking to the
  daemon over a Unix socket, starting one if none is running.
* `mirage daemon` — the supervisor: owns every session, serves the control
  socket, and (with `--features webui`) the HTTP dashboard.

## Crates

| Crate               | Role                                                                 |
| ------------------- | -------------------------------------------------------------------- |
| `mirage` (root)     | The unified binary: parses argv, owns a Tokio runtime, dispatches.   |
| `mirage_ctl`        | The control plane: the `CtlCmd` subcommand tree, `dispatch`, and the `RpcClient` that talks to the daemon. |
| `mirage_core`       | Shared types (`ProfileDef`, `SessionDef`, `ExecDef`, …), XDG path resolution for config, atomic state I/O, the async `MirageCtl` trait, the client/daemon wire protocol, and the emulator/registry traits. |
| `mirage_supervisor` | The engine: owns sessions, execs and processes in memory; spawns, supervises and reaps workloads. |
| `mirage_daemon`     | The daemon: hosts a `SessionManager`, serves the Unix-socket control plane, and optionally the HTTP/WebSocket API. |
| `mirage_container`  | Container provider abstraction (podman/docker) for containerised sessions. |
| `mirage_builtin`    | Embedded builtin agents, topologies, and profiles, plus their unpackers. |
| `mirage_dashboard`  | The embedded React/Vite single-page app served by `mirage_daemon`.   |
| `mirage_noop`       | The `noop` backend: runs commands directly with no emulation.        |
| `mirage_rocjitsu`   | The `rocjitsu` (and `rocjitsu-dbt`) backend.                         |
| `mirage_hotswap`    | The `hotswap` load-time ISA-rewriting backend.                       |
| `rocjitsu_sys`      | FFI bindings to `librocjitsu.so`, plus safe RAII wrappers over them.  |

Emulator backends are **link-only** dependencies: each registers itself
into the emulator registry via [`inventory`] at link time. The binary
never names a backend directly, so enabling or disabling a backend's Cargo
feature simply adds or removes it from `mirage emulators`.

[`inventory`]: https://docs.rs/inventory

## Data flow

```text
┌──────────────────────────────────────────────────────────────────┐
│ mirage <ctl-command>            (mirage_ctl::dispatch)           │
│  • parses argv (clap)                                            │
│  • owns the Tokio runtime in the CLI process                     │
│  • renders text / JSON                                           │
│  • drives a MirageCtl implementation (RpcClient)                 │
└───────────────┬──────────────────────────────────────────────────┘
                │ length-delimited JSON over a Unix socket
                │ (one connection per operation; attach is duplex)
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ mirage daemon                   (mirage_daemon + mirage_supervisor)│
│  • one per user; auto-started, exits when idle                   │
│  • SessionManager: every session, exec and process, in memory    │
│  • the same manager also backs the HTTP API                      │
└───────────────┬──────────────────────────────────────────────────┘
                │ tokio::process — piped stdio, own process group,
                │ always waited on
                ▼
          workload processes (or `provider exec` into a container)

┌──────────────────────────────────────────────────────────────────┐
│ Filesystem: configuration only                                   │
│  profiles, agents, topologies  (config dir)                      │
│  per-session emulator scratch  (runtime dir)                     │
└──────────────────────────────────────────────────────────────────┘
```

Both the CLI and the daemon read configuration from disk. Session state is
never on disk. See [`state-layout.md`](state-layout.md) for what remains
and [`daemon.md`](daemon.md) for the process model.

## Why a daemon rather than files

Mirage previously ran a detached `mirage host` process per session and had
the CLI and that host communicate through the session directory: an exec
was a `def.json` the host polled for, output was a file to tail, stdin was
a FIFO, completion was a `status.json`.

The appeal was inspectability — `ls`, `cat`, `jq`. The cost was that a
file cannot answer the question a control plane needs answered: *is the
thing that wrote this still alive?*

* The host was detached, so it was reparented to init. Nothing owned it
  and nothing reaped it, so its children became zombies.
* Liveness had to be inferred from a heartbeat file's timestamp, with a
  `ready` → `stalled` → `dead` ladder that was an elaborate way of not
  knowing.
* Teardown signalled pids read from files and then deleted the directory,
  so a signal that did not land became invisible: the state was gone but
  the process was not.

With one owner:

* **Liveness is a fact.** A session exists exactly as long as the daemon
  holds it. No heartbeat, no staleness ladder.
* **Cleanup is closed-loop.** `session destroy` terminates each process
  group, waits for each child, and returns only when the process table
  agrees.
* **Shutdown is total.** Stopping the daemon stops everything it started.
* **The surfaces agree.** The CLI and the dashboard read the same
  in-memory manager, so they cannot disagree about what exists.

The trade is that inspection goes through `mirage` (or the HTTP API)
rather than `cat`. In exchange, what you are told is true.

## Concurrency model

The CLI process runs a single Tokio runtime created in `main`; the daemon
runs a multi-threaded one.

* **Sessions** are values in a map behind a `std::sync::RwLock`. Every
  critical section is a lookup or an insert and none of them awaits, so an
  async lock would only add suspension points to operations that never
  block.
* **Health** is a `tokio::sync::watch` channel, so waiters are woken
  rather than polling. It is published with `send_replace`, not `send`:
  `send` fails and *leaves the value unchanged* when no receiver exists,
  which would strand a session at `starting` if it became ready before
  anyone asked.
* **Execs** own their process grid through one supervising task per
  process. Each races `Child::wait` against a `CancellationToken`; both
  arms end with the child reaped.
* **Output** flows through an `OutputHub`: a bounded replay buffer plus a
  broadcast channel, updated under one lock so a subscriber can neither
  miss a packet nor see it twice. Attaching to a finished exec replays its
  whole output and its exit code.
* **Blocking work** — container providers, emulator daemon startup and
  shutdown — runs on `spawn_blocking`, never inline on the runtime.
* **Stdio** is pipes or a pseudo-terminal, per exec. Pipes keep stdout and
  stderr distinct and byte-exact; a terminal is what makes an interactive
  shell work. See [`daemon.md`](daemon.md).

## Safety

Every crate is `#![forbid(unsafe_code)]` through the workspace lint table,
with one exception: `rocjitsu_sys`, the FFI layer to `librocjitsu.so`,
which carries an equivalent lint table of its own with `unsafe` permitted.
Safe RAII wrappers over the C API (e.g. the daemon handle) live in that
crate too, so the `unsafe` and the invariants justifying it sit in the same
file and are reviewed together.

Mirage previously hand-rolled `unsafe` in three places outside the FFI
layer. Each is now either unnecessary or delegated to a crate built for
it:

| Was | Now |
| --- | --- |
| `pre_exec` calling `setsid` to put a child in its own process group | the safe `Command::process_group(0)` |
| `pre_exec` calling `setsid` + `TIOCSCTTY` to attach a pty | [`pty-process`], which owns that `unsafe` |
| hand-written `termios` and `TIOCGWINSZ` in the attach path | `nix`'s safe `termios` wrappers and [`terminal_size`] |

The rule is enforced, not aspirational: `forbid` cannot be relaxed by a
later `allow`, so an `unsafe` block anywhere outside `rocjitsu_sys` fails
the build.

[`pty-process`]: https://docs.rs/pty-process
[`terminal_size`]: https://docs.rs/terminal_size

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
  validation, path resolution, atomic state I/O, option parsing, profile
  overrides, the wire protocol's round-tripping, output fan-out, and
  process spawn/kill/reap semantics.
* **Supervisor integration tests** (`supervisor/tests/manager.rs`) drive
  the real engine with real processes but no daemon and no socket, so a
  failure points at the lifecycle rather than the transport.
* **End-to-end tests** under `tests/` drive the actual binary through the
  CLI, the Unix socket, and the HTTP/WebSocket surface. Each gets a
  private XDG root and its own daemon, so they are independent and
  parallel-safe.
* **Interactive tests** (`tests/interactive.rs`) drive the CLI through a
  real pseudo-terminal and assert on what appears on the screen: that
  `bash` prints a prompt, echoes input, answers `read -p`, honours Ctrl-C,
  and sees the right window size before and after a resize. A test that
  piped stdin would prove none of this, because a shell decides how to
  behave by calling `isatty`.
* **Strain tests** (`tests/strain.rs`) rapidly create, kill and delete
  sessions and execs under concurrency, then assert against the operating
  system's process table — no leaked processes, no zombies, no leaked file
  descriptors. Cleanup bugs are rarely visible in a single clean run, and
  internal bookkeeping agreeing with itself is exactly the failure mode
  the old design had.
