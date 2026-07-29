# mirage

**mirage** is the user-facing UX — a command-line tool and an optional web
dashboard — for the [`rocjitsu`][rocjitsu] GPU emulator and other emulator
backends. It lets you run real ROCm applications on top of an emulated GPU
without changing the application, and inspect, script, and recover from the
emulation as easily as you read a file.

```sh
$ mirage profile create cdna4 --emulator rocjitsu --agent MI350X
$ mirage run --profile cdna4 -- ./my-rocm-app --flag
```

## Why mirage

* **Nothing is left behind.** Every workload process is owned by a
  supervisor that always waits on it, runs it in its own process group, and
  escalates `SIGTERM` to `SIGKILL` on teardown. `mirage session stop`
  returns only once the process table agrees, so destroying a session
  really destroys it — no orphans, no zombies.
* **Configuration lives on disk** in standard [XDG locations][xdg]:
  profiles, agents and topologies are files you can read, edit and check
  into version control. Session state does not — it is owned in memory by
  the daemon, so what you are told about a running session is true rather
  than inferred from a file someone left behind.
* **Interactive when you need it.** `mirage run -- bash` is a real shell:
  prompt, echo, line editing, Ctrl-C and job control all work, and
  resizing your window resizes the workload's terminal. Redirected runs
  stay on pipes, so their output is byte-exact with stdout and stderr
  separate.
* **Easy to script.** Every list/show command accepts `--json` for
  machine-readable output, and exit codes are predictable.
* **A drop-in for `rocjitsu`.** `mirage --config cfg.json -- ./app` works just
  like the upstream `rocjitsu` CLI, so existing scripts keep running.

## Core concepts

| Concept      | What it is                                                                 |
| ------------ | -------------------------------------------------------------------------- |
| **Emulator** | A backend that runs GPU code (`rocjitsu`, `rocjitsu-dbt`, `hotswap`). |
| **Agent**    | A hardware GPU definition (e.g. `MI300X`, `MI350X`, `MI450X`).             |
| **Topology** | A rack/node/GPU layout that references an agent.                           |
| **Profile**  | A reusable preset binding an emulator + topology + options.               |
| **Session**  | A long-lived context that hosts an emulator and runs execs. Owned by the daemon, so it outlives the command that created it. |
| **Exec**     | A single command invocation inside a session, with fully redirected stdio. |

A typical flow is: pick or create a **profile**, start a **session** from it,
and run one or more **execs** in that session. The `mirage run` shortcut does
all three (create → run → clean up) in a single command.

## Quick start

```sh
# See which emulator backends are available on this machine.
mirage emulators

# Create a profile targeting an MI350X with the rocjitsu emulator.
mirage profile create cdna4 --emulator rocjitsu --agent MI350X

# One-shot: create a transient session, run a command, attach, clean up.
mirage run --profile cdna4 -- ./my-rocm-app --flag

# Or manage the lifecycle yourself:
sid=$(mirage session start --profile cdna4)
mirage exec start "$sid" -- ./my-rocm-app
mirage session stop "$sid"
```

No physical GPU is needed: `rocjitsu` emulates one in software. You do
need its runtime library — see
[`docs/building.md`](docs/building.md) — and `mirage emulators` reports
whether this machine has it.

## Building

mirage is a single Cargo workspace. One build produces the unified `mirage`
binary:

```sh
cd emulation/mirage
cargo build --workspace          # debug build -> target/debug/mirage
./target/debug/mirage --help
```

By default the `rocjitsu` backend is compiled in. Backends are selected with
Cargo features, and the optional web dashboard needs **Node.js 20.19+**. See
[`docs/building.md`](docs/building.md) for the full guide, including building
`rocjitsu` and the dashboard.

## Testing

```sh
cargo test --workspace
```

The end-to-end tests in `tests/` drive the full lifecycle (create session →
start exec → attach → signal → stop) through the public CLI and HTTP surfaces.
The rocjitsu-backed e2e tests require a working rocjitsu runtime; without it
they are expected to fail with a "KMD preload library not found" message.

## The daemon

Sessions outlive the command that created them, so something has to own
them in between. That is the supervisor daemon — one per user, started
automatically the first time a command needs it, and gone again after ten
idle minutes. You normally never think about it:

```sh
mirage daemon status             # pid, uptime, sessions
mirage daemon stop               # destroy every session and exit
```

## Web dashboard (optional)

The daemon can also serve an HTTP/WebSocket dashboard, backed by the same
sessions the CLI sees:

```sh
cargo build --workspace --features webui
mirage daemon --addr 127.0.0.1:5174   # serve the dashboard
mirage daemon install --addr 127.0.0.1:5174   # as a systemd user service
```

## Documentation

* [`docs/cli.md`](docs/cli.md) — complete CLI reference.
* [`docs/architecture.md`](docs/architecture.md) — design and crate overview.
* [`docs/building.md`](docs/building.md) — building mirage, the dashboard, and rocjitsu.
* [`docs/daemon.md`](docs/daemon.md) — the supervisor daemon and its process model.
* [`docs/state-layout.md`](docs/state-layout.md) — authoritative on-disk layout reference.

[rocjitsu]: ../rocjitsu/
[xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
