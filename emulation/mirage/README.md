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

* **All state lives on disk** in standard [XDG locations][xdg]. There is no
  required background daemon to keep state alive: the CLI and the per-session
  host processes coordinate exclusively through well-defined files. You can
  inspect everything with `ls`, `cat`, and `tail -f`.
* **Easy to script.** Every list/show command accepts `--json` for
  machine-readable output, and exit codes are predictable.
* **Crash-resilient.** Because state is on disk, a crashed CLI or host can be
  restarted and pick up where it left off.
* **A drop-in for `rocjitsu`.** `mirage --config cfg.json -- ./app` works just
  like the upstream `rocjitsu` CLI, so existing scripts keep running.

## Core concepts

| Concept      | What it is                                                                 |
| ------------ | -------------------------------------------------------------------------- |
| **Emulator** | A backend that runs GPU code (`rocjitsu`, `rocjitsu-dbt`, `hotswap`, `noop`). |
| **Agent**    | A hardware GPU definition (e.g. `MI300X`, `MI350X`, `MI450X`).             |
| **Topology** | A rack/node/GPU layout that references an agent.                           |
| **Profile**  | A reusable preset binding an emulator + topology + options.               |
| **Session**  | A long-lived context (one host process) that hosts an emulator and runs execs. |
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

If you have no GPU and just want to exercise the tooling, use the `noop`
emulator, which runs commands directly with no emulation:

```sh
mirage profile create local --emulator noop
mirage run --profile local -- echo "hello from mirage"
```

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

## Web dashboard (optional)

mirage ships an optional cross-session HTTP/WebSocket dashboard:

```sh
cargo build --workspace --features webui
mirage webui                     # serve on 127.0.0.1:5174 by default
mirage webui install             # register it as a systemd user service
```

## Documentation

* [`docs/cli.md`](docs/cli.md) — complete CLI reference.
* [`docs/architecture.md`](docs/architecture.md) — design and crate overview.
* [`docs/building.md`](docs/building.md) — building mirage, the dashboard, and rocjitsu.
* [`docs/host.md`](docs/host.md) — what the per-session host does and how to extend it.
* [`docs/state-layout.md`](docs/state-layout.md) — authoritative on-disk layout reference.

[rocjitsu]: ../rocjitsu/
[xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
