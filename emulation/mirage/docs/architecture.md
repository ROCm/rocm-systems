# Architecture

mirage is a single Cargo workspace that builds one unified `mirage`
executable. The binary bundles three roles — the control plane (`ctl`), the
per-session host (`host`), and the optional web dashboard (`daemon`) — plus a
set of pluggable emulator backends. Which role runs is decided by the
subcommand:

* `mirage <ctl-command>` — every user-facing verb (`profile`, `topology`,
  `agent`, `emulators`, `session`, `exec`, `run`, `attach`, `logs`, `state`,
  `paths`, `about`).
* `mirage host --session <id>` — the per-session host (spawned internally).
* `mirage webui` — the cross-session HTTP/WebSocket dashboard.

## Crates

| Crate              | Role                                                                 |
| ------------------ | ------------------------------------------------------------------- |
| `mirage` (root)    | The unified binary: parses argv, owns the Tokio runtime, dispatches to a role. |
| `mirage_ctl`       | The control plane: the `CtlCmd` subcommand tree and `dispatch`.     |
| `mirage_core`      | Shared types (`ProfileDef`, `SessionDef`, `ExecDef`, …), XDG path resolution, atomic state I/O, `FileCtl`, the attach stream, and the emulator/registry traits. |
| `mirage_host`      | The per-session host: watches `exec/`, runs commands, publishes health. |
| `mirage_container` | Container provider abstraction (podman/docker) for containerised sessions. |
| `mirage_builtin`   | Embedded builtin agents, topologies, and profiles, plus their unpackers. |
| `mirage_daemon`    | The web UI server (HTTP API + WebSocket), built only with the `daemon`/`webui` features. |
| `mirage_dashboard` | The embedded React/Vite single-page app served by `mirage_daemon`.  |
| `mirage_noop`      | The `noop` backend: runs commands directly with no emulation.       |
| `mirage_rocjitsu`  | The `rocjitsu` (and `rocjitsu-dbt`) backend and its asset embedding. |
| `mirage_hotswap`   | The `hotswap` load-time ISA-rewriting backend.                      |
| `rocjitsu_sys`     | Low-level FFI bindings used by the rocjitsu backend.                |

Emulator backends are **link-only** dependencies: each registers itself into
the emulator registry via [`inventory`] at link time. The binary never names a
backend directly, so enabling or disabling a backend's Cargo feature simply
adds or removes it from `mirage emulators`.

[`inventory`]: https://docs.rs/inventory

## Data flow

```text
┌──────────────────────────────────────────────────────────────────┐
│ mirage <ctl-command>            (mirage_ctl::dispatch)            │
│  • parses argv (clap)                                             │
│  • owns the only Tokio runtime in the CLI process                │
│  • renders text / JSON                                           │
│  • drives a MirageCtl implementation (FileCtl by default)        │
└───────────────┬──────────────────────────────────────────────────┘
                │ trait MirageCtl  (read/write on-disk state)
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Filesystem under the mirage config / runtime / state directories │
│  profiles, agents, topologies (config)                           │
│  session/<id>/...           (runtime)                            │
└───────────────▲──────────────────────────────────────────────────┘
                │ writes def.json + exec defs; reads health/status/stdout
                │
┌───────────────┴──────────────────────────────────────────────────┐
│ mirage host --session <id>      (mirage_host)                    │
│  • one detached process per session                              │
│  • spawned by `session start` / `run`                           │
│  • watches exec/ for new defs, runs them through the emulator    │
│  • writes health.json continuously                              │
└──────────────────────────────────────────────────────────────────┘
```

The CLI and the host never talk over a socket or network — only through the
session directory. See [`state-layout.md`](state-layout.md) for the exact
files and formats.

## Why file-backed?

* **Inspectability.** Every interesting state can be read with `ls`, `cat`,
  and `jq`. Debugging is far easier than tracing through an IPC daemon.
* **Recoverability.** If any process crashes, the others keep going.
  Restarting the CLI is free; restarting a host is safe.
* **No daemon required.** Read-only commands (list, show, paths) work with no
  host running at all.
* **Simple lifetime.** `$XDG_RUNTIME_DIR` is cleared on logout, so sessions do
  not leak across reboots.

## Why a per-session host?

* **Isolation.** A buggy emulator in one session cannot affect another.
* **Containerisation.** A host can be the entry point of a container while the
  CLI on the outside still drives it through the bind-mounted session
  directory.
* **Clean shutdown.** `SIGTERM` to a host means exactly "stop this session";
  nothing else needs to know.

## Concurrency model

The CLI process runs a single Tokio runtime created in `main`. Each host
process runs its own runtime.

* **Attach** streams output via `tokio_stream`'s `ReceiverStream` over a tail
  task that polls each node's `stdout` and `exit_code`. Interactive attaches
  put the terminal into raw mode and forward stdin to the exec's node-0 FIFO.
* **The host** scans `exec/` on a fixed tick, runs each discovered exec in its
  own task, and shares a `tokio::sync::Notify` between its signal handler and
  main loop for clean shutdown.

## Adding an emulator backend

1. Create a crate that implements the `mirage_core::emulator::Emulator` trait.
2. Register it into the registry with an `inventory::submit!` in its `lib.rs`.
3. Add it as an optional, feature-gated dependency of the root `mirage` crate
   and reference it with `extern crate … as _` so the linker keeps the
   registration.

No code in the binary needs to name the backend; it appears automatically in
`mirage emulators` and becomes selectable via `--emulator`.

## Testing strategy

* **Unit tests** live in each crate's `*::tests` modules and cover id
  validation, path resolution, atomic state I/O, option parsing, profile
  overrides, and the `FileCtl` surface. Tests that mutate process-wide env
  vars serialize through a shared lock.
* **End-to-end tests** under `tests/` exercise the full stack through the
  public CLI (`assert_cmd`) and the HTTP/WebSocket surface. Each runs in a
  fresh tempdir with overridden mirage directories, so they are independent
  and parallel-safe even though they spawn real host processes. The
  rocjitsu-backed suites require a working rocjitsu runtime; the `noop`-backed
  ones do not.
