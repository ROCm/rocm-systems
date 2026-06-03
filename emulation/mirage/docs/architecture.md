# Architecture

Mirage is split into a small set of crates with clear responsibilities:

```text
┌─────────────────────────────────────────────────────────────────┐
│ mirage (root binary)            $ mirage ...                    │
│ thin wrapper that calls into mirage_cli::main()                 │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ mirage_cli                                                      │
│  • parses argv (clap)                                           │
│  • holds the only Tokio runtime                                 │
│  • renders text/JSON                                            │
│  • delegates to MirageCtl                                       │
└────────────┬────────────────────────────────────────────────────┘
             │ trait MirageCtl
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ mirage_core                                                     │
│  • types: SessionDef, ExecDef, ProfileDef, ...                  │
│  • paths: XDG resolution                                        │
│  • state: atomic JSON read/write                                │
│  • ctl::FileCtl: file-backed MirageCtl                          │
│  • attach: streaming tail-f over stdout/stderr/exit_code        │
└────────────┬────────────────────────────────────────────────────┘
             │ reads/writes
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Filesystem under $XDG_CONFIG_HOME and $XDG_RUNTIME_DIR          │
└─────────────────────────────────────────────────────────────────┘
                          ▲
                          │ writes def.json
                          │ reads exec/, writes status/pid/exit_code
┌─────────────────────────┴───────────────────────────────────────┐
│ mirage-host (mirage_host crate)                                 │
│  • one process per session                                      │
│  • spawned by `mirage session start` (detached)                 │
│  • watches exec/ for new defs, runs them                        │
│  • writes health.json continuously                              │
└─────────────────────────────────────────────────────────────────┘
```

## Why file-backed?

* **Inspectability**: every interesting state can be read with `ls`,
  `cat`, `jq`. Debugging is dramatically easier than tracing through
  an IPC daemon.
* **Recoverability**: if any process crashes, the others can keep
  going.  Restarting the CLI is free; restarting the host is safe.
* **No daemon required**: most CLI commands (list, show, paths) work
  even with no host running.
* **Simple lifetime**: `XDG_RUNTIME_DIR` is automatically cleaned by
  systemd on logout, so we don't leak state across reboots.

## Why a per-session host?

* Isolation between sessions: a buggy emulator in one session can't
  hurt another.
* Optional containerization: each host can be the entry point of a
  container, while the CLI on the outside still drives it via the
  shared session directory (which is bind-mounted into the container).
* Clean shutdown semantics: `SIGTERM` to the host means "stop this
  session"; nothing else has to know.

## Concurrency model

The CLI uses a single Tokio runtime, started in `mirage_cli::main`. The
host also has a single Tokio runtime. The two never talk over a
network or socket — only through the session directory.

Within the CLI:

* `attach` uses `tokio_stream::wrappers::ReceiverStream` over a tail
  task that polls `<node>/stdout`, `<node>/stderr`, and
  `<node>/exit_code` on a 50ms tick.
* All other ops are blocking filesystem calls wrapped in async fns
  for uniformity. The runtime is single-threaded by default.

Within the host:

* The main loop scans `exec/` on a 50ms tick.
* Each discovered exec runs in its own `tokio::spawn` task that owns
  the child via `tokio::process::Command::spawn`.
* Signal handling (SIGTERM/SIGINT) shares a `tokio::sync::Notify` with
  the main loop.

## Testing strategy

* **Unit tests** in `mirage_core::*::tests` cover id validation, path
  resolution, atomic state I/O, and every method of `FileCtl`.
  Process-wide env vars are serialized with a static mutex
  (`paths::test_env_lock`).
* **End-to-end tests** in `tests/e2e.rs` exercise the full stack
  through the public CLI surface (`assert_cmd`). Each test runs in a
  fresh tempdir with overridden XDG vars, so they are independent and
  parallel-safe even though they spawn real host processes.
* The e2e tests cover: profiles CRUD, run, attach, detach, signal,
  cleanup, JSON output, invalid id rejection, duplicate id rejection.
