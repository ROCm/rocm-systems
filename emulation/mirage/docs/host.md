# The mirage host process

The host is the per-session worker that brings a session to life. There
is exactly one host per session; when it exits, the session is
considered stopped.

## Lifecycle

```text
                ┌─────────────┐
   CLI calls    │ session     │
   session start│ create()    │
        ────────▶ writes def  │
                └──────┬──────┘
                       │
                       ▼
              ┌─────────────────┐    ┌────────────────────────┐
              │ mirage-host     │───▶│ writes host.pid         │
              │  --session ID   │    │ writes health.json      │
              │                 │    │   {healthy: true,       │
              │ poll exec/ dir  │    │    state: "ready"}      │
              └─────────────────┘    └────────────────────────┘
                       │
                       │  new exec/<id>/def.json appears
                       ▼
              ┌─────────────────┐
              │ for each node n │
              │   mkfifo stdin  │
              │   open stdout   │
              │   spawn(child)  │
              │   write pid     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ wait(child)     │
              │ write exit_code │
              │ update status   │
              │ if !keep, rm    │
              └─────────────────┘
```

On `SIGTERM`/`SIGINT`:

1. Set health `state="stopping", healthy=false`.
2. `SIGTERM` every per-node `pid` recorded under `exec/*/node/*/pid`.
3. Wait up to ~2 seconds, then `SIGKILL` anyone still alive.
4. Set health `state="stopped"` and exit cleanly.

## Polling cadence

The host uses a fixed `50ms` polling tick to scan `exec/` for new
definitions. This was chosen over `inotify` for portability and
simplicity; `50ms` is fast enough to feel interactive and cheap enough
to be invisible on idle systems.

## File watching

The host watches:

* `exec/<id>/def.json` — appearance triggers a new exec launch.
* `exec/<id>/status.json` — the host (re)writes this; presence with
  `ended=true` marks the exec done.

It does not watch profiles (those are read by the CLI on session
create and inlined into `def.json` as needed by the emulator).

## Recovery

If the host crashes, restarting it (`mirage-host --session <id>`) is
safe:

* `host.pid` is overwritten.
* Already-finished execs (with `status.json: ended=true`) are skipped.
* In-progress execs whose child process is still alive are *not*
  re-adopted (their stdio FDs were owned by the old host); they remain
  visible as `started=true, ended=false` with a stale pid. A future
  enhancement could move adoption into a `PR_SET_CHILD_SUBREAPER`
  pattern.

## Containerized sessions

`SessionDef.container` is reserved for running the host inside a
container provider (docker/podman). When set, `mirage session start`
will spawn the host inside the container with appropriate bind-mounts
of the session directory.

This wiring is not yet implemented; the field is present so on-disk
state remains forward-compatible.

## Extending the host

The host is a thin orchestrator. The interesting per-emulator logic
lives behind the `mirage_core::emulator::Emulator` trait:

* `injection_def()` returns env vars, files, and `LD_PRELOAD` entries
  that should be added to every child.
* `health()` returns a snapshot the host periodically publishes.

To add a new emulator backend, implement `Emulator` for it and wire
it into the host's `match` on `EmulatorDef::emulator`.

## Environment

| Variable          | Effect                                                   |
| ----------------- | -------------------------------------------------------- |
| `MIRAGE_LOG`      | Tracing filter, e.g. `mirage_host=debug`.                |
| `XDG_RUNTIME_DIR` | Where to look for the session directory.                 |

The host inherits no other variables for the spawned children except
`PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`, `TERM`, and `TMPDIR`. To
add more, set them in `ExecDef.exec.env` (or supply them via the
emulator's `injection_def()`).
