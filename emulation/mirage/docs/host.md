# The mirage host process

The host is the per-session worker that brings a session to life. There is
exactly one host per session (the orchestrator host on the real machine);
when it exits, the session is considered stopped. It is the same `mirage`
binary, invoked as `mirage host --session <id>`, and is normally spawned for
you by `mirage session start` and `mirage run`.

## Lifecycle

```text
                 ┌──────────────────┐
   CLI calls     │ session create() │
   session start │ writes def.json  │
        ─────────▶                  │
                 └────────┬─────────┘
                          │ spawn detached, new process group
                          ▼
        ┌───────────────────────┐    ┌──────────────────────────┐
        │ mirage host           │───▶│ writes node/<rank>/pid    │
        │   --session <id>      │    │ writes health.json        │
        │                       │    │   { healthy: true,        │
        │ poll exec/ directory  │    │     state: "ready" }      │
        └───────────┬───────────┘    └──────────────────────────┘
                    │  new exec/<id>/def.json appears
                    ▼
        ┌───────────────────────┐
        │ for each node n:      │
        │   mkfifo stdin        │
        │   open stdout         │
        │   spawn child (PTY)   │
        │   write node/<n>/pid  │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │ wait(child)           │
        │ write exit_code       │
        │ update status.json    │
        │ if !keep, rm exec dir │
        └───────────────────────┘
```

On `SIGTERM`/`SIGINT` the host:

1. Sets health `state="stopping", healthy=false`.
2. Signals every per-node child recorded under `exec/*/node/*/pid`.
3. Waits briefly, then escalates to `SIGKILL` for anything still alive.
4. Sets health `state="stopped"` and exits cleanly.

## Health states

`health.json` carries a `state` that is one of `starting`, `pulling`
(container image), `ready`, `degraded`, `stopping`, `stopped`, or `error`,
together with `healthy` and `terminal` booleans and an optional `message`.

`mirage session wait` and the `run`/`session start` ready-wait poll
`health.json` until `healthy=true` or the session becomes terminal — a
terminal-but-unhealthy session (a bad image, a node that won't start, a
missing emulator asset) is treated as a hard failure so the CLI never submits
an exec that no host will ever run.

## Polling cadence

The host polls `exec/` on a fixed, sub-second tick to discover new exec
definitions. Polling (rather than `inotify`) was chosen for portability and
simplicity; it is fast enough to feel interactive and cheap enough to be
invisible on idle systems.

## What the host watches

* `exec/<id>/def.json` — its appearance triggers a new exec launch.
* `exec/<id>/status.json` — the host (re)writes this; `ended=true` marks the
  exec done.

It does not watch profiles: a profile is resolved by the CLI on session
create and inlined into the session's `def.json` as needed.

## Containerised sessions

When a profile (or `--image`/`--mount`/`--container-provider` override) requests
containerisation, the orchestrator host brings up one container per node
through the configured provider (podman or docker, autodetected), bind-mounts
the session directory in, and launches a per-node `mirage host` *inside* each
container. Those inner hosts run only their own node's execs and never manage
containers themselves. The provider, image, and per-node container ids are
recorded in the session state and surfaced by `mirage session list`.

The container provider can be selected with `--container-provider` or the
`MIRAGE_CONTAINER_PROVIDER` environment variable.

## Recovery

The session host is re-execable. Restarting `mirage host --session <id>` is
safe:

* The node's `pid` file is overwritten.
* Already-finished execs (`status.json: ended=true`) are skipped.
* In-progress execs whose child is still alive are not re-adopted (their stdio
  FDs were owned by the previous host); they remain visible as
  `started=true, ended=false` with a stale pid.

## Extending the host

The host is a thin orchestrator. Per-emulator behaviour lives behind the
`mirage_core::emulator::Emulator` trait:

* `injection_def()` returns env vars, files, and `LD_PRELOAD` entries added to
  every child.
* `validate_profile()` checks a profile is runnable (reported at
  `profile create` time).
* `health()` returns a snapshot the host periodically publishes.

To add a backend, implement `Emulator`, register it via `inventory`, and add
its Cargo feature (see [`architecture.md`](architecture.md)). No `match` arm in
the host needs editing.

## Environment

| Variable          | Effect                                                |
| ----------------- | ----------------------------------------------------- |
| `MIRAGE_LOG`      | Tracing filter, e.g. `debug` or `mirage_host=debug`.  |
| `MIRAGE_RUNTIME`  | Override the runtime dir holding the session directory (else `$XDG_RUNTIME_DIR/mirage`). |

A `-v`/`-vv` passed to the CLI is propagated to the detached host via
`MIRAGE_LOG`, so host events land in `node/<rank>/host.log` at the requested
verbosity. Children inherit only a curated set of variables (`PATH`, `HOME`,
`USER`, `LANG`, `LC_ALL`, `TERM`, `TMPDIR`); to add more, set them in the
exec's `--env` or via the emulator's `injection_def()`.
