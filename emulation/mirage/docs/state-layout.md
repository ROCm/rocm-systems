# Mirage on-disk state layout

All mirage state lives on disk in standard [XDG Base Directory][xdg]
locations. This document is the authoritative reference for the layout
and the file formats; tools that want to interoperate with mirage
should read and write these files directly.

[xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

## Base directories

| Resource              | Variable           | Default                | Subpath                       |
| --------------------- | ------------------ | ---------------------- | ----------------------------- |
| Profiles              | `XDG_CONFIG_HOME`  | `~/.config`            | `mirage/profile/<name>.json`  |
| Sessions (runtime)    | `XDG_RUNTIME_DIR`  | `/run/user/<uid>`      | `mirage/session/<id>/...`     |
| State (reserved)      | `XDG_STATE_HOME`   | `~/.local/state`       | `mirage/`                     |

`XDG_RUNTIME_DIR` is preferred because the kernel/systemd guarantees
this directory exists, is writable only by the owning user, and is
cleared on logout — exactly the lifetime we want for sessions.

## Profiles

A profile is a JSON document conforming to the `ProfileDef` schema.
Profiles are content-addressable by filename:

```text
$XDG_CONFIG_HOME/mirage/profile/<name>.json
```

Example (`cdna3.json`):

```json
{
  "name": "cdna3",
  "description": "A single-node rocjitsu emulator targeting CDNA3.",
  "emulator": {
    "emulator": "rocjitsu",
    "plugins": {},
    "nodes": 1,
    "gpus_per_node": 1,
    "exec_mode": "Functional",
    "options": {},
    "topology": "rocjitsu/cdna3"
  }
}
```

Use `mirage profile show <name>` to print an existing profile.

## Sessions

A session is a directory under `$XDG_RUNTIME_DIR/mirage/session/<id>`:

```text
<session>/
├── def.json          # SessionDef (immutable after create)
├── health.json       # SessionHealth (rewritten by the host on changes)
├── host.pid          # ASCII pid of the host process
├── host.log          # the host's stderr (tracing output)
└── exec/
    └── <exec-id>/    # one directory per exec
```

### `def.json`

A `SessionDef`:

```json
{
  "id": "s-20260530-153012-abcd",
  "profile": "cdna3",
  "container": null,
  "workdir": "/home/me/work",
  "created_at": "2026-05-30T15:30:12.000Z"
}
```

`profile` may be a string (the name of a profile in
`$XDG_CONFIG_HOME/mirage/profile`) or an inline `ProfileDef` object.

### `health.json`

A `SessionHealth`:

```json
{
  "timestamp": "2026-05-30T15:30:12.250Z",
  "healthy": true,
  "state": "ready",
  "terminal": false,
  "message": null
}
```

`state` is one of `starting`, `pulling`, `ready`, `degraded`,
`stopping`, `stopped`, `error`. When `terminal=true` the host has
given up and the session must be discarded.

The CLI's `session wait` polls `health.json` until it sees
`healthy=true` or `terminal=true`.

### `host.pid` and `host.log`

`host.pid` is the pid of the host process. It is rewritten on each
host startup and is what `mirage session stop` signals. `host.log` is
appended to by the host with tracing output (controlled by
`MIRAGE_LOG=`).

## Execs

Each exec gets a directory under `<session>/exec/<exec-id>/`:

```text
<exec>/
├── def.json          # ExecDef (immutable)
├── status.json       # ExecStatus (rewritten as nodes start/finish)
└── node/
    └── <node-id>/
        ├── stdin     # FIFO (named pipe), read by the child
        ├── stdout    # file, child stdout+stderr (merged by the PTY) appended here
        ├── pid       # ASCII pid of the child process
        └── exit_code # ASCII exit code, written after the child exits
```

### Exec id format

`<exec-id>` is `e-<n>` where `<n>` is a zero-padded counter (e.g.
`e-000000`). Ids are allocated monotonically by inspecting the
existing entries; if you delete an exec directory and then create a
new one, the new one will reuse the slot.

### `def.json`

```json
{
  "timestamp": "2026-05-30T15:31:00.000Z",
  "session": "s-20260530-153012-abcd",
  "exec": {
    "command": "/bin/sh",
    "args": ["-c", "echo hi"],
    "env": { "EXTRA": "1" },
    "workdir": null
  },
  "worker_exec": null,
  "keep": true
}
```

The host watches `exec/` for new directories that contain a `def.json`
but no `status.json` yet. As soon as one appears, it creates the
per-node directory structure and spawns the child.

### `status.json`

```json
{
  "started": true,
  "ended": true,
  "exit_code": 0,
  "started_at": "2026-05-30T15:31:00.100Z",
  "ended_at": "2026-05-30T15:31:00.250Z",
  "nodes": {
    "0": { "pid": 110891, "exit_code": 0 }
  }
}
```

`exit_code` at the top level is the worst (largest absolute) exit
code across nodes. Clients should consult per-node entries for full
detail.

### Per-node files

* `stdin` is a Unix FIFO created with `mkfifo(2)`. To write to a
  running exec's stdin: `printf 'data\n' > <node>/stdin`. The CLI
  exposes this via `MirageCtl::session_stdin`.
* `stdout` is a plain file opened with `O_APPEND` by the
  host (the PTY merges the child's stderr into it). Concurrent readers
  can `tail -f` it safely.
* `pid` is written before the child runs; readers that need a pid can
  poll the file. The file is removed when the host removes the exec
  directory (only when `keep=false`).
* `exit_code` is written after the child exits, atomically (write-then-
  rename). Its presence indicates the child is done.

## Atomicity guarantees

* All JSON writes are atomic (`<path>.tmp.<pid>` followed by `rename`).
* `stdin` (FIFO) and `stdout` (append-only file) are created
  before the child is spawned, so readers can attach as soon as the
  per-node directory exists.
* `status.json` may be rewritten many times; each rewrite is atomic.
* The host writes its own pid file before publishing healthy, so any
  client that sees `health.json:healthy=true` can rely on `host.pid`.

## Cleanup

* When `keep=false` (the default for `exec start` without `--keep` and
  for `run`), the host removes the entire `<exec>/` directory after
  `exit_code` has been written for every node.
* `mirage session stop` removes the entire `<session>/` directory after
  the host exits.
* `XDG_RUNTIME_DIR` is itself cleared on logout, so leaked sessions
  do not survive across reboots/logins.
