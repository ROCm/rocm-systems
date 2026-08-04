# mirage on-disk state layout

All mirage state lives on disk in standard [XDG Base Directory][xdg]
locations. This document is the authoritative reference for the layout and the
file formats; tools that interoperate with mirage may read and write these
files directly.

[xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

## Base directories

| Resource          | Override         | XDG fallback        | Subpath                      |
| ----------------- | ---------------- | ------------------- | ---------------------------- |
| Config            | `MIRAGE_CONFIG`  | `$XDG_CONFIG_HOME`  | `mirage/` (profiles, agents, topologies) |
| Sessions (runtime)| `MIRAGE_RUNTIME` | `$XDG_RUNTIME_DIR`  | `mirage/session/<id>/...`    |
| Persistent state  | `MIRAGE_STATE`   | `$XDG_STATE_HOME`   | `mirage/`                    |

Each `MIRAGE_*` variable, when set, fully overrides the corresponding
directory; otherwise the XDG variable (or its standard default) is used.
`mirage paths` prints the resolved directories. `$XDG_RUNTIME_DIR` is
preferred for sessions because it is per-user, writable only by its owner, and
cleared on logout — exactly the lifetime sessions want.

The config directory holds three resource trees:

```text
<config>/mirage/
├── profile/<name>.json     # ProfileDef
├── agent/<name>.json       # AgentDef   (hardware GPU definition)
└── topology/<name>.json    # TopologyDef (rack/node/GPU layout)
```

## Profiles

A profile is a JSON `ProfileDef` named by its filename:

```text
<config>/mirage/profile/<name>.json
```

```json
{
  "name": "cdna4",
  "description": "Single-node rocjitsu targeting MI350X.",
  "emulator": {
    "emulator": "rocjitsu",
    "plugins": {},
    "exec_mode": "Functional",
    "options": {},
    "topology": {
      "num_nodes": 1,
      "gpus_per_node": 1,
      "agent": "MI350X"
    }
  }
}
```

* `emulator.topology` may be an inline object (as above) or a string naming a
  topology in `<config>/mirage/topology/`.
* `topology.agent` may likewise be an inline object or a string naming an
  agent in `<config>/mirage/agent/`.
* A containerised profile additionally carries a `containerize` object
  (`image`, optional `provider`, and `mounts`).

Use `mirage profile show <name>` to print an existing profile.

## Sessions

A session is a directory under `<runtime>/mirage/session/<id>`:

```text
<session>/
├── def.json              # SessionDef (immutable after create)
├── health.json           # SessionHealth (rewritten by the host)
├── node/                 # per-node host runtime state (one dir per rank)
│   └── <rank>/
│       ├── pid           # pid of the node's host process
│       └── host.log      # the node host's stderr (tracing output)
└── exec/
    └── <exec-id>/        # one directory per exec
```

### `def.json`

A `SessionDef`:

```json
{
  "id": "s-20260616-191636-f6c1",
  "profile": "cdna4",
  "workdir": "/home/me/work",
  "daemon": false,
  "created_at": "2026-06-16T19:16:36.951820571Z"
}
```

* `profile` may be a string (a profile name under `<config>/mirage/profile`)
  or an inline `ProfileDef` object (used when CLI overrides are applied).
* `daemon` selects out-of-process emulation when `true`.

### `health.json`

A `SessionHealth`, rewritten by the host as the session progresses:

```json
{
  "timestamp": "2026-06-16T19:16:36.972372050Z",
  "healthy": true,
  "state": "ready",
  "terminal": false
}
```

`state` is one of `starting`, `pulling`, `ready`, `degraded`, `stopping`,
`stopped`, `error`. When `terminal=true` the host has given up and the session
must be discarded. An optional `message` carries detail (image-pull progress,
node bring-up status, crash diagnostics). `mirage session wait` polls this
file until `healthy=true` or `terminal=true`.

### Host pid and log

The per-session (node 0) host writes its pid to `node/0/pid` on startup and
appends tracing output to `node/0/host.log`. Each additional node rank has its
own `node/<rank>/` directory with the same two files. `mirage session stop`
signals the recorded host pids.

## Execs

Each exec gets a directory under `<session>/exec/<exec-id>/`:

```text
<exec>/
├── def.json              # ExecDef (immutable)
├── status.json           # ExecStatus (rewritten as nodes start/finish)
└── node/
    └── <node-id>/
        ├── stdin         # FIFO (named pipe), bridged to the child's PTY
        ├── stdout        # plain file, merged child stdout+stderr (PTY)
        ├── pid           # pid of the child process
        └── exit_code     # exit code, written after the child exits
```

### Exec id format

`<exec-id>` is `e-<n>` with a zero-padded counter (e.g. `e-000000`). Ids are
allocated monotonically from the existing entries.

### `def.json`

```json
{
  "timestamp": "2026-06-16T19:31:00.000Z",
  "session": "s-20260616-191636-f6c1",
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

The host watches for `exec/<id>/def.json` directories with no `status.json`
yet; when one appears it creates the per-node structure and spawns the child.

### `status.json`

```json
{
  "started": true,
  "ended": true,
  "exit_code": 0,
  "started_at": "2026-06-16T19:31:00.100Z",
  "ended_at": "2026-06-16T19:31:00.250Z",
  "nodes": {
    "0": { "pid": 110891, "exit_code": 0 }
  }
}
```

The top-level `exit_code` summarises across nodes; consult the per-node
entries for full detail.

### Per-node files

* `stdin` is a Unix FIFO created with `mkfifo(2)`. Writing to it forwards to
  the running exec's stdin: `printf 'data\n' > <node>/stdin`. The CLI exposes
  this via `MirageCtl::session_stdin` (used by interactive `attach`). It is
  created when the exec starts.
* `stdout` is a plain file opened `O_APPEND`; the PTY merges the child's
  stderr into it, so concurrent readers can `tail -f` safely.
* `pid` is written before the child runs.
* `exit_code` is written atomically after the child exits; its presence marks
  the child done.

## Atomicity guarantees

* All JSON writes are atomic (`<path>.tmp.<pid>` then `rename`).
* `stdin` and `stdout` exist before the child is spawned, so readers can
  attach as soon as the per-node directory appears.
* `status.json` is rewritten many times; each rewrite is atomic.
* The host writes its pid before publishing `healthy=true`, so any client that
  sees `healthy=true` can rely on the recorded pid.

## Cleanup

* When `keep=false` (the default for `exec start` without `--keep`, and for
  `run`), the host removes the entire `<exec>/` directory once `exit_code` has
  been written for every node. Attaching to such an exec after it finishes is
  therefore racy; use `--keep` if you need its logs afterward.
* `mirage session stop` removes the entire `<session>/` directory after the
  host exits.
* `mirage state purge` removes the runtime and state directories (and,
  with `--all`, the config directory too).
* `$XDG_RUNTIME_DIR` is cleared on logout, so leaked sessions do not survive
  across reboots.
