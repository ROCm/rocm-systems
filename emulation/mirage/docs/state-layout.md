# mirage on-disk state layout

Mirage's **configuration** lives on disk in standard
[XDG Base Directory][xdg] locations. This document is the authoritative
reference for that layout and its file formats; tools that interoperate
with mirage may read and write these files directly.

Mirage's **session state does not live on disk at all**. Sessions, execs,
their process tables, their output and their health are held in memory by
the supervisor daemon and reached over its Unix socket. See
[Why sessions are not files](#why-sessions-are-not-files) below, and
[the daemon](daemon.md) for the process model.

[xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

## Base directories

| Resource            | Override         | XDG fallback        | Subpath                                  |
| ------------------- | ---------------- | ------------------- | ---------------------------------------- |
| Config              | `MIRAGE_CONFIG`  | `$XDG_CONFIG_HOME`  | `mirage/` (profiles, agents, topologies) |
| Daemon socket + log | `MIRAGE_RUNTIME` | `$XDG_RUNTIME_DIR`  | `mirage/`                                |
| Emulator scratch    | `MIRAGE_RUNTIME` | `$XDG_RUNTIME_DIR`  | `mirage/session/<id>/`                   |
| Persistent state    | `MIRAGE_STATE`   | `$XDG_STATE_HOME`   | `mirage/`                                |

Each `MIRAGE_*` variable, when set, fully overrides the corresponding
directory; otherwise the XDG variable (or its standard default) is used.
`mirage paths` prints the resolved directories. `$XDG_RUNTIME_DIR` is
preferred for runtime files because it is per-user, writable only by its
owner, and cleared on logout.

The runtime directory holds:

```text
<runtime>/mirage/
├── mirage.sock          # the daemon's control socket
├── mirage.lock          # exclusive lock held by the running daemon
├── daemon.log           # the daemon's stderr, when auto-started by the CLI
└── session/<id>/        # per-session emulator scratch (see below)
```

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

## Why sessions are not files

Mirage used to keep every session on disk: a `def.json`, a `health.json`
rewritten by a heartbeat, per-node `pid` files, a stdin FIFO, a stdout
file, a `status.json`, and a `signal` request file. The CLI and a detached
per-session `mirage host` process communicated by writing and polling
them.

It was inspectable with `ls` and `cat`, which was the appeal. It was also
the source of the lifecycle bugs, because a file cannot answer the
question the control plane actually needs answered: *is the thing that
wrote this still alive?*

* A `health.json` saying `healthy: true` says nothing about its author. To
  guess, the design added a heartbeat and a staleness ladder
  (`ready` → `stalled` → `dead`), which is an elaborate way of not knowing.
* A `pid` file records a number, not a process. By the time anything reads
  it the pid may have been recycled.
* Teardown removed the directory, so a kill that did not land became
  invisible: the state was gone, but the process was not.

State now has an owner. A session exists exactly as long as the daemon
holds it; every child process is owned by a task that always waits on it;
and `session destroy` returns only when the process table agrees. Clients
ask the daemon rather than reading files, which is why `mirage session
list` and `mirage logs` work through the socket.

## Emulator scratch directories

One runtime directory per session survives, at
`<runtime>/mirage/session/<id>/`. It is **not** a channel between mirage
processes — it exists because emulator runtimes are configured by path.
rocjitsu's `LD_PRELOAD` interposer, for instance, discovers its
`SimulationConfig` by reading a file from `$ROCJITSU_RUNTIME_DIR`, and its
daemon binds a socket in the same place:

```text
<runtime>/mirage/session/<id>/
├── rj_config.json            # synthesised rocjitsu SimulationConfig
└── rocjitsu/
    ├── config_path           # discovery file the interposer reads
    └── daemon.sock           # the emulator daemon's socket
```

Mirage never reads these back to answer a control-plane query. The
supervisor creates the directory before bring-up and removes it wholesale
during teardown, so it cannot outlive its session.

For a containerised session the directory is bind-mounted into each node
container at `/mnt/mirage/runtime`, so an in-container emulator runtime
resolves the same assets.

## Inspecting sessions

Since sessions are not files, use the CLI (or the HTTP API) to inspect
them. Every command below is answered by the daemon:

| Question | Command |
| --- | --- |
| What sessions exist? | `mirage session list` (`--json` for full state) |
| What is this session doing? | `mirage session show <id>` |
| What execs are in it? | `mirage exec list <id>` |
| How did an exec end? | `mirage exec show <id> <exec>` |
| What did it print? | `mirage logs <id> <exec>` (`-f` to follow) |
| Where is the emulator scratch? | `mirage session dir <id>` |
| Is a daemon running? | `mirage daemon status` |

## Atomicity guarantees

* All configuration writes are atomic (`<path>.tmp.<pid>` then `rename`),
  so a reader never observes a truncated profile, agent or topology.
* Emulator scratch files are written the same way, because an emulator
  runtime may read them while mirage is rewriting them.

## Cleanup

* `mirage session stop` destroys the session: every workload process is
  terminated and reaped, containers and the per-session network are
  removed, and the scratch directory is deleted. It returns only once all
  of that has happened.
* Stopping the daemon (`mirage daemon stop`, `SIGTERM`, or the idle
  timeout) destroys every session it owns first, so no workload outlives
  it.
* `mirage state purge` stops the daemon and removes the runtime and state
  directories (and, with `--all`, the config directory too).
* `$XDG_RUNTIME_DIR` is cleared on logout, so nothing survives a reboot.
