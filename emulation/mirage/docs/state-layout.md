# mirage on-disk state layout

Mirage's **configuration** lives on disk in standard
[XDG Base Directory][xdg] locations. This document is the authoritative
reference for that layout and its file formats; tools that interoperate
with mirage may read and write these files directly.

Mirage's **session state does not live on disk at all**. A session, its
process table, its output and its health are held in memory by the
`mirage run` process that created them, and they cease to exist when it
exits. See [Why sessions are not files](#why-sessions-are-not-files)
below.

[xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

## Base directories

| Resource            | Override         | XDG fallback        | Subpath                                  |
| ------------------- | ---------------- | ------------------- | ---------------------------------------- |
| Config              | `MIRAGE_CONFIG`  | `$XDG_CONFIG_HOME`  | `mirage/` (profiles, agents, topologies) |
| Run sockets         | `MIRAGE_RUNTIME` | `$XDG_RUNTIME_DIR`  | `mirage/run/`                            |
| Emulator scratch    | `MIRAGE_RUNTIME` | `$XDG_RUNTIME_DIR`  | `mirage/session/<id>/`                   |

Each `MIRAGE_*` variable, when set, fully overrides the corresponding
directory; otherwise the XDG variable (or its standard default) is used.
`MIRAGE_RUNTIME` is also an *output*: mirage stamps the resolved runtime
directory on every process it starts and every container it creates, so
that reclamation can tell its own leftovers from the live sessions of a
mirage using a different one. See [Cleanup](#cleanup).
`mirage paths` prints the resolved directories — config, runtime,
profiles, sessions and runs. `$XDG_RUNTIME_DIR` is preferred for runtime
files because it is per-user, writable only by its owner, and cleared on
logout.

There is no persistent *state* directory. `$XDG_STATE_HOME` and
`MIRAGE_STATE` are not consulted at all: nothing mirage writes has to
survive a reboot beyond its configuration, because everything else
belongs to a process.

The runtime directory holds:

```text
<runtime>/mirage/
├── run/<session>.sock   # one socket per live `mirage run`
└── session/<id>/        # per-session emulator scratch (see below)
```

One socket per run, named after its session, rather than one well-known
socket for a server. That is not a detail. With a single shared path,
"who owns this session?" needs a registry and a lock protocol, and a
socket file left behind by a crashed process is indistinguishable from a
live one. Here the socket *is* the registration: connecting to it either
reaches the owner or fails, and failing is how a stale entry is
recognised. `mirage run` applies that test at bind time — if something
answers on the path, another run already owns the id and it refuses; if
nothing does, the file is a corpse and is unlinked. `mirage exec` applies
the same test from the other side, and reports "no `mirage run` is
serving session …" when nobody answers.

The `run/` directory is created `0700` and each socket `0600`, stated
rather than inherited from the umask: anyone who can connect learns how
to start processes in the session, and the `$TMPDIR` fallback used when
`$XDG_RUNTIME_DIR` is unset is not already private.

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
per-session host process communicated by writing and polling them.

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

State now has an owner, and the owner is a process the user can see. A
session exists exactly as long as the `mirage run` that created it; every
workload process is that run's child, owned by a task that always waits
on it; and the run's teardown finishes before the command returns. There
is nothing to ask about liveness because there is nothing to ask *of*:
the run is in a terminal, in the foreground, and when it is gone so is
the session.

The only survivor of the old design is the pid file, and only inside a
container — for the opposite reason. A containerised rank runs in the
container's PID namespace, where `podman exec` neither forwards signals
to the workload nor reports its pid, so the wrapper records `$$` to a
file in the scratch directory before `exec`ing the real program. The
supervisor reads it off the host filesystem with no provider round trip.
It names a process mirage cannot otherwise reach; it is not a channel and
nothing consults it to decide whether a session is alive.

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
├── rocjitsu/
│   ├── config_path           # discovery file the interposer reads
│   └── daemon.sock           # the emulator daemon's socket
└── exec/<exec>/<rank>.pid    # containerised ranks only (see above)
```

Mirage never reads these back to answer a control-plane query. The
session creates the directory before bring-up and removes it wholesale
during teardown, so it cannot outlive its session.

For a containerised session the directory is bind-mounted into each node
container at `/mnt/mirage/runtime`, so an in-container emulator runtime
resolves the same assets — and so the in-container wrapper's pid file
lands somewhere the host can read it.

## Finding a live run

Since sessions are not files, there is no session listing to read and no
server to query. A run's output is in the terminal running it, and its
existence is its socket:

| Question | Answer |
| --- | --- |
| Where do run sockets live? | `mirage paths` (the `runs:` line) |
| Which sessions are live? | the `.sock` files in that directory that answer when connected to |
| How do I start something in one? | `mirage exec -- <command>`, or `mirage exec -s <id> -- <command>` when several runs are up |
| What did it print? | the terminal its `mirage run` is in; a multi-node job labels every rank's lines automatically |

`mirage exec` may omit `--session` because the common case is one run in
one terminal and an exec in another; when the guess would be ambiguous it
lists the candidates rather than picking one.

The candidates are runs that *answer*, not socket files. A `SIGKILL`ed run
leaves its socket behind — the expected leak `mirage cleanup` exists for —
so counting files meant a single `kill -9` broke auto-selection outright:
beside a live run it reported "several runs are live" and offered a dead
session, and on its own it picked the corpse and then failed to connect.
Each candidate is therefore probed, and a socket nothing answers on is
unlinked on the way past so the next caller does not re-test it. Only two
errors are read as death — nothing listening, or the file already gone.
Anything else is about *this* process rather than the run, and is read as
"still alive", because unlinking a live run's socket on a full accept
backlog would leave it invisible and its session eligible for reclamation.

Naming a session explicitly is the exception, and deliberately so:
`mirage exec --session <id>` never consults that list, so a session that
has died is reported as gone rather than as never having existed.

## Atomicity guarantees

* All configuration writes are atomic (`<path>.tmp.<pid>` then `rename`),
  so a reader never observes a truncated profile, agent or topology.
* Emulator scratch files are written the same way, because an emulator
  runtime may read them while mirage is rewriting them.

## Cleanup

* A `mirage run` tears its session down on every exit path, including the
  failing ones: it terminates and reaps every workload process, stops the
  emulator daemon, removes the containers and the per-session network,
  deletes the scratch directory, and unlinks its socket. Ctrl-C goes
  through the same path rather than around it, which is why an
  interrupted run still leaves nothing behind.
* Containers are started with `--rm` and are not detached, so the
  container engine removes them even in the cases teardown never reaches.
* Nothing covers a `SIGKILL`ed run, an OOM kill, or a machine that lost
  power: no signal handler, no `Drop` and no cancelled task of mirage's
  runs, so the containers keep running, the workloads are reparented to
  init and the scratch directory stays. `mirage cleanup` is the recovery.
  It finds them by the marks left on the resources themselves — the
  `mirage.owner`, `mirage.session` and `mirage.runtime` labels on
  containers and networks, `MIRAGE_SESSION` and `MIRAGE_RUNTIME` in every
  workload's environment — since the session's own record died with the
  run. Sessions whose run still answers are skipped, so it is safe to run
  while other work is in flight; `--dry-run` previews it.
* Cleanup is scoped to one runtime directory, which is why the runtime
  mark exists. The sockets under `<runtime>/mirage/run/` are the whole
  registry of what is live *there*, so a cleanup cannot see the live
  sessions of a mirage running under a different `MIRAGE_RUNTIME`, and
  without the mark it would read them as a crashed run's leftovers. It
  therefore reclaims only what its own runtime directory created, and
  skips anything that records no runtime directory at all — including
  work left behind by a mirage older than the mark, which has to be
  removed by hand.
* `mirage state purge` does the same and then removes the whole runtime
  directory. Unlike `cleanup` it refuses while any run is still live:
  killing someone else's foreground command from a state-cleanup
  subcommand would be a surprise, and Ctrl-C in that terminal is the
  right way to stop it. With `--all` it removes the config directory too.
* `$XDG_RUNTIME_DIR` is cleared on logout, so nothing survives a reboot.
