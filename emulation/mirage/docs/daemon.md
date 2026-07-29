# The mirage supervisor daemon

The daemon owns every mirage session. It is the same `mirage` binary,
invoked as `mirage daemon`, and the CLI starts one for you the first time
a command needs it.

```text
  mirage run / session start / exec start / attach
                    │
                    │  Unix socket ($XDG_RUNTIME_DIR/mirage/mirage.sock)
                    ▼
        ┌──────────────────────────────┐
        │ mirage daemon                │
        │                              │
        │  SessionManager              │   in-memory, no on-disk state
        │   ├── Session "s-1"          │
        │   │    ├── health (watch)    │
        │   │    ├── containers        │
        │   │    ├── emulator daemon   │
        │   │    └── Exec "e-000000"   │
        │   │         ├── OutputHub    │   replay buffer + live broadcast
        │   │         └── process grid │   one task per process
        │   └── Session "s-2" …        │
        └──────────────────────────────┘
                    │
                    │  tokio::process, process_group(0), always waited on
                    ▼
              workload processes
```

## Why there is a daemon

Sessions outlive the command that created them: `mirage session start` in
one shell and `mirage exec start` in another have to reach the same
processes. Something must own those processes in between.

Mirage used to make that owner a detached `mirage host` process per
session, with the filesystem as the channel between it and the CLI — an
exec was requested by writing a `def.json` into a directory the host
polled, output arrived by tailing a file, and completion was a
`status.json` the CLI polled for.

That design leaked processes, for three structural reasons:

* **No single owner.** The host was detached from the CLI that spawned
  it, so it was reparented to init. Nothing was responsible for it, and
  nothing reaped it.
* **Files outlive their writers.** A `health.json` saying
  `healthy: true` tells you nothing about whether its author is still
  alive, so the design needed a heartbeat and a staleness ladder to
  *guess* — and a guess is a poor basis for deciding whether to kill
  something.
* **Kill paths were open-loop.** Teardown signalled pids read from files
  and then deleted the directory. If a signal did not land, or a
  grandchild had escaped the process group, nothing noticed: the state
  was gone, so the leak was invisible.

One daemon, holding sessions as values and children as awaited tasks,
turns each of those from an inference into a fact.

## Lifecycle

| Event | What happens |
|---|---|
| First CLI command | The CLI starts a daemon and waits for its socket. |
| Racing CLIs | Startup takes an exclusive `flock` *before* binding, so exactly one daemon wins; the others connect to it. |
| Idle | With no sessions and no clients for `--idle-timeout` seconds (default 600), the daemon exits. |
| `mirage daemon stop` | Destroys every session, then exits. |
| `SIGTERM` / `SIGINT` | Same path as `stop`. |
| Socket deleted | If the socket file is removed or replaced, no client can reach the daemon again. It shuts down rather than lingering invisibly while holding a process tree nobody can stop. |
| `SIGKILL` | No cleanup is possible, but the kernel releases the `flock`, so the next daemon recognises the leftover socket as stale and reclaims it. |

Terminating the daemon terminates every workload it started. That is the
point of it owning them.

## Process handling

Three properties are maintained for every workload process, each a
response to a way the previous design leaked:

1. **Every child is waited on.** A child is owned by exactly one task
   whose only exit path runs through `Child::wait`. A child that is never
   waited on becomes a zombie.
2. **Every child leads its own process group** (`process_group(0)`, a
   safe API — the old design used an `unsafe` `pre_exec` calling
   `setsid`). A workload that forks puts its whole tree in one group that
   can be signalled as a unit; signalling only the direct child would
   leave grandchildren running.
3. **Termination escalates, then confirms.** `SIGTERM` to the group, a
   bounded grace period, then `SIGKILL`, and the call does not return
   until the child has actually been reaped. `SIGKILL` cannot be caught,
   so the second stage always ends.

`session_destroy` returns only once all of that has happened, so a caller
told a session is destroyed can rely on it.

## Stdio

Workloads get ordinary pipes. stdout and stderr stay distinct, so
redirection works and clients can render them separately. An earlier
design ran every workload under a pseudo-terminal, which merged them
irreversibly and required `unsafe` to set up.

`mirage attach` therefore does the terminal work on the *client* side: it
puts the local tty into raw mode (via `nix`'s safe `termios` wrappers) and
forwards keystrokes, so typing works even though the remote end is a
pipe. Closing the client's stdin closes the workload's, so
`echo hi | mirage run -- cat` terminates.

## Which commands need it

Not all of them, deliberately. Anything touching a session or an exec
(`run`, `session`, `exec`, `attach`, `logs`, `state purge`) goes through
the daemon and starts one if needed. Everything else — `paths`,
`emulators`, and all profile, topology and agent commands — is answered in
this process from the config store and the link-time backend registry.

Starting a background process to read a directory would be a surprise
nobody asked for. There is no cache to keep coherent either: the daemon
reads a profile off disk when a session is created, so a profile the CLI
wrote a moment earlier is already visible to it.

## Talking to it

* `mirage daemon` — run in the foreground.
* `mirage daemon status` — pid, uptime, session count.
* `mirage daemon stop` — destroy everything and exit.
* `mirage daemon install` — install a systemd user service (with the idle
  timeout disabled and a stop timeout long enough to finish teardown).

`MIRAGE_SOCKET` selects a non-default socket, which is how the test suite
gives every test a private daemon. `MIRAGE_AUTOSTART=0` makes the CLI
report that no daemon is running instead of starting one.

## Where the HTTP API fits

With `--features webui`, `mirage daemon --addr 127.0.0.1:5174` also serves
the dashboard and its JSON/WebSocket API — backed by the *same*
`SessionManager`. The dashboard and the CLI cannot disagree about what
exists, because there is only one set of sessions.
