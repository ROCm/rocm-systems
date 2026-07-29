# mirage CLI reference

`mirage` is organized into a small set of verbs, each managing one kind of
resource. Every command supports `--help`, and most support `--json` for
machine-readable output.

```text
mirage <command> [options]
mirage <command> --help        # per-command help
```

## Global options

| Flag          | Description                                                       |
| ------------- | ---------------------------------------------------------------- |
| `--json`      | Emit JSON output where applicable.                               |
| `-v`, `-vv`   | Increase logging verbosity (`-v` = info, `-vv` = debug).         |
| `--help`      | Print help (top level or per command).                           |
| `--version`   | Print the mirage version.                                        |

Global flags may appear before or after the subcommand, e.g.
`mirage --json profile list`.

## Command summary

| Command            | Purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| `mirage emulators` | List emulator backends and their install / support status.    |
| `mirage profile`   | Manage profiles (reusable emulator presets).                   |
| `mirage topology`  | Manage topologies (rack/node/GPU layouts).                     |
| `mirage agent`     | Manage agents (hardware GPU definitions).                      |
| `mirage session`   | Manage sessions.                                               |
| `mirage exec`      | Manage execs inside a session.                                 |
| `mirage run`       | Create a session, run a command, attach, and clean up.         |
| `mirage attach`    | Re-attach to a running (or finished) exec's streams.           |
| `mirage logs`      | Show or follow an exec's stdout.                               |
| `mirage state`     | Manage mirage's on-disk state (builtins, purge).              |
| `mirage paths`     | Print where mirage stores its state.                          |
| `mirage about`     | Show version, copyright, and third-party licenses.            |
| `mirage daemon`    | Run or manage the supervisor daemon (usually auto-started).   |

## `mirage emulators`

List the emulator backends compiled into this build, whether each one's
runtime is installed, and whether this machine's hardware supports it.

```text
mirage emulators [-l|--long]
```

* The default emulator for new profiles is marked with `*`.
* `-l` shows a detailed block per backend, including the support reason.
* `--json` emits the full backend descriptors.

```sh
$ mirage emulators
NAME          INSTALLED  SUPPORTED  DESCRIPTION
noop*         yes        yes        no-op emulator: runs commands directly with no GPU emulation
rocjitsu      no         yes        ROCm just-in-time GPU emulator (cycle-accurate or functional)
...
* = default emulator for new profiles
```

## `mirage profile`

Profiles are reusable emulator presets stored in
`$XDG_CONFIG_HOME/mirage/profile/<name>.json`.

```text
mirage profile list [-l|--long]
mirage profile show <name>
mirage profile create [<name>] [--emulator NAME] [--agent NAME]
                      [--num-nodes N] [--gpus-per-node N]
                      [--description TEXT]
                      [--image IMAGE] [--mount SPEC]... [--container-provider PROV]
                      [--no-input]
mirage profile import <file>          # use '-' for stdin
mirage profile delete <name> [-f|--force]
```

* `--emulator` defaults to the first installed backend (rocjitsu if present,
  otherwise noop). `--agent` defaults to `MI350X`.
* `--image` containerises the profile (every node runs inside a container
  built from that image). `--mount HOST[:CONTAINER[:ro|rw]]` and `--container-provider`
  (`podman`, `docker`, or a path) require `--image`.
* On a terminal, `profile create` interactively prompts for any field not
  passed as a flag. Pass `--no-input` (or pipe/redirect stdin) to use defaults
  and stay non-interactive.
* Profiles are validated against their emulator at creation time, so an
  unusable setup is reported immediately.

## `mirage topology`

Topologies describe a rack/node/GPU layout and reference an agent. Builtin
topologies are written on first run.

```text
mirage topology list
mirage topology show <name>
mirage topology create <name> [--agent NAME] [--num-nodes N] [--gpus-per-node N]
mirage topology import <name> <file>   # use '-' for stdin
mirage topology delete <name> [-f|--force]
```

## `mirage agent`

Agents are hardware GPU definitions (e.g. `MI300X`, `MI350X`, `MI450X`).
Builtin agents are written on first run.

```text
mirage agent list
mirage agent show <name>
mirage agent import <name> <file>      # use '-' for stdin
mirage agent delete <name> [-f|--force]
```

## `mirage session`

Sessions are the long-lived contexts in which execs run. They are owned by
the [supervisor daemon](daemon.md), which the CLI starts on demand — so a
session created in one shell is usable from another.

```text
mirage session list
mirage session show <id>
mirage session wait <id> [--timeout SECONDS]   # default 30
mirage session start [--profile NAME] [--id ID] [--workdir DIR]
                     [--no-wait] [--ready-timeout SECONDS]
                     [--image IMAGE] [--mount SPEC]... [--container-provider PROV]
                     [--exec-mode functional|clocked] [--option KEY=VALUE]...
                     [--config PATH] [--daemon] [--no-input]
mirage session stop <id> [-f|--force]
mirage session dir  <id>                        # emulator scratch directory
```

* `session start` creates the session and waits up to `--ready-timeout`
  seconds for it to become healthy, then prints its id (or the full state
  with `--json`). A session that fails to come up is destroyed and the
  reason reported, rather than left behind for you to clean up.
* On a terminal it prompts for the profile (and id, workdir, ready-timeout)
  when not passed; `--no-input` requires `--profile` and uses defaults.
* `--no-wait` returns as soon as the session is registered, without waiting
  for bring-up to finish.
* `session stop` returns only once every workload process has been
  terminated and reaped, every container removed, and the scratch directory
  deleted.
* `-o`/`--option KEY=VALUE` overrides emulator options; `--exec-mode` selects
  functional (default) or clocked emulation; `--config PATH` uses an explicit
  emulator config file; `--daemon` runs the emulator out-of-process.
* `session stop` signals the host (`SIGTERM`, escalating to `SIGKILL`) and
  removes the session directory.

## `mirage exec`

```text
mirage exec list   <session>
mirage exec show   <session> <exec>
mirage exec start  <session> [--keep] [--detach] [--env KEY=VALUE]... -- <cmd> [args...]
mirage exec signal <session> <exec> [SIG]        # default: TERM
mirage exec remove <session> <exec>
```

`exec start` is the workhorse:

* By default it attaches to the new exec, forwards stdin/stdout/stderr, exits
  with the exec's exit code, and then removes the on-disk exec directory.
* `--detach` submits the exec and prints its id without attaching.
* `--keep` preserves the exec directory after it exits so you can re-read its
  logs with `mirage logs` or `mirage exec show`. Without `--keep`, a finished
  exec is reaped by the host — so attaching to a detached, non-`--keep` exec
  is racy; combine `--detach` with `--keep` if you intend to attach later.
* `--env KEY=VALUE` injects extra environment variables (repeatable).
* Everything after `--` is passed verbatim to the command.
* `exec signal` accepts a signal name (`TERM`, `KILL`, `INT`, `HUP`, `QUIT`,
  `USR1`, `USR2`, with or without the `SIG` prefix) or a number.

## `mirage attach <session> <exec>`

Re-attach to an exec that is still running (or to a finished, `--keep`-ed exec,
in which case buffered output is replayed and the saved exit code is returned).
Useful after a `--detach`.

## `mirage logs <session> <exec> [-f|--follow]`

Print the exec's stdout (concatenated across nodes). With `-f`, follow the
stream as it is appended (using the same attach machinery).

## `mirage run [--profile NAME] -- <cmd> [args...]`

Convenience verb that combines `session start`, `exec start`, and (for
transient sessions) `session stop` into one call:

```sh
mirage run --profile cdna4 -- pytest -x my_tests/
```

```text
mirage run [--profile NAME] [--emulator NAME]
           [--session ID] [--keep-session] [--workdir DIR]
           [--env KEY=VALUE]...
           [--image IMAGE] [--mount SPEC]... [--container-provider PROV]
           [--exec-mode functional|clocked] [--option KEY=VALUE]...
           [--config PATH] [--daemon] [--tty auto|always|never]
           -- <cmd> [args...]
```

* `--profile` defaults to the `mi450x` builtin. `--emulator` overrides the
  profile's backend.
* `--session ID` reuses an existing session instead of creating a transient
  one; `--keep-session` keeps a transient session running after the exec
  finishes.
* The container, emulator, and option overrides mirror `session start`.

### Interactive workloads and `--tty`

`--tty auto` (the default) allocates a pseudo-terminal for the workload
when `mirage`'s own stdin and stdout are both terminals. That is what
makes an interactive program work:

```sh
mirage run --profile cdna4 -- bash        # a real shell: prompt, echo,
                                          # line editing, Ctrl-C, job control
mirage run --profile cdna4 -- python3     # a working REPL
```

Resizing your window resizes the workload's terminal, so full-screen
programs redraw correctly.

When output is redirected or piped, `auto` stays on pipes instead, which
keeps it byte-exact and keeps stdout separate from stderr:

```sh
mirage run --profile cdna4 -- ./app > out.txt 2> err.txt   # streams stay split
```

Force either with `--tty always` or `--tty never`. Note that a terminal
has only one stream by construction, so under `--tty always` stderr is
merged into stdout and newlines become CRLF — that is what a terminal is,
not a mirage limitation.

## `mirage state`

```text
mirage state builtins                 # (re)write builtin agents/topologies/profiles
mirage state purge [-f|--force] [--all]
```

* mirage writes any *missing* builtins on every run. `state builtins`
  additionally **overwrites** existing builtins (useful after upgrading).
* `state purge` stops every running session and removes the runtime and state
  directories. The config directory (profiles, topologies, agents)
  is left alone unless `--all` is given.

## `mirage paths`

Print the resolved mirage directories. Useful in scripts and for verifying
that environment overrides took effect.

```sh
$ mirage paths
config:   /home/me/.config/mirage
runtime:  /run/user/1000/mirage
state:    /home/me/.local/state/mirage
profiles: /home/me/.config/mirage/profile
sessions: /run/user/1000/mirage/session
```

## `mirage about`

Print the mirage version, copyright, and the third-party crates (with
licenses) the binary is built from.

## `mirage daemon`

The supervisor daemon owns every session. The CLI starts one automatically
the first time a command needs it, so you normally never run this — see
[`daemon.md`](daemon.md) for the model.

Only session-related commands need it. `paths`, `emulators`, and the
profile/topology/agent commands are answered in-process and start no
daemon.

```text
mirage daemon [serve] [--socket PATH] [--idle-timeout SECONDS] [--addr ADDR]
mirage daemon status                  # pid, uptime, session count
mirage daemon stop                    # destroy every session and exit
mirage daemon install [--addr ADDR]   # register a systemd user service
```

* `--idle-timeout` (default 600) is how long the daemon stays up with no
  sessions and no clients before exiting; `0` disables it. The systemd unit
  installs with it disabled.
* `--addr` additionally serves the HTTP/WebSocket API, and with
  `--features webui` the bundled dashboard SPA. It is backed by the same
  sessions the CLI sees. Also settable with `MIRAGE_WEBUI_ADDR`.
* `webui` is accepted as an alias of `daemon`.

## Drop-in `rocjitsu` mode

For compatibility with the upstream `rocjitsu` CLI, a bare invocation with a
`--` application separator and no recognised subcommand is routed to
`mirage run`:

```sh
mirage --config cfg.json -- ./app           # == mirage run --config cfg.json -- ./app
mirage --attach --config cfg.json -- ./app  # --attach maps to --daemon
```

Invocations that name a subcommand, or that have no `--` separator (so
`--help`/`--version` keep working), are left untouched.

## Environment variables

| Variable                     | Purpose                                                          |
| ---------------------------- | ---------------------------------------------------------------- |
| `MIRAGE_LOG`                 | Tracing-subscriber filter, e.g. `debug` or `mirage_supervisor=debug`. Inherited by an auto-started daemon. |
| `MIRAGE_BIN`                 | Path to the `mirage` binary used to start the daemon. Defaults to the current executable, falling back to `mirage` on `$PATH`. |
| `MIRAGE_SOCKET`              | Control socket path (else `<runtime>/mirage/mirage.sock`). Give each isolated mirage instance its own. |
| `MIRAGE_AUTOSTART`           | Set to `0` to make the CLI report that no daemon is running instead of starting one. |
| `MIRAGE_CONTAINER_PROVIDER`  | Default container provider when none is given (`podman`/`docker`/path). |
| `MIRAGE_CONFIG`              | Override the config dir (else `$XDG_CONFIG_HOME/mirage`).         |
| `MIRAGE_RUNTIME`             | Override the runtime dir — daemon socket, log, and emulator scratch (else `$XDG_RUNTIME_DIR/mirage`). |
| `MIRAGE_STATE`               | Override the state dir (else `$XDG_STATE_HOME/mirage`).           |
| `MIRAGE_WEBUI_ADDR`          | Default bind address for the daemon's HTTP API.                  |
| `XDG_CONFIG_HOME` / `XDG_RUNTIME_DIR` / `XDG_STATE_HOME` | Standard XDG base directories used when the `MIRAGE_*` overrides are unset. |

rocjitsu discovery additionally honours `ROCM_HOME` and the ROCm SDK
install root reported by `rocm-sdk path --root` (see
[`building.md`](building.md)).

## Exit codes

| Code | Meaning                                                              |
| ---- | ------------------------------------------------------------------- |
| 0    | Success.                                                            |
| 1    | A mirage-level error (bad arguments, resource not found, …).        |
| 2    | Argument parse error (clap), or `session wait` timed out/unhealthy. |
| N    | For `run`, `exec start`, and `attach`: the exit code of the exec.   |

## JSON output

`--json` makes list/show commands emit a parseable JSON document:

```sh
$ mirage --json profile list
[
  "cdna4"
]
$ mirage --json session list
[
  {
    "def": { "id": "s-20260616-191636-f6c1", "profile": "cdna4", ... },
    "health": { "healthy": true, "state": "ready", "terminal": false }
  }
]
```
