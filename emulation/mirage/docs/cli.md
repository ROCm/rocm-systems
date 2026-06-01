# mirage CLI reference

`mirage` is split into a small set of verbs that each manage a single
kind of resource. Every command supports `--help` and most support
`--json` for machine-readable output.

## Global options

| Flag          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `--json`      | Emit JSON output where applicable.                           |
| `-v`, `-vv`   | Increase logging verbosity (also `MIRAGE_LOG=info\|debug`).  |
| `--help`      | Per-command help.                                            |
| `--version`   | Print version.                                               |

## `mirage paths`

Print the resolved XDG paths.  Useful for scripts and for verifying
that overrides like `XDG_RUNTIME_DIR` are taking effect.

```sh
$ mirage paths
config:   /home/me/.config
runtime:  /run/user/1000
state:    /home/me/.local/state
profiles: /home/me/.config/mirage/profile
sessions: /run/user/1000/mirage/session
```

## `mirage profile`

Profiles are reusable emulator presets stored in
`$XDG_CONFIG_HOME/mirage/profile/<name>.json`.

```text
mirage profile list [-l]
mirage profile show <name>
mirage profile create <name> [--emulator NAME] [--nodes N]
                              [--gpus-per-node N] [--description TEXT]
mirage profile import <file>            # use '-' for stdin
mirage profile delete <name> [-f]
```

`profile import` accepts any valid `ProfileDef` JSON document.

## `mirage session`

Sessions are the long-lived contexts in which execs run.

```text
mirage session list
mirage session show <id>
mirage session wait <id> [--timeout SECONDS]
mirage session start --profile NAME [--id ID] [--workdir DIR]
                     [--no-host] [--host-bin PATH]
                     [--ready-timeout SECONDS]
mirage session stop  <id> [-f]
mirage session dir   <id>     # print the on-disk directory
```

* `session start` creates the session and spawns `mirage-host --session
  <id>` in a new process group, then waits up to `--ready-timeout`
  seconds for the host to publish `healthy=true`.
* `--no-host` is useful for tests or when you want to start the host
  yourself (for example, in a container).
* `session stop` sends `SIGTERM` to the host, waits briefly, escalates
  to `SIGKILL` if needed, and removes the session directory.

## `mirage exec`

```text
mirage exec list   <session>
mirage exec show   <session> <exec>
mirage exec start  <session> [--keep] [--detach] -- <cmd> [args...]
mirage exec signal <session> <exec> [SIG]        # default: TERM
mirage exec remove <session> <exec>
```

`exec start` is the workhorse:

* By default it attaches to the new exec's stdout/stderr and exits
  with the exec's exit code, then cleans up the on-disk exec directory.
* `--detach` returns the exec id immediately without attaching.
* `--keep` preserves the exec directory after it exits so you can
  re-read its logs with `mirage logs` or `mirage exec show`.
* Everything after `--` is passed verbatim to the command.

## `mirage attach <session> <exec>`

Re-attach to an exec that is still running (or to one that has
finished, in which case the buffered output is replayed and the saved
exit code is returned). Useful after a `--detach`.

## `mirage logs <session> <exec> [-f] [--stdout|--stderr]`

Show the contents of `node/0/stdout` and `node/0/stderr`. With `-f`,
follow the streams (uses the same attach machinery).

## `mirage run --profile NAME -- <cmd> [args...]`

Convenience verb that combines `session start`, `exec start`, and
`session stop` into one call. Useful for one-shot invocations:

```sh
mirage run --profile cdna3 -- pytest -x my_tests/
```

Flags:

* `--session ID` reuse an existing session instead of creating one.
* `--keep-session` keeps the (transient) session running after the exec
  finishes. Only meaningful when mirage created the session.
* `--workdir DIR` sets the working directory.

## Environment variables

| Variable               | Purpose                                                   |
| ---------------------- | --------------------------------------------------------- |
| `MIRAGE_HOST_BIN`      | Override the `mirage-host` binary path used when spawning |
|                        | per-session hosts. Defaults to the binary next to         |
|                        | `mirage`, falling back to `mirage-host` on `$PATH`.       |
| `MIRAGE_LOG`           | Tracing-subscriber filter (e.g. `mirage_host=debug`).     |
| `XDG_CONFIG_HOME`      | Where profiles are stored.                                |
| `XDG_RUNTIME_DIR`      | Where sessions are stored.                                |
| `XDG_STATE_HOME`       | Reserved for future persistent-state needs.               |

## Exit codes

| Code | Meaning                                                           |
| ---- | ----------------------------------------------------------------- |
| 0    | Success.                                                          |
| 1    | A mirage-level error (bad arguments, session not found, …).      |
| 2    | `session wait` timed out without the host becoming healthy.       |
| N    | For `run`, `exec start`, and `attach`: the exit code of the exec. |

## JSON output

`--json` makes every list/show command emit a parseable JSON document.
For example:

```sh
$ mirage --json profile list
[
  "cdna3"
]
$ mirage --json session list
[
  {
    "def": { "id": "s-20260530-153012-abcd", ... },
    "health": { "healthy": true, "state": "ready", ... }
  }
]
```
