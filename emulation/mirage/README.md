# mirage

**mirage** is the user-facing UX — a single command-line tool — for the
[`rocjitsu`][rocjitsu] GPU emulator and other emulator backends. It lets you
run real ROCm applications on top of an emulated GPU without changing the
application, and inspect, script, and recover from the emulation as easily
as you read a file.

```sh
$ mirage profile create cdna4 --emulator rocjitsu --agent MI350X
$ mirage run --profile cdna4 -- ./my-rocm-app --flag
```

That second command is the whole system. There is nothing to start
beforehand and nothing to clean up afterwards: `mirage run` brings the
emulated machine up inside its own process, runs your application in your
terminal, and takes everything with it when it exits.

## Why mirage

* **One command, no daemon.** `mirage run` *is* the runtime. There is no
  background supervisor, no service to install, no control socket that
  outlives your shell. The session lives in the address space of the
  command you typed, which is why what mirage tells you about it is true
  rather than inferred from a file someone left behind.
* **Nothing is left behind.** Every workload process is owned by a
  supervisor that always waits on it, runs it in its own process group,
  and escalates `SIGTERM` to `SIGKILL` on teardown. Containers are
  launched with `--rm` and are children of the run, so Ctrl-C, a `kill`,
  or a crashed run all end the same way: no orphans, no zombies, no
  stray containers.
* **Your terminal, not a pipe.** A workload inherits your real stdin,
  stdout and stderr, so `mirage run -- bash` is a real shell: prompt,
  echo, line editing and Ctrl-C all work. There is no pseudo-terminal in
  the way, no output relay, and no stdin forwarding — which is also why
  redirected runs are byte-exact, with stdout and stderr still separate.
* **A second terminal when you want one.** While a run is up it serves a
  socket naming its session, so `mirage exec -- <command>` from another
  window starts a process in that same emulated machine — in *that*
  window, as a child of *that* command. With one run live you don't even
  name it.
* **Configuration lives on disk** in standard [XDG locations][xdg]:
  profiles, agents and topologies are files you can read, edit and check
  into version control. Session state does not: it is owned by the run
  process and disappears with it.
* **Easy to script.** Every list/show command accepts `--json` for
  machine-readable output, and `mirage run` exits with the workload's own
  exit code.
* **A drop-in for `rocjitsu`.** `mirage --config cfg.json -- ./app` works
  just like the upstream `rocjitsu` CLI, so existing scripts keep running.

## Core concepts

| Concept      | What it is                                                                 |
| ------------ | -------------------------------------------------------------------------- |
| **Emulator** | A backend that runs GPU code (`rocjitsu`, `rocjitsu-dbt`, `hotswap`). |
| **Agent**    | A hardware GPU definition (e.g. `MI300X`, `MI350X`, `MI450X`).             |
| **Topology** | A rack/node/GPU layout that references an agent.                           |
| **Profile**  | A reusable preset binding an emulator + topology + options.               |
| **Session**  | The emulated machine a workload runs on. Owned by the `mirage run` that created it, and alive exactly as long as that command is. |
| **Exec**     | One command invocation in a session, running in the terminal that launched it. |

A typical flow is: pick or create a **profile** and `mirage run` a command
against it. That single command creates the session, runs the **exec** in
it, and tears the session down again. If you want a second command in the
same session while the first is still running, `mirage exec` it from
another terminal.

## Quick start

```sh
# See which emulator backends are available on this machine.
mirage emulators

# Create a profile targeting an MI350X with the rocjitsu emulator.
mirage profile create cdna4 --emulator rocjitsu --agent MI350X

# Run a workload. This terminal owns the session for as long as it runs.
mirage run --profile cdna4 -- ./my-rocm-app --flag

# Or get a shell on the emulated machine — a real, interactive one.
mirage run --profile cdna4 -- bash
```

`mirage run` prints the session id it created on stderr
(`mirage: session <id>`). From a **second terminal**, while that run is
still up:

```sh
# Start another process in the same session. It runs here, in this
# terminal, as a child of this command.
mirage exec -- rocm-smi

# With several runs live, say which one you mean.
mirage exec --session <id> -- python -c 'import torch; print(torch.cuda.device_count())'
```

`--session` may be omitted whenever exactly one `mirage run` is live —
one terminal running the job, another exec'ing into it, which is the
case that matters. When it would be ambiguous mirage lists the
candidates instead of guessing.

No physical GPU is needed: `rocjitsu` emulates one in software. You do
need its runtime library — see
[`docs/building.md`](docs/building.md) — and `mirage emulators` reports
whether this machine has it.

## Multiple nodes and ranks

A profile's topology sets how many nodes and GPUs the emulated machine
has; `--num-nodes` and `--gpus-per-node` override it for one run, and
`--nproc-per-node` launches several workload processes per node (like
`torchrun`). Each process gets a distinct `LOCAL_RANK` and global `RANK`,
and `WORLD_SIZE` becomes `num_nodes * nproc_per_node`, so
`torch.distributed` runs without a separate launcher:

```sh
mirage run --profile cdna4 --num-nodes 2 --nproc-per-node 4 \
    --capture-all -- python train.py
```

`--capture-all` is what makes that readable. By default every rank writes
straight to your terminal, which is what keeps a single interactive run
interactive and its output byte-exact — but several ranks writing at once
interleave with nothing to say which wrote what. With `--capture-all`
every rank's output is piped through mirage and printed a line at a time,
prefixed with `[<rank>] `:

```
[0] step 10 loss=6.812
[1] step 10 loss=6.809
[0] step 20 loss=6.114
```

The cost is stdin: under `--capture-all` no rank gets one. Without it,
rank 0 inherits your stdin and the other ranks get `/dev/null`.
`--capture-all` is accepted by both `mirage run` and `mirage exec`.

## Where things live

```sh
mirage paths        # config, runtime, profiles, sessions, runs
```

The `runs` directory is the interesting one: a live run publishes a
socket there named after its session
(`$XDG_RUNTIME_DIR/mirage/run/<session>.sock`), and that is the entire
registry of what is running. One socket per run rather than one
well-known socket for a daemon, so "who owns this session?" is answered
by the filesystem and a run that dies takes its own entry with it.

`mirage state purge` removes the runtime and state directories and
reclaims any container resources a run that died abruptly could not. It
refuses while a run is still live: stopping someone's foreground command
from a cleanup subcommand would be a surprise, and Ctrl-C in its own
terminal already does the job.

## Building

mirage is a single Cargo workspace with no Node.js, no SPA and no
optional server components. One build produces the `mirage` binary:

```sh
cd emulation/mirage
cargo build --workspace          # debug build -> target/debug/mirage
./target/debug/mirage --help
```

By default the `rocjitsu` backend is compiled in. Backends are selected
with Cargo features (`--no-default-features --features hotswap`, and so
on). See [`docs/building.md`](docs/building.md) for the full guide,
including building `rocjitsu` itself.

## Testing

```sh
cargo test --workspace
```

The end-to-end tests in `tests/` drive the real binary as a subprocess
against a private XDG root, so what they exercise is the whole stack: CLI
→ session bring-up → supervisor → real processes. What they check is the
observable contract — a run's streams, its exit code, the socket it
serves while it is alive, `mirage exec` borrowing that session from
another terminal, and that nothing survives the run. The rocjitsu-backed
e2e tests require a working rocjitsu runtime; without it they are
expected to fail with a "KMD preload library not found" message.

## Documentation

* [`docs/cli.md`](docs/cli.md) — complete CLI reference.
* [`docs/architecture.md`](docs/architecture.md) — design and crate overview.
* [`docs/building.md`](docs/building.md) — building mirage and rocjitsu.
* [`docs/state-layout.md`](docs/state-layout.md) — authoritative on-disk layout reference.

[rocjitsu]: ../rocjitsu/
[xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
