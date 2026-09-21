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
  and escalates `SIGTERM` to `SIGKILL` on teardown — the whole group, so
  a workload that forked and then exited cleanly does not leave its
  children running. Containers are launched with `--rm` and are children
  of the run, and teardown waits for a bring-up still in flight before it
  decides what to remove, so a Ctrl-C during an image pull takes the
  containers that pull was still creating with it. Every exit path a
  signal can reach ends the same way: no orphans, no zombies, no stray
  containers. `SIGHUP` is one of them, so closing the terminal window on
  a run you have lost interest in takes the run down with it — that is
  the commonest way a run is abandoned, and it used to be the one that
  left everything behind. The remaining exception is deliberate and
  cannot be otherwise: a `SIGKILL` runs no code of mirage's at all, so
  its leftovers are *recovered* rather than prevented — see
  `mirage cleanup` below.
* **Your terminal, not a pipe.** A workload inherits your real stdin,
  stdout and stderr, so `mirage run -- bash` is a real shell: prompt,
  echo, line editing and Ctrl-C all work. There is no pseudo-terminal in
  the way, no output relay, and no stdin forwarding — which is also why
  redirected runs are byte-exact, with stdout and stderr still separate.
  Your environment comes with it too: whatever you exported is what the
  workload sees, with the emulator's variables layered on top. Pass
  `--clear-env-vars` for the strict, ambient-free version.
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
  exit code. Exit codes distinguish the answers a script has to act on
  differently: a confirmation you declined exits 2 rather than 0, so
  "you said no" is not mistaken for "it is done", and a prompt that
  reaches end-of-file with nobody to answer it is an error naming
  `--force` rather than a silent "no".
* **A drop-in for `rocjitsu`.** `mirage --config cfg.json -- ./app` works
  just like the upstream `rocjitsu` CLI, so existing scripts keep
  running — including `--attach`, which mirage takes as a spelling of
  `--daemon`. A flag before the `--` that mirage does not accept is a
  typo rather than a program to run, and is refused before anything
  boots.

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
candidates instead of guessing. "Live" means a run that answers when
connected to, not a socket file left on disk, so a run you `kill -9`ed
neither breaks the guess nor turns up among the candidates.

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
mirage run --profile cdna4 --num-nodes 2 --nproc-per-node 4 -- python train.py
```

### One process gets a terminal; many get labels

The shape of the job decides how output works — there is no flag,
because there is no version of this you would want the other way.

**One process** — `mirage run -- bash` on a one-node profile — gets your
terminal whole. Its stdin, stdout and stderr *are* yours, so a shell
prints a prompt, echoes what you type and edits its line; redirection is
byte-exact and stdout stays separate from stderr. Mirage is not in the
middle at all.

**More than one** and every rank's output is piped through mirage and
printed a line at a time, prefixed with the rank that wrote it — the same
thing `docker compose up` does, for the same reason:

```
[0] step 10 loss=6.812
[1] step 10 loss=6.809
[0] step 20 loss=6.114
```

No rank gets stdin in that mode. One terminal cannot be shared between
readers, and quietly handing it to rank 0 would mean keystrokes going
somewhere you cannot see. If you piped something in, mirage says once
that nothing will read it: an immediate end-of-file in every rank is
otherwise indistinguishable from a broken pipe or a bad path.

### A terminal on one node of a multi-node job

Start a second one there:

```sh
mirage exec --node 2 -- bash
```

That leaves a single process, so it takes the first branch: a real
interactive shell, in the window you ran it from, on node 2 of the
running session. The process still believes it is that node — same rank
variables, same `WORLD_SIZE`, same rendezvous as its neighbours — so it
is a shell *inside* the job rather than beside it.

`--node` says *where*, not *how many*. Add `--nproc-per-node 3` and you
get three processes on node 2, captured and labelled like any other
grid — the process count decides the terminal, and it wins over having
named a node.

## Where things live

```sh
mirage paths        # config, runtime, profiles, topologies, agents,
                    # sessions, runs
```

The `runs` directory is the interesting one: a live run publishes a
socket there named after its session
(`$XDG_RUNTIME_DIR/mirage/run/<session>.sock`), and that is the entire
registry of what is running. One socket per run rather than one
well-known socket for a daemon, so "who owns this session?" is answered
by the filesystem and a run that dies takes its own entry with it.

`mirage cleanup` reclaims what a run that died abruptly could not: its
containers and per-session network, its workload processes, and its
scratch directory. That is not a hypothetical tidy-up — a `SIGKILL`ed
run, an OOM kill, or a machine losing power runs no code of mirage's at
all, so the leak is expected and this is the recovery. Sessions whose run
still answers are left completely alone, as is anything mirage did not
create, so it is safe to run at any time; `--dry-run` shows what it would
remove first. Everything it looks for is marked on the resource itself —
`mirage.owner`, `mirage.session` and `mirage.runtime` labels on containers
and networks, `MIRAGE_SESSION` and `MIRAGE_RUNTIME` in every workload's
environment — because that is what survives the death of the only process
that knew about it. Every container engine on the machine is asked, not
just the first one autodetection finds, and each line of the report says
which engine is holding what.

The runtime mark is what keeps this safe when two mirages share a
machine. The sockets in one runtime directory are the whole registry of
what is live under it, so a cleanup run there has never heard of a
session belonging to a different `MIRAGE_RUNTIME` and would otherwise
read a perfectly healthy job as wreckage. It reclaims only what its own
runtime directory created, and skips anything that does not say which
runtime directory that was — including work left by a mirage older than
the mark, which has to be cleaned up by hand.

`mirage state purge` is the blunter version: it does all of the above and
then removes the runtime directory (`--all` takes the config directory
with it). There is no separate state directory: mirage writes nothing
that has to survive a reboot beyond its configuration. It refuses while a
run is still live, where `cleanup` simply skips it: stopping someone's
foreground command from a cleanup subcommand would be a surprise, and
Ctrl-C in its own terminal already does the job.

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
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The lint policy is denied, not advised — but only when clippy is what you
ran, so the last two lines are not optional. Both are `ctest` cases too;
see [`docs/building.md`](docs/building.md).

The end-to-end tests in `tests/` drive the real binary as a subprocess
against a private XDG root, so what they exercise is the whole stack: CLI
→ session bring-up → supervisor → real processes. What they check is the
observable contract — a run's streams, its exit code, the socket it
serves while it is alive, `mirage exec` borrowing that session from
another terminal, and that nothing survives the run. The rocjitsu-backed
e2e tests require a working rocjitsu runtime; without it they print
`SKIP: the rocjitsu runtime was not found` and one guard test per suite
fails, so a build without the emulator cannot quietly report a green
suite. See [`docs/building.md`](docs/building.md) for
`MIRAGE_E2E_ALLOW_SKIP` / `-DMIRAGE_ALLOW_TEST_SKIP=ON`, which turn that
guard off deliberately.

## Documentation

* [`docs/cli.md`](docs/cli.md) — complete CLI reference.
* [`docs/architecture.md`](docs/architecture.md) — design and crate overview.
* [`docs/building.md`](docs/building.md) — building mirage and rocjitsu.
* [`docs/state-layout.md`](docs/state-layout.md) — authoritative on-disk layout reference.
* [`docs/ddp-training.md`](docs/ddp-training.md) — tutorial: PyTorch DDP
  across several emulated MI350X GPUs, with `torchrun`.
* [`docs/demo-debugging-at-scale.md`](docs/demo-debugging-at-scale.md) —
  presentation deck on debugging ROCm at scale with mirage and rocjitsu.

[rocjitsu]: ../rocjitsu/
[xdg]: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
