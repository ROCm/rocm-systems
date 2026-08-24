# Distributed training with PyTorch DDP on mirage

This tutorial trains a small MLP with PyTorch
[`DistributedDataParallel`](https://pytorch.org/docs/stable/notes/ddp.html)
(DDP) across **multiple emulated MI350X GPUs**, launched with
[`torchrun`](https://pytorch.org/docs/stable/elastic/run.html). No
physical GPU is required — [`rocjitsu`](architecture.md) emulates the
devices, so the same launch command you would use on a real 8-GPU box
runs on a laptop.

By the end you will have run:

```sh
mirage run --profile mi350x --gpus-per-node 2 \
  -- torchrun --standalone --nproc_per_node=2 ddp_mlp.py
```

and watched two ranks all-reduce their gradients and converge to
byte-identical weights.

The ready-to-run pieces live in the repo:

| File | Purpose |
|------|---------|
| [`tests/fixtures/ml/ddp_mlp.py`](../tests/fixtures/ml/ddp_mlp.py) | the DDP training workload (one process per rank) |
| [`tests/run_ddp_mlp_mi350.sh`](../tests/run_ddp_mlp_mi350.sh) | end-to-end runner: sets up a venv and launches the demo |

## TL;DR

```sh
cd emulation/mirage
./tests/run_ddp_mlp_mi350.sh          # first run installs a venv; prints ddp_mlp_ok
SKIP_INSTALL=1 ./tests/run_ddp_mlp_mi350.sh   # reuse the venv on later runs
```

A successful run ends with:

```text
[rank 0] loss: 30.5666 -> 0.2634 over 50 steps
[rank 0] all ranks converged with identical replicas
ddp_mlp_ok
==> PASS: DDP MLP trained on 2 emulated mi350x GPUs via torchrun
```

## How the pieces fit together

```mermaid
flowchart LR
  subgraph host["mirage run (1 emulated node)"]
    tr["torchrun --nproc_per_node=2"]
    subgraph r0["rank 0"]
      p0["python ddp_mlp.py"] --> g0["emulated GPU 0"]
    end
    subgraph r1["rank 1"]
      p1["python ddp_mlp.py"] --> g1["emulated GPU 1"]
    end
    tr --> p0
    tr --> p1
    p0 <-- "all-reduce grads (RCCL)" --> p1
  end
```

Three things make this work:

1. **Multiple emulated GPUs.** `mirage run --gpus-per-node N` tells
   rocjitsu to synthesize `N` KFD device nodes, so inside the workload
   `torch.cuda.device_count() == N`. Every rank must pin itself to a
   *distinct* device — RCCL refuses to run if two ranks land on the same
   GPU (`Duplicate GPU detected`). The fixture picks
   `rank % torch.cuda.device_count()`, so each rank owns its own GPU.

2. **`torchrun` rendezvous works out of the box.** mirage exports the
   standard `torch.distributed` variables `MASTER_ADDR` and
   `MASTER_PORT` on every node (aliasing mirage's own
   `MIRAGE_HEAD_ADDR` / `MIRAGE_HEAD_PORT`). For a single node,
   `torchrun --standalone` picks its own loopback rendezvous; for
   multi-node you can point `torchrun --rdzv-endpoint` at
   `$MASTER_ADDR:$MASTER_PORT`.

3. **One emulator shared by every rank (default).** mirage hosts a single
   emulator daemon that every rank talks to over a socket, so the rank
   processes share GPU memory through it — which is what lets RCCL set up
   its transports across ranks. It is *out of the workload's* process,
   not a service of its own: the daemon is `dlopen`ed into the `mirage
   run` process itself, so there is nothing else to start, nothing to
   leave behind, and the session lives and dies with that one command. The mode is the
   default; `--daemon` asks for it explicitly (and is the spelling the
   `rocjitsu` drop-in accepts). Pass `mirage run --in-process` to instead
   give every process its own in-process emulator (no shared GPU memory;
   multi-GPU RCCL cannot work in that mode).

## Step by step

### 1. Set up a venv

The demo uses PyTorch and the ROCm SDK (which ships rocjitsu) from the
`gfx950-dcgpu` (MI350) nightly index. The runner script does this for
you, but to do it by hand:

```sh
cd emulation/mirage
python3 -m venv .venv-mi350
source .venv-mi350/bin/activate
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ \
  "rocm[libraries,devel]" torch numpy
rocm-sdk init      # unpacks librocjitsu.so + configs into the venv
```

`rocm-sdk init` is required: the `devel` package ships its contents
compressed, and rocjitsu's KMD interposer only appears under
`site-packages/_rocm_sdk_devel` after this step. mirage auto-detects
that library from the active venv — no `LD_LIBRARY_PATH` wiring needed.

### 2. Launch the training

From the workspace (so mirage's own rocjitsu discovery works), run:

```sh
cargo run --quiet -- run \
  --profile mi350x \
  --gpus-per-node 2 \
  -- .venv-mi350/bin/torchrun --standalone --nproc_per_node=2 \
     tests/fixtures/ml/ddp_mlp.py
```

The first `--` ends `cargo run`'s arguments; the second ends mirage's,
so everything after it is the workload `torchrun` launches once per
rank. mirage already runs the shared out-of-process emulator by default
(pass `--in-process` to opt out).

The command runs in the foreground and *is* the session: mirage prints
`mirage: session <id>` on stderr as it comes up, and everything it
brought up is gone when the command returns.

### 3. Read the output

By default the workload inherits this terminal. Its processes write to
your stdout and stderr directly — mirage never sees the bytes — so
redirection behaves exactly as it would without mirage (`> train.log`
captures stdout and leaves the emulator's warnings on stderr), and rank 0
also inherits stdin, which is what makes `mirage run -- bash` an
interactive shell.

That is the right default here: `torchrun` is what forks the ranks, so
mirage is supervising a single process, and the fixture tags its own
lines with `[rank N]`. As soon as *mirage* is the thing launching several
ranks — the multi-node case below — mirage labels every rank's lines
itself.

Each rank logs the loss it started and ended with, and rank 0 confirms
that every replica converged to identical weights before printing
`ddp_mlp_ok`. The fixture fails loudly (non-zero exit) if the loss does
not drop or any rank's weights diverge.

## What the workload does

[`ddp_mlp.py`](../tests/fixtures/ml/ddp_mlp.py) is a standard,
self-contained DDP program:

1. Reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` (set by `torchrun`) and
   `MASTER_ADDR` / `MASTER_PORT` (set by mirage).
2. Pins to GPU `LOCAL_RANK` and joins the process group.
3. Builds an identical MLP on every rank, wraps it in
   `DistributedDataParallel` (which broadcasts the initial weights and
   all-reduces gradients every backward pass).
4. Trains on a per-rank shard of a fixed synthetic regression task, so
   the gradients genuinely differ and the all-reduce matters.
5. Verifies the loss dropped **and** that an `all_gather` of the final
   weight checksum is identical on every rank — proof that DDP kept the
   replicas in lock-step.

## Tuning the run

The runner script and fixture honor these environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `NPROC` | `2` | GPUs / ranks per node (`torchrun --nproc_per_node`, `mirage --gpus-per-node`). |
| `STEPS` | `50` | Optimizer steps. |
| `PROFILE` | `mi350x` | mirage profile / emulated GPU. |
| `VENV` | `.venv-mi350` | venv location. |
| `SKIP_INSTALL` | unset | set to `1` to reuse an already-populated venv. |

Examples:

```sh
# 4 emulated GPUs, 100 steps
NPROC=4 STEPS=100 SKIP_INSTALL=1 ./tests/run_ddp_mlp_mi350.sh
```

## Multi-node DDP (no launcher)

mirage emulates each *node* as its own rank process and exports the full
set of `torch.distributed` `env://` variables — `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT` — on every node. That means
you can run the workload **directly**, with no `torchrun` launcher: each
node runs `python ddp_mlp.py` once and rendezvouses through `env://`.

```sh
mirage run --profile mi350x --num-nodes 2 --gpus-per-node 2 \
  --env NCCL_P2P_DISABLE=1 --env NCCL_SHM_DISABLE=1 --env NCCL_SOCKET_IFNAME=lo \
  -- .venv-mi350/bin/python3 tests/fixtures/ml/ddp_mlp.py
```

- `--num-nodes 2` makes `WORLD_SIZE == 2` and runs the script twice, once
  per rank, with distinct `RANK`s.
- `--gpus-per-node 2` exposes two GPUs so each rank can pin to a distinct
  device (`rank % device_count`).
- Every rank's output is piped through mirage, which prefixes
  each line with the rank that wrote it.
- The `NCCL_*` variables force RCCL onto its loopback socket transport,
  which is what the shared emulator supports.

### Why the output is labelled here

Without it, every rank writes straight to the terminal. That is what
keeps output byte-exact and a shell interactive, but two ranks writing at
once interleave with nothing to say which wrote what — and "which rank
stopped printing" is the single most useful fact when a collective
stalls. Capturing labels every line:

```text
[0] [rank 0] joined process group: world_size=2 device=cuda:0 master=127.0.0.1:29500
[1] [rank 1] joined process group: world_size=2 device=cuda:1 master=127.0.0.1:29500
[0] [rank 0] loss: 30.5666 -> 0.2634 over 50 steps
[1] [rank 1] loss: 28.9142 -> 0.2571 over 50 steps
[0] [rank 0] all ranks converged with identical replicas
[0] ddp_mlp_ok
```

The outer `[0]`/`[1]` is mirage's label — the global rank of the process
it read the line from. The inner `[rank 0]` is the fixture printing its
own `RANK`; they agree here because each node runs one process.

Two properties are worth knowing before you reach for it:

- **stdout and stderr stay separate.** Lines are labelled on the way
  through and written back to the stream they came from, so `2>` still
  splits them.
- **No rank gets stdin.** Capturing costs you the terminal, so
  a multi-rank job is not something you can be interactive with — use
  `mirage exec --node N -- bash` for a terminal on one of its nodes.

Increase `--nproc-per-node` and the labels keep counting globally:
`--num-nodes 2 --nproc-per-node 2` gives `WORLD_SIZE == 4` and ranks
`[0]` through `[3]`, with ranks 0–1 on node 0 and 2–3 on node 1. Give
each node at least `--nproc-per-node` GPUs so every process can pin its
own device.

### Quick connectivity check

Before the full training run, a minimal smoke test
([`tests/fixtures/ml/dist_smoke.py`](../tests/fixtures/ml/dist_smoke.py))
does a single `all_reduce` and prints `dist_smoke_ok`. It is heavily
logged with timestamps so any stall is easy to localize:

```sh
mirage run --profile mi350x --num-nodes 2 --gpus-per-node 2 \
  --env NCCL_P2P_DISABLE=1 --env NCCL_SHM_DISABLE=1 --env NCCL_SOCKET_IFNAME=lo \
  -- .venv-mi350/bin/python3 tests/fixtures/ml/dist_smoke.py
```

A successful run shows each rank on its own device and the reduced sum:

```text
[0] [rank 0] current device = 0 (cuda:0)
[1] [rank 1] current device = 1 (cuda:1)
[0] [rank 0] post all_reduce: tensor = 3.0
[0] dist_smoke_ok
```

If it stalls instead, the mirage label tells you which rank went quiet,
and the fixture's own timestamps tell you where.

### Poking at a live run from another terminal

A session exists exactly as long as the `mirage run` that created it. For
as long as one is up, another terminal can start a process inside it:

```sh
# terminal 1: owns the session for the length of the training run
mirage run --profile mi350x --num-nodes 2 --gpus-per-node 2 \
  -- .venv-mi350/bin/python3 tests/fixtures/ml/ddp_mlp.py

# terminal 2: joins it while it runs
mirage exec -- .venv-mi350/bin/python3 -c \
  'import torch; print(torch.cuda.device_count())'
```

`mirage exec` needs no session id while exactly one run is live; name one
with `-s <id>` (the id `mirage run` printed) when several are. It takes
the same `--node`, `--env`, `--nproc-per-node` and `--workdir`
flags as `run`, and the process it starts is a child of *terminal 2*, on
terminal 2's streams — so `mirage exec -- bash` really is an interactive
shell inside the emulated node. When the training in terminal 1 exits,
the session goes with it and further execs fail.

## Troubleshooting

- **`torch.cuda.is_available()` is False.** The venv is missing the ROCm
  runtime or `rocm-sdk init` was not run. Re-run the install step in the
  active venv.
- **A rank SIGSEGV/SIGABRTs immediately with `--nproc_per_node > 1`.**
  You are likely in `--in-process` mode, where each rank has its own
  emulator and cannot share GPU memory. Drop `--in-process` so the
  default shared, out-of-process emulator is used.
- **Several ranks' lines are interleaved and you cannot tell them
  apart.** When mirage is the thing launching the ranks it already
  labels every line with the rank that wrote it. A job `torchrun` forks
  is a single process as far as mirage can see, so any labels there are
  the fixture's own — launch the ranks with `--num-nodes` /
  `--nproc-per-node` instead if you want mirage's. There is no flag
  either way: the shape of the job decides it, and the price of the
  labels is that no rank gets stdin.
- **`Duplicate GPU detected` / `ncclInvalidUsage` at the first
  collective.** Two ranks pinned to the same emulated GPU. Give the
  run at least as many GPUs as ranks (`--gpus-per-node N`) so each
  rank can own a distinct device (the fixtures use
  `rank % torch.cuda.device_count()`).
- **Multi-GPU run hangs / times out after a few steps.** Known
  limitation: RCCL initializes and a single collective completes (the
  `dist_smoke.py` smoke test passes), but DDP's repeated, bucketed
  collectives stall on a later one over the shared emulator. Single-rank
  runs train end to end. The `[rank]` labels show which rank
  stopped first.
- **`no mirage run is running` from `mirage exec`.** Nothing owns a
  session: there is no background service to fall back on, so a run has
  to be live in another terminal before you can exec into it.
- **`KMD preload library not found`.** You are running an installed
  `mirage` from `PATH` instead of the workspace build. Run via
  `cargo run --` from `emulation/mirage` (see
  [building.md](building.md)).
