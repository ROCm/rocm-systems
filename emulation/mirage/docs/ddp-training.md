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
  subgraph host["mirage host (1 emulated node)"]
    tr["torchrun --nproc_per_node=2"]
    subgraph r0["rank 0"]
      p0["python ddp_mlp.py"] --> g0["emulated GPU 0"]
    end
    subgraph r1["rank 1"]
      p1["python ddp_mlp.py"] --> g1["emulated GPU 1"]
    end
    tr --> p0
    tr --> p1
    p0 <-- "all-reduce grads (gloo)" --> p1
  end
```

Three things make this work:

1. **Multiple emulated GPUs.** `mirage run --gpus-per-node N` tells
   rocjitsu to synthesize `N` KFD device nodes, so inside the workload
   `torch.cuda.device_count() == N`. Each rank pins itself to GPU
   `LOCAL_RANK`.

2. **`torchrun` rendezvous works out of the box.** mirage exports the
   standard `torch.distributed` variables `MASTER_ADDR` and
   `MASTER_PORT` on every node (aliasing mirage's own
   `MIRAGE_HEAD_ADDR` / `MIRAGE_HEAD_PORT`). For a single node,
   `torchrun --standalone` picks its own loopback rendezvous; for
   multi-node you can point `torchrun --rdzv-endpoint` at
   `$MASTER_ADDR:$MASTER_PORT`.

3. **The collective backend.** The model trains on the **emulated GPU**,
   while gradients are all-reduced with the `gloo` backend (host-staged
   collectives). This is the combination that works across *multiple*
   emulated GPUs today — see [Backends](#backends-gloo-vs-nccl).

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
rocm-sdk init      # unpacks librocjitsu_kmd.so + configs into the venv
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
  --env DDP_BACKEND=gloo \
  --env DDP_DEVICE=cuda \
  -- .venv-mi350/bin/torchrun --standalone --nproc_per_node=2 \
     tests/fixtures/ml/ddp_mlp.py
```

The first `--` ends `cargo run`'s arguments; the second ends mirage's,
so everything after it is the workload `torchrun` launches once per
rank.

### 3. Read the output

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

## Backends: gloo vs nccl

`DDP_BACKEND` selects how gradients are all-reduced:

| Backend | 1 emulated GPU | N emulated GPUs | Notes |
|---------|:--:|:--:|-------|
| `gloo` (default) | ✅ | ✅ | Collectives staged through host sockets. Model still runs on the GPU. |
| `nccl` (RCCL) | ✅ | ❌ | Multi-GPU RCCL collectives are not yet serviced by the functional emulator (ranks fault during the first collective). |

Single-GPU `nccl` works, so you can exercise the RCCL init path:

```sh
mirage run --profile mi350x --gpus-per-node 1 \
  --env DDP_BACKEND=nccl \
  -- torchrun --standalone --nproc_per_node=1 ddp_mlp.py
```

For multi-GPU runs, keep the default `gloo` backend — the model still
trains on the emulated GPUs; only the gradient exchange goes over host
sockets.

## Tuning the run

The runner script and fixture honor these environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `NPROC` | `2` | GPUs / ranks per node (`torchrun --nproc_per_node`, `mirage --gpus-per-node`). |
| `BACKEND` | `gloo` | Collective backend (`gloo` or `nccl`). |
| `DEVICE` | `cuda` | Where tensors live (`cuda`, `cpu`, or `auto`). |
| `STEPS` | `50` | Optimizer steps. |
| `PROFILE` | `mi350x` | mirage profile / emulated GPU. |
| `VENV` | `.venv-mi350` | venv location. |
| `SKIP_INSTALL` | unset | set to `1` to reuse an already-populated venv. |

Examples:

```sh
# 4 emulated GPUs, 100 steps
NPROC=4 STEPS=100 SKIP_INSTALL=1 ./tests/run_ddp_mlp_mi350.sh

# CPU-only DDP (no GPU emulation in the math, pure gloo)
DEVICE=cpu SKIP_INSTALL=1 ./tests/run_ddp_mlp_mi350.sh
```

## Multi-node DDP

mirage emulates each node as its own host process. To run DDP across
emulated *nodes* (rather than multiple GPUs in one node), give the
profile more nodes and use a c10d rendezvous pointed at mirage's head:

```sh
mirage run --profile mi350x --num-nodes 2 --gpus-per-node 1 \
  -- torchrun --nnodes=2 --nproc_per_node=1 \
     --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
     ddp_mlp.py
```

`MASTER_ADDR` / `MASTER_PORT` are populated by mirage on every node, so
the workers find the head without extra configuration.

## Troubleshooting

- **`torch.cuda.is_available()` is False.** The venv is missing the ROCm
  runtime or `rocm-sdk init` was not run. Re-run the install step in the
  active venv.
- **A rank SIGSEGV/SIGABRTs during the first collective with `nccl` and
  `--nproc_per_node > 1`.** Expected — multi-GPU RCCL is not yet
  emulated. Use the default `gloo` backend.
- **`KMD preload library not found`.** You are running an installed
  `mirage` from `PATH` instead of the workspace build. Run via
  `cargo run --` from `emulation/mirage` (see
  [building.md](building.md)).
