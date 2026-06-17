"""Train a tiny MLP with PyTorch DistributedDataParallel (DDP).

This is the workload half of the DDP-on-mirage demo. It is launched once
per rank -- normally by ``torchrun`` -- and each rank:

  1. Reads its rank / world-size / local-rank from the standard
     ``torch.distributed`` environment variables (``RANK``,
     ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``, ``MASTER_PORT``).
     ``torchrun`` populates the first three; mirage exports
     ``MASTER_ADDR``/``MASTER_PORT`` on every node so multi-node
     rendezvous works without extra flags.
  2. Pins itself to GPU ``LOCAL_RANK`` (one emulated GPU per rank).
  3. Builds an identical MLP, wraps it in ``DistributedDataParallel`` so
     gradients are all-reduced across ranks every backward pass.
  4. Trains on a fixed synthetic regression task for a few steps.
  5. Sanity-checks that training actually worked: the loss dropped and
     every rank ended with byte-identical weights (DDP keeps replicas in
     lock-step). Rank 0 then prints ``ddp_mlp_ok``.

Exit codes:
  * 0 + rank 0 prints ``ddp_mlp_ok``  -> success
  * non-zero on any failure (no silent fallback)

Environment knobs:
  * ``DDP_BACKEND``  collective backend: ``nccl`` (RCCL on ROCm) or
    ``gloo`` (default; CPU-staged collectives). ``gloo`` still trains the
    model on the GPU -- only the gradient all-reduce goes through host
    sockets -- which is the combination that runs across *multiple*
    emulated GPUs today.
  * ``DDP_DEVICE``   where tensors live: ``auto`` (default; GPU when one
    is visible, else CPU), ``cuda``, or ``cpu``.
  * ``DDP_STEPS``    number of optimizer steps (default: 50).
  * ``DDP_SEED``     base RNG seed (default: 0).
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def log(msg: str) -> None:
    rank = os.environ.get("RANK", "?")
    print(f"[rank {rank}] {msg}", flush=True)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


class MLP(nn.Module):
    """A small 3-layer perceptron for a synthetic regression task."""

    def __init__(self, in_features: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> int:
    backend = os.environ.get("DDP_BACKEND", "gloo").lower()
    steps = env_int("DDP_STEPS", 50)
    seed = env_int("DDP_SEED", 0)

    # torchrun sets these; fall back to a single-process run otherwise.
    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)

    # The collective backend (how gradients are all-reduced) is chosen
    # independently of where the model runs. On the emulator the model
    # trains on the GPU while `gloo` carries the collective over host
    # sockets -- the combination that works across multiple emulated GPUs.
    want = os.environ.get("DDP_DEVICE", "auto").lower()
    use_cuda = want != "cpu" and torch.cuda.is_available()
    if want == "cuda" and not use_cuda:
        print("FAIL: DDP_DEVICE=cuda but torch.cuda.is_available() is False", file=sys.stderr)
        return 2
    if use_cuda:
        if local_rank >= torch.cuda.device_count():
            print(
                f"FAIL: local_rank {local_rank} >= device_count "
                f"{torch.cuda.device_count()}",
                file=sys.stderr,
            )
            return 2
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # Rendezvous. With WORLD_SIZE==1 we still go through the process group
    # so the DDP code path is identical to the multi-rank case.
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    log(
        f"joined process group: backend={backend} world_size={world_size} "
        f"device={device} master={os.environ.get('MASTER_ADDR')}:"
        f"{os.environ.get('MASTER_PORT')}"
    )

    try:
        # Every rank builds the SAME initial model (same seed) so DDP's
        # broadcast-on-construction has matching replicas to align.
        torch.manual_seed(seed)
        model = MLP().to(device)
        ddp_model = DDP(
            model,
            device_ids=[local_rank] if use_cuda else None,
            output_device=local_rank if use_cuda else None,
        )

        # A fixed linear target so the loss is deterministic and must drop.
        torch.manual_seed(1234)
        weight = torch.randn(32, 1, device=device)
        bias = torch.randn(1, device=device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-2)

        # Each rank trains on its own shard of synthetic data, seeded by
        # rank, so gradients genuinely differ and the all-reduce matters.
        batch = 64
        first_loss = None
        last_loss = None
        for step in range(steps):
            torch.manual_seed(seed + 1000 * rank + step)
            x = torch.randn(batch, 32, device=device)
            y = x @ weight + bias

            optimizer.zero_grad()
            pred = ddp_model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            if first_loss is None:
                first_loss = last_loss

        log(f"loss: {first_loss:.4f} -> {last_loss:.4f} over {steps} steps")

        # Sanity check 1: the model actually learned (loss decreased).
        if not (last_loss < first_loss):
            print(
                f"FAIL: loss did not decrease ({first_loss:.4f} -> {last_loss:.4f})",
                file=sys.stderr,
            )
            return 3

        # Sanity check 2: DDP kept every replica in lock-step. After
        # synchronized SGD all ranks must hold byte-identical parameters.
        # All-reduce a checksum of the flattened weights and compare to
        # rank 0's value broadcast to everyone.
        with torch.no_grad():
            checksum = sum(p.sum() for p in ddp_model.parameters()).to(device)
            gathered = [torch.zeros_like(checksum) for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather(gathered, checksum)
                ref = gathered[0]
                for r, value in enumerate(gathered):
                    if not torch.allclose(value, ref, rtol=0, atol=1e-5):
                        print(
                            f"FAIL: rank {r} weights diverged from rank 0 "
                            f"({value.item()} != {ref.item()})",
                            file=sys.stderr,
                        )
                        return 4

        dist.barrier()
        if rank == 0:
            log("all ranks converged with identical replicas")
            print("ddp_mlp_ok", flush=True)
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
