"""Ultra-simple torch.distributed smoke test with verbose logging.

Launched once per node by mirage (one process per node). mirage exports
RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT on every node,
so this script uses the standard ``env://`` rendezvous with no launcher.

Each rank:
  1. Logs the distributed environment it sees.
  2. Pins to its GPU and initializes the NCCL (RCCL on ROCm) group.
  3. Runs a single all_reduce and checks the result.
  4. Rank 0 prints ``dist_smoke_ok`` on success.

Every step is logged with a timestamp so a hang is easy to localize.
"""

import datetime as _dt
import os
import sys

import torch
import torch.distributed as dist


def log(msg: str) -> None:
    rank = os.environ.get("RANK", "?")
    ts = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [rank {rank}] {msg}", flush=True)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def main() -> int:
    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)
    master_addr = os.environ.get("MASTER_ADDR", "<unset>")
    master_port = os.environ.get("MASTER_PORT", "<unset>")

    log(
        f"env: RANK={rank} WORLD_SIZE={world_size} LOCAL_RANK={local_rank} "
        f"MASTER_ADDR={master_addr} MASTER_PORT={master_port}"
    )

    log("torch.cuda.is_available() ...")
    cuda_ok = torch.cuda.is_available()
    log(f"torch.cuda.is_available() = {cuda_ok}")
    if not cuda_ok:
        log("FAIL: CUDA/HIP device not available")
        return 2

    device_count = torch.cuda.device_count()
    log(f"device_count = {device_count}")
    # Pin each rank to its own GPU. With one GPU per node this is device 0;
    # when several ranks share a host that exposes multiple GPUs, spread
    # them across distinct devices so each rank owns a different GPU.
    device_index = local_rank % device_count if device_count else 0
    log(f"setting cuda device {device_index} ...")
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    log(f"current device = {torch.cuda.current_device()} ({device})")

    log("init_process_group(backend='nccl', init_method='env://') ...")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device_index)
    log("init_process_group returned")

    log(f"is_initialized={dist.is_initialized()} "
        f"get_rank={dist.get_rank()} get_world_size={dist.get_world_size()}")

    try:
        # Build a tensor whose value depends on rank, all_reduce(SUM), and
        # check it equals the sum 0+1+...+(world_size-1).
        log("allocating tensor on device ...")
        t = torch.ones(1, device=device) * (rank + 1)
        log(f"pre  all_reduce: tensor = {t.item()}")

        log("all_reduce(SUM) ...")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        log(f"post all_reduce: tensor = {t.item()}")

        expected = sum(r + 1 for r in range(world_size))
        if abs(t.item() - expected) > 1e-3:
            log(f"FAIL: all_reduce = {t.item()}, expected {expected}")
            return 3
        log(f"all_reduce correct (= {expected})")

        log("barrier ...")
        dist.barrier()
        log("barrier returned")

        if rank == 0:
            log("all ranks reduced successfully")
            print("dist_smoke_ok", flush=True)
        return 0
    finally:
        log("destroy_process_group ...")
        dist.destroy_process_group()
        log("destroy_process_group returned")


if __name__ == "__main__":
    sys.exit(main())
