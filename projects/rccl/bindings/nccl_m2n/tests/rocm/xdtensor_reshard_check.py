# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end DTensor reshard over RCCL, checked against a torch reference.

Trainer TP=N/2 on ranks [0, N/2) -> generator DP=2 x TP=N/4 on [N/2, N)
(8 ranks: TP=4 on 0-3 -> DP=2 x TP=2 on 4-7). Run under MPI with one GPU per
rank and only one librccl in the process (LD_PRELOAD the RCCL libnccl_m2n was
built against):

    mpirun -np 8 python tests/rocm/xdtensor_reshard_check.py --comm rccl
    mpirun -np 8 python tests/rocm/xdtensor_reshard_check.py --comm pg

``--comm rccl`` creates a dedicated ``nccl.core.Communicator`` (nccl4py);
``--comm pg`` reuses the default ProcessGroupNCCL communicator
(``_comm_ptr()``).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

import nccl
import nccl.core as nccl_core
import nccl.m2n as m2n

CASES = [
    # name, src placements, dst placements (on the 2-D dst mesh), global shape
    ("tp_row_to_dp_tp_col", [Shard(0)], [Replicate(), Shard(1)], (4096, 2048)),
    ("tp_col_to_dp_tp_col", [Shard(1)], [Replicate(), Shard(1)], (4096, 2048)),
    ("tp_row_to_dp_tp_row", [Shard(0)], [Replicate(), Shard(0)], (1024, 64, 96)),
]
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int8": torch.int8}


def _env_rank() -> tuple[int, int, int]:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", 0)))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
    local = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", 0))
    )
    return rank, world, local


def _reference(shape, dtype) -> torch.Tensor:
    numel = 1
    for d in shape:
        numel *= d
    base = torch.arange(numel, dtype=torch.int64, device="cuda")
    if dtype.is_floating_point:
        return ((base % 4093).to(torch.float32) * 0.25 - 511.0).to(dtype).view(shape)
    return ((base * 7) % 251 - 125).to(dtype).view(shape)


def _local_slice(full: torch.Tensor, mesh: DeviceMesh, placements, rank: int) -> torch.Tensor:
    ranks = mesh.mesh.flatten().tolist()
    coords = (mesh.mesh == rank).nonzero()[0].tolist()
    out = full
    for axis, placement in enumerate(placements):
        if isinstance(placement, Shard):
            out = out.chunk(mesh.mesh.shape[axis], dim=placement.dim)[coords[axis]]
    assert rank in ranks
    return out.contiguous()


def _mapped_libraries(pattern: str) -> list[str]:
    with open("/proc/self/maps") as maps:
        return sorted({f.split()[5] for f in maps if len(f.split()) >= 6 and pattern in f.split()[5]})


def _rccl_comm(rank: int, world: int) -> nccl_core.Communicator:
    uid = nccl_core.get_unique_id(empty=(rank != 0))
    obj = [uid.as_bytes if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    uid = nccl_core.UniqueId.from_bytes(obj[0])
    return nccl_core.Communicator.init(nranks=world, rank=rank, unique_id=uid)


def _pg_comm() -> int:
    pg = dist.distributed_c10d._get_default_group()
    backend = pg._get_backend(torch.device("cuda"))
    return int(backend._comm_ptr())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm", choices=("rccl", "pg"), default="rccl")
    parser.add_argument("--dtypes", default="bfloat16,float32,int8")
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    rank, world, local = _env_rank()
    assert world % 4 == 0, "need a multiple of 4 ranks"
    torch.cuda.set_device(local)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, device_id=torch.device("cuda", local)
    )

    maps = _mapped_libraries("librccl.so")
    if rank == 0:
        print(f"torch {torch.__version__} hip {torch.version.hip}")
        print(f"torch.cuda.nccl.version() = {torch.cuda.nccl.version()}")
        print(f"nccl.get_version()        = {nccl.get_version()}")
        print(f"librccl mapped: {maps}")
    all_maps = [None] * world
    dist.all_gather_object(all_maps, maps)
    single_rccl = all(len(m) == 1 and m == all_maps[0] for m in all_maps)

    half = world // 2
    src_mesh = DeviceMesh("cuda", torch.arange(0, half))
    dst_mesh = DeviceMesh("cuda", torch.arange(half, world).view(2, half // 2))
    is_src = rank < half

    if args.comm == "pg":
        dist.barrier()
        comm = _pg_comm()
    else:
        comm = _rccl_comm(rank, world)
    if rank == 0:
        print(f"comm source: {args.comm}, ncclComm_t = {getattr(comm, 'ptr', comm):#x}")

    stream = torch.cuda.current_stream()
    failures = 0
    with m2n.init() as handle:
        for dtype_name in args.dtypes.split(","):
            dtype = DTYPES[dtype_name]
            for name, src_pl, dst_pl, shape in CASES:
                full = _reference(shape, dtype)
                ok = True
                for _ in range(args.iters):
                    if is_src:
                        src = DTensor.from_local(
                            _local_slice(full, src_mesh, src_pl, rank),
                            src_mesh, src_pl, run_check=False,
                        )
                        m2n.xdtensor_reshard(
                            src, None, comm, stream,
                            dst_mesh=dst_mesh.mesh.tolist(), dst_placements=dst_pl,
                            handle=handle,
                        )
                    else:
                        expect = _local_slice(full, dst_mesh, dst_pl, rank)
                        dst = DTensor.from_local(
                            torch.full_like(expect, 0), dst_mesh, dst_pl, run_check=False
                        )
                        m2n.xdtensor_reshard(
                            None, dst, comm, stream,
                            src_mesh=src_mesh.mesh.tolist(), src_placements=src_pl,
                            handle=handle,
                        )
                    stream.synchronize()
                    if not is_src:
                        ok &= torch.equal(dst.to_local(), expect)
                flags = [None] * world
                dist.all_gather_object(flags, ok)
                passed = all(flags)
                failures += 0 if passed else 1
                if rank == 0:
                    print(
                        f"{'PASS' if passed else 'FAIL'} {name} {dtype_name} "
                        f"global={shape} src={src_pl} dst={dst_pl}"
                    )
                    sys.stdout.flush()

    dist.barrier()
    # Recheck after the reshard calls: nccl4py, the communicator and m2n.init() load native code lazily.
    maps = _mapped_libraries("librccl.so")
    dist.all_gather_object(all_maps, maps)
    single_rccl = all(len(m) == 1 and m == all_maps[0] for m in all_maps)
    if isinstance(comm, nccl_core.Communicator):
        comm.destroy()
    ok = failures == 0 and single_rccl
    if rank == 0:
        print(f"librccl mapped after reshard: {maps}")
        print(f"single librccl across ranks: {single_rccl}")
        print(f"RESULT {'PASS' if ok else 'FAIL'} failures={failures}")
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
