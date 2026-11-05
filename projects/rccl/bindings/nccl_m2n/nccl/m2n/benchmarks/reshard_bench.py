# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Python NCCL M2N reshard benchmark.

This mirrors ``benchmarks/reshard_bench.cc`` closely enough for
apples-to-apples C++/Python comparisons.  The default path reuses one
``nccl.m2n.Handle`` and calls ``Handle.reshard`` in the timed loop, matching
the C++ ``reshard_bench --api default`` lifecycle.  ``--python-api reshard``
measures the public ``reshard`` helper separately.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time
from collections.abc import Sequence

import mpi4py

mpi4py.rc.thread_level = "single"
from mpi4py import MPI

try:
    import torch
except ImportError:  # pragma: no cover - exercised without PyTorch installed.
    torch = None

try:
    from cuda.bindings import runtime as cudart
    import cuda.core as cuda_core
except ImportError:  # pragma: no cover - raw CUDA buffers require cuda-python.
    cudart = None
    cuda_core = None

import nccl.core as nccl
from nccl.m2n import DistTensor, Handle, Mesh, Replicate, Shard, finalize, reshard


def _parse_mesh_dims(value: str) -> tuple[int, int]:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("mesh dims must be '<rep>,<shard>'")
    dims = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(f"mesh dims must be positive, got {dims}")
    return dims


def _parse_tensor_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part) for part in value.replace("x", ",").split(","))
    if len(dims) not in (2, 3):
        raise argparse.ArgumentTypeError("tensor dims must be 2-D or 3-D")
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(f"tensor dims must be positive, got {dims}")
    return dims


def _local_rank(comm: MPI.Comm) -> int:
    for name in (
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
        "SLURM_LOCALID",
    ):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue

    rank = comm.Get_rank()
    host = socket.gethostname()
    hosts = comm.allgather(host)
    return sum(1 for peer_host in hosts[:rank] if peer_host == host)


def _shared_unique_id(comm: MPI.Comm) -> object:
    rank = comm.Get_rank()
    uid = nccl.get_unique_id(empty=(rank != 0))
    comm.Bcast([uid.as_ndarray, MPI.BYTE], root=0)
    return uid


def _cuda_call(result: object, name: str) -> object | None:
    if isinstance(result, tuple):
        err = result[0]
        values = result[1:]
    else:
        err = result
        values = ()
    if int(err) != 0:
        raise RuntimeError(f"{name} failed with CUDA error {err}")
    if len(values) == 1:
        return values[0]
    if values:
        return values
    return None


def _require_cudart() -> None:
    if cudart is None:
        raise RuntimeError("raw buffer provider requires cuda.bindings.runtime")


def _set_cuda_device(comm: MPI.Comm, buffer_provider: str) -> int:
    local_rank = _local_rank(comm)
    if buffer_provider == "torch":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("torch buffer provider requires torch with CUDA")
        device_count = torch.cuda.device_count()
    else:
        _require_cudart()
        device_count = int(_cuda_call(cudart.cudaGetDeviceCount(), "cudaGetDeviceCount"))
    if device_count <= 0:
        raise RuntimeError("no visible CUDA devices")
    device_id = local_rank % device_count
    if buffer_provider == "torch":
        torch.cuda.set_device(device_id)
    else:
        _cuda_call(cudart.cudaSetDevice(device_id), "cudaSetDevice")
    if cuda_core is not None:
        cuda_core.Device(device_id).set_current()
    return device_id


def _product(values: Sequence[int]) -> int:
    return math.prod(int(value) for value in values)


def _local_shape(
    global_shape: Sequence[int],
    shard_dim: int,
    shard_count: int,
) -> tuple[int, ...]:
    shape = list(global_shape)
    if shape[shard_dim] % shard_count != 0:
        raise ValueError(
            f"tensor dim {shard_dim} size {shape[shard_dim]} is not divisible "
            f"by shard count {shard_count}"
        )
    shape[shard_dim] //= shard_count
    return tuple(shape)


def _pattern(
    local_shape: Sequence[int],
    shard_dim: int,
    shard_idx: int,
    shard_count: int,
    device: torch.device,
) -> torch.Tensor:
    ndims = len(local_shape)
    dims = list(local_shape)
    dim0 = int(dims[0])
    dim1 = int(dims[1])
    dim2 = int(dims[2]) if ndims == 3 else 1
    total = _product(local_shape)

    idx = torch.arange(total, device=device, dtype=torch.int64)
    if ndims == 3:
        d2 = idx.remainder(dim2)
        rem = torch.div(idx, dim2, rounding_mode="floor")
    else:
        d2 = 0
        rem = idx
    d1 = rem.remainder(dim1)
    d0 = torch.div(rem, dim1, rounding_mode="floor")

    global_start = [0, 0, 0]
    global_dims = [dim0, dim1, dim2]
    global_start[shard_dim] = shard_idx * int(local_shape[shard_dim])
    global_dims[shard_dim] = int(local_shape[shard_dim]) * shard_count

    values = (
        d0
        + global_start[0]
        + (d1 + global_start[1]) * global_dims[1]
        + (d2 + global_start[2]) * global_dims[1] * global_dims[2]
    ).remainder(256).to(torch.int16)
    values = torch.where(values >= 128, values - 256, values)
    return values.to(torch.int8).view(tuple(local_shape))


def _init_source_data(
    buffer: torch.Tensor,
    local_shape: Sequence[int],
    shard_dim: int,
    shard_idx: int,
    shard_count: int,
) -> None:
    view = buffer[: _product(local_shape)].view(tuple(local_shape))
    view.copy_(
        _pattern(
            local_shape,
            shard_dim,
            shard_idx,
            shard_count,
            torch.device("cuda", torch.cuda.current_device()),
        )
    )


def _validate_dest_data(
    buffer: torch.Tensor,
    local_shape: Sequence[int],
    shard_dim: int,
    shard_idx: int,
    shard_count: int,
    rank: int,
) -> bool:
    view = buffer[: _product(local_shape)].view(tuple(local_shape))
    expected = _pattern(
        local_shape,
        shard_dim,
        shard_idx,
        shard_count,
        torch.device("cuda", torch.cuda.current_device()),
    )
    mismatch = view != expected
    if not bool(mismatch.any().item()):
        print(f"[Rank {rank}] VALIDATION PASSED: {_product(local_shape)} bytes correct")
        return True

    flat_mismatch = mismatch.flatten()
    first_error_idx = int(torch.nonzero(flat_mismatch, as_tuple=False)[0].item())
    actual = int(view.flatten()[first_error_idx].item()) & 0xFF
    expect = int(expected.flatten()[first_error_idx].item()) & 0xFF
    errors = int(flat_mismatch.sum().item())
    print(
        f"[Rank {rank}] VALIDATION FAILED: {errors} errors, first at idx "
        f"{first_error_idx} (expected 0x{expect:02x}, got 0x{actual:02x})"
    )
    return False


def _allocate_buffer(api: str, size: int, device_id: int, buffer_provider: str) -> object:
    if buffer_provider == "raw":
        if api == "window":
            raise RuntimeError("raw buffer provider supports --api default only")
        _require_cudart()
        _cuda_call(cudart.cudaSetDevice(device_id), "cudaSetDevice")
        ptr = int(_cuda_call(cudart.cudaMalloc(size), "cudaMalloc"))
        _cuda_call(cudart.cudaMemset(ptr, 0xDE, size), "cudaMemset")
        _cuda_call(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        return ptr

    if api == "window":
        return nccl.torch.empty(size, dtype=torch.int8, device=f"cuda:{device_id}")
    return torch.empty(size, dtype=torch.int8, device=f"cuda:{device_id}")


def _make_dist_tensors(
    buffer: object,
    is_source: bool,
    is_dest: bool,
    src_shape: Sequence[int],
    dst_shape: Sequence[int],
    src_mesh: Mesh,
    dst_mesh: Mesh,
    src_shard_dim: int,
    dst_shard_dim: int,
) -> tuple[DistTensor, DistTensor]:
    src_desc = DistTensor(
        buffer if is_source else None,
        local_shape=src_shape,
        dtype="int8",
        mesh=src_mesh,
        placements=[Replicate(), Shard(src_shard_dim)],
    )
    dst_desc = DistTensor(
        buffer if is_dest else None,
        local_shape=dst_shape,
        dtype="int8",
        mesh=dst_mesh,
        placements=[Replicate(), Shard(dst_shard_dim)],
    )
    return src_desc, dst_desc


def _make_public_tensors(
    buffer: object,
    is_source: bool,
    is_dest: bool,
    src_shape: Sequence[int],
    dst_shape: Sequence[int],
) -> tuple[object | None, object | None]:
    if isinstance(buffer, int):
        return (buffer if is_source else None, buffer if is_dest else None)

    src = buffer[: _product(src_shape)].view(tuple(src_shape)) if is_source else None
    dst = buffer[: _product(dst_shape)].view(tuple(dst_shape)) if is_dest else None
    return src, dst


def _synchronize(stream: object | None, buffer_provider: str) -> None:
    if buffer_provider == "raw":
        _require_cudart()
        if stream is None:
            _cuda_call(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        else:
            _cuda_call(cudart.cudaStreamSynchronize(int(stream)), "cudaStreamSynchronize")
        return

    if stream is None:
        torch.cuda.synchronize()
    else:
        stream.synchronize()


def _free_buffer(buffer: object, buffer_provider: str) -> None:
    if buffer_provider == "raw" and isinstance(buffer, int):
        _require_cudart()
        _cuda_call(cudart.cudaFree(buffer), "cudaFree")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Python NCCL M2N reshard benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-mesh-dims", required=True, type=_parse_mesh_dims)
    parser.add_argument("--dst-mesh-dims", required=True, type=_parse_mesh_dims)
    parser.add_argument("--tensor-dims", required=True, type=_parse_tensor_dims)
    parser.add_argument("--src-shard-dim", required=True, type=int)
    parser.add_argument("--dst-shard-dim", required=True, type=int)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--algorithm", choices=("ring", "direct"), default="ring")
    parser.add_argument("--api", choices=("default", "window"), default="default")
    parser.add_argument(
        "--python-api",
        choices=("lowlevel", "reshard"),
        default="lowlevel",
        help=(
            "lowlevel reuses one nccl.m2n.Handle; reshard calls the "
            "public reshard helper once per iteration"
        ),
    )
    parser.add_argument("--copy-algorithm", choices=("direct", "pack", "pipe"))
    parser.add_argument("--lb-mode", choices=("uniform", "node"), default="uniform")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-all-ranks", action="store_true")
    parser.add_argument("--use-default-stream", action="store_true")
    parser.add_argument(
        "--buffer-provider",
        choices=("torch", "raw"),
        default="torch",
        help="Use torch tensors or raw CUDA Runtime pointers for benchmark buffers",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if args.api == "window" and args.python_api == "reshard":
        if mpi_rank == 0:
            print("ERROR: --python-api reshard only supports --api default")
        return 1
    if args.buffer_provider == "raw":
        if args.api != "default":
            if mpi_rank == 0:
                print("ERROR: --buffer-provider raw only supports --api default")
            return 1
        if args.validate:
            if mpi_rank == 0:
                print("ERROR: --buffer-provider raw does not support --validate")
            return 1
        if not args.use_default_stream:
            if mpi_rank == 0:
                print(
                    "ERROR: --buffer-provider raw requires --use-default-stream"
                )
            return 1

    src_mesh_dims = args.src_mesh_dims
    dst_mesh_dims = args.dst_mesh_dims
    global_shape = args.tensor_dims
    ndims = len(global_shape)
    if not 0 <= args.src_shard_dim < ndims or not 0 <= args.dst_shard_dim < ndims:
        if mpi_rank == 0:
            print("ERROR: shard dimensions must be in range for tensor rank")
        return 1
    if args.iterations <= 0 or args.warmup < 0:
        if mpi_rank == 0:
            print("ERROR: --iterations must be positive and --warmup non-negative")
        return 1

    src_total = src_mesh_dims[0] * src_mesh_dims[1]
    dst_total = dst_mesh_dims[0] * dst_mesh_dims[1]
    total_expected = src_total + dst_total
    if mpi_size != total_expected:
        if mpi_rank == 0:
            print(
                f"ERROR: Expected {total_expected} processes "
                f"(src={src_total} + dst={dst_total}), got {mpi_size}"
            )
        return 1

    os.environ["NCCL_RESHARD_ALGORITHM"] = args.algorithm.upper()
    os.environ["NCCL_RESHARD_LB_MODE"] = (
        "NODE_AWARE" if args.lb_mode == "node" else "UNIFORM"
    )
    if args.verbose:
        os.environ["NCCL_RESHARD_LOG_LEVEL"] = "DEBUG"
    if args.api == "default":
        if args.copy_algorithm is not None:
            os.environ["NCCL_RESHARD_COPY_ALGORITHM"] = (
                args.copy_algorithm.upper()
            )
        elif "NCCL_RESHARD_COPY_ALGORITHM" not in os.environ:
            os.environ["NCCL_RESHARD_COPY_ALGORITHM"] = (
                "DIRECT" if args.algorithm == "direct" else "PIPE"
            )

    src_shape = _local_shape(global_shape, args.src_shard_dim, src_mesh_dims[1])
    dst_shape = _local_shape(global_shape, args.dst_shard_dim, dst_mesh_dims[1])
    src_buffer_size = _product(src_shape)
    dst_buffer_size = _product(dst_shape)
    alloc_size = max(src_buffer_size, dst_buffer_size, 4096)

    device_id = _set_cuda_device(mpi_comm, args.buffer_provider)
    uid = _shared_unique_id(mpi_comm)
    comm = nccl.Communicator.init(nranks=mpi_size, rank=mpi_rank, unique_id=uid)
    m2n_handle: Handle | None = None
    window = None
    buffer: object | None = None

    is_source = mpi_rank < src_total
    is_dest = mpi_rank >= src_total
    src_mesh = Mesh(src_mesh_dims, start_rank=0)
    dst_mesh = Mesh(dst_mesh_dims, start_rank=src_total)
    stream = None if args.use_default_stream else torch.cuda.Stream()
    call_stream = stream

    validation_rc = 0
    try:
        buffer = _allocate_buffer(args.api, alloc_size, device_id, args.buffer_provider)
        if args.buffer_provider == "torch":
            buffer.fill_(-34)
            torch.cuda.synchronize()

        if args.api == "window":
            window = comm.register_window(buffer, flags=nccl.WindowFlag.COLL_SYMMETRIC)
            if window is None:
                raise RuntimeError("NCCL window registration returned None")

        src_desc, dst_desc = _make_dist_tensors(
            buffer,
            is_source,
            is_dest,
            src_shape,
            dst_shape,
            src_mesh,
            dst_mesh,
            args.src_shard_dim,
            args.dst_shard_dim,
        )
        public_src, public_dst = _make_public_tensors(
            buffer,
            is_source,
            is_dest,
            src_shape,
            dst_shape,
        )

        if is_source and args.validate:
            local_rank = mpi_rank - src_mesh.start_rank
            shard_idx = local_rank % src_mesh_dims[1]
            _init_source_data(
                buffer,
                src_shape,
                args.src_shard_dim,
                shard_idx,
                src_mesh_dims[1],
            )
            torch.cuda.synchronize()

        mpi_comm.Barrier()

        if args.python_api == "lowlevel":
            m2n_handle = Handle.create()

        if mpi_rank == 0:
            print("=== Python Tensor Reshard Benchmark ===")
            if args.python_api == "reshard":
                print("Using: nccl.m2n.reshard (public Python API)")
            elif args.api == "default":
                ca = os.environ.get("NCCL_RESHARD_COPY_ALGORITHM", "PIPE")
                print(
                    "Using: nccl.m2n.Handle.reshard "
                    f"(default API, copy-algorithm={ca})"
                )
            else:
                print(
                    "Using: nccl.m2n.Handle.reshard_with_window "
                    "(user window API)"
                )
            print(f"Global tensor: {list(global_shape)} ({ndims}D)")
            suffix = (
                " (same-dim)"
                if args.src_shard_dim == args.dst_shard_dim
                else " (CROSS-DIM!)"
            )
            print(
                f"Source shard dim: {args.src_shard_dim}, "
                f"Dest shard dim: {args.dst_shard_dim}{suffix}"
            )
            print(
                f"Source: {src_total} ranks = {src_mesh_dims[0]} reps x "
                f"{src_mesh_dims[1]} shards, local={list(src_shape)}"
            )
            print(
                f"Dest: {dst_total} ranks = {dst_mesh_dims[0]} reps x "
                f"{dst_mesh_dims[1]} shards, local={list(dst_shape)}"
            )
            print(f"Algorithm: {args.algorithm.upper()}")
            if args.algorithm == "ring":
                print(
                    "Load Balance Mode: "
                    f"{'NODE_AWARE' if args.lb_mode == 'node' else 'UNIFORM'}"
                )
            print(
                f"Iterations: {args.iterations} (warmup: {args.warmup}), "
                f"Validate: {'yes' if args.validate else 'no'}"
            )
            sys.stdout.flush()

        def run_one_iteration() -> None:
            if args.python_api == "reshard":
                raw_kwargs = {}
                if args.buffer_provider == "raw":
                    if public_src is not None:
                        raw_kwargs["src_local_shape"] = src_shape
                        raw_kwargs["src_dtype"] = "int8"
                    if public_dst is not None:
                        raw_kwargs["dst_local_shape"] = dst_shape
                        raw_kwargs["dst_dtype"] = "int8"
                reshard(
                    public_src,
                    public_dst,
                    comm,
                    stream=call_stream,
                    src_mesh=src_mesh,
                    src_placements=[Replicate(), Shard(args.src_shard_dim)],
                    dst_mesh=dst_mesh,
                    dst_placements=[Replicate(), Shard(args.dst_shard_dim)],
                    **raw_kwargs,
                )
            elif args.api == "default":
                if m2n_handle is None:
                    raise AssertionError("unreachable")
                m2n_handle.reshard(comm, src_desc, dst_desc, stream=call_stream)
            else:
                if m2n_handle is None or window is None:
                    raise AssertionError("unreachable")
                m2n_handle.reshard_with_window(
                    comm,
                    window,
                    src_desc,
                    dst_desc,
                    stream=call_stream,
                )

        def run_one_iteration_checked() -> None:
            try:
                run_one_iteration()
                _synchronize(stream, args.buffer_provider)
                mpi_comm.Barrier()
            except Exception as exc:
                print(
                    f"[Rank {mpi_rank}] ERROR: benchmark iteration failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                mpi_comm.Abort(1)
                raise

        if mpi_rank == 0:
            print(f"\nRunning {args.warmup} warmup iterations...")
            sys.stdout.flush()

        for _ in range(args.warmup):
            run_one_iteration_checked()

        if mpi_rank == 0:
            print("Warmup complete.")
            sys.stdout.flush()

        if args.validate:
            if args.warmup == 0:
                run_one_iteration_checked()

            local_valid = True
            if is_dest:
                local_rank = mpi_rank - dst_mesh.start_rank
                shard_idx = local_rank % dst_mesh_dims[1]
                local_valid = _validate_dest_data(
                    buffer,
                    dst_shape,
                    args.dst_shard_dim,
                    shard_idx,
                    dst_mesh_dims[1],
                    mpi_rank,
                )

            global_result = mpi_comm.allreduce(1 if local_valid else 0, op=MPI.MIN)
            if global_result == 0:
                if mpi_rank == 0:
                    print("\n*** VALIDATION FAILED ***\n")
                validation_rc = 1
            else:
                if mpi_rank == 0:
                    print("\n*** VALIDATION PASSED ***\n")

            if is_dest:
                buffer[:dst_buffer_size].fill_(-34)
                torch.cuda.synchronize()
            mpi_comm.Barrier()

        if mpi_rank == 0:
            print(f"\nRunning {args.iterations} timed iterations...")
            sys.stdout.flush()

        mpi_comm.Barrier()
        start = time.perf_counter()
        for _ in range(args.iterations):
            run_one_iteration_checked()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        avg_time_ms = elapsed_ms / args.iterations

        total_data = _product(global_shape)
        my_data = src_buffer_size if is_source else dst_buffer_size
        my_bw_gbps = float(my_data) / (avg_time_ms / 1000.0) / (1024.0**3)

        bw_min = mpi_comm.reduce(my_bw_gbps, op=MPI.MIN, root=0)
        bw_max = mpi_comm.reduce(my_bw_gbps, op=MPI.MAX, root=0)
        bw_sum = mpi_comm.reduce(my_bw_gbps, op=MPI.SUM, root=0)
        time_min = mpi_comm.reduce(avg_time_ms, op=MPI.MIN, root=0)
        time_max = mpi_comm.reduce(avg_time_ms, op=MPI.MAX, root=0)

        source_bw_min = mpi_comm.reduce(
            my_bw_gbps if is_source else 1.0e20, op=MPI.MIN, root=0
        )
        source_bw_max = mpi_comm.reduce(
            my_bw_gbps if is_source else -1.0e20, op=MPI.MAX, root=0
        )
        source_bw_sum = mpi_comm.reduce(
            my_bw_gbps if is_source else 0.0, op=MPI.SUM, root=0
        )
        source_time_min = mpi_comm.reduce(
            avg_time_ms if is_source else 1.0e20, op=MPI.MIN, root=0
        )
        source_time_max = mpi_comm.reduce(
            avg_time_ms if is_source else -1.0e20, op=MPI.MAX, root=0
        )

        dest_bw_min = mpi_comm.reduce(
            my_bw_gbps if is_dest else 1.0e20, op=MPI.MIN, root=0
        )
        dest_bw_max = mpi_comm.reduce(
            my_bw_gbps if is_dest else -1.0e20, op=MPI.MAX, root=0
        )
        dest_bw_sum = mpi_comm.reduce(
            my_bw_gbps if is_dest else 0.0, op=MPI.SUM, root=0
        )
        dest_time_min = mpi_comm.reduce(
            avg_time_ms if is_dest else 1.0e20, op=MPI.MIN, root=0
        )
        dest_time_max = mpi_comm.reduce(
            avg_time_ms if is_dest else -1.0e20, op=MPI.MAX, root=0
        )

        if args.print_all_ranks:
            for rank in range(mpi_size):
                if mpi_rank == rank:
                    role = "Source" if is_source else "Dest  "
                    print(
                        f"[Rank {mpi_rank:3d}] {role}: time={avg_time_ms:.3f} ms, "
                        f"bw={my_bw_gbps:.2f} GB/s"
                    )
                    sys.stdout.flush()
                mpi_comm.Barrier()

        if mpi_rank == 0:
            total_throughput_gbps = (
                float(total_data) / (time_max / 1000.0) / (1024.0**3)
            )

            print("\n=================================")
            print("       BENCHMARK RESULTS")
            print("=================================")
            print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
            print(
                f"Total data: {total_data} bytes "
                f"({total_data / (1024.0 * 1024.0):.2f} MB)"
            )
            print(f"Sources: {src_total} ranks, Destinations: {dst_total} ranks")
            shard_kind = (
                "same-dim"
                if args.src_shard_dim == args.dst_shard_dim
                else "cross-dim"
            )
            print(
                f"Sharding: src_dim={args.src_shard_dim}, "
                f"dst_dim={args.dst_shard_dim} ({shard_kind})"
            )

            print("\n--- Overall (all ranks) ---")
            print(
                "Time per iteration (ms):  "
                f"Min={time_min:.3f}  Max={time_max:.3f}"
            )
            print(
                "Bandwidth (GB/s):         "
                f"Min={bw_min:.2f}  Max={bw_max:.2f}  Avg={bw_sum / mpi_size:.2f}"
            )

            print(f"\n--- Sources only ({src_total} ranks) ---")
            print(
                "Time per iteration (ms):  "
                f"Min={source_time_min:.3f}  Max={source_time_max:.3f}"
            )
            print(
                "Bandwidth (GB/s):         "
                f"Min={source_bw_min:.2f}  Max={source_bw_max:.2f}  "
                f"Avg={source_bw_sum / src_total:.2f}"
            )

            print(f"\n--- Destinations only ({dst_total} ranks) ---")
            print(
                "Time per iteration (ms):  "
                f"Min={dest_time_min:.3f}  Max={dest_time_max:.3f}"
            )
            print(
                "Bandwidth (GB/s):         "
                f"Min={dest_bw_min:.2f}  Max={dest_bw_max:.2f}  "
                f"Avg={dest_bw_sum / dst_total:.2f}"
            )

            print("\n--- Effective bandwidth ---")
            print(f"Total data throughput: {total_throughput_gbps:.2f} GB/s")
            print("=================================")
            print(
                "CSV_RESULT,"
                f"python,{args.python_api},{args.api},{args.algorithm},"
                f"{'same' if shard_kind == 'same-dim' else 'cross'},"
                f"\"{','.join(str(dim) for dim in global_shape)}\","
                f"{total_throughput_gbps:.6f},{bw_sum / mpi_size:.6f},"
                f"{time_max:.6f},{'PASS' if validation_rc == 0 else 'FAIL'}"
            )
            sys.stdout.flush()

    finally:
        if m2n_handle is not None:
            m2n_handle.destroy()
        finalize()
        if window is not None:
            window.close()
        if buffer is not None:
            _free_buffer(buffer, args.buffer_provider)
        comm.destroy()

    if mpi_rank == 0:
        if validation_rc == 0:
            print("\nBenchmark completed successfully!")
        else:
            print("\nBenchmark completed with VALIDATION FAILURES.")
    return validation_rc


if __name__ == "__main__":
    raise SystemExit(main())
