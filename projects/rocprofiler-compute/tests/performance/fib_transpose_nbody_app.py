#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""HipRTC Fibonacci ping-pong of tiled transpose and mini-nbody.

Shared device buffers: LDS tiled matrix transpose writes dst from src then
swaps; update_bodies runs in-place on the current src (nbody input =
transpose output; nbody output is the next transpose input). Schedule is
1t,1n,2t,2n,3t,3n,5t,5n,... until --iters launches.

The transpose kernel follows the LDS tiling approach from
ROCm/rocm-examples tiling_matrix_transpose (tile_dim=16).
"""

import argparse
import math
import random
import sys
from ctypes import POINTER, byref, c_float, c_int, c_void_p, cast
from pathlib import Path
from typing import Any, List, Optional, Tuple

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import utils.hip_interface as hip  # noqa: E402
import utils.hiprtc_interface as hiprtc  # noqa: E402

_KERNEL_SOURCE = r"""
constexpr int kTileDim = 16;

// LDS tiled matrix transpose (coalesced load/store via shared tile).
extern "C" __global__ void transpose_lds(
    float *in, float *out, int width, int height)
{
    __shared__ float tile[kTileDim][kTileDim];

    int x_tile_index = blockIdx.x * kTileDim;
    int y_tile_index = blockIdx.y * kTileDim;

    int in_index =
        (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
    int out_index =
        (x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

    tile[threadIdx.y][threadIdx.x] = in[in_index];

    __syncthreads();

    out[out_index] = tile[threadIdx.x][threadIdx.y];
}

constexpr int kBlockSize = 128;
constexpr float kSoftening = 1e-9f;

extern "C" __global__ void update_bodies(
    const float *pos_x,
    const float *pos_y,
    const float *pos_z,
    float *vel_x,
    float *vel_y,
    float *vel_z,
    float step,
    int body_count)
{
    __shared__ float3 tile_positions[kBlockSize];

    int body = blockDim.x * blockIdx.x + threadIdx.x;
    bool valid_body = body < body_count;
    float force_x = 0.0f;
    float force_y = 0.0f;
    float force_z = 0.0f;
    float origin_x = valid_body ? pos_x[body] : 0.0f;
    float origin_y = valid_body ? pos_y[body] : 0.0f;
    float origin_z = valid_body ? pos_z[body] : 0.0f;

    for (int tile_start = 0; tile_start < body_count; tile_start += kBlockSize) {
        int other = tile_start + threadIdx.x;
        int tile_size = body_count - tile_start;
        tile_size = tile_size < kBlockSize ? tile_size : kBlockSize;
        if (threadIdx.x < tile_size) {
            tile_positions[threadIdx.x] =
                make_float3(pos_x[other], pos_y[other], pos_z[other]);
        }
        __syncthreads();

        if (valid_body) {
            for (int tile_index = 0; tile_index < tile_size; ++tile_index) {
                float delta_x = tile_positions[tile_index].x - origin_x;
                float delta_y = tile_positions[tile_index].y - origin_y;
                float delta_z = tile_positions[tile_index].z - origin_z;
                float distance_squared =
                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
                    + kSoftening;
                float inverse_distance = rsqrtf(distance_squared);
                float scale =
                    inverse_distance * inverse_distance * inverse_distance;
                force_x += delta_x * scale;
                force_y += delta_y * scale;
                force_z += delta_z * scale;
            }
        }
        __syncthreads();
    }

    if (valid_body) {
        vel_x[body] += step * force_x;
        vel_y[body] += step * force_y;
        vel_z[body] += step * force_z;
    }
}
"""

_COORDINATES_PER_BODY = 6
_TILE_DIM = 16
_NBODY_BLOCK = 128
_STEP = 0.01
# 32768^2 floats x 4 bytes x 2 ping-pong buffers = 8 GiB device memory.
_DEFAULT_MATRIX_DIM = 32768


class HiprtcProgram:
    """Compile HipRTC source and resolve kernels by name."""

    def __init__(self, source: str) -> None:
        self._prog = hiprtc.hiprtcCreateProgram(source, "fib_transpose_nbody")
        try:
            hiprtc.hiprtcCompileProgram(self._prog)
        except hiprtc.HIPRTCError:
            log = hiprtc.hiprtcGetProgramLog(self._prog)
            print(f"HipRTC compile log:\n{log}", file=sys.stderr)
            raise
        self._code = hiprtc.hiprtcGetCode(self._prog)
        self._module = hip.hipModuleLoadData(self._code)

    def get_kernel(self, name: str) -> Any:
        return hip.hipModuleGetFunction(self._module, name)


def build_fibonacci_launch_kinds(total_launches: int) -> List[str]:
    """Return kinds for 1t,1n,2t,2n,3t,3n,5t,... truncated to total_launches."""
    kinds = []  # type: List[str]
    fib = 1
    next_fib = 2
    while len(kinds) < total_launches:
        for _ in range(fib):
            if len(kinds) >= total_launches:
                break
            kinds.append("transpose")
        for _ in range(fib):
            if len(kinds) >= total_launches:
                break
            kinds.append("nbody")
        fib, next_fib = next_fib, fib + next_fib
    return kinds


def matrix_side_for_bodies(body_count: int, tile_dim: int) -> int:
    """Smallest tile-aligned square side that holds body SoA floats."""
    min_elements = body_count * _COORDINATES_PER_BODY
    side = int(math.ceil(math.sqrt(float(min_elements))))
    return ((side + tile_dim - 1) // tile_dim) * tile_dim


def transpose_launch_geometry(
    matrix_dim: int,
    tile_dim: int,
    max_grid_size: Tuple[int, int, int],
    max_threads_per_block: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Grid/block that covers every matrix element; rejects undersized launches."""
    if matrix_dim < tile_dim or matrix_dim % tile_dim != 0:
        raise ValueError(
            f"matrix_dim={matrix_dim} must be a positive multiple of "
            f"tile_dim={tile_dim}"
        )
    threads_per_block = tile_dim * tile_dim
    if threads_per_block > max_threads_per_block:
        raise RuntimeError(
            f"transpose block {tile_dim}x{tile_dim}={threads_per_block} threads "
            f"exceeds device maxThreadsPerBlock={max_threads_per_block}"
        )
    blocks_per_side = matrix_dim // tile_dim
    grid = (blocks_per_side, blocks_per_side, 1)
    block = (tile_dim, tile_dim, 1)
    covered_elements = (blocks_per_side * tile_dim) * (blocks_per_side * tile_dim)
    matrix_elements = matrix_dim * matrix_dim
    if covered_elements != matrix_elements:
        raise RuntimeError(
            f"transpose launch covers {covered_elements} elements, "
            f"matrix has {matrix_elements}"
        )
    if blocks_per_side > max_grid_size[0] or blocks_per_side > max_grid_size[1]:
        raise RuntimeError(
            f"transpose grid {grid} exceeds device maxGridSize="
            f"({max_grid_size[0]}, {max_grid_size[1]}, {max_grid_size[2]}); "
            f"reduce --matrix-dim"
        )
    return grid, block


def _device_byte_offset(memory: hip.HIPDeviceMemory, float_offset: int) -> c_void_p:
    base = memory.ptr.value
    if base is None:
        raise RuntimeError("device pointer is null")
    return c_void_p(base + float_offset * 4)


def _soa_pointers(memory: hip.HIPDeviceMemory, body_count: int) -> List[c_void_p]:
    return [
        _device_byte_offset(memory, body_count * index)
        for index in range(_COORDINATES_PER_BODY)
    ]


def _launch_kernel(
    func: Any,
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    args: List[Any],
) -> None:
    converted = []  # type: List[Any]
    for arg in args:
        if isinstance(arg, int):
            converted.append(c_int(arg))
        elif isinstance(arg, float):
            converted.append(c_float(arg))
        elif isinstance(arg, hip.HIPDeviceMemory):
            converted.append(arg.ptr)
        else:
            converted.append(arg)
    normalized = [cast(byref(arg), c_void_p) for arg in converted]
    args_ptr = (c_void_p * len(args))(*normalized)
    hip.hipModuleLaunchKernel(
        func,
        grid[0],
        grid[1],
        grid[2],
        block[0],
        block[1],
        block[2],
        0,
        None,
        args_ptr,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("HipRTC Fibonacci tiled transpose + nbody ping-pong workload")
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5000,
        help="Total kernel launches (default: 5000)",
    )
    parser.add_argument(
        "--bodies",
        type=int,
        default=81920,
        help="N-body particle count / SoA length (default: 81920)",
    )
    parser.add_argument(
        "--matrix-dim",
        type=int,
        default=_DEFAULT_MATRIX_DIM,
        help=(
            "Square matrix side for tiled transpose "
            "(default: 32768, ~8 GiB for two buffers; multiple of 16)"
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="HIP device id (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.iters < 1:
        print("--iters must be >= 1", file=sys.stderr)
        return 1
    if args.bodies < 1:
        print("--bodies must be >= 1", file=sys.stderr)
        return 1
    if args.matrix_dim < _TILE_DIM or args.matrix_dim % _TILE_DIM != 0:
        print(
            f"--matrix-dim must be a positive multiple of {_TILE_DIM}",
            file=sys.stderr,
        )
        return 1

    soa_floats = args.bodies * _COORDINATES_PER_BODY
    matrix_elements = args.matrix_dim * args.matrix_dim
    if matrix_elements < soa_floats:
        min_side = matrix_side_for_bodies(args.bodies, _TILE_DIM)
        print(
            f"--matrix-dim={args.matrix_dim} is too small for "
            f"--bodies={args.bodies} (need side >= {min_side})",
            file=sys.stderr,
        )
        return 1

    hip.hipSetDevice(args.device)
    device_props = hip.hipGetDeviceProperties(args.device)
    max_grid_size = (
        int(device_props.maxGridSize[0]),
        int(device_props.maxGridSize[1]),
        int(device_props.maxGridSize[2]),
    )
    try:
        transpose_grid, transpose_block = transpose_launch_geometry(
            args.matrix_dim,
            _TILE_DIM,
            max_grid_size,
            int(device_props.maxThreadsPerBlock),
        )
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    buffer_bytes = matrix_elements * 4
    buffer_gib = (2 * buffer_bytes) / float(1024**3)
    covered_elements = (
        transpose_grid[0] * transpose_block[0] * transpose_grid[1] * transpose_block[1]
    )
    if covered_elements != matrix_elements:
        print(
            f"internal error: launch covers {covered_elements} of "
            f"{matrix_elements} elements",
            file=sys.stderr,
        )
        return 1

    program = HiprtcProgram(_KERNEL_SOURCE)
    transpose = program.get_kernel("transpose_lds")
    update_bodies = program.get_kernel("update_bodies")

    # Avoid a multi-GiB host staging buffer: zero the matrices on device, then
    # upload only the SoA prefix used by nbody.
    buffer_a = hip.hipMalloc(buffer_bytes)
    buffer_b = hip.hipMalloc(buffer_bytes)
    hip.hipMemset(buffer_a, 0, buffer_bytes)
    hip.hipMemset(buffer_b, 0, buffer_bytes)
    host_soa = (c_float * soa_floats)()
    rng = random.Random(0)
    for index in range(soa_floats):
        host_soa[index] = 2.0 * rng.random() - 1.0
    soa_bytes = soa_floats * 4
    hip.hipMemcpyHtoD(buffer_a, cast(host_soa, POINTER(c_float)), soa_bytes)
    hip.hipMemcpyHtoD(buffer_b, cast(host_soa, POINTER(c_float)), soa_bytes)

    src = buffer_a
    dst = buffer_b
    kinds = build_fibonacci_launch_kinds(args.iters)
    nbody_grid = (
        (args.bodies + _NBODY_BLOCK - 1) // _NBODY_BLOCK,
        1,
        1,
    )

    for kind in kinds:
        if kind == "transpose":
            _launch_kernel(
                transpose,
                transpose_grid,
                transpose_block,
                [src, dst, args.matrix_dim, args.matrix_dim],
            )
            src, dst = dst, src
        else:
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = _soa_pointers(src, args.bodies)
            _launch_kernel(
                update_bodies,
                nbody_grid,
                (_NBODY_BLOCK, 1, 1),
                [
                    pos_x,
                    pos_y,
                    pos_z,
                    vel_x,
                    vel_y,
                    vel_z,
                    _STEP,
                    args.bodies,
                ],
            )

    hip.hipDeviceSynchronize()
    print(
        f"fib_transpose_nbody: launches={len(kinds)} "
        f"matrix={args.matrix_dim}x{args.matrix_dim} "
        f"buffers={buffer_gib:.3f}GiB "
        f"transpose_grid={transpose_grid[0]}x{transpose_grid[1]} "
        f"transpose_block={transpose_block[0]}x{transpose_block[1]} "
        f"cover_elems={covered_elements} bodies={args.bodies} "
        f"transpose={kinds.count('transpose')} nbody={kinds.count('nbody')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
