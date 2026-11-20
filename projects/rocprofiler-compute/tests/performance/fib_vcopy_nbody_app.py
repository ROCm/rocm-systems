#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""HipRTC Fibonacci ping-pong of vecCopy and mini-nbody update_bodies.

Shared SoA buffers: vecCopy writes dst from src then swaps; update_bodies
runs in-place on the current src (nbody input = vcopy output; nbody output
is the next vcopy input). Schedule is 1v,1n,2v,2n,3v,3n,5v,5n,... until
--iters launches.
"""

from __future__ import annotations

import argparse
import random
import sys
from ctypes import POINTER, byref, c_float, c_int, c_void_p, cast
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import utils.hip_interface as hip  # noqa: E402
import utils.hiprtc_interface as hiprtc  # noqa: E402

_KERNEL_SOURCE = r"""
extern "C" __global__ void vecCopy(
    float *a, float *b, float *c, int n, int stride)
{
    (void)b;
    (void)stride;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id];
    }
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
_VCOPY_BLOCK = 256
_NBODY_BLOCK = 128
_STEP = 0.01


class HiprtcProgram:
    """Compile HipRTC source and resolve kernels by name."""

    def __init__(self, source: str) -> None:
        self._prog = hiprtc.hiprtcCreateProgram(source, "fib_vcopy_nbody")
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


def build_fibonacci_launch_kinds(total_launches: int) -> list[str]:
    """Return kinds for 1v,1n,2v,2n,3v,3n,5v,... truncated to total_launches."""
    kinds: list[str] = []
    fib = 1
    next_fib = 2
    while len(kinds) < total_launches:
        for _ in range(fib):
            if len(kinds) >= total_launches:
                break
            kinds.append("vcopy")
        for _ in range(fib):
            if len(kinds) >= total_launches:
                break
            kinds.append("nbody")
        fib, next_fib = next_fib, fib + next_fib
    return kinds


def _device_byte_offset(memory: hip.HIPDeviceMemory, float_offset: int) -> c_void_p:
    base = memory.ptr.value
    if base is None:
        raise RuntimeError("device pointer is null")
    return c_void_p(base + float_offset * 4)


def _soa_pointers(memory: hip.HIPDeviceMemory, body_count: int) -> list[c_void_p]:
    return [
        _device_byte_offset(memory, body_count * index)
        for index in range(_COORDINATES_PER_BODY)
    ]


def _launch_kernel(
    func: Any,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: list[Any],
) -> None:
    converted: list[Any] = []
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HipRTC Fibonacci vcopy + nbody ping-pong workload"
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
        "--device",
        type=int,
        default=0,
        help="HIP device id (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.iters < 1:
        print("--iters must be >= 1", file=sys.stderr)
        return 1
    if args.bodies < 1:
        print("--bodies must be >= 1", file=sys.stderr)
        return 1

    hip.hipSetDevice(args.device)
    program = HiprtcProgram(_KERNEL_SOURCE)
    vec_copy = program.get_kernel("vecCopy")
    update_bodies = program.get_kernel("update_bodies")

    element_count = args.bodies * _COORDINATES_PER_BODY
    byte_count = element_count * 4
    host_state = (c_float * element_count)()
    rng = random.Random(0)
    for index in range(element_count):
        host_state[index] = 2.0 * rng.random() - 1.0

    buffer_a = hip.hipMalloc(byte_count)
    buffer_b = hip.hipMalloc(byte_count)
    hip.hipMemcpyHtoD(buffer_a, cast(host_state, POINTER(c_float)), byte_count)
    hip.hipMemcpyHtoD(buffer_b, cast(host_state, POINTER(c_float)), byte_count)

    src = buffer_a
    dst = buffer_b
    kinds = build_fibonacci_launch_kinds(args.iters)
    vcopy_grid = (
        (element_count + _VCOPY_BLOCK - 1) // _VCOPY_BLOCK,
        1,
        1,
    )
    nbody_grid = (
        (args.bodies + _NBODY_BLOCK - 1) // _NBODY_BLOCK,
        1,
        1,
    )

    for kind in kinds:
        if kind == "vcopy":
            _launch_kernel(
                vec_copy,
                vcopy_grid,
                (_VCOPY_BLOCK, 1, 1),
                [src, src, dst, element_count, 1],
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
        f"fib_vcopy_nbody: launches={len(kinds)} bodies={args.bodies} "
        f"vcopy={kinds.count('vcopy')} nbody={kinds.count('nbody')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
