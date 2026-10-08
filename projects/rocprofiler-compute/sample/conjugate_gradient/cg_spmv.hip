// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Built as its own shared library so that the runtime loads it as a code object
// separate from the one holding kernel_cg_update_reduce.

#include "cg_args.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

// extern "C" keeps the symbol unmangled, so profiler output names the kernel
// the same way this file does.
extern "C" __global__ __launch_bounds__(cg_block_size) void kernel_spmv_csr(CgArgs arguments)
{
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= arguments.rows)
    {
        return;
    }

    for (std::uint32_t round = 0; round < arguments.rounds; ++round)
    {
        float sum = 0.0F;
        for (std::uint32_t entry = arguments.row_offsets[row]; entry < arguments.row_offsets[row + 1];
             ++entry)
        {
            sum += arguments.values[entry] * arguments.p[arguments.column_indices[entry]];
        }
        arguments.q[row] = sum;
    }
}

extern "C" void cg_launch(const CgArgs& arguments)
{
    const std::uint32_t grid_size = (arguments.rows + cg_block_size - 1) / cg_block_size;
    hipLaunchKernelGGL(kernel_spmv_csr, dim3(grid_size), dim3(cg_block_size), 0, nullptr, arguments);
}
