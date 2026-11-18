// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Built as its own shared library so that the runtime loads it as a code object
// separate from the one holding kernel_spmv_csr.

#include "cg_args.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

// extern "C" keeps the symbol unmangled, so profiler output names the kernel
// the same way this file does.
extern "C" __global__ __launch_bounds__(cg_block_size) void kernel_cg_update_reduce(CgArgs arguments)
{
    __shared__ float reduction_values[cg_block_size];

    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    for (std::uint32_t round = 0; round < arguments.rounds; ++round)
    {
        float squared_residual = 0.0F;
        if (row < arguments.rows)
        {
            const float alpha = 0.5F + static_cast<float>(round) * 1.0E-6F;
            arguments.x[row] += alpha * arguments.p[row];
            arguments.r[row] -= alpha * arguments.q[row];
            squared_residual = arguments.r[row] * arguments.r[row];
        }

        reduction_values[threadIdx.x] = squared_residual;
        __syncthreads();

        for (std::uint32_t stride = cg_block_size / 2; stride > 0; stride /= 2)
        {
            if (threadIdx.x < stride)
            {
                reduction_values[threadIdx.x] += reduction_values[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            arguments.partials[blockIdx.x] = reduction_values[0];
        }
    }
}

extern "C" void cg_launch(const CgArgs& arguments)
{
    const std::uint32_t grid_size = (arguments.rows + cg_block_size - 1) / cg_block_size;
    hipLaunchKernelGGL(
        kernel_cg_update_reduce, dim3(grid_size), dim3(cg_block_size), 0, nullptr, arguments);
}
