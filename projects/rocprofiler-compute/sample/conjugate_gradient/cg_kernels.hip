// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "cg_args.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

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
