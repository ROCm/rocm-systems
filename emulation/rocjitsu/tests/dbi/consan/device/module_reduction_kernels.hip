// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Caller-supplied LDS kernels for the hipModuleLoad regression fixture.

#include <hip/hip_runtime.h>

template <bool SynchronizeInitial>
__device__ void module_reduction_body(const uint32_t *input, uint32_t *output, uint32_t count) {
  __shared__ uint32_t values[256];
  const uint32_t tid = threadIdx.x;
  const uint32_t index = blockIdx.x * blockDim.x + tid;
  values[tid] = index < count ? input[index] : 0u;
  if constexpr (SynchronizeInitial)
    __syncthreads();

  for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
    if (tid < stride)
      values[tid] += values[tid + stride];
    __syncthreads();
  }
  if (tid == 0u)
    output[blockIdx.x] = values[0];
}

extern "C" __global__ __launch_bounds__(256) void module_reduction_correct(const uint32_t *input,
                                                                           uint32_t *output,
                                                                           uint32_t count) {
  module_reduction_body<true>(input, output, count);
}

extern "C" __global__ __launch_bounds__(256) void module_reduction_incorrect(const uint32_t *input,
                                                                             uint32_t *output,
                                                                             uint32_t count) {
  module_reduction_body<false>(input, output, count);
}
