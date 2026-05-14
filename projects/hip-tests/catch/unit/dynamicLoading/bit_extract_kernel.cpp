/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
extern "C" __global__ void bit_extract_kernel(uint32_t* C_d, const uint32_t* A_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < N; i += stride) {
#if HT_AMD
    C_d[i] = __bitextract_u32(A_d[i], 8, 4);
#else /* defined __HIP_PLATFORM_NVIDIA__ or other path */
    C_d[i] = ((A_d[i] & 0xf00) >> 8);
#endif
  }
}
