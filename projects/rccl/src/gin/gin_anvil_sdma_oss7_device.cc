/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#if defined(__HIPCC__) || defined(__CUDACC__)

namespace gin_anvil {
namespace sdma {

// Device-side OSS7 toggle for SDMA packet selection (COPY_LINEAR_PHY_MI4 vs legacy).
__device__ int gin_anvil_sdma_oss7_enabled = 1;

}  // namespace sdma
}  // namespace gin_anvil

#endif  // __HIPCC__ || __CUDACC__

#endif  // ENABLE_ROCSHMEM_GIN
