/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#if defined(__HIPCC__) || defined(__CUDACC__)

namespace anvil {

// Device-side OSS7 toggle for SDMA packet selection (COPY_LINEAR_PHY_MI4 vs legacy).
// Host may upload via ginAnvilSdmaOss7Upload() when wired; default enabled.
__device__ int gin_anvil_sdma_oss7_enabled = 1;

}  // namespace anvil

#endif  // __HIPCC__ || __CUDACC__

#endif  // ENABLE_ROCSHMEM_GIN
