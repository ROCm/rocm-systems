/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// AllToAll fabric LL vs gin.put/SDMA total-size gate (bytes, nRanks * per-peer).
// If RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL is set (explicit 0 is honored),
// use it; otherwise fall back to RCCL_DDA_LL_THRESHOLD.

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_POLICY_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_POLICY_H_

#include <cstddef>

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#include <cstdlib>
#endif

namespace gin {
namespace fabric {

inline size_t pickGinFabricLLThresholdAlltoAll(bool alltoallSet, unsigned long long alltoallVal,
                                               size_t ddaLLThreshold) {
  if (alltoallSet) return (size_t)alltoallVal;
  return ddaLLThreshold;
}

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)

inline bool parseGinFabricLLThresholdEnv(const char* name, unsigned long long* val) {
  const char* e = getenv(name);
  if (!e || !e[0] || *e == '-') return false;
  char* end = nullptr;
  unsigned long long v = strtoull(e, &end, 10);
  if (end == e) return false;
  *val = v;
  return true;
}

inline size_t resolveGinFabricLLThresholdAlltoAll(size_t ddaLLThreshold) {
  unsigned long long alltoallVal = 0;
  const bool alltoallSet =
      parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &alltoallVal) ||
      parseGinFabricLLThresholdEnv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &alltoallVal);
  return pickGinFabricLLThresholdAlltoAll(alltoallSet, alltoallVal, ddaLLThreshold);
}

#endif // host

} // namespace fabric
} // namespace gin

#endif // _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_POLICY_H_
