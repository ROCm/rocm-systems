/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Device-side abortFlag polling for DDA kernels. Generic RCCL collectives
 * observe comm->abortFlagDev via ncclShmem; DDA kernels are standalone and
 * must take that pointer as an argument (or via FabricGpuBarrier /
 * IpcGpuBarrier) so ncclCommRevoke can unblock an incomplete collective.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstdint>

namespace dda::common {

// Same cadence as NCCL_SPINS_BEFORE_CHECK_ABORT in src/device/primitives.h.
constexpr int kDdaSpinsBeforeCheckAbort = 10000;

__device__ __forceinline__ bool ddaAbortFlagSet(const uint32_t* abortFlag) {
  return abortFlag != nullptr && __atomic_load_n(abortFlag, __ATOMIC_SEQ_CST) != 0;
}

// Count one spin. Every kDdaSpinsBeforeCheckAbort spins, load abortFlag.
// Returns true when the communicator has been revoked/aborted.
__device__ __forceinline__ bool ddaAbortSpinTick(const uint32_t* abortFlag, int& spins) {
  if (++spins < kDdaSpinsBeforeCheckAbort) return false;
  spins = 0;
  return ddaAbortFlagSet(abortFlag);
}

} // namespace dda::common
