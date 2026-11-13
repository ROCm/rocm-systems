/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Per-launch parameters shared by every DDA kernel. Generic RCCL collectives
 * observe comm->abortFlagDev via ncclShmem; DDA kernels are standalone and
 * take it in a DdaKernelParams argument (or via FabricGpuBarrier /
 * IpcGpuBarrier) so ncclCommRevoke can unblock an incomplete collective.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstdint>

namespace dda::common {

// Same cadence as NCCL_SPINS_BEFORE_CHECK_ABORT in src/device/primitives.h.
constexpr int kDdaSpinsBeforeCheckAbort = 10000;

// Abort flag a DDA kernel takes by value. Keep it one pointer: gfx1250 preloads that into SGPRs,
// but a larger struct is read from memory and ends the preload of every later kernel argument.
struct DdaKernelParams {
  const uint32_t* flag; // comm->abortFlagDev; null never reports an abort

  // #11245 taught every DDA wait to sample this flag. That behavior is reverted:
  // the pointer stays one kernarg so later kernels keep their launch signature,
  // but a wait never gives up on it.
  __device__ __forceinline__ bool aborted() const {
    (void)flag;
    return false;
  }

  __device__ __forceinline__ bool abortSpinTick(int& spins) const {
    (void)spins;
    return false;
  }
};
static_assert(sizeof(DdaKernelParams) == sizeof(void*), "DdaKernelParams must stay one pointer (kernarg preload)");

} // namespace dda::common
