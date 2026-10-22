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

// Communicator state a DDA kernel needs at run time, currently its abort flag.
// Passed by value as a kernel argument and held by FabricGpuBarrier /
// IpcGpuBarrier. As a lone pointer it is passed as a pointer argument, which
// gfx1250 preloads into SGPRs; a second member makes it an aggregate that is
// read from the kernarg segment and ends the preload of every later argument.
// Pass further per-launch values as separate scalar arguments.
struct DdaKernelParams {
  const uint32_t* flag; // comm->abortFlagDev; null never reports an abort

  __device__ __forceinline__ bool aborted() const {
    return flag != nullptr && __atomic_load_n(flag, __ATOMIC_SEQ_CST) != 0;
  }

  // Count one spin. Every kDdaSpinsBeforeCheckAbort spins, load the flag.
  // Returns true when the communicator has been revoked/aborted.
  __device__ __forceinline__ bool abortSpinTick(int& spins) const {
    if (++spins < kDdaSpinsBeforeCheckAbort) return false;
    spins = 0;
    return aborted();
  }
};
static_assert(sizeof(DdaKernelParams) == sizeof(void*), "DdaKernelParams must stay one pointer (kernarg preload)");

} // namespace dda::common
