/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_MEM_BARRIER_H_
#define _NCCL_DEVICE_MEM_BARRIER_H_
#include "impl/core__types.h"
#include "core_tmp.h"

// Upstream NCCL had:  #undef __CUDACC__  /  #define __CUDACC__ 0
// before the device template guard, expecting the next translation unit to
// re-enable it.  hipify converted those identifiers to __HIPCC__, which IS
// the actual compiler-defined macro on AMD - undefining it inside the
// build broke all subsequent device-template visibility on RCCL builds.
// Drop the corruption of __CUDACC__/__HIPCC__ entirely; on AMD HIP builds
// the device template should always be visible (gated by __HIPCC__ which
// the compiler defines).  Use __HIPCC__ directly so the file compiles on
// HIP without further hipify mangling.

struct ncclLsaBarrierHandle;

NCCL_EXTERN_C __host__ ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

#if __HIPCC__
template<typename Coop>
struct ncclLsaBarrierSession_internal;

template<typename Coop>
struct ncclLsaBarrierSession: ncclLsaBarrierSession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeam, ncclLsaBarrierHandle, uint32_t index, bool multimem=false, ncclMultimemHandle mmHandle={});

  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeamTagLsa, uint32_t index, bool multimem=false);

  NCCL_DEVICE_INLINE ~ncclLsaBarrierSession();

  ncclLsaBarrierSession(ncclLsaBarrierSession const&) = delete; // Sessions are not copyable

#if __HIP_PLATFORM_AMD__
  NCCL_DEVICE_INLINE void arrive(Coop, std::memory_order);
  NCCL_DEVICE_INLINE void wait(Coop, std::memory_order);
  NCCL_DEVICE_INLINE void sync(Coop, std::memory_order);
#else
  NCCL_DEVICE_INLINE void arrive(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE void wait(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order);
#endif
};
#endif

#endif // _NCCL_DEVICE_MEM_BARRIER_H_
