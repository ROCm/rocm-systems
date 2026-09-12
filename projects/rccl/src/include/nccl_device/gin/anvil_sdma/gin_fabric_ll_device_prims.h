/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Installed LL packet/epoch primitives for fabric DDA AllToAll. Safe for
// rccl-tests and other ENABLE_DEVICE_API consumers; no algorithms/dda includes.

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_DEVICE_PRIMS_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LL_DEVICE_PRIMS_H_

#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "../../rccl_ptr.h"

namespace dda {
namespace common {

constexpr size_t kDdaLLMaxBytes = (size_t)16 * 1024 * 1024;

union LLPacket16 {
  struct {
    uint32_t data0;
    uint32_t flag0;
    uint32_t data1;
    uint32_t flag1;
  };
  uint4 raw;
};
static_assert(sizeof(LLPacket16) == 16, "LLPacket16 must be exactly 16 bytes");

__device__ __forceinline__ void ddaLLStoreLineB128(uint32_t* dst, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    uint32_t w[4];
  } u;
  u.w[0] = a0;
  u.w[1] = a1;
  u.w[2] = a2;
  u.w[3] = a3;
  __builtin_amdgcn_global_store_b128((v4u_gptr)dst, u.v, RCCL_SYSTEM_SYNCSCOPE);
#else
  __builtin_nontemporal_store(a0, (u32_gptr)dst + 0);
  __builtin_nontemporal_store(a1, (u32_gptr)dst + 1);
  __builtin_nontemporal_store(a2, (u32_gptr)dst + 2);
  __builtin_nontemporal_store(a3, (u32_gptr)dst + 3);
#endif
  asm volatile("" ::: "memory");
}

__device__ __forceinline__ void ddaLLLoadLineB128(const uint32_t* src, uint32_t& o0, uint32_t& o1, uint32_t& o2,
                                                  uint32_t& o3) {
  asm volatile("" ::: "memory");
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    uint32_t w[4];
  } u;
  u.v = __builtin_amdgcn_global_load_b128((v4u_gptr)src, RCCL_SYSTEM_SYNCSCOPE);
  o0 = u.w[0];
  o1 = u.w[1];
  o2 = u.w[2];
  o3 = u.w[3];
#else
  o0 = __builtin_nontemporal_load((u32_gptr)src + 0);
  o1 = __builtin_nontemporal_load((u32_gptr)src + 1);
  o2 = __builtin_nontemporal_load((u32_gptr)src + 2);
  o3 = __builtin_nontemporal_load((u32_gptr)src + 3);
#endif
}

__device__ __forceinline__ uint32_t ddaGetLLEpochInc(const uint32_t* __restrict__ epochDev, int flatBlockId,
                                                     uint32_t inc) {
  uint32_t flag = epochDev[flatBlockId] + inc;
  if (flag == 0u) flag = 2u;
  __syncthreads();
  return flag;
}

__device__ __forceinline__ void ddaSetLLEpoch(uint32_t* __restrict__ epochDev, int epochLen, int flatBlockId, int total,
                                              uint32_t flag) {
  for (int e = flatBlockId + (int)threadIdx.x * total; e < epochLen; e += total * (int)blockDim.x) {
    epochDev[e] = flag;
  }
}

} // namespace common
} // namespace dda

#endif
