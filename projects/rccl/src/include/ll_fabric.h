/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Low-latency (LL) packet primitives for the DDA fabric/VMM remote-write path.
 *
 * Each 16-byte LL packet carries 8 bytes of payload as two {4B data, 4B flag}
 * pairs. The flag travels inline with the data, so a reader can detect arrival
 * by polling the flag instead of using a separate cross-GPU barrier. The flag
 * value is a monotonically increasing per-op epoch; the recv buffer is cleared
 * once at init, so no reset is needed between ops.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstdint>

namespace meta::comms {

// 16B LL packet: two (4B data, 4B flag) pairs -> 8B payload per packet.
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

// 8 payload bytes per 16B packet.
constexpr size_t kLLDataBytesPerPacket = 8;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
typedef __attribute__((address_space(1))) uint64_t* ll_u64_gptr;

// Two 8B system-scope atomic stores: publish a full 16B packet with cross-GPU
// visibility (data + flag together).
__device__ __forceinline__ void llStoreLine(
    uint32_t* dst, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
  union {
    uint64_t u64;
    uint32_t u32[2];
  } p0, p1;
  p0.u32[0] = a0;
  p0.u32[1] = a1;
  p1.u32[0] = a2;
  p1.u32[1] = a3;
  __hip_atomic_store(
      (ll_u64_gptr)dst, p0.u64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(
      (ll_u64_gptr)dst + 1, p1.u64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  asm volatile("" ::: "memory");
}

// Two 8B system-scope atomic loads of a 16B packet.
__device__ __forceinline__ void llLoadLine(
    const uint32_t* src, uint32_t& o0, uint32_t& o1, uint32_t& o2,
    uint32_t& o3) {
  asm volatile("" ::: "memory");
  union {
    uint64_t u64;
    uint32_t u32[2];
  } p0, p1;
  p0.u64 = __hip_atomic_load(
      (ll_u64_gptr)src, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  p1.u64 = __hip_atomic_load(
      (ll_u64_gptr)src + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  o0 = p0.u32[0];
  o1 = p0.u32[1];
  o2 = p1.u32[0];
  o3 = p1.u32[1];
}
#endif // __HIP_PLATFORM_AMD__

} // namespace meta::comms
