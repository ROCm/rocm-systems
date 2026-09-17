// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

/// Lane zero writes the first LDS word, lanes one and two collide on the
/// second, and every later lane selects its own word. The only multi-lane
/// address group is therefore reached after the producer removes the first
/// group from its pending EXEC mask.
__launch_bounds__(64) __global__ void second_address_group_race_for_instrumentation(uint32_t *out) {
  __shared__ volatile uint32_t lds[64];
  const uint32_t lane = static_cast<uint32_t>(threadIdx.x) % warpSize;
  const uint32_t index = lane == 0u ? 0u : (lane <= 2u ? 1u : lane - 1u);
  const auto lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[index]) & 0xffffu);
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("ds_write_b32 %0, %1" : : "v"(lds_address), "v"(lane) : "memory");
#else
  asm volatile("ds_store_b32 %0, %1" : : "v"(lds_address), "v"(lane) : "memory");
#endif
  __syncthreads();
  if (lane == 0u)
    out[0] = lds[0];
}

/// Every lane writes a distinct 16-byte LDS region with the same native B128
/// instruction. ConSan must preserve the effective ranges it observes:
/// grouping the whole wave by the instruction's coarse static range would
/// manufacture a same-wave write/write conflict.
__launch_bounds__(32) __global__ void disjoint_lane_b128_stores_for_instrumentation(uint32_t *out) {
  __shared__ alignas(16) uint8_t lds[32 * 16];
  const uint32_t lane = static_cast<uint32_t>(threadIdx.x);
  const auto lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[lane * 16u]) & 0xffffu);
  asm volatile("v_mov_b32 v20, %1\n\t"
               "v_mov_b32 v21, %2\n\t"
               "v_mov_b32 v22, %3\n\t"
               "v_mov_b32 v23, %4\n\t"
#if defined(__gfx942__) || defined(__gfx950__)
               "ds_write_b128 %0, v[20:23]\n\t"
#else
               "ds_store_b128 %0, v[20:23]\n\t"
#endif
               ".rept 32\n\t"
               "s_nop 0\n\t"
               ".endr\n\t"
#if defined(__gfx942__) || defined(__gfx950__)
               "s_waitcnt lgkmcnt(0)"
#else
               "s_wait_dscnt 0"
#endif
               :
               : "v"(lds_address), "v"(lane), "v"(lane + 1u), "v"(lane + 2u), "v"(lane + 3u)
               : "v20", "v21", "v22", "v23", "memory");
  if (lane == 0u)
    out[0] = reinterpret_cast<uint32_t *>(lds)[0];
}
