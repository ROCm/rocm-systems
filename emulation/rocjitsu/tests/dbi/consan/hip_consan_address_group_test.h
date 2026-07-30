// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

/// Lane zero writes the first LDS word, lanes one and two collide on the
/// second, and every later lane selects its own word. The only multi-lane
/// address group is therefore reached after the producer removes the first
/// group from its pending EXEC mask.
__launch_bounds__(64) __global__
    void moi_second_address_group_race_for_instrumentation(uint32_t *out) {
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
