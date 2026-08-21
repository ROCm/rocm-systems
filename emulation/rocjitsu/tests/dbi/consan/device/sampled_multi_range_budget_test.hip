// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Two dual-address LDS sites distilled from PyTorch radix-sort kernels.
///
/// The Sampled test configuration deliberately caps instrumentation at two
/// sites. Each site consumes two ranges and multiple report banks, but those
/// report slots must not consume the site-count budget.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t observed[128];
  uint32_t control[128];
};

__device__ __forceinline__ void store_lds_pair(volatile uint32_t *address, uint32_t first,
                                               uint32_t second) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)
  asm volatile("ds_write2_b32 %0, %1, %2 offset0:0 offset1:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(lds_address), "v"(first), "v"(second)
               : "memory");
#else
  asm volatile("ds_store_2addr_b32 %0, %1, %2 offset0:0 offset1:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(lds_address), "v"(first), "v"(second)
               : "memory");
#endif
}

__device__ __forceinline__ uint2 load_lds_pair(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  register uint32_t first asm("v20") = 0u;
  register uint32_t second asm("v21") = 0u;
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("ds_read2_b32 v[20:21], %2 offset0:0 offset1:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_waitcnt lgkmcnt(0)\n\t"
               : "+v"(first), "+v"(second)
               : "v"(lds_address)
               : "memory");
#elif defined(__gfx1250__)
  asm volatile("ds_read2_b32 v[20:21], %2 offset0:0 offset1:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_wait_dscnt 0\n\t"
               : "+v"(first), "+v"(second)
               : "v"(lds_address)
               : "memory");
#elif defined(__gfx1201__)
  asm volatile("ds_load_2addr_b32 v[20:21], %2 offset0:0 offset1:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_wait_dscnt 0\n\t"
               : "+v"(first), "+v"(second)
               : "v"(lds_address)
               : "memory");
#else
  asm volatile("ds_load_2addr_b32 v[20:21], %2 offset0:0 offset1:1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               "s_waitcnt lgkmcnt(0)\n\t"
               : "+v"(first), "+v"(second)
               : "v"(lds_address)
               : "memory");
#endif
  return make_uint2(first, second);
}

__global__ void sampled_multi_range_budget_for_consan(DeviceState *state) {
  __shared__ uint32_t words[128];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  const uint32_t first = 0x61000000u | lane;
  const uint32_t second = 0x62000000u | lane;

  state->control[tid] = 0x63000000u | tid;
  if (wave == 0u)
    store_lds_pair(&words[2u * lane], first, second);
  if (state->synchronize)
    __syncthreads();
  if (wave == 1u) {
    const uint2 pair = load_lds_pair(&words[2u * lane]);
    state->observed[2u * lane] = pair.x;
    state->observed[2u * lane + 1u] = pair.y;
  }
}

void run_sampled_multi_range_budget(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  sampled_multi_range_budget_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x63000000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0u; lane < wave_size; ++lane) {
      EXPECT_EQ(state.observed[2u * lane], 0x61000000u | lane) << "lane=" << lane;
      EXPECT_EQ(state.observed[2u * lane + 1u], 0x62000000u | lane) << "lane=" << lane;
    }
  }
}

TEST(ConSanDeviceSampledMultiRangeBudgetTest, Correct) { run_sampled_multi_range_budget(true); }

TEST(ConSanDeviceSampledMultiRangeBudgetTest, Incorrect) { run_sampled_multi_range_budget(false); }

} // namespace
