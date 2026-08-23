// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Cross-wave LDS compare/exchange stage distilled from framework sort kernels.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t low_value[64];
  uint32_t low_index[64];
  uint32_t high_value[64];
  uint32_t high_index[64];
  uint32_t control[128];
};

__host__ __device__ uint32_t input_value(uint32_t index) {
  return ((index * 37u + 11u) ^ (index * 0x45d9f3bu)) & 0x0000ffffu;
}

__global__ __launch_bounds__(128) void sort_network_exchange_for_consan(DeviceState *state) {
  __shared__ uint32_t values[128];
  __shared__ uint32_t indices[128];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x8a000000u | tid;

  values[tid] = input_value(tid);
  indices[tid] = tid;
  asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  __syncthreads();

  // One wave owns a disjoint set of cross-wave pairs, as in an in-place
  // sorting-network stage. It publishes both the reordered values and their
  // original indices for the next wave to consume.
  if (wave == 0u) {
    uint32_t low_value = values[lane];
    uint32_t low_index = indices[lane];
    uint32_t high_value = values[warpSize + lane];
    uint32_t high_index = indices[warpSize + lane];
    if (low_value > high_value) {
      const uint32_t saved_value = low_value;
      const uint32_t saved_index = low_index;
      low_value = high_value;
      low_index = high_index;
      high_value = saved_value;
      high_index = saved_index;
    }
    values[lane] = low_value;
    indices[lane] = low_index;
    values[warpSize + lane] = high_value;
    indices[warpSize + lane] = high_index;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }

  // The adjacent pair differs only at this stage-publication edge. The E2E
  // torch.mode specialization that motivated it happens to launch one wave
  // on gfx950, making its analogous barrier redundant there; this two-wave
  // transport preserves the algorithmic idea while making the missing edge a
  // genuine sanitizer contract on every supported architecture.
  if (state->synchronize)
    __syncthreads();

  if (wave == 1u) {
    state->low_value[lane] = values[lane];
    state->low_index[lane] = indices[lane];
    state->high_value[lane] = values[warpSize + lane];
    state->high_index[lane] = indices[warpSize + lane];
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
}

void run_sort_network_exchange(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_size = wave_size;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  sort_network_exchange_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_size, wave_size);
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x8a000000u | tid) << "thread=" << tid;
  if (!synchronize)
    return;

  for (uint32_t lane = 0; lane < wave_size; ++lane) {
    const uint32_t left_value = input_value(lane);
    const uint32_t right_value = input_value(wave_size + lane);
    const bool left_is_low = left_value <= right_value;
    EXPECT_EQ(state.low_value[lane], left_is_low ? left_value : right_value) << "lane=" << lane;
    EXPECT_EQ(state.low_index[lane], left_is_low ? lane : wave_size + lane) << "lane=" << lane;
    EXPECT_EQ(state.high_value[lane], left_is_low ? right_value : left_value) << "lane=" << lane;
    EXPECT_EQ(state.high_index[lane], left_is_low ? wave_size + lane : lane) << "lane=" << lane;
  }
}

TEST(ConSanDeviceSortNetworkExchangeTest, Correct) { run_sort_network_exchange(true); }

TEST(ConSanDeviceSortNetworkExchangeTest, Incorrect) { run_sort_network_exchange(false); }

} // namespace
