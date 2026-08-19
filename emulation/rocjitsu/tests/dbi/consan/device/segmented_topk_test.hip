// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Segmented value/index selection with non-wave-multiple tails.

#include "consan_device_test_support.h"

#include <algorithm>

namespace {

constexpr uint32_t kRowCount = 3u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t top_value[kRowCount];
  uint32_t top_index[kRowCount];
  uint32_t control[kRowCount * 128u];
};

__global__ __launch_bounds__(128) void segmented_topk_for_consan(DeviceState *state) {
  __shared__ __align__(8) uint64_t pairs[64];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  const uint32_t row = blockIdx.x;
  const uint32_t valid = warpSize - (3u + 5u * row);
  state->control[row * blockDim.x + tid] = 0x7a000000u | (row << 12u) | tid;

  if (wave == 1u && lane < valid) {
    const uint32_t value = row * 1000u + ((37u * lane + 11u) % valid) + 1u;
    pairs[lane] = (static_cast<uint64_t>(value) << 32u) | lane;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  if (state->synchronize)
    __syncthreads();

  if (wave == 0u) {
    uint64_t selected = lane < valid ? pairs[lane] : 0u;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
#pragma unroll 1
    for (uint32_t offset = warpSize / 2u; offset != 0u; offset /= 2u) {
      const uint64_t peer = __shfl_down(selected, offset, warpSize);
      if (lane + offset < warpSize && peer > selected)
        selected = peer;
    }
    if (lane == 0u) {
      state->top_value[row] = static_cast<uint32_t>(selected >> 32u);
      state->top_index[row] = static_cast<uint32_t>(selected);
    }
  }
}

void run_segmented_topk(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_size = wave_size;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  segmented_topk_for_consan<<<kRowCount, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_size, wave_size);
  for (uint32_t row = 0; row < kRowCount; ++row) {
    const uint32_t valid = wave_size - (3u + 5u * row);
    uint64_t expected = 0u;
    for (uint32_t lane = 0; lane < valid; ++lane) {
      const uint32_t value = row * 1000u + ((37u * lane + 11u) % valid) + 1u;
      expected = std::max(expected, (static_cast<uint64_t>(value) << 32u) | lane);
    }
    if (synchronize) {
      EXPECT_EQ(state.top_value[row], static_cast<uint32_t>(expected >> 32u)) << "row=" << row;
      EXPECT_EQ(state.top_index[row], static_cast<uint32_t>(expected)) << "row=" << row;
    }
    for (uint32_t tid = 0; tid < thread_count; ++tid) {
      EXPECT_EQ(state.control[row * thread_count + tid], 0x7a000000u | (row << 12u) | tid)
          << "row=" << row << " thread=" << tid;
    }
  }
}

TEST(ConSanDeviceSegmentedTopKTest, Correct) { run_segmented_topk(true); }

TEST(ConSanDeviceSegmentedTopKTest, Incorrect) { run_segmented_topk(false); }

} // namespace
