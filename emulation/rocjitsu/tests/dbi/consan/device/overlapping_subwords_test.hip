// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Adjacent byte stores followed by overlapping two-byte reads.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t observed[128];
  uint32_t control[128];
};

__global__ __launch_bounds__(128) void overlapping_subwords_for_consan(DeviceState *state) {
  __shared__ uint8_t bytes[64];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  const uint32_t value = (lane * 37u + 11u) & 0xffu;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (wave == 0)
    bytes[lane] = static_cast<uint8_t>(value);
  state->control[tid] = 0x74000000u | value;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (state->synchronize)
    __syncthreads();

  const uint32_t next = (lane + 1u) % warpSize;
  if (wave == 1)
    state->observed[lane] =
        static_cast<uint32_t>(bytes[lane]) | (static_cast<uint32_t>(bytes[next]) << 8u);
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
}

void run_overlapping_subwords(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  overlapping_subwords_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < thread_count; ++tid) {
    const uint32_t low = ((tid % (thread_count / 2u)) * 37u + 11u) & 0xffu;
    EXPECT_EQ(state.control[tid], 0x74000000u | low) << "thread=" << tid;
  }
  if (synchronize)
    for (uint32_t lane = 0; lane < thread_count / 2u; ++lane) {
      const uint32_t low = (lane * 37u + 11u) & 0xffu;
      const uint32_t high = (((lane + 1u) % (thread_count / 2u)) * 37u + 11u) & 0xffu;
      EXPECT_EQ(state.observed[lane], low | (high << 8u)) << "lane=" << lane;
    }
}

TEST(ConSanDeviceOverlappingSubwordsTest, Correct) { run_overlapping_subwords(true); }

TEST(ConSanDeviceOverlappingSubwordsTest, Incorrect) { run_overlapping_subwords(false); }

} // namespace
