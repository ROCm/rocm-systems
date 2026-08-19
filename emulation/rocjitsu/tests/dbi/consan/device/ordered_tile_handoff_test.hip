// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Cross-wave LDS publication distilled from tiled attention and matmul.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t observed[64];
  uint32_t control[128];
};

__global__ __launch_bounds__(128) void ordered_tile_handoff_for_consan(DeviceState *state) {
  __shared__ uint32_t tile[64];
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t wave = threadIdx.x / warpSize;
  const uint32_t control = 0x71000000u + threadIdx.x;

  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (wave == 0)
    tile[lane] = 0x1000u + lane;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (state->synchronize)
    __syncthreads();
  if (wave == 1)
    state->observed[lane] = tile[lane];
  state->control[threadIdx.x] = control;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
}

void run_ordered_tile_handoff(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  ordered_tile_handoff_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x71000000u + tid) << "thread=" << tid;
  if (synchronize)
    for (uint32_t lane = 0; lane < wave_size; ++lane)
      EXPECT_EQ(state.observed[lane], 0x1000u + lane) << "lane=" << lane;
}

TEST(ConSanDeviceOrderedTileHandoffTest, Correct) { run_ordered_tile_handoff(true); }

TEST(ConSanDeviceOrderedTileHandoffTest, Incorrect) { run_ordered_tile_handoff(false); }

} // namespace
