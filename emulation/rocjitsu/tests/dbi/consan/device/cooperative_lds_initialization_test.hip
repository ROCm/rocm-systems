// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Cooperative LDS initialization distilled from PyTorch radix kernels.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t initialized_by_peer[64];
  uint32_t control[128];
};

__global__
__launch_bounds__(128) void cooperative_lds_initialization_for_consan(DeviceState *state) {
  __shared__ uint32_t scratch[128];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0x7b000000u | tid;

  // Radix and histogram kernels cooperatively initialize LDS before each
  // wave consumes slots initialized by a peer. The adjacent members differ
  // only in this publication edge.
  scratch[tid] = 0x81000000u | tid;
  asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  if (state->synchronize)
    __syncthreads();

  const uint32_t peer_tid = ((wave ^ 1u) * warpSize) + lane;
  const uint32_t peer_value = scratch[peer_tid];
  asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  if (wave == 0u)
    state->initialized_by_peer[lane] = peer_value;
}

void run_cooperative_lds_initialization(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_size = wave_size;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  cooperative_lds_initialization_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_size, wave_size);
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x7b000000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      EXPECT_EQ(state.initialized_by_peer[lane], 0x81000000u | (wave_size + lane))
          << "lane=" << lane;
    }
  }
}

TEST(ConSanDeviceCooperativeLdsInitializationTest, Correct) {
  run_cooperative_lds_initialization(true);
}

TEST(ConSanDeviceCooperativeLdsInitializationTest, Incorrect) {
  run_cooperative_lds_initialization(false);
}

} // namespace
