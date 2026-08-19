// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Shared noinline helper reached from multiple generated-kernel owners.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t observed[128];
  uint32_t control[256];
};

__device__ __attribute__((noinline)) void shared_owner_helper_for_consan(DeviceState *state,
                                                                         uint32_t *tile,
                                                                         uint32_t owner_index,
                                                                         uint32_t owner_tag) {
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t wave = threadIdx.x / warpSize;
  const uint32_t value = owner_tag | threadIdx.x;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (wave == 0)
    tile[lane] = owner_tag | lane;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (state->synchronize)
    __syncthreads();
  if (wave == 1)
    state->observed[owner_index * warpSize + lane] = tile[lane];
  state->control[owner_index * blockDim.x + threadIdx.x] = value;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
}

__global__ __launch_bounds__(128) void shared_owner_a_for_consan(DeviceState *state) {
  __shared__ uint32_t tile[64];
  shared_owner_helper_for_consan(state, tile, 0u, 0xa0000000u);
}

__global__ __launch_bounds__(128) void shared_owner_b_for_consan(DeviceState *state) {
  __shared__ uint32_t tile[64];
  shared_owner_helper_for_consan(state, tile, 1u, 0xb0000000u);
}

void run_shared_helper_owners(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  shared_owner_a_for_consan<<<1, thread_count>>>(device_state);
  shared_owner_b_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < thread_count; ++tid) {
    EXPECT_EQ(state.control[tid], 0xa0000000u | tid) << "owner A thread=" << tid;
    EXPECT_EQ(state.control[thread_count + tid], 0xb0000000u | tid) << "owner B thread=" << tid;
  }
  if (synchronize)
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      EXPECT_EQ(state.observed[lane], 0xa0000000u | lane) << "owner A lane=" << lane;
      EXPECT_EQ(state.observed[wave_size + lane], 0xb0000000u | lane) << "owner B lane=" << lane;
    }
}

TEST(ConSanDeviceSharedHelperOwnersTest, Correct) { run_shared_helper_owners(true); }

TEST(ConSanDeviceSharedHelperOwnersTest, Incorrect) { run_shared_helper_owners(false); }

} // namespace
