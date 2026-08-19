// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Three-dimensional workgroup identity with an intra-workgroup LDS handoff.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t observed[8 * 64];
  uint32_t control[8 * 128];
};

__global__ __launch_bounds__(128) void independent_workgroups_for_consan(DeviceState *state) {
  __shared__ uint32_t tile[64];
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t wave = threadIdx.x / warpSize;
  const uint32_t group = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const uint32_t group_value = (group + 1u) * 1000u + lane;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (wave == 0)
    tile[lane] = group_value;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (state->synchronize)
    __syncthreads();
  if (wave == 1)
    state->observed[group * warpSize + lane] = tile[lane];
  state->control[group * blockDim.x + threadIdx.x] = 0x73000000u + group * 256u + threadIdx.x;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
}

void run_independent_workgroups(bool synchronize) {
  constexpr uint32_t kGridX = 2;
  constexpr uint32_t kGridY = 2;
  constexpr uint32_t kGridZ = 2;
  constexpr uint32_t kWorkgroups = kGridX * kGridY * kGridZ;
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  independent_workgroups_for_consan<<<dim3(kGridX, kGridY, kGridZ), thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t group = 0; group < kWorkgroups; ++group) {
    for (uint32_t tid = 0; tid < thread_count; ++tid)
      EXPECT_EQ(state.control[group * thread_count + tid], 0x73000000u + group * 256u + tid)
          << "group=" << group << " thread=" << tid;
    if (synchronize)
      for (uint32_t lane = 0; lane < wave_size; ++lane)
        EXPECT_EQ(state.observed[group * wave_size + lane], (group + 1u) * 1000u + lane)
            << "group=" << group << " lane=" << lane;
  }
}

TEST(ConSanDeviceIndependentWorkgroupsTest, Correct) { run_independent_workgroups(true); }

TEST(ConSanDeviceIndependentWorkgroupsTest, Incorrect) { run_independent_workgroups(false); }

} // namespace
