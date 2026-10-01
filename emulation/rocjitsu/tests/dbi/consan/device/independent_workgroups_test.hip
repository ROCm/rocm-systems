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

void run_phase(DeviceState *device_state, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
               uint32_t thread_count, bool synchronize) {
  const uint32_t wave_size = thread_count / 2u;
  const uint32_t workgroup_count = grid_x * grid_y * grid_z;
  DeviceState state{};
  state.synchronize = synchronize;
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  independent_workgroups_for_consan<<<dim3(grid_x, grid_y, grid_z), thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  for (uint32_t group = 0; group < workgroup_count; ++group) {
    for (uint32_t tid = 0; tid < thread_count; ++tid)
      EXPECT_EQ(state.control[group * thread_count + tid], 0x73000000u + group * 256u + tid)
          << "group=" << group << " thread=" << tid;
    if (synchronize)
      for (uint32_t lane = 0; lane < wave_size; ++lane)
        EXPECT_EQ(state.observed[group * wave_size + lane], (group + 1u) * 1000u + lane)
            << "group=" << group << " lane=" << lane;
  }
}

void run_independent_workgroups(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));

  // Preserve the three-dimensional identity coverage, then use one designated
  // workgroup so the diagnostic oracle cannot depend on which independently
  // scheduled workgroups win the two bounded access windows.
  run_phase(device_state, /*grid_x=*/2u, /*grid_y=*/2u, /*grid_z=*/2u, thread_count, synchronize);
  run_phase(device_state, /*grid_x=*/1u, /*grid_y=*/1u, /*grid_z=*/1u, thread_count, synchronize);

  HIP_ASSERT(hipFree(device_state));
}

TEST(ConSanDeviceIndependentWorkgroupsTest, Correct) { run_independent_workgroups(true); }

TEST(ConSanDeviceIndependentWorkgroupsTest, Incorrect) { run_independent_workgroups(false); }

} // namespace
