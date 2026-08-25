// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Sparse-workgroup Record/Replay sampling distilled from a user-reported
/// multi-stage LDS reduction. The correct and incorrect members differ only
/// in the initial publication barrier.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kThreads = 256u;
constexpr uint32_t kMaximumGroups = 8u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t reduction[kMaximumGroups];
  uint32_t control[kMaximumGroups * kThreads];
};

__global__ __launch_bounds__(kThreads) void sampling_reduction_for_consan(DeviceState *state) {
  __shared__ uint32_t values[kThreads];
  const uint32_t tid = threadIdx.x;
  const uint32_t group = blockIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  if (wave == 1u)
    values[lane] = 1u;
  state->control[group * kThreads + tid] = 0x9a000000u + group * kThreads + tid;
  if (state->synchronize)
    __syncthreads();

#pragma unroll 1
  for (uint32_t stride = warpSize / 2u; stride != 0u; stride /= 2u) {
    if (wave == 0u && lane < stride)
      values[lane] += values[lane + stride];
    __syncthreads();
  }
  if (tid == 0u)
    state->reduction[group] = values[0];
}

void run_sampling_reduction(uint32_t group_count, bool synchronize) {
  ASSERT_LE(group_count, kMaximumGroups);
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  sampling_reduction_for_consan<<<group_count, kThreads>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  for (uint32_t group = 0u; group < group_count; ++group) {
    for (uint32_t tid = 0u; tid < kThreads; ++tid) {
      EXPECT_EQ(state.control[group * kThreads + tid], 0x9a000000u + group * kThreads + tid)
          << "group=" << group << " thread=" << tid;
    }
    if (synchronize)
      EXPECT_EQ(state.reduction[group], consan_device_test::active_wave_size())
          << "group=" << group;
  }
}

TEST(ConSanDeviceRecordReplaySamplingReductionTest, DefaultCorrect) {
  run_sampling_reduction(/*group_count=*/1u, /*synchronize=*/true);
}

TEST(ConSanDeviceRecordReplaySamplingReductionTest, DefaultIncorrect) {
  run_sampling_reduction(/*group_count=*/1u, /*synchronize=*/false);
}

TEST(ConSanDeviceRecordReplaySamplingReductionTest, SparseCorrect) {
  run_sampling_reduction(/*group_count=*/8u, /*synchronize=*/true);
}

TEST(ConSanDeviceRecordReplaySamplingReductionTest, SparseIncorrect) {
  run_sampling_reduction(/*group_count=*/8u, /*synchronize=*/false);
}

} // namespace
