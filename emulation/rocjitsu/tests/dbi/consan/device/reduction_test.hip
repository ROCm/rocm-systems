// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Barrier reduction distilled from softmax, norm, and generated reductions.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t reduction;
  uint32_t control[128];
};

__global__ __launch_bounds__(128) void shared_reduction_for_consan(DeviceState *state) {
  extern __shared__ uint32_t values[];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (wave == 1)
    values[lane] = lane + 1u;
  state->control[tid] = 0x72000000u + tid;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (state->synchronize)
    __syncthreads();

#pragma unroll 1
  for (uint32_t stride = warpSize / 2u; stride != 0; stride /= 2u) {
    if (wave == 0 && lane < stride)
      values[lane] += values[lane + stride];
    asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
    __syncthreads();
  }
  if (tid == 0)
    state->reduction = values[0];
}

void run_reduction(bool synchronize_initial) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  DeviceState state{};
  state.synchronize = synchronize_initial;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  shared_reduction_for_consan<<<1, thread_count, thread_count * sizeof(uint32_t)>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x72000000u + tid) << "thread=" << tid;
  if (synchronize_initial)
    EXPECT_EQ(state.reduction, thread_count / 2u * (thread_count / 2u + 1u) / 2u);
}

TEST(ConSanDeviceReductionTest, Correct) { run_reduction(true); }

TEST(ConSanDeviceReductionTest, Incorrect) { run_reduction(false); }

} // namespace
