// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Max-then-sum reduction distilled from fused and split softmax kernels.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_count;
  float maximum;
  float denominator;
  uint32_t control[128];
};

__global__ __launch_bounds__(128) void two_stage_softmax_for_consan(DeviceState *state) {
  __shared__ float partial_max[4];
  __shared__ float other_wave_sum;
  __shared__ float row_max;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  const float input = -static_cast<float>(tid & 7u);
  state->control[tid] = 0x7b000000u | tid;

  float maximum = input;
#pragma unroll 1
  for (uint32_t offset = warpSize / 2u; offset != 0u; offset /= 2u)
    maximum = fmaxf(maximum, __shfl_down(maximum, offset, warpSize));
  if (lane == 0u) {
    partial_max[wave] = maximum;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  __syncthreads();
  if (tid == 0u) {
    float value = partial_max[0];
    for (uint32_t producer = 1u; producer < state->wave_count; ++producer)
      value = fmaxf(value, partial_max[producer]);
    row_max = value;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  __syncthreads();

  const float weight = exp2f(input - row_max);
  float sum = weight;
#pragma unroll 1
  for (uint32_t offset = warpSize / 2u; offset != 0u; offset /= 2u)
    sum += __shfl_down(sum, offset, warpSize);
  // With exactly two waves, lane zero of the second wave publishes its
  // partial to lane zero of the first. This is the small-scale form of the
  // inter-wave handoff used by split softmax kernels, and keeps the broken
  // edge explicit instead of hiding it in a later loop iteration.
  if (wave == 1u && lane == 0u) {
    other_wave_sum = sum;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  // The incorrect member preserves both numerical stages but removes the
  // publication edge for the per-wave denominators.
  if (state->synchronize)
    __syncthreads();
  if (tid == 0u) {
    state->maximum = row_max;
    state->denominator = sum + other_wave_sum;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
}

void run_two_stage_softmax(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  state.wave_count = 2u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  two_stage_softmax_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_count, 2u);
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], 0x7b000000u | tid) << "thread=" << tid;
  if (synchronize) {
    EXPECT_EQ(state.maximum, 0.0f);
    EXPECT_EQ(state.denominator, static_cast<float>(thread_count / 8u) * (255.0f / 128.0f));
  }
}

TEST(ConSanDeviceTwoStageSoftmaxTest, Correct) { run_two_stage_softmax(true); }

TEST(ConSanDeviceTwoStageSoftmaxTest, Incorrect) { run_two_stage_softmax(false); }

} // namespace
