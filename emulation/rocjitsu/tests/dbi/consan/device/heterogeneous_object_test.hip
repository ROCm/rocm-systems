// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// A small heterogeneous framework object distilled from Qwen, PyTorch, and
/// Aorta. The object combines attention-like tile publication, a
/// mixed-precision optimizer stage, MoE-style atomic routing, a shared helper,
/// and an unexecuted high-pressure kernel. The adjacent members differ only in
/// the attention publication edge; every other dispatched result must remain
/// exact. This exercises whole-object transformation without checking in a
/// production-sized model or depending on the prototype's placement layout.

#include "consan_device_test_support.h"

#include <hip/hip_fp16.h>

namespace {

constexpr uint32_t kMaximumLanes = 64u;
constexpr uint32_t kMaximumThreads = 128u;
constexpr uint32_t kExpertCount = 4u;

struct DeviceState {
  uint32_t synchronize_attention;
  uint32_t wave_size;
  uint32_t attention[kMaximumLanes];
  float optimizer[kMaximumLanes];
  uint32_t expert_counts[kExpertCount];
  uint32_t control[3][kMaximumThreads];
};

__device__ __noinline__ uint32_t shared_epilogue(uint32_t value, uint32_t lane) {
  return consan_device_test::rotate_left(value ^ (0x9e3779b9u + lane), (lane & 7u) + 1u);
}

uint32_t shared_epilogue_reference(uint32_t value, uint32_t lane) {
  return consan_device_test::rotate_left(value ^ (0x9e3779b9u + lane), (lane & 7u) + 1u);
}

__global__ __launch_bounds__(128) void attention_tile_for_consan(DeviceState *state) {
  __shared__ uint32_t tile[kMaximumLanes];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[0][tid] = 0xa1000000u | tid;

  if (wave == 1u)
    consan_device_test::store_lds_word(&tile[lane], 0x12500000u | lane);

  // The incorrect member removes only this publication edge. The later
  // optimizer and routing dispatches are identical in both members.
  if (state->synchronize_attention)
    __syncthreads();

  if (wave == 0u) {
    const uint32_t value = consan_device_test::load_lds_word(&tile[lane]);
    state->attention[lane] = shared_epilogue(value, lane);
  }
}

__global__ __launch_bounds__(128) void optimizer_stage_for_consan(DeviceState *state) {
  __shared__ __half gradients[kMaximumLanes];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[1][tid] = 0xa2000000u | tid;

  if (wave == 1u) {
    gradients[lane] = __float2half(static_cast<float>((lane & 7u) + 1u) * 0.125f);
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  __syncthreads();
  if (wave == 0u) {
    const float gradient = __half2float(gradients[lane]);
    const float parameter = 4.0f + static_cast<float>(lane & 7u) * 0.125f;
    state->optimizer[lane] = parameter - 0.25f * gradient - 0.125f * parameter;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
}

__global__ __launch_bounds__(128) void moe_route_for_consan(DeviceState *state) {
  __shared__ uint32_t bins[kExpertCount];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[2][tid] = 0xa3000000u | tid;

  if (tid < kExpertCount)
    bins[tid] = 0u;
  __syncthreads();
  if (wave == 1u && lane < 13u) {
    const uint32_t route = lane % 3u;
    const uint32_t expert = route == 2u ? 3u : route;
    atomicAdd(&bins[expert], 1u);
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  __syncthreads();
  if (tid < kExpertCount)
    state->expert_counts[tid] = bins[tid];
}

// This kernel is deliberately present but not dispatched. It gives the same
// loaded object an independent, register-rich owner with a private value array
// and multiple LDS phases, as generated framework libraries do. Its
// existence may affect transformation planning; it must not affect runtime
// attribution or the exact results of the three selected kernels.
__global__ __launch_bounds__(128) void unexecuted_pressure_for_consan(DeviceState *state,
                                                                      uint32_t seed) {
  __shared__ uint32_t staged[kMaximumLanes];
  uint32_t values[16];
#pragma unroll
  for (uint32_t i = 0u; i < 16u; ++i)
    values[i] = shared_epilogue(seed + 17u * i + threadIdx.x, i);
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  if (threadIdx.x < kMaximumLanes)
    consan_device_test::store_lds_word(&staged[threadIdx.x], values[threadIdx.x & 15u]);
  __syncthreads();
  if (threadIdx.x < kMaximumLanes) {
    const uint32_t staged_value = consan_device_test::load_lds_word(&staged[threadIdx.x]);
    state->attention[threadIdx.x] = staged_value ^ values[(threadIdx.x + 3u) & 15u];
  }
}

void run_heterogeneous_object(bool synchronize_attention) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize_attention = synchronize_attention;
  state.wave_size = wave_size;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  attention_tile_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  optimizer_stage_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  moe_route_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_size, wave_size);
  for (uint32_t stage = 0u; stage < 3u; ++stage) {
    for (uint32_t tid = 0u; tid < thread_count; ++tid)
      EXPECT_EQ(state.control[stage][tid], (0xa1000000u + stage * 0x01000000u) | tid)
          << "stage=" << stage << " thread=" << tid;
  }
  for (uint32_t lane = 0u; lane < wave_size; ++lane) {
    const float gradient = static_cast<float>((lane & 7u) + 1u) * 0.125f;
    const float parameter = 4.0f + static_cast<float>(lane & 7u) * 0.125f;
    EXPECT_FLOAT_EQ(state.optimizer[lane], parameter - 0.25f * gradient - 0.125f * parameter)
        << "lane=" << lane;
    if (synchronize_attention) {
      EXPECT_EQ(state.attention[lane], shared_epilogue_reference(0x12500000u | lane, lane))
          << "lane=" << lane;
    }
  }
  EXPECT_EQ(state.expert_counts[0], 5u);
  EXPECT_EQ(state.expert_counts[1], 4u);
  EXPECT_EQ(state.expert_counts[2], 0u);
  EXPECT_EQ(state.expert_counts[3], 4u);
}

TEST(ConSanDeviceHeterogeneousObjectTest, Correct) { run_heterogeneous_object(true); }

TEST(ConSanDeviceHeterogeneousObjectTest, Incorrect) { run_heterogeneous_object(false); }

} // namespace
