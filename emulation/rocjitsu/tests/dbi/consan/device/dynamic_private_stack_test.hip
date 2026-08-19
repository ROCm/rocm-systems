// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Dynamic private allocation retained around an LDS publication edge.

#include "consan_device_test_support.h"

namespace {

struct DeviceState {
  uint32_t synchronize;
  uint32_t observed[64];
  uint32_t control[128];
  uint32_t epilogue_input[128];
  uint32_t epilogue_output[128];
};

__global__ __launch_bounds__(128) void dynamic_private_stack_for_consan(DeviceState *state) {
  __shared__ uint32_t tile[64];
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t wave = threadIdx.x / warpSize;
  const uint32_t private_count = (threadIdx.x & 15u) + 1u;
  auto *private_values =
      static_cast<volatile uint32_t *>(__builtin_alloca(private_count * sizeof(uint32_t)));
  for (uint32_t i = 0; i < private_count; ++i)
    private_values[i] = (threadIdx.x + 1u) * 100u + i;
  const uint32_t selected = private_values[(threadIdx.x * 7u + 3u) % private_count];
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (wave == 0)
    tile[lane] = selected;
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (state->synchronize)
    __syncthreads();
  if (wave == 1)
    state->observed[lane] = tile[lane];
  state->control[threadIdx.x] = selected;
  state->epilogue_output[threadIdx.x] =
      consan_device_test::fused_epilogue_mix(state->epilogue_input, threadIdx.x, 127u);
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
}

uint32_t expected_private_value(uint32_t tid) {
  const uint32_t private_count = (tid & 15u) + 1u;
  return (tid + 1u) * 100u + (tid * 7u + 3u) % private_count;
}

void run_dynamic_private_stack(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize;
  for (uint32_t i = 0; i < 128u; ++i)
    state.epilogue_input[i] = 0xc0010000u ^ (i * 0x1021u);
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  dynamic_private_stack_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(state.control[tid], expected_private_value(tid)) << "thread=" << tid;
  for (uint32_t tid = 0; tid < thread_count; ++tid) {
    EXPECT_EQ(state.epilogue_output[tid],
              consan_device_test::fused_epilogue_mix_reference(state.epilogue_input, tid, 127u))
        << "thread=" << tid;
  }
  if (synchronize)
    for (uint32_t lane = 0; lane < wave_size; ++lane)
      EXPECT_EQ(state.observed[lane], expected_private_value(lane)) << "lane=" << lane;
}

TEST(ConSanDeviceDynamicPrivateStackTest, Correct) { run_dynamic_private_stack(true); }

TEST(ConSanDeviceDynamicPrivateStackTest, Incorrect) { run_dynamic_private_stack(false); }

} // namespace
