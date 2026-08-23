// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Ordinary (non-cluster) CDNA5 tensor-data movement distilled from the MXF4
/// staging path. One wave transfers global words directly into LDS through
/// GLOBAL_LOAD_ASYNC_TO_LDS while a peer wave consumes them. The adjacent
/// pair differs only in the workgroup publication barrier.

#include "consan_device_test_support.h"

#include <array>

namespace {

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;

struct DeviceState {
  uint32_t synchronize;
  uint32_t observed[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void global_load_async_to_lds_b32(uint32_t lds_address,
                                                             uint32_t global_offset,
                                                             const uint32_t *global_base) {
  asm volatile("global_load_async_to_lds_b32 %0, %1, %2\n\t"
               "s_wait_asynccnt 0\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(lds_address), "v"(global_offset), "s"(global_base)
               : "memory");
}

__global__
__launch_bounds__(kThreadCount) void gfx1250_global_async_to_lds_for_consan(const uint32_t *input,
                                                                            DeviceState *state) {
  __shared__ uint32_t staged[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  if (tid < kWaveSize) {
    const uint32_t lds_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&staged[lane]) & 0xffffu);
    global_load_async_to_lds_b32(lds_address, lane * sizeof(uint32_t), input);
  }
  if (state->synchronize != 0u)
    __syncthreads();
  if (tid >= kWaveSize)
    state->observed[lane] = consan_device_test::load_lds_word(&staged[lane]);
  state->control[tid] = 0x1250a000u | tid;
}

void run_gfx1250_global_async_to_lds(bool synchronize) {
  std::array<uint32_t, kWaveSize> input{};
  for (uint32_t lane = 0u; lane < kWaveSize; ++lane)
    input[lane] = 0xa5100000u | (lane * 29u + 7u);
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  uint32_t *device_input = nullptr;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_input, sizeof(input)));
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_input, input.data(), sizeof(input), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  gfx1250_global_async_to_lds_for_consan<<<1, kThreadCount>>>(device_input, device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  HIP_ASSERT(hipFree(device_input));
  for (uint32_t tid = 0u; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0x1250a000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0u; lane < kWaveSize; ++lane)
      EXPECT_EQ(result.observed[lane], input[lane]) << "lane=" << lane;
  }
}

TEST(ConSanDeviceGfx1250GlobalAsyncToLdsTest, Correct) { run_gfx1250_global_async_to_lds(true); }

TEST(ConSanDeviceGfx1250GlobalAsyncToLdsTest, Incorrect) { run_gfx1250_global_async_to_lds(false); }

} // namespace
