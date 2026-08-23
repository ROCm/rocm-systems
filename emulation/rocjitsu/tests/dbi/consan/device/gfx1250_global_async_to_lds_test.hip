// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Ordinary (non-cluster) CDNA5 tensor-data movement distilled from the MXF4
/// staging and RCCL AsyncDataCopier paths. One wave transfers four disjoint
/// global regions directly into LDS through every supported
/// GLOBAL_LOAD_ASYNC_TO_LDS width while a peer wave consumes them. The
/// adjacent pair differs only in the workgroup publication barrier.

#include "consan_device_test_support.h"

#include <array>

namespace {

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr uint32_t kCanary = 0xccccccccu;

using GlobalChar = __attribute__((address_space(1))) char;
using LdsChar = __attribute__((address_space(3))) char;
using GlobalI32 = __attribute__((address_space(1))) int32_t;
using LdsI32 = __attribute__((address_space(3))) int32_t;
using I32x2 = __attribute__((ext_vector_type(2))) int32_t;
using I32x4 = __attribute__((ext_vector_type(4))) int32_t;
using GlobalI32x2 = __attribute__((address_space(1))) I32x2;
using LdsI32x2 = __attribute__((address_space(3))) I32x2;
using GlobalI32x4 = __attribute__((address_space(1))) I32x4;
using LdsI32x4 = __attribute__((address_space(3))) I32x4;

struct alignas(16) WideWords {
  uint32_t word[4];
};

struct InputState {
  uint8_t input_b8[kWaveSize];
  uint32_t input_b32[kWaveSize];
  uint64_t input_b64[kWaveSize];
  WideWords input_b128[kWaveSize];
};

struct DeviceState {
  uint32_t synchronize;
  uint8_t observed_b8[kWaveSize];
  uint32_t observed_b32[kWaveSize];
  uint64_t observed_b64[kWaveSize];
  WideWords observed_b128[kWaveSize];
  uint32_t control[kThreadCount];
};

__global__
__launch_bounds__(kThreadCount) void gfx1250_global_async_to_lds_for_consan(const InputState *input,
                                                                            DeviceState *state) {
  __shared__ uint8_t staged_b8[kWaveSize];
  __shared__ uint32_t staged_b32[kWaveSize];
  __shared__ uint64_t staged_b64[kWaveSize];
  __shared__ WideWords staged_b128[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  if (tid < kWaveSize) {
    __builtin_amdgcn_global_load_async_to_lds_b8((GlobalChar *)(&input->input_b8[lane]),
                                                 (LdsChar *)(&staged_b8[lane]), 0, 0);
    __builtin_amdgcn_global_load_async_to_lds_b32((GlobalI32 *)(&input->input_b32[lane]),
                                                  (LdsI32 *)(&staged_b32[lane]), 0, 0);
    __builtin_amdgcn_global_load_async_to_lds_b64((GlobalI32x2 *)(&input->input_b64[lane]),
                                                  (LdsI32x2 *)(&staged_b64[lane]), 0, 0);
    __builtin_amdgcn_global_load_async_to_lds_b128((GlobalI32x4 *)(&input->input_b128[lane]),
                                                   (LdsI32x4 *)(&staged_b128[lane]), 0, 0);
    __builtin_amdgcn_s_wait_asynccnt(0);
  }
  if (state->synchronize != 0u)
    __syncthreads();
  if (tid >= kWaveSize) {
    state->observed_b8[lane] = staged_b8[lane];
    state->observed_b32[lane] = consan_device_test::load_lds_word(&staged_b32[lane]);
    state->observed_b64[lane] = staged_b64[lane];
    state->observed_b128[lane] = staged_b128[lane];
  }
  state->control[tid] = 0x1250a000u | tid;
}

void run_gfx1250_global_async_to_lds(bool synchronize) {
  InputState input{};
  for (uint32_t lane = 0u; lane < kWaveSize; ++lane) {
    input.input_b8[lane] = static_cast<uint8_t>(0x40u + lane);
    input.input_b32[lane] = 0x32000000u | (lane * 17u + 3u);
    input.input_b64[lane] = 0x6400000000000000ull | (static_cast<uint64_t>(lane) * 31u + 5u);
    for (uint32_t word = 0u; word < 4u; ++word)
      input.input_b128[lane].word[word] = 0x12800000u | (lane << 8u) | word;
  }
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  for (uint32_t lane = 0u; lane < kWaveSize; ++lane) {
    initial.observed_b8[lane] = static_cast<uint8_t>(kCanary);
    initial.observed_b32[lane] = kCanary;
    initial.observed_b64[lane] = (static_cast<uint64_t>(kCanary) << 32u) | kCanary;
    for (uint32_t word = 0u; word < 4u; ++word)
      initial.observed_b128[lane].word[word] = kCanary;
  }

  InputState *device_input = nullptr;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_input, sizeof(InputState)));
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_input, &input, sizeof(InputState), hipMemcpyHostToDevice));
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
    for (uint32_t lane = 0u; lane < kWaveSize; ++lane) {
      EXPECT_EQ(result.observed_b8[lane], input.input_b8[lane]) << "lane=" << lane;
      EXPECT_EQ(result.observed_b32[lane], input.input_b32[lane]) << "lane=" << lane;
      EXPECT_EQ(result.observed_b64[lane], input.input_b64[lane]) << "lane=" << lane;
      for (uint32_t word = 0u; word < 4u; ++word)
        EXPECT_EQ(result.observed_b128[lane].word[word], input.input_b128[lane].word[word])
            << "lane=" << lane << " word=" << word;
    }
  }
}

TEST(ConSanDeviceGfx1250GlobalAsyncToLdsTest, Correct) { run_gfx1250_global_async_to_lds(true); }

TEST(ConSanDeviceGfx1250GlobalAsyncToLdsTest, Incorrect) { run_gfx1250_global_async_to_lds(false); }

} // namespace
