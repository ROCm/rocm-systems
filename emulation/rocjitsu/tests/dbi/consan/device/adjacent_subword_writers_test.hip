// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Two waves publish adjacent FP16-sized LDS halves of one dword. Both members
/// keep those byte ranges disjoint. The correct member synchronizes a
/// third-wave consumer, while the incorrect member omits that barrier and must
/// diagnose the resulting read/write race.

#include "consan_device_test_support.h"

#include <hip/hip_fp16.h>

namespace {

constexpr uint32_t kMaximumThreadCount = 3u * 64u;

struct DeviceState {
  uint32_t synchronize;
  float observed[2];
  uint32_t control[kMaximumThreadCount];
};

__global__ void adjacent_subword_writers_for_consan(DeviceState *state) {
  __shared__ __half halves[2];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  state->control[tid] = 0xd2000000u | tid;

  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (wave == 0u && lane == 0u)
    halves[0] = __float2half(1.5f);
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
  if (wave == 1u && lane == 0u)
    halves[1] = __float2half(2.5f);
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");

  if (state->synchronize != 0u)
    __syncthreads();
  if (wave == 2u && lane == 0u) {
    state->observed[0] = __half2float(halves[0]);
    state->observed[1] = __half2float(halves[1]);
  }
  asm volatile(RJ_CONSAN_DEVICE_CAVE ::: "memory");
}

void run_adjacent_subword_writers(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));

  const uint32_t thread_count = 3u * consan_device_test::active_wave_size();
  adjacent_subword_writers_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    EXPECT_EQ(result.control[tid], 0xd2000000u | tid) << "thread=" << tid;
  if (synchronize) {
    EXPECT_FLOAT_EQ(result.observed[0], 1.5f);
    EXPECT_FLOAT_EQ(result.observed[1], 2.5f);
  }
}

TEST(ConSanDeviceAdjacentSubwordWritersTest, Correct) { run_adjacent_subword_writers(true); }

TEST(ConSanDeviceAdjacentSubwordWritersTest, Incorrect) { run_adjacent_subword_writers(false); }

} // namespace
