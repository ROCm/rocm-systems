// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Cooperative multi-wave LDS staging distilled from the generated rocBLAS
/// SGEMM tile selected by the physical gfx950 validation row. The portable
/// behavioral idea applies to both wave32 and wave64 targets: every wave
/// publishes a disjoint part of a 16-KiB tile and consumes a peer wave's part.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kThreadCount = 256u;
constexpr uint32_t kWordsPerThread = 16u;
constexpr uint32_t kTileWords = kThreadCount * kWordsPerThread;

struct DeviceState {
  uint32_t synchronize;
  uint32_t wave_size;
  uint32_t observed[kThreadCount];
  uint32_t control[kThreadCount];
};

__host__ __device__ __forceinline__ uint32_t staged_value(uint32_t producer, uint32_t element) {
  return 0xa5000000u ^ (producer * 0x1021u) ^ (element * 0x45d9u);
}

__host__ __device__ __forceinline__ uint32_t accumulate_value(uint32_t accumulator, uint32_t value,
                                                              uint32_t element) {
  return consan_device_test::rotate_left(accumulator ^ value, (element % 13u) + 1u) + 0x9e3779b9u;
}

__global__ __launch_bounds__(kThreadCount) void multiwave_gemm_tile_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint32_t tile[kTileWords];

  const uint32_t tid = threadIdx.x;
  const uint32_t wave_size = warpSize;
  const uint32_t peer_tid = (tid + wave_size) % kThreadCount;

#pragma unroll
  for (uint32_t element = 0; element < kWordsPerThread; ++element)
    tile[tid * kWordsPerThread + element] = staged_value(tid, element);
  asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");

  // This branch deliberately mirrors the generated SGEMM control shape: both
  // paths reconverge immediately before the workgroup publication edge.
  if ((state->wave_size & 1u) != 0u)
    state->control[tid] = 0u;
  else
    state->control[tid] = 0xb6000000u | tid;

  if (state->synchronize != 0u)
    __syncthreads();

  uint32_t accumulator = 0x13579bdfu ^ tid;
#pragma unroll
  for (uint32_t element = 0; element < kWordsPerThread; ++element) {
    const uint32_t value = tile[peer_tid * kWordsPerThread + element];
    accumulator = accumulate_value(accumulator, value, element);
  }
  asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  state->observed[tid] = accumulator;
}

uint32_t expected_accumulator(uint32_t tid, uint32_t wave_size) {
  const uint32_t peer_tid = (tid + wave_size) % kThreadCount;
  uint32_t accumulator = 0x13579bdfu ^ tid;
  for (uint32_t element = 0; element < kWordsPerThread; ++element)
    accumulator = accumulate_value(accumulator, staged_value(peer_tid, element), element);
  return accumulator;
}

void run_multiwave_gemm_tile(bool synchronize) {
  DeviceState state{};
  state.synchronize = synchronize ? 1u : 0u;
  state.wave_size = consan_device_test::active_wave_size();

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  multiwave_gemm_tile_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  EXPECT_EQ(state.wave_size, consan_device_test::active_wave_size());
  for (uint32_t tid = 0; tid < kThreadCount; ++tid) {
    EXPECT_EQ(state.control[tid], 0xb6000000u | tid) << "thread=" << tid;
    if (synchronize)
      EXPECT_EQ(state.observed[tid], expected_accumulator(tid, state.wave_size))
          << "thread=" << tid;
  }
}

TEST(ConSanDeviceMultiWaveGemmTileTest, Correct) { run_multiwave_gemm_tile(true); }

TEST(ConSanDeviceMultiWaveGemmTileTest, Incorrect) { run_multiwave_gemm_tile(false); }

} // namespace
