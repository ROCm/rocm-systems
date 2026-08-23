// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Native gfx1250 FP16 WMMA staging distilled from the target's Jakub matmul
/// and attention workloads. A producer wave publishes one 32-byte K=32
/// fragment per lane through LDS; a consumer wave keeps an earlier WMMA result
/// live across the handoff, consumes the fragment, and checks the exact second
/// matrix accumulation.

#include "consan_device_test_support.h"

namespace {

using f16x16_t = _Float16 __attribute__((ext_vector_type(16)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr float kExpectedResult = 64.0f;

struct DeviceState {
  uint32_t synchronize;
  uint32_t head_canary;
  float output[kWaveSize * 8u];
  uint32_t tail_canary;
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ f32x8_t wmma(f16x16_t lhs, f16x16_t rhs, f32x8_t accumulator) {
  return __builtin_amdgcn_wmma_f32_16x16x32_f16(false, lhs, false, rhs, static_cast<short>(0),
                                                accumulator, false, false);
}

__device__ __forceinline__ f16x16_t fp16_ones() {
  f16x16_t value;
#pragma unroll
  for (uint32_t i = 0; i < 16u; ++i)
    value[i] = static_cast<_Float16>(1.0f);
  return value;
}

__global__
__launch_bounds__(kThreadCount) void gfx1250_fp16_wmma_pipeline_for_consan(DeviceState *state) {
  __shared__ __align__(32) f16x16_t staged[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const f16x16_t ones = fp16_ones();
  f32x8_t accumulator{};

  // Keep target-native matrix state live across the publication edge. Each
  // invocation contributes exactly 32 products of 1.0 to every accumulator.
  accumulator = wmma(ones, ones, accumulator);
  if (tid < kWaveSize) {
    staged[lane] = ones;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }
  if (state->synchronize != 0u)
    __syncthreads();
  if (tid >= kWaveSize) {
    const f16x16_t consumed = staged[lane];
    accumulator = wmma(ones, consumed, accumulator);
#pragma unroll
    for (uint32_t word = 0; word < 8u; ++word)
      state->output[lane * 8u + word] = accumulator[word];
  }
  state->control[tid] = 0x1250f000u | tid;
}

void run_gfx1250_fp16_wmma_pipeline(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  initial.head_canary = 0x13579bdfu;
  initial.tail_canary = 0x2468ace0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  gfx1250_fp16_wmma_pipeline_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(result.head_canary, 0x13579bdfu);
  EXPECT_EQ(result.tail_canary, 0x2468ace0u);
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0x1250f000u | tid) << "thread=" << tid;
  if (!synchronize)
    return;
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    for (uint32_t word = 0; word < 8u; ++word)
      EXPECT_EQ(result.output[lane * 8u + word], kExpectedResult)
          << "lane=" << lane << " result word=" << word;
}

TEST(ConSanDeviceGfx1250Fp16WmmaPipelineTest, Correct) { run_gfx1250_fp16_wmma_pipeline(true); }

TEST(ConSanDeviceGfx1250Fp16WmmaPipelineTest, Incorrect) { run_gfx1250_fp16_wmma_pipeline(false); }

} // namespace
