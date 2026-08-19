// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// MFMA accumulator liveness across a cross-wave LDS publication edge,
/// distilled from the CDNA3/CDNA4 hip-moi attention workloads.

#include "consan_device_test_support.h"

#include <hip/hip_fp16.h>

namespace {

using f16x4_t = _Float16 __attribute__((ext_vector_type(4)));
using f32x4_t = float __attribute__((ext_vector_type(4)));

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr uint32_t kAccumulatorElements = 4u;

struct DeviceState {
  uint32_t synchronize;
  float output[kThreadCount * kAccumulatorElements];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ f16x4_t splat_half(float value) {
  f16x4_t result;
#pragma unroll
  for (uint32_t i = 0; i < kAccumulatorElements; ++i)
    result[i] = static_cast<_Float16>(value);
  return result;
}

__device__ __forceinline__ void begin_accvgpr_mfma(f16x4_t lhs, f16x4_t rhs) {
  const uint32_t zero = 0u;
  asm volatile("v_accvgpr_write_b32 a0, %2\n\t"
               "v_accvgpr_write_b32 a1, %2\n\t"
               "v_accvgpr_write_b32 a2, %2\n\t"
               "v_accvgpr_write_b32 a3, %2\n\t"
               "v_mfma_f32_16x16x16_f16 a[0:3], %0, %1, a[0:3]\n\t"
               :
               : "v"(lhs), "v"(rhs), "v"(zero)
               : "a0", "a1", "a2", "a3", "memory");
}

__device__ __forceinline__ f32x4_t finish_accvgpr_mfma(f16x4_t lhs, f16x4_t rhs) {
  f32x4_t result;
  asm volatile("v_mfma_f32_16x16x16_f16 a[0:3], %4, %5, a[0:3]\n\t"
               "s_nop 15\n\t"
               "v_accvgpr_read_b32 %0, a0\n\t"
               "v_accvgpr_read_b32 %1, a1\n\t"
               "v_accvgpr_read_b32 %2, a2\n\t"
               "v_accvgpr_read_b32 %3, a3\n\t"
               : "=v"(result[0]), "=v"(result[1]), "=v"(result[2]), "=v"(result[3])
               : "v"(lhs), "v"(rhs)
               : "a0", "a1", "a2", "a3", "memory");
  return result;
}

__global__ __launch_bounds__(kThreadCount) void cdna_mfma_pipeline_for_consan(DeviceState *state) {
  __shared__ f16x4_t published[kWaveSize];

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const f16x4_t ones = splat_half(1.0f);

  // Keep a realistic ordinary-register allocation below the accumulator
  // boundary. The fixed MFMA operands alone otherwise let the compiler place
  // a0 immediately above a very small v-register window that cannot hold even
  // a spill-backed sanitizer probe.
  asm volatile("v_mov_b32 v31, 0" ::: "v31");

  // Keep one accumulator live across the publication edge. Instrumentation
  // must neither allocate through it nor change the architectural AccVGPR
  // boundary selected for the owning kernel.
  begin_accvgpr_mfma(ones, ones);

  if (tid < kWaveSize) {
    published[lane] = ones;
    asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  }

  if (state->synchronize != 0u)
    __syncthreads();

  const f16x4_t consumed = published[lane];
  asm volatile(RJ_CONSAN_DEVICE_LARGE_CAVE ::: "memory");
  const f32x4_t accumulator = finish_accvgpr_mfma(ones, consumed);

#pragma unroll
  for (uint32_t i = 0; i < kAccumulatorElements; ++i)
    state->output[tid * kAccumulatorElements + i] = accumulator[i];
  state->control[tid] = 0xcd000000u | tid;
}

void run_cdna_mfma_pipeline(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;

  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));

  cdna_mfma_pipeline_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  for (uint32_t tid = 0; tid < kThreadCount; ++tid) {
    EXPECT_EQ(result.control[tid], 0xcd000000u | tid) << "tid=" << tid;
    if (!synchronize)
      continue;
    for (uint32_t i = 0; i < kAccumulatorElements; ++i)
      EXPECT_EQ(result.output[tid * kAccumulatorElements + i], 32.0f)
          << "tid=" << tid << " element=" << i;
  }
}

TEST(ConSanDeviceCdnaMfmaPipelineTest, Correct) { run_cdna_mfma_pipeline(true); }

TEST(ConSanDeviceCdnaMfmaPipelineTest, Incorrect) { run_cdna_mfma_pipeline(false); }

} // namespace
