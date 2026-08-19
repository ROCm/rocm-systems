// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// A compact production-shaped FP16 WMMA pipeline with a cross-wave LDS stage,
/// dynamic private state, helper ownership, and a full architectural VGPR
/// allocation. The first accumulator remains live across the publication edge.

#include "consan_device_test_support.h"

namespace {

using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));
#if defined(__gfx1100__)
using wmma_input_t = _Float16 __attribute__((ext_vector_type(16)));
#else
using wmma_input_t = f16x8_t;
#endif

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr uint32_t kFragmentWords = 4u;

struct DeviceState {
  uint32_t synchronize;
  float output[kWaveSize * 8u];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ f32x8_t wmma(wmma_input_t lhs, wmma_input_t rhs, f32x8_t accumulator) {
#if defined(__gfx1100__)
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(lhs, rhs, accumulator);
#else
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(lhs, rhs, accumulator);
#endif
}

__device__ __noinline__ void publish_fragment(uint32_t address, uint4 value) {
  register uint32_t value0 asm("v20") = value.x;
  register uint32_t value1 asm("v21") = value.y;
  register uint32_t value2 asm("v22") = value.z;
  register uint32_t value3 asm("v23") = value.w;
  asm volatile("ds_store_b128 %0, v[20:23]\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(address), "v"(value0), "v"(value1), "v"(value2), "v"(value3)
               : "memory");
}

__device__ __noinline__ uint4 consume_fragment(uint32_t address) {
  register uint32_t value0 asm("v20") = 0u;
  register uint32_t value1 asm("v21") = 0u;
  register uint32_t value2 asm("v22") = 0u;
  register uint32_t value3 asm("v23") = 0u;
  asm volatile("ds_load_b128 v[20:23], %4\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
#if defined(__gfx1201__)
               "s_wait_dscnt 0\n\t"
#else
               "s_waitcnt lgkmcnt(0)\n\t"
#endif
               : "+v"(value0), "+v"(value1), "+v"(value2), "+v"(value3)
               : "v"(address)
               : "memory");
  return make_uint4(value0, value1, value2, value3);
}

__global__ __launch_bounds__(kThreadCount) void rdna_wmma_pipeline_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint4 staged[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t private_count = (tid & 7u) + 1u;
  auto *private_words =
      static_cast<volatile uint32_t *>(__builtin_alloca(private_count * sizeof(uint32_t)));
  for (uint32_t i = 0; i < private_count; ++i)
    private_words[i] = 0x71000000u ^ (tid * 17u + i);
  const uint32_t selected = private_words[(tid * 3u + 1u) % private_count];

  wmma_input_t ones;
  f32x8_t accumulator{};
#pragma unroll
  for (uint32_t i = 0; i < sizeof(wmma_input_t) / sizeof(_Float16); ++i)
    ones[i] = static_cast<_Float16>(1.0f);
  accumulator = wmma(ones, ones, accumulator);

  // Leave only a narrow tail on gfx11 so persistent dispatch identity still
  // fits while the much larger per-site probe must spill. gfx12 exercises the
  // fully allocated bank and its scalar persistent-state fallback.
#if defined(__gfx1100__)
  asm volatile("v_mov_b32 v247, %0\n\t" : : "v"(selected) : "v247");
#else
  asm volatile("v_mov_b32 v255, %0\n\t" : : "v"(selected) : "v255");
#endif

  const uint32_t address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&staged[lane]) & 0xffffu);
  const uint4 packed_ones = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u);
  if (tid < kWaveSize)
    publish_fragment(address, packed_ones);
  if (state->synchronize != 0u)
    __syncthreads();
  if (tid >= kWaveSize) {
    const uint4 packed = consume_fragment(address);
    const f16x8_t consumed8 = __builtin_bit_cast(f16x8_t, packed);
    wmma_input_t consumed;
#pragma unroll
    for (uint32_t i = 0; i < sizeof(wmma_input_t) / sizeof(_Float16); ++i)
      consumed[i] = consumed8[i & 7u];
    accumulator = wmma(ones, consumed, accumulator);
#pragma unroll
    for (uint32_t i = 0; i < 8u; ++i)
      state->output[lane * 8u + i] = accumulator[i];
  }
  state->control[tid] = selected;
}

uint32_t expected_private_value(uint32_t tid) {
  const uint32_t private_count = (tid & 7u) + 1u;
  return 0x71000000u ^ (tid * 17u + (tid * 3u + 1u) % private_count);
}

void run_rdna_wmma_pipeline(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  rdna_wmma_pipeline_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], expected_private_value(tid)) << "thread=" << tid;
  if (!synchronize)
    return;
  for (uint32_t i = 0; i < kWaveSize * 8u; ++i)
    EXPECT_EQ(result.output[i], 32.0f) << "element=" << i;
}

TEST(ConSanDeviceRdnaWmmaPipelineTest, Correct) { run_rdna_wmma_pipeline(true); }
TEST(ConSanDeviceRdnaWmmaPipelineTest, Incorrect) { run_rdna_wmma_pipeline(false); }

} // namespace
