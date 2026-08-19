// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Production-shaped gfx12 FP8 WMMA staging reduced from the empirical FP8
/// matmul: B128 LDS traffic, split workgroup synchronization, live WMMA state,
/// register pressure, exact outputs, and surrounding canaries.

#include "consan_device_test_support.h"

namespace {

using i32x2_t = int32_t __attribute__((ext_vector_type(2)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;

struct DeviceState {
  uint32_t synchronize;
  uint32_t head_canary;
  float output[kWaveSize * 8u];
  uint32_t tail_canary;
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ f32x8_t fp8_wmma(i32x2_t lhs, i32x2_t rhs, f32x8_t accumulator) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(lhs, rhs, accumulator);
}

__device__ __forceinline__ void store_b128(uint32_t address, uint4 value) {
  register uint32_t value0 asm("v20") = value.x;
  register uint32_t value1 asm("v21") = value.y;
  register uint32_t value2 asm("v22") = value.z;
  register uint32_t value3 asm("v23") = value.w;
  asm volatile("ds_store_b128 %0, v[20:23]\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(address), "v"(value0), "v"(value1), "v"(value2), "v"(value3)
               : "v20", "v21", "v22", "v23", "memory");
}

__device__ __forceinline__ uint4 load_b128(uint32_t address) {
  register uint32_t value0 asm("v20") = 0u;
  register uint32_t value1 asm("v21") = 0u;
  register uint32_t value2 asm("v22") = 0u;
  register uint32_t value3 asm("v23") = 0u;
  asm volatile("ds_load_b128 v[20:23], %4\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               : "+v"(value0), "+v"(value1), "+v"(value2), "+v"(value3)
               : "v"(address)
               : "memory");
  return make_uint4(value0, value1, value2, value3);
}

__global__
__launch_bounds__(kThreadCount) void rdna_fp8_staged_matmul_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint4 staged[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t private_count = (tid & 7u) + 1u;
  auto *private_words =
      static_cast<volatile uint32_t *>(__builtin_alloca(private_count * sizeof(uint32_t)));
  for (uint32_t i = 0; i < private_count; ++i)
    private_words[i] = 0xf8000000u ^ (tid * 13u + i);
  const uint32_t selected = private_words[(tid + 1u) % private_count];

  // The gfx12 WMMA FP8 encoding represents 1.0 as 0x38. Each operand contains eight
  // packed elements, hence two identical 32-bit words.
  const i32x2_t ones = {static_cast<int32_t>(0x38383838u), static_cast<int32_t>(0x38383838u)};
  f32x8_t accumulator{};
  accumulator = fp8_wmma(ones, ones, accumulator);
  asm volatile("v_mov_b32 v191, %0" : : "v"(selected) : "v191");

  const uint32_t address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&staged[lane]) & 0xffffu);
  const uint4 packed = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  if (tid < kWaveSize)
    store_b128(address, packed);
  if (state->synchronize != 0u)
    __syncthreads();
  if (tid >= kWaveSize) {
    const uint4 consumed = load_b128(address);
    const i32x2_t rhs = {static_cast<int32_t>(consumed.x), static_cast<int32_t>(consumed.y)};
    accumulator = fp8_wmma(ones, rhs, accumulator);
#pragma unroll
    for (uint32_t i = 0; i < 8u; ++i)
      state->output[lane * 8u + i] = accumulator[i];
  }
  state->control[tid] = selected;
}

uint32_t expected_private_value(uint32_t tid) {
  const uint32_t count = (tid & 7u) + 1u;
  return 0xf8000000u ^ (tid * 13u + (tid + 1u) % count);
}

void run_rdna_fp8_staged_matmul(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  initial.head_canary = 0x13579bdfu;
  initial.tail_canary = 0x2468ace0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  rdna_fp8_staged_matmul_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(result.head_canary, 0x13579bdfu);
  EXPECT_EQ(result.tail_canary, 0x2468ace0u);
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], expected_private_value(tid)) << "thread=" << tid;
  if (!synchronize)
    return;
  for (uint32_t i = 0; i < kWaveSize * 8u; ++i)
    EXPECT_EQ(result.output[i], 32.0f) << "element=" << i;
}

TEST(ConSanDeviceRdnaFp8StagedMatmulTest, Correct) { run_rdna_fp8_staged_matmul(true); }
TEST(ConSanDeviceRdnaFp8StagedMatmulTest, Incorrect) { run_rdna_fp8_staged_matmul(false); }

} // namespace
