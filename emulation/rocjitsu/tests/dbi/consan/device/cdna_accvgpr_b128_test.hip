// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// CDNA4 native B128 LDS load directly into AccVGPRs. Independent accumulator
/// state and VCC remain live across the access so either class being clobbered
/// changes an exact output.

#include "consan_device_test_support.h"

namespace {

using f16x4_t = _Float16 __attribute__((ext_vector_type(4)));

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;

struct DeviceState {
  uint32_t synchronize;
  uint32_t loaded[kWaveSize * 4u];
  float live_accumulator[kWaveSize * 4u];
  uint32_t vcc_survived[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ f16x4_t splat_half(float value) {
  f16x4_t result;
#pragma unroll
  for (uint32_t i = 0; i < 4u; ++i)
    result[i] = static_cast<_Float16>(value);
  return result;
}

__device__ __forceinline__ void begin_live_mfma(f16x4_t lhs, f16x4_t rhs) {
  const uint32_t zero = 0u;
  asm volatile("v_accvgpr_write_b32 a4, %2\n\t"
               "v_accvgpr_write_b32 a5, %2\n\t"
               "v_accvgpr_write_b32 a6, %2\n\t"
               "v_accvgpr_write_b32 a7, %2\n\t"
               "v_mfma_f32_16x16x16_f16 a[4:7], %0, %1, a[4:7]\n\t"
               "s_nop 15\n\t"
               :
               : "v"(lhs), "v"(rhs), "v"(zero)
               : "a4", "a5", "a6", "a7", "memory");
}

__device__ __forceinline__ void store_b128(uint32_t address, uint4 value) {
  register uint32_t value0 asm("v20") = value.x;
  register uint32_t value1 asm("v21") = value.y;
  register uint32_t value2 asm("v22") = value.z;
  register uint32_t value3 asm("v23") = value.w;
  asm volatile("ds_write_b128 %0, v[20:23]\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(address), "v"(value0), "v"(value1), "v"(value2), "v"(value3)
               : "v20", "v21", "v22", "v23", "memory");
}

__device__ __forceinline__ void load_acc_b128_with_live_state(uint32_t address, uint4 &loaded,
                                                              float4 &live_accumulator,
                                                              uint32_t &vcc_survived) {
  asm volatile("v_cmp_eq_u32_e32 vcc, %9, %9\n\t"
               "ds_read_b128 a[0:3], %9\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_waitcnt lgkmcnt(0)\n\t"
               "v_accvgpr_read_b32 %0, a0\n\t"
               "v_accvgpr_read_b32 %1, a1\n\t"
               "v_accvgpr_read_b32 %2, a2\n\t"
               "v_accvgpr_read_b32 %3, a3\n\t"
               "v_accvgpr_read_b32 %4, a4\n\t"
               "v_accvgpr_read_b32 %5, a5\n\t"
               "v_accvgpr_read_b32 %6, a6\n\t"
               "v_accvgpr_read_b32 %7, a7\n\t"
               "s_cbranch_vccz 1f\n\t"
               "v_mov_b32 %8, 1\n\t"
               "1:\n\t"
               : "=v"(loaded.x), "=v"(loaded.y), "=v"(loaded.z), "=v"(loaded.w),
                 "=v"(live_accumulator.x), "=v"(live_accumulator.y), "=v"(live_accumulator.z),
                 "=v"(live_accumulator.w), "+v"(vcc_survived)
               : "v"(address)
               : "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "vcc", "memory");
}

__global__ __launch_bounds__(kThreadCount) void cdna_accvgpr_b128_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint4 staged[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&staged[lane]) & 0xffffu);
  const uint4 expected =
      make_uint4(0xac000000u | lane, 0xac100000u | lane, 0xac200000u | lane, 0xac300000u | lane);
  const f16x4_t ones = splat_half(1.0f);
  begin_live_mfma(ones, ones);
  asm volatile("v_mov_b32 v31, 0" ::: "v31");

  if (tid < kWaveSize)
    store_b128(address, expected);
  if (state->synchronize != 0u)
    __syncthreads();
  if (tid >= kWaveSize) {
    uint4 loaded{};
    float4 live_accumulator{};
    uint32_t vcc_survived = 0u;
    load_acc_b128_with_live_state(address, loaded, live_accumulator, vcc_survived);
    const uint32_t base = lane * 4u;
    state->loaded[base + 0u] = loaded.x;
    state->loaded[base + 1u] = loaded.y;
    state->loaded[base + 2u] = loaded.z;
    state->loaded[base + 3u] = loaded.w;
    state->live_accumulator[base + 0u] = live_accumulator.x;
    state->live_accumulator[base + 1u] = live_accumulator.y;
    state->live_accumulator[base + 2u] = live_accumulator.z;
    state->live_accumulator[base + 3u] = live_accumulator.w;
    state->vcc_survived[lane] = vcc_survived;
  }
  state->control[tid] = 0xaccc0000u | tid;
}

void run_cdna_accvgpr_b128(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  cdna_accvgpr_b128_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0xaccc0000u | tid) << "thread=" << tid;
  if (!synchronize)
    return;
  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
    const uint32_t base = lane * 4u;
    EXPECT_EQ(result.loaded[base + 0u], 0xac000000u | lane) << "lane=" << lane;
    EXPECT_EQ(result.loaded[base + 1u], 0xac100000u | lane) << "lane=" << lane;
    EXPECT_EQ(result.loaded[base + 2u], 0xac200000u | lane) << "lane=" << lane;
    EXPECT_EQ(result.loaded[base + 3u], 0xac300000u | lane) << "lane=" << lane;
    for (uint32_t i = 0; i < 4u; ++i)
      EXPECT_EQ(result.live_accumulator[base + i], 16.0f) << "lane=" << lane << " i=" << i;
    EXPECT_EQ(result.vcc_survived[lane], 1u) << "lane=" << lane;
  }
}

TEST(ConSanDeviceCdnaAccvgprB128Test, Correct) { run_cdna_accvgpr_b128(true); }
TEST(ConSanDeviceCdnaAccvgprB128Test, Incorrect) { run_cdna_accvgpr_b128(false); }

} // namespace
