// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Two disconnected CDNA owners distilled from the gfx950 HIP-matmul object.
/// The fixed-stack owner keeps a real accumulator bank immediately above its
/// ordinary VGPR allocation; the dynamic-stack owner reaches the corresponding
/// ordinary-VGPR boundary without allocating accumulators. The fixed owner
/// retains the complete scalar tail, while the dynamic owner mirrors the E2E
/// object's pressure through s81 and leaves s82:s89 available for transient
/// instrumentation state. The correct/incorrect pair differs only in the
/// dynamic owner's cross-wave publication edge and checks exact independent
/// state from both owners, without prescribing ConSan's persistent-state
/// representation.

#include "consan_device_test_support.h"

namespace {

using f16x4_t = _Float16 __attribute__((ext_vector_type(4)));

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;

struct DeviceState {
  uint32_t synchronize_dynamic;
  uint32_t fixed_observed[kWaveSize];
  uint32_t dynamic_observed[kWaveSize];
  float fixed_accumulator[kWaveSize * 4u];
  uint32_t fixed_pressure[kThreadCount];
  uint32_t dynamic_pressure[kThreadCount];
  uint32_t dynamic_private[kThreadCount];
  uint32_t fixed_control[kThreadCount];
  uint32_t dynamic_control[kThreadCount];
};

// Preserve the complete compiler-addressable scalar tail at a generated LDS
// owner. The physical self-moves do not change program values. s32:s35 remain
// compiler-reserved and are read but intentionally omitted from the clobbers;
// s90:s91 are physical VCC on these targets.
#define RJ_CONSAN_MIXED_SCALAR_USES_THROUGH_81                                                     \
  ".irp r,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,"           \
  "29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,"              \
  "56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81\n\t"              \
  "s_mov_b32 s\\r, s\\r\n\t"                                                                       \
  ".endr\n\t"

#define RJ_CONSAN_MIXED_FULL_SCALAR_USES                                                           \
  RJ_CONSAN_MIXED_SCALAR_USES_THROUGH_81                                                           \
  ".irp r,82,83,84,85,86,87,88,89\n\t"                                                             \
  "s_mov_b32 s\\r, s\\r\n\t"                                                                       \
  ".endr\n\t"

#define RJ_CONSAN_MIXED_SCALAR_CLOBBERS_THROUGH_81                                                 \
  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",  \
      "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31", "s36", "s37", "s38",   \
      "s39", "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48", "s49", "s50", "s51",   \
      "s52", "s53", "s54", "s55", "s56", "s57", "s58", "s59", "s60", "s61", "s62", "s63", "s64",   \
      "s65", "s66", "s67", "s68", "s69", "s70", "s71", "s72", "s73", "s74", "s75", "s76", "s77",   \
      "s78", "s79", "s80", "s81"

#define RJ_CONSAN_MIXED_FULL_SCALAR_CLOBBERS                                                       \
  RJ_CONSAN_MIXED_SCALAR_CLOBBERS_THROUGH_81, "s82", "s83", "s84", "s85", "s86", "s87", "s88", "s89"

__device__ __forceinline__ f16x4_t splat_half(float value) {
  f16x4_t result;
#pragma unroll
  for (uint32_t index = 0u; index < 4u; ++index)
    result[index] = static_cast<_Float16>(value);
  return result;
}

__device__ __forceinline__ void begin_live_accumulator(f16x4_t lhs, f16x4_t rhs) {
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

__device__ __forceinline__ float4 read_live_accumulator() {
  float4 result{};
  asm volatile("v_accvgpr_read_b32 %0, a4\n\t"
               "v_accvgpr_read_b32 %1, a5\n\t"
               "v_accvgpr_read_b32 %2, a6\n\t"
               "v_accvgpr_read_b32 %3, a7\n\t"
               : "=v"(result.x), "=v"(result.y), "=v"(result.z), "=v"(result.w)
               :
               : "memory");
  return result;
}

__device__ __forceinline__ uint32_t retain_fixed_pressure(uint32_t value) {
  uint32_t result = 0u;
  asm volatile("v_mov_b32 v63, %1\n\t" RJ_CONSAN_MIXED_FULL_SCALAR_USES "v_mov_b32 %0, v63\n\t"
               : "=v"(result)
               : "v"(value)
               : "v63", RJ_CONSAN_MIXED_FULL_SCALAR_CLOBBERS, "memory");
  return result;
}

__device__ __forceinline__ uint32_t retain_dynamic_pressure(uint32_t value) {
  uint32_t result = 0u;
  asm volatile("v_mov_b32 v125, %1\n\t" RJ_CONSAN_MIXED_SCALAR_USES_THROUGH_81
               "v_mov_b32 %0, v125\n\t"
               : "=v"(result)
               : "v"(value)
               : "v125", RJ_CONSAN_MIXED_SCALAR_CLOBBERS_THROUGH_81, "memory");
  return result;
}

__global__
__launch_bounds__(kThreadCount) void fixed_accumulator_owner_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t tile[kThreadCount];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
  const f16x4_t ones = splat_half(1.0f);
  begin_live_accumulator(ones, ones);
  const uint32_t pressure = retain_fixed_pressure(0xf1000000u | tid);

  consan_device_test::store_lds_word(&tile[tid], 0xf2000000u | tid);
  __syncthreads();
  const uint32_t other_wave_base = (1u - wave) * kWaveSize;
  const uint32_t observed = consan_device_test::load_lds_word(&tile[other_wave_base + lane]);
  if (wave == 1u)
    state->fixed_observed[lane] = observed;

  const float4 accumulator = read_live_accumulator();
  if (wave == 1u) {
    const uint32_t base = lane * 4u;
    state->fixed_accumulator[base + 0u] = accumulator.x;
    state->fixed_accumulator[base + 1u] = accumulator.y;
    state->fixed_accumulator[base + 2u] = accumulator.z;
    state->fixed_accumulator[base + 3u] = accumulator.w;
  }
  state->fixed_pressure[tid] = pressure;
  state->fixed_control[tid] = 0xf3000000u | tid;
}

__global__ __launch_bounds__(kThreadCount) void dynamic_empty_accumulator_owner_for_consan(
    DeviceState *state) {
  __shared__ volatile uint32_t tile[2u * kThreadCount];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
  const uint32_t private_count = (tid & 15u) + 1u;
  auto *private_values =
      static_cast<volatile uint32_t *>(__builtin_alloca(private_count * sizeof(uint32_t)));
  for (uint32_t index = 0u; index < private_count; ++index)
    private_values[index] = 0xd1000000u ^ (tid * 97u + index);
  const uint32_t selected = private_values[(tid * 5u + 3u) % private_count];
  const uint32_t pressure = retain_dynamic_pressure(0xd2000000u | tid);

  // Give each wave a disjoint publication tile, then have it consume the
  // other wave's tile. The leading band keeps this disconnected owner's
  // recorded LDS ranges separate from the fixed owner's ranges; repeated
  // dispatch identity has its own focused contract. This preserves the E2E
  // cross-wave publication idiom without coupling this persistent-state
  // regression to that concern or to immediately-after-saveexec lowering.
  consan_device_test::store_lds_word(&tile[kThreadCount + tid], 0xd3000000u | tid);
  if (state->synchronize_dynamic != 0u)
    __syncthreads();
  const uint32_t other_wave_base = kThreadCount + (1u - wave) * kWaveSize;
  const uint32_t observed = consan_device_test::load_lds_word(&tile[other_wave_base + lane]);
  if (wave == 1u)
    state->dynamic_observed[lane] = observed;

  state->dynamic_private[tid] = selected;
  state->dynamic_pressure[tid] = pressure;
  state->dynamic_control[tid] = 0xd4000000u | tid;
}

uint32_t expected_private_value(uint32_t tid) {
  const uint32_t count = (tid & 15u) + 1u;
  return 0xd1000000u ^ (tid * 97u + (tid * 5u + 3u) % count);
}

void run_cdna_mixed_persistent_state(bool synchronize_dynamic) {
  DeviceState initial{};
  initial.synchronize_dynamic = synchronize_dynamic ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  fixed_accumulator_owner_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  dynamic_empty_accumulator_owner_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0u; tid < kThreadCount; ++tid) {
    EXPECT_EQ(result.fixed_pressure[tid], 0xf1000000u | tid) << "thread=" << tid;
    EXPECT_EQ(result.dynamic_pressure[tid], 0xd2000000u | tid) << "thread=" << tid;
    EXPECT_EQ(result.dynamic_private[tid], expected_private_value(tid)) << "thread=" << tid;
    EXPECT_EQ(result.fixed_control[tid], 0xf3000000u | tid) << "thread=" << tid;
    EXPECT_EQ(result.dynamic_control[tid], 0xd4000000u | tid) << "thread=" << tid;
  }
  for (uint32_t lane = 0u; lane < kWaveSize; ++lane) {
    EXPECT_EQ(result.fixed_observed[lane], 0xf2000000u | lane) << "lane=" << lane;
    const uint32_t base = lane * 4u;
    for (uint32_t index = 0u; index < 4u; ++index)
      EXPECT_EQ(result.fixed_accumulator[base + index], 16.0f)
          << "lane=" << lane << " index=" << index;
    if (synchronize_dynamic)
      EXPECT_EQ(result.dynamic_observed[lane], 0xd3000000u | lane) << "lane=" << lane;
  }
}

TEST(ConSanDeviceCdnaMixedPersistentStateTest, Correct) { run_cdna_mixed_persistent_state(true); }

TEST(ConSanDeviceCdnaMixedPersistentStateTest, Incorrect) {
  run_cdna_mixed_persistent_state(false);
}

} // namespace
