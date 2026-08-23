// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Sparse Record/Replay selection composed with access density, broad scalar
/// liveness, and spill-backed VGPR pressure. Qwen-style generated objects
/// combine these properties:
/// a rewrite must preserve the uninstrumented guest path and the selected
/// workgroup's evidence even when every access probe needs spill-backed scratch.
/// Correct/Incorrect differ only in the cross-wave publication edge.

#include "consan_device_test_support.h"

namespace {

#if defined(RJ_CONSAN_SPARSE_SPILL_WAVE32)
constexpr uint32_t kWaveSize = 32u;
#else
constexpr uint32_t kWaveSize = 64u;
#endif
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr uint32_t kWorkgroupCount = 2u;

#if defined(RJ_CONSAN_SPARSE_SPILL_MODERN_LDS)
#define RJ_CONSAN_SPARSE_SPILL_STORE "ds_store_b32"
#define RJ_CONSAN_SPARSE_SPILL_LOAD "ds_load_b32"
#define RJ_CONSAN_SPARSE_SPILL_WAIT "s_wait_dscnt 0\n\t"
#else
#define RJ_CONSAN_SPARSE_SPILL_STORE "ds_write_b32"
#define RJ_CONSAN_SPARSE_SPILL_LOAD "ds_read_b32"
#define RJ_CONSAN_SPARSE_SPILL_WAIT "s_waitcnt lgkmcnt(0)\n\t"
#endif

// Preserve broad scalar liveness across every dense LDS site while leaving a
// legal high-bank control window. This matches the scalar fallback available
// to the motivating Qwen kernels without conflating the separate all-SGPR
// branch-only ABI covered by RecordReplayLongRangeFullPressure.
#if defined(RJ_CONSAN_SPARSE_SPILL_GFX1250_COMPACT_SCALAR_PRESSURE)
// Tensile's wave32 Stream-K kernels keep every ordinary scalar window live
// while leaving only a PC pair and two routing scalars. This makes a sanitizer
// borrow and preserve guest scalar state instead of growing the descriptor,
// without requiring any particular sanitizer spill representation.
#define RJ_CONSAN_SPARSE_SPILL_SCALAR_USES                                                         \
  ".irp r,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,"           \
  "29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,"           \
  "57,58,59,60,61,62,63,64,65,68,69,70,71,72,73,74,77,78,79,80,81,82,83,84,85,86,87,88,"           \
  "89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105\n\t"                                   \
  "s_mov_b32 s\\r, s\\r\n\t"                                                                       \
  ".endr\n\t"
#else
#define RJ_CONSAN_SPARSE_SPILL_SCALAR_USES                                                         \
  ".irp r,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,"           \
  "29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,"           \
  "57,58,59,60,61,62,63\n\t"                                                                       \
  "s_mov_b32 s\\r, s\\r\n\t"                                                                       \
  ".endr\n\t"
#endif

#define RJ_CONSAN_SPARSE_SPILL_SCALAR_CLOBBERS                                                     \
  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",  \
      "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31", "s36", "s37", "s38",   \
      "s39", "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48", "s49", "s50", "s51",   \
      "s52", "s53", "s54", "s55", "s56", "s57", "s58", "s59", "s60", "s61", "s62", "s63"

// The TopK-shaped companion keeps the complete portable ordinary scalar bank
// live. It owns the distinct many-site branch-only-pressure contract; the
// sparse-spill pair above deliberately retains the high-bank control window
// available to Qwen.
#define RJ_CONSAN_SPARSE_SPILL_FULL_SCALAR_USES                                                    \
  RJ_CONSAN_SPARSE_SPILL_SCALAR_USES                                                               \
  ".irp r,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89\n\t"       \
  "s_mov_b32 s\\r, s\\r\n\t"                                                                       \
  ".endr\n\t"

#define RJ_CONSAN_SPARSE_SPILL_FULL_SCALAR_CLOBBERS                                                \
  RJ_CONSAN_SPARSE_SPILL_SCALAR_CLOBBERS, "s64", "s65", "s66", "s67", "s68", "s69", "s70", "s71",  \
      "s72", "s73", "s74", "s75", "s76", "s77", "s78", "s79", "s80", "s81", "s82", "s83", "s84",   \
      "s85", "s86", "s87", "s88", "s89"

// One live value in every adjacent VGPR pair prevents a new contiguous probe
// window without changing the guest's values. This is target-neutral pressure,
// not an assertion about which spill representation ConSan must choose.
#define RJ_CONSAN_SPARSE_SPILL_VECTOR_USES                                                         \
  ".irp r,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,"         \
  "57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,"             \
  "109,111,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,"           \
  "151,153,155,157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,187,189,191,"           \
  "193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,"           \
  "235,237,239,241,243,245,247,249,251,253,255\n\t"                                                \
  "v_mov_b32 v\\r, v\\r\n\t"                                                                       \
  ".endr\n\t"

struct DeviceState {
  uint32_t synchronize;
  uint32_t warmup[kWorkgroupCount][kThreadCount];
  uint32_t observed[kWorkgroupCount][kWaveSize];
  uint32_t control[kWorkgroupCount][kThreadCount];
};

__device__ __forceinline__ uint32_t lds_address(volatile uint32_t *address) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
}

__device__ __forceinline__ uint32_t pressured_dense_roundtrip(volatile uint32_t *address,
                                                              uint32_t value) {
  const uint32_t address_value = lds_address(address);
  uint32_t observed = 0u;
  asm volatile(RJ_CONSAN_SPARSE_SPILL_WAIT
                   RJ_CONSAN_SPARSE_SPILL_SCALAR_USES RJ_CONSAN_SPARSE_SPILL_VECTOR_USES
               ".rept 64\n\t" RJ_CONSAN_SPARSE_SPILL_STORE " %1, %2\n\t" RJ_CONSAN_SPARSE_SPILL_LOAD
               " %0, %1\n\t" RJ_CONSAN_SPARSE_SPILL_WAIT
               ".endr\n\t" RJ_CONSAN_SPARSE_SPILL_VECTOR_USES RJ_CONSAN_SPARSE_SPILL_SCALAR_USES
               : "=v"(observed)
               : "v"(address_value), "v"(value)
               : RJ_CONSAN_SPARSE_SPILL_SCALAR_CLOBBERS, "memory");
  return observed;
}

__device__ __forceinline__ uint32_t
full_scalar_pressured_dense_roundtrip(volatile uint32_t *address, uint32_t value) {
  const uint32_t address_value = lds_address(address);
  uint32_t observed = 0u;
  asm volatile(
      RJ_CONSAN_SPARSE_SPILL_WAIT
          RJ_CONSAN_SPARSE_SPILL_FULL_SCALAR_USES RJ_CONSAN_SPARSE_SPILL_VECTOR_USES
      ".rept 64\n\t" RJ_CONSAN_SPARSE_SPILL_STORE " %1, %2\n\t" RJ_CONSAN_SPARSE_SPILL_LOAD
      " %0, %1\n\t" RJ_CONSAN_SPARSE_SPILL_WAIT
      ".endr\n\t" RJ_CONSAN_SPARSE_SPILL_VECTOR_USES RJ_CONSAN_SPARSE_SPILL_FULL_SCALAR_USES
      : "=v"(observed)
      : "v"(address_value), "v"(value)
      : RJ_CONSAN_SPARSE_SPILL_FULL_SCALAR_CLOBBERS, "memory");
  return observed;
}

__device__ __forceinline__ void store_lds(volatile uint32_t *address, uint32_t value) {
  const uint32_t address_value = lds_address(address);
  asm volatile(RJ_CONSAN_SPARSE_SPILL_STORE " %0, %1\n\t"
               :
               : "v"(address_value), "v"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t load_lds(volatile uint32_t *address) {
  const uint32_t address_value = lds_address(address);
  uint32_t value = 0u;
  asm volatile(RJ_CONSAN_SPARSE_SPILL_LOAD " %0, %1\n\t" RJ_CONSAN_SPARSE_SPILL_WAIT
               : "=v"(value)
               : "v"(address_value)
               : "memory");
  return value;
}

template <bool kFullScalarPressure>
__global__ __launch_bounds__(kThreadCount) void record_replay_sparse_spill_pressure_for_consan(
    DeviceState *state) {
  __shared__ uint32_t warmup[kThreadCount];
  __shared__ uint32_t tile[kWaveSize];
  const uint32_t group = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
  const uint32_t warmup_value = 0xe1000000u | (group << 16u) | tid;

  if constexpr (kFullScalarPressure) {
    state->warmup[group][tid] = full_scalar_pressured_dense_roundtrip(&warmup[tid], warmup_value);
  } else {
    state->warmup[group][tid] = pressured_dense_roundtrip(&warmup[tid], warmup_value);
  }
  if (wave == 0u)
    store_lds(&tile[lane], 0xe2000000u | (group << 16u) | lane);
  if (state->synchronize)
    __syncthreads();
  if (wave == 1u)
    state->observed[group][lane] = load_lds(&tile[lane]);
  state->control[group][tid] = 0xe3000000u | (group << 16u) | tid;
}

template <bool kFullScalarPressure> void run_record_replay_sparse_spill_pressure(bool synchronize) {
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  record_replay_sparse_spill_pressure_for_consan<kFullScalarPressure>
      <<<kWorkgroupCount, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));

  for (uint32_t group = 0u; group < kWorkgroupCount; ++group) {
    for (uint32_t tid = 0u; tid < kThreadCount; ++tid) {
      EXPECT_EQ(state.warmup[group][tid], 0xe1000000u | (group << 16u) | tid)
          << "group=" << group << " thread=" << tid;
      EXPECT_EQ(state.control[group][tid], 0xe3000000u | (group << 16u) | tid)
          << "group=" << group << " thread=" << tid;
    }
    if (synchronize) {
      for (uint32_t lane = 0u; lane < kWaveSize; ++lane) {
        EXPECT_EQ(state.observed[group][lane], 0xe2000000u | (group << 16u) | lane)
            << "group=" << group << " lane=" << lane;
      }
    }
  }
}

TEST(ConSanDeviceRecordReplaySparseSpillPressureTest, Correct) {
  run_record_replay_sparse_spill_pressure<false>(true);
}

TEST(ConSanDeviceRecordReplaySparseSpillPressureTest, Incorrect) {
  run_record_replay_sparse_spill_pressure<false>(false);
}

TEST(ConSanDeviceRecordReplayDenseBranchOnlyPressureTest, Correct) {
  run_record_replay_sparse_spill_pressure<true>(true);
}

TEST(ConSanDeviceRecordReplayDenseBranchOnlyPressureTest, Incorrect) {
  run_record_replay_sparse_spill_pressure<true>(false);
}

} // namespace

#undef RJ_CONSAN_SPARSE_SPILL_STORE
#undef RJ_CONSAN_SPARSE_SPILL_LOAD
#undef RJ_CONSAN_SPARSE_SPILL_WAIT
#undef RJ_CONSAN_SPARSE_SPILL_SCALAR_USES
#undef RJ_CONSAN_SPARSE_SPILL_SCALAR_CLOBBERS
#undef RJ_CONSAN_SPARSE_SPILL_FULL_SCALAR_USES
#undef RJ_CONSAN_SPARSE_SPILL_FULL_SCALAR_CLOBBERS
#undef RJ_CONSAN_SPARSE_SPILL_VECTOR_USES
