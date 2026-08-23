// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Record/Replay cross-wave LDS publication in a long, full-resource-pressure
/// code object. The shape is distilled from generated PyTorch TopK kernels;
/// the adjacent Correct/Incorrect cases retain a user-visible synchronization
/// contract across CDNA3/4/5 and RDNA3/4 without prescribing how an
/// implementation reaches the access.

#include "consan_device_test_support.h"

namespace {

#if defined(RJ_CONSAN_RECORD_REPLAY_PRESSURE_WAVE32)
constexpr uint32_t kWaveSize = 32u;
#else
constexpr uint32_t kWaveSize = 64u;
#endif
constexpr uint32_t kThreadCount = 2u * kWaveSize;

#if defined(RJ_CONSAN_RECORD_REPLAY_PRESSURE_MODERN_LDS)
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_STORE "ds_store_b32"
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_LOAD "ds_load_b32"
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_WAIT "s_wait_dscnt 0\n\t"
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_BARRIER                                                   \
  "s_barrier_signal -1\n\t"                                                                        \
  "s_barrier_wait -1\n\t"
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_BARRIER_REPEATS "128"
#else
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_STORE "ds_write_b32"
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_LOAD "ds_read_b32"
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_WAIT "s_waitcnt lgkmcnt(0)\n\t"
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_BARRIER "s_barrier\n\t"
#define RJ_CONSAN_RECORD_REPLAY_PRESSURE_BARRIER_REPEATS "4682"
#endif

struct DeviceState {
  uint32_t synchronize;
  uint32_t warmup[kThreadCount];
  uint32_t observed[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ uint32_t load_lds_without_cave(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
  asm volatile(RJ_CONSAN_RECORD_REPLAY_PRESSURE_LOAD
               " %0, %1\n\t" RJ_CONSAN_RECORD_REPLAY_PRESSURE_WAIT
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
  return value;
}

__device__ __forceinline__ void store_lds_without_cave(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  asm volatile(RJ_CONSAN_RECORD_REPLAY_PRESSURE_STORE " %0, %1\n\t"
               :
               : "v"(lds_address), "v"(value)
               : "memory");
}

// Keep a broad scalar bank live at the first LDS access. These physical
// self-moves preserve any compiler-assigned guest values. On CDNA the highest
// ordinary guest reference remains s89, below physical VCC at s90:s91. The
// same portable low-bank pressure is valuable on wave32 targets even though
// their architectural tail differs. Compiler-reserved s32:s35 are physically
// read and preserved but deliberately omitted from the clobber declaration.
#define RJ_CONSAN_FULL_SCALAR_USES                                                                 \
  ".irp r,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,"           \
  "29,30,31,32,33,34,35,36,37,"                                                                    \
  "38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,"                       \
  "62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89\n\t"        \
  "s_mov_b32 s\\r, s\\r\n\t"                                                                       \
  ".endr\n\t"

// Make the compiler descriptor allocate the same s0:s89 guest tail that the
// emitted physical references require. The low ABI inputs remain available to
// the compiler; the asm itself preserves every listed register.
#define RJ_CONSAN_FULL_SCALAR_CLOBBERS                                                             \
  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",  \
      "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31", "s36", "s37", "s38",   \
      "s39", "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48", "s49", "s50", "s51",   \
      "s52", "s53", "s54", "s55", "s56", "s57", "s58", "s59", "s60", "s61", "s62", "s63", "s64",   \
      "s65", "s66", "s67", "s68", "s69", "s70", "s71", "s72", "s73", "s74", "s75", "s76", "s77",   \
      "s78", "s79", "s80", "s81", "s82", "s83", "s84", "s85", "s86", "s87", "s88", "s89"

// One live register in every adjacent VGPR pair of the portable low bank rules
// out descriptor-grown contiguous probe scratch without changing guest
// values. This is the compact device-code analogue of TopK's FP64 vector
// pressure. gfx1250's separate high-bank pair covers its additional bank.
#define RJ_CONSAN_FULL_VECTOR_USES                                                                 \
  ".irp r,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,"         \
  "57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,"         \
  "111,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,151,"           \
  "153,155,157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,187,189,191,193,"           \
  "195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,"           \
  "237,239,241,243,245,247,249,251,253,255\n\t"                                                    \
  "v_mov_b32 v\\r, v\\r\n\t"                                                                       \
  ".endr\n\t"

__global__ __launch_bounds__(kThreadCount) void record_replay_long_range_full_pressure_for_consan(
    DeviceState *state) {
  __shared__ uint32_t warmup[kThreadCount];
  __shared__ uint32_t tile[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
  const uint32_t warmup_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&warmup[tid]) & 0xffffu);
  const uint32_t warmup_value = 0xc4000000u | tid;

  // Put the first full-pressure LDS site more than two SOPP hops from the
  // appended probe. Ordinary scalar instructions in this block are safe
  // instruction-reservoir donors. Executing their relocated copies also makes
  // this a dynamic correctness contract rather than only a patch-inventory
  // assertion.
  asm volatile(RJ_CONSAN_RECORD_REPLAY_PRESSURE_WAIT RJ_CONSAN_FULL_SCALAR_USES
                   RJ_CONSAN_FULL_VECTOR_USES RJ_CONSAN_RECORD_REPLAY_PRESSURE_STORE
               " %0, %1\n\t"
               ".rept 100000\n\t"
               "s_mov_b32 s20, s20\n\t"
               ".endr\n\t" RJ_CONSAN_FULL_VECTOR_USES RJ_CONSAN_FULL_SCALAR_USES
               :
               : "v"(warmup_address), "v"(warmup_value)
               : RJ_CONSAN_FULL_SCALAR_CLOBBERS, "memory");

  state->warmup[tid] = load_lds_without_cave(&warmup[tid]);
  if (wave == 0u)
    store_lds_without_cave(&tile[lane], 0xc5000000u | lane);
  if (state->synchronize)
    __syncthreads();
  if (wave == 1u)
    state->observed[lane] = load_lds_without_cave(&tile[lane]);
  state->control[tid] = 0xc6000000u | tid;
}

// Keep a dormant user kernel in the same code object with a dense E2E-shaped
// synchronization inventory. The behavioral pair below never launches it:
// unrelated synchronization sites must not make the exercised long-range
// access uninstrumentable or corrupt its result. Modern split barriers expose
// every signal/wait pair as two static sites, so their bounded inventory still
// exercises dense planning without dominating this quick checked-in contract.
// Legacy monolithic barriers retain the larger TopK-derived inventory that the
// analyzer canonicalizes to its representative synchronization obligation.
__global__ void record_replay_long_barrier_inventory_for_consan() {
  asm volatile(".rept " RJ_CONSAN_RECORD_REPLAY_PRESSURE_BARRIER_REPEATS
               "\n\t" RJ_CONSAN_RECORD_REPLAY_PRESSURE_BARRIER ".endr\n\t"
               :
               :
               : "memory");
}

void run_record_replay_long_range_full_pressure(bool synchronize) {
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  record_replay_long_range_full_pressure_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid) {
    EXPECT_EQ(state.warmup[tid], 0xc4000000u | tid) << "thread=" << tid;
    EXPECT_EQ(state.control[tid], 0xc6000000u | tid) << "thread=" << tid;
  }
  if (synchronize)
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
      EXPECT_EQ(state.observed[lane], 0xc5000000u | lane) << "lane=" << lane;
}

TEST(ConSanDeviceRecordReplayLongRangeFullPressureTest, Correct) {
  run_record_replay_long_range_full_pressure(true);
}

TEST(ConSanDeviceRecordReplayLongRangeFullPressureTest, Incorrect) {
  run_record_replay_long_range_full_pressure(false);
}

} // namespace

#undef RJ_CONSAN_RECORD_REPLAY_PRESSURE_STORE
#undef RJ_CONSAN_RECORD_REPLAY_PRESSURE_LOAD
#undef RJ_CONSAN_RECORD_REPLAY_PRESSURE_WAIT
#undef RJ_CONSAN_RECORD_REPLAY_PRESSURE_BARRIER
#undef RJ_CONSAN_RECORD_REPLAY_PRESSURE_BARRIER_REPEATS
