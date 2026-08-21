// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Independent Sampled EXEC-save and dispatch-identity placement under the
/// scalar-pressure shape distilled from hip-moi's D128 attention workload.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr uint32_t kAccessCount = 18u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t access_marker[kThreadCount];
  uint32_t barrier_marker[kThreadCount];
  uint32_t warmup[kThreadCount][kAccessCount];
  uint32_t exec_observed[kWaveSize];
  uint32_t exec_control[kThreadCount];
  uint32_t dispatch_observed[kWaveSize];
  uint32_t dispatch_control[kThreadCount];
};

__device__ __forceinline__ void store_lds_without_cave(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  asm volatile("ds_write_b32 %0, %1\n\t" : : "v"(lds_address), "v"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_lds_without_cave(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
  asm volatile("ds_read_b32 %0, %1\n\t"
               "s_waitcnt lgkmcnt(0)\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
  return value;
}

// Keep a broad live range around every instrumented site, but deliberately
// leave one transient scalar window available to the liveness-based planner.
#define RJ_CONSAN_LIVE_SGPR_USES                                                                   \
  "s_mov_b64 s[8:9], s[8:9]\n\t"                                                                   \
  "s_mov_b64 s[10:11], s[10:11]\n\t"                                                               \
  "s_mov_b64 s[12:13], s[12:13]\n\t"                                                               \
  "s_mov_b64 s[14:15], s[14:15]\n\t"                                                               \
  "s_mov_b64 s[16:17], s[16:17]\n\t"                                                               \
  "s_mov_b64 s[18:19], s[18:19]\n\t"                                                               \
  "s_mov_b64 s[20:21], s[20:21]\n\t"                                                               \
  "s_mov_b64 s[22:23], s[22:23]\n\t"                                                               \
  "s_mov_b64 s[24:25], s[24:25]\n\t"                                                               \
  "s_mov_b64 s[26:27], s[26:27]\n\t"                                                               \
  "s_mov_b64 s[28:29], s[28:29]\n\t"                                                               \
  "s_mov_b64 s[30:31], s[30:31]\n\t"                                                               \
  "s_mov_b64 s[36:37], s[36:37]\n\t"                                                               \
  "s_mov_b64 s[38:39], s[38:39]\n\t"                                                               \
  "s_mov_b64 s[40:41], s[40:41]\n\t"                                                               \
  "s_mov_b32 s43, s43\n\t"                                                                         \
  "s_mov_b64 s[44:45], s[44:45]\n\t"                                                               \
  "s_mov_b64 s[46:47], s[46:47]\n\t"                                                               \
  "s_mov_b64 s[48:49], s[48:49]\n\t"                                                               \
  "s_mov_b64 s[50:51], s[50:51]\n\t"                                                               \
  "s_mov_b64 s[52:53], s[52:53]\n\t"                                                               \
  "s_mov_b64 s[54:55], s[54:55]\n\t"                                                               \
  "s_mov_b64 s[56:57], s[56:57]\n\t"                                                               \
  "s_mov_b64 s[58:59], s[58:59]\n\t"                                                               \
  "s_mov_b64 s[60:61], s[60:61]\n\t"                                                               \
  "s_mov_b64 s[62:63], s[62:63]\n\t"                                                               \
  "s_mov_b64 s[64:65], s[64:65]\n\t"                                                               \
  "s_mov_b64 s[66:67], s[66:67]\n\t"                                                               \
  "s_mov_b64 s[68:69], s[68:69]\n\t"                                                               \
  "s_mov_b64 s[70:71], s[70:71]\n\t"

#define RJ_CONSAN_LIVE_SGPR_CLOBBERS                                                               \
  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",  \
      "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31", "s36", "s37", "s38",   \
      "s39", "s40", "s41", "s43", "s44", "s45", "s46", "s47", "s48", "s49", "s50", "s51", "s52",   \
      "s53", "s54", "s55", "s56", "s57", "s58", "s59", "s60", "s61", "s62", "s63", "s64", "s65",   \
      "s66", "s67", "s68", "s69", "s70", "s71", "s82", "s83", "s84", "s85", "s86", "s87", "s88",   \
      "s89"

__global__
__launch_bounds__(kThreadCount) void cdna_liveness_dead_exec_for_consan(DeviceState *state) {
  __shared__ uint32_t warmup[kThreadCount][kAccessCount];
  __shared__ uint32_t tile[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
  const uint32_t value = 0xd0000000u | tid;
  const uint32_t warmup_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&warmup[tid][0]) & 0xffffu);
  const uint32_t synchronize = state->synchronize;

  if (wave == 0u)
    store_lds_without_cave(&tile[lane], 0xd2000000u | lane);

  uint32_t access_marker = 0u;
  uint32_t barrier_marker = 0u;
  asm volatile(RJ_CONSAN_LIVE_SGPR_USES "s_cmp_eq_u32 %4, %4\n\t"
                                        "ds_write_b32 %2, %3 offset:0\n\t"
                                        "ds_write_b32 %2, %3 offset:4\n\t"
                                        "ds_write_b32 %2, %3 offset:8\n\t"
                                        "ds_write_b32 %2, %3 offset:12\n\t"
                                        "ds_write_b32 %2, %3 offset:16\n\t"
                                        "ds_write_b32 %2, %3 offset:20\n\t"
                                        "ds_write_b32 %2, %3 offset:24\n\t"
                                        "ds_write_b32 %2, %3 offset:28\n\t"
                                        "ds_write_b32 %2, %3 offset:32\n\t"
                                        "ds_write_b32 %2, %3 offset:36\n\t"
                                        "ds_write_b32 %2, %3 offset:40\n\t"
                                        "ds_write_b32 %2, %3 offset:44\n\t"
                                        "ds_write_b32 %2, %3 offset:48\n\t"
                                        "ds_write_b32 %2, %3 offset:52\n\t"
                                        "ds_write_b32 %2, %3 offset:56\n\t"
                                        "ds_write_b32 %2, %3 offset:60\n\t"
                                        "ds_write_b32 %2, %3 offset:64\n\t"
                                        "ds_write_b32 %2, %3 offset:68\n\t"
                                        "s_cbranch_scc0 1f\n\t"
                                        "v_mov_b32 %0, 0xd1000001\n\t"
                                        "s_branch 2f\n\t"
                                        "1: v_mov_b32 %0, 0xd1000000\n\t"
                                        "2: s_cmp_eq_u32 %4, 1\n\t"
                                        "s_cbranch_scc0 3f\n\t"
                                        "s_cmp_eq_u32 %4, 0\n\t"
                                        "s_barrier\n\t"
                                        "3: s_cbranch_scc1 4f\n\t"
                                        "v_mov_b32 %1, 0xd3000000\n\t"
                                        "s_branch 5f\n\t"
                                        "4: v_mov_b32 %1, 0xd3000001\n\t"
                                        "5:\n\t"
                                        ".rept 36000\n\t"
                                        "s_mov_b32 s8, s8\n\t"
                                        ".endr\n\t" RJ_CONSAN_LIVE_SGPR_USES
                                        // Raise the owner's complete scalar tail without making
                                        // this def-only range live across an instrumented site.
                                        "s_mov_b64 s[82:83], 0\n\t"
                                        "s_mov_b64 s[84:85], 0\n\t"
                                        "s_mov_b64 s[86:87], 0\n\t"
                                        "s_mov_b64 s[88:89], 0\n\t"
               : "=&v"(access_marker), "=&v"(barrier_marker)
               : "v"(warmup_address), "v"(value), "s"(synchronize)
               : RJ_CONSAN_LIVE_SGPR_CLOBBERS, "memory");

  state->access_marker[tid] = access_marker;
  state->barrier_marker[tid] = barrier_marker;
  for (uint32_t index = 0u; index < kAccessCount; ++index)
    state->warmup[tid][index] = load_lds_without_cave(&warmup[tid][index]);
  if (wave == 1u)
    state->exec_observed[lane] = load_lds_without_cave(&tile[lane]);
  state->exec_control[tid] = 0xd4000000u | tid;
}

// The high hard pair makes dispatch placement choose a literal identity for
// this independent owner. That choice must not erase or strengthen the EXEC
// proof already established for the first owner.
__global__
__launch_bounds__(kThreadCount) void cdna_dispatch_pressure_for_consan(DeviceState *state) {
  __shared__ uint32_t tile[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;

  if (wave == 0u)
    store_lds_without_cave(&tile[lane], 0xd5000000u | lane);
  if (state->synchronize)
    __syncthreads();
  if (wave == 1u)
    state->dispatch_observed[lane] = load_lds_without_cave(&tile[lane]);
  asm volatile("s_mov_b64 s[96:97], s[96:97]\n\t" ::: "s96", "s97", "memory");
  state->dispatch_control[tid] = 0xd6000000u | tid;
}

void run_cdna_sampled_independent_scalar_proofs(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  // At stride two, these two-workgroup launches guarantee that one workgroup
  // remains sampled independent of the persistent dispatch identity.
  cdna_liveness_dead_exec_for_consan<<<2, kThreadCount>>>(device_state);
  cdna_dispatch_pressure_for_consan<<<2, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0u; tid < kThreadCount; ++tid) {
    EXPECT_EQ(result.access_marker[tid], 0xd1000001u) << "thread=" << tid;
    EXPECT_EQ(result.barrier_marker[tid], 0xd3000000u) << "thread=" << tid;
    EXPECT_EQ(result.exec_control[tid], 0xd4000000u | tid) << "thread=" << tid;
    EXPECT_EQ(result.dispatch_control[tid], 0xd6000000u | tid) << "thread=" << tid;
    for (uint32_t index = 0u; index < kAccessCount; ++index)
      EXPECT_EQ(result.warmup[tid][index], 0xd0000000u | tid)
          << "thread=" << tid << " access=" << index;
  }
  if (synchronize) {
    for (uint32_t lane = 0u; lane < kWaveSize; ++lane) {
      EXPECT_EQ(result.exec_observed[lane], 0xd2000000u | lane) << "lane=" << lane;
      EXPECT_EQ(result.dispatch_observed[lane], 0xd5000000u | lane) << "lane=" << lane;
    }
  }
}

TEST(ConSanDeviceCdnaSampledIndependentScalarProofsTest, Correct) {
  run_cdna_sampled_independent_scalar_proofs(true);
}

TEST(ConSanDeviceCdnaSampledIndependentScalarProofsTest, Incorrect) {
  run_cdna_sampled_independent_scalar_proofs(false);
}

} // namespace
