// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Portable long-range LDS instrumentation with live scalar condition state.
///
/// Generated PyTorch and Aorta kernels often separate an access from useful
/// relocation space by more than a direct SOPP branch range. They also use SCC
/// to control later synchronization. This pair keeps SCC live across an
/// instrumented LDS instruction and a 36,000-instruction scalar span, then
/// validates both the scalar control flow and cross-wave publication.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kMaximumWaveSize = 64u;
constexpr uint32_t kMaximumThreadCount = 2u * kMaximumWaveSize;

struct DeviceState {
  uint32_t synchronize;
  uint32_t warmup[kMaximumThreadCount];
  uint32_t observed[kMaximumWaveSize];
  uint32_t scc_marker[kMaximumThreadCount];
  uint32_t control[kMaximumThreadCount];
};

__device__ __forceinline__ uint32_t load_lds_word_with_distinct_output(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
#if defined(__gfx942__) || defined(__gfx950__)
  asm volatile("ds_read_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_waitcnt lgkmcnt(0)\n\t"
               : "=&v"(value)
               : "v"(lds_address)
               : "memory");
#elif defined(__gfx1250__)
  asm volatile("ds_read_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               : "=&v"(value)
               : "v"(lds_address)
               : "memory");
#elif defined(__gfx1201__)
  asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               : "=&v"(value)
               : "v"(lds_address)
               : "memory");
#else
  asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_waitcnt lgkmcnt(0)\n\t"
               : "=&v"(value)
               : "v"(lds_address)
               : "memory");
#endif
  return value;
}

__global__
__launch_bounds__(kMaximumThreadCount) void long_range_live_scc_for_consan(DeviceState *state) {
  __shared__ volatile uint32_t warmup[kMaximumThreadCount];
  __shared__ volatile uint32_t tile[kMaximumThreadCount];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wave = tid / warpSize;
  const uint32_t warmup_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&warmup[tid]) & 0xffffu);
  const uint32_t warmup_value = 0xb1000000u | tid;
  const uint32_t tile_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&tile[tid]) & 0xffffu);
  const uint32_t tile_value = 0xb2000000u | tid;
  const uint32_t synchronize = state->synchronize;

  uint32_t scc_marker = 0u;
#if defined(__gfx1100__) || defined(__gfx1201__)
#define RJ_CONSAN_PORTABLE_LONG_RANGE_STORE                                                        \
  "ds_store_b32 %1, %2\n\t"                                                                        \
  ".rept 36000\n\t"                                                                                \
  "s_mov_b32 s20, s20\n\t"                                                                         \
  ".endr\n\t"                                                                                      \
  "ds_store_b32 %3, %4\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
#else
#define RJ_CONSAN_PORTABLE_LONG_RANGE_STORE                                                        \
  "ds_write_b32 %1, %2\n\t"                                                                        \
  ".rept 36000\n\t"                                                                                \
  "s_mov_b32 s20, s20\n\t"                                                                         \
  ".endr\n\t"                                                                                      \
  "ds_write_b32 %3, %4\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
#endif
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1100__)
#define RJ_CONSAN_PORTABLE_LONG_RANGE_BARRIER "s_barrier\n\t"
#else
#define RJ_CONSAN_PORTABLE_LONG_RANGE_BARRIER                                                      \
  "s_barrier_signal -1\n\t"                                                                        \
  "s_barrier_wait -1\n\t"
#endif
  asm volatile(
      "s_cmp_eq_u32 %5, 1\n\t" RJ_CONSAN_PORTABLE_LONG_RANGE_STORE
      "s_cbranch_scc0 1f\n\t" RJ_CONSAN_PORTABLE_LONG_RANGE_BARRIER "v_mov_b32 %0, 0xb3000001\n\t"
      "s_branch 2f\n\t"
      "1: v_mov_b32 %0, 0xb3000000\n\t"
      "2:\n\t"
      : "=&v"(scc_marker)
      : "v"(warmup_address), "v"(warmup_value), "v"(tile_address), "v"(tile_value), "s"(synchronize)
      : "s20", "memory");
#undef RJ_CONSAN_PORTABLE_LONG_RANGE_STORE
#undef RJ_CONSAN_PORTABLE_LONG_RANGE_BARRIER

  state->warmup[tid] = consan_device_test::load_lds_word_without_cave(&warmup[tid]);
  if (wave == 1u)
    state->observed[lane] = load_lds_word_with_distinct_output(&tile[lane]);
  state->scc_marker[tid] = scc_marker;
  state->control[tid] = 0xb4000000u | tid;
}

void run_long_range_live_scc(bool synchronize) {
  const uint32_t wave_size = consan_device_test::active_wave_size();
  const uint32_t thread_count = 2u * wave_size;
  DeviceState state{};
  state.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  long_range_live_scc_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0u; tid < thread_count; ++tid) {
    EXPECT_EQ(state.warmup[tid], 0xb1000000u | tid) << "thread=" << tid;
    EXPECT_EQ(state.scc_marker[tid], synchronize ? 0xb3000001u : 0xb3000000u) << "thread=" << tid;
    EXPECT_EQ(state.control[tid], 0xb4000000u | tid) << "thread=" << tid;
  }
  if (synchronize)
    for (uint32_t lane = 0u; lane < wave_size; ++lane)
      EXPECT_EQ(state.observed[lane], 0xb2000000u | lane) << "lane=" << lane;
}

TEST(ConSanDeviceLongRangeLiveSccTest, Correct) { run_long_range_live_scc(true); }

TEST(ConSanDeviceLongRangeLiveSccTest, Incorrect) { run_long_range_live_scc(false); }

} // namespace
