// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Cross-wave LDS publication in a code object large enough to require
/// SuperCollider relay reservoirs. The long scalar block is distilled from
/// large generated PyTorch kernels and keeps their relocation geometry. The
/// behavioral contract is shared across wave64 CDNA3/4 and wave32
/// RDNA3/4/CDNA5, using each generation's native LDS spelling and wait.

#include "consan_device_test_support.h"

namespace {

#if defined(RJ_CONSAN_LARGE_TEXT_WAVE32)
constexpr uint32_t kWaveSize = 32u;
#else
constexpr uint32_t kWaveSize = 64u;
#endif
constexpr uint32_t kThreadCount = 2u * kWaveSize;

#if defined(RJ_CONSAN_LARGE_TEXT_MODERN_LDS)
#define RJ_CONSAN_LARGE_TEXT_STORE "ds_store_b32 %0, %1\n\t"
#define RJ_CONSAN_LARGE_TEXT_LOAD "ds_load_b32 %0, %1\n\t"
#define RJ_CONSAN_LARGE_TEXT_WAIT "s_wait_dscnt 0\n\t"
#else
#define RJ_CONSAN_LARGE_TEXT_STORE "ds_write_b32 %0, %1\n\t"
#define RJ_CONSAN_LARGE_TEXT_LOAD "ds_read_b32 %0, %1\n\t"
#define RJ_CONSAN_LARGE_TEXT_WAIT "s_waitcnt lgkmcnt(0)\n\t"
#endif

struct DeviceState {
  uint32_t synchronize;
  uint32_t warmup[kThreadCount];
  uint32_t observed[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void store_lds_without_cave(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  asm volatile(RJ_CONSAN_LARGE_TEXT_STORE : : "v"(lds_address), "v"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_lds_without_cave(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
  asm volatile(RJ_CONSAN_LARGE_TEXT_LOAD RJ_CONSAN_LARGE_TEXT_WAIT
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
  return value;
}

__global__ __launch_bounds__(kThreadCount) void large_text_relay_for_consan(DeviceState *state) {
  __shared__ uint32_t warmup[kThreadCount];
  __shared__ uint32_t tile[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;

  store_lds_without_cave(&warmup[tid], 0xc0000000u | tid);

  // Put the first LDS site outside direct branch range of the appended probe.
  // Ordinary scalar instructions in this block are valid relay-reservoir
  // donors. They also exercise the relocated guest instructions at runtime.
  asm volatile(".rept 36000\n\t"
               "s_mov_b32 s20, s20\n\t"
               ".endr\n\t" ::
                   : "s20", "memory");

  state->warmup[tid] = load_lds_without_cave(&warmup[tid]);
  if (wave == 0u)
    store_lds_without_cave(&tile[lane], 0xc1000000u | lane);
  if (state->synchronize)
    __syncthreads();
  if (wave == 1u)
    state->observed[lane] = load_lds_without_cave(&tile[lane]);
  state->control[tid] = 0xc2000000u | tid;
}

void run_large_text_relay(bool synchronize) {
  DeviceState state{};
  state.synchronize = synchronize;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device_state, &state, sizeof(state), hipMemcpyHostToDevice));
  large_text_relay_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(&state, device_state, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid) {
    EXPECT_EQ(state.warmup[tid], 0xc0000000u | tid) << "thread=" << tid;
    EXPECT_EQ(state.control[tid], 0xc2000000u | tid) << "thread=" << tid;
  }
  if (synchronize)
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
      EXPECT_EQ(state.observed[lane], 0xc1000000u | lane) << "lane=" << lane;
}

TEST(ConSanDeviceLargeTextRelayTest, Correct) { run_large_text_relay(true); }

TEST(ConSanDeviceLargeTextRelayTest, Incorrect) { run_large_text_relay(false); }

} // namespace

#undef RJ_CONSAN_LARGE_TEXT_STORE
#undef RJ_CONSAN_LARGE_TEXT_LOAD
#undef RJ_CONSAN_LARGE_TEXT_WAIT
