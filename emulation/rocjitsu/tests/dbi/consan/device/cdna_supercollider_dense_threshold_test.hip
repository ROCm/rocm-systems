// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Production-scale SuperCollider dense dispatch distilled from PyTorch's
/// generated radix-sort library. The checked-in kernel crosses the
/// 1,024-stranded-site routing threshold while executing the stress sites with
/// one active lane. Target-native LDS spellings retain the same behavioral
/// contract across CDNA3/4/5 and RDNA3/4.

#include "consan_device_test_support.h"

namespace {

#if defined(RJ_CONSAN_DENSE_THRESHOLD_WAVE32)
constexpr uint32_t kWaveSize = 32u;
#else
constexpr uint32_t kWaveSize = 64u;
#endif
constexpr uint32_t kThreadCount = 2u * kWaveSize;

#if defined(RJ_CONSAN_DENSE_THRESHOLD_MODERN_LDS)
#define RJ_CONSAN_DENSE_THRESHOLD_STORE "ds_store_b32"
#define RJ_CONSAN_DENSE_THRESHOLD_LOAD "ds_load_b32"
#define RJ_CONSAN_DENSE_THRESHOLD_WAIT "s_wait_dscnt 0\n\t"
#else
#define RJ_CONSAN_DENSE_THRESHOLD_STORE "ds_write_b32"
#define RJ_CONSAN_DENSE_THRESHOLD_LOAD "ds_read_b32"
#define RJ_CONSAN_DENSE_THRESHOLD_WAIT "s_waitcnt lgkmcnt(0)\n\t"
#endif

struct DeviceState {
  uint32_t synchronize;
  uint32_t dense_value;
  uint32_t observed[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void store_lds(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  asm volatile(RJ_CONSAN_DENSE_THRESHOLD_STORE " %0, %1\n\t"
               :
               : "v"(lds_address), "v"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t load_lds(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
  asm volatile(RJ_CONSAN_DENSE_THRESHOLD_LOAD " %0, %1\n\t" RJ_CONSAN_DENSE_THRESHOLD_WAIT
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
  return value;
}

__global__ void supercollider_dense_sites_for_consan(DeviceState *state) {
  __shared__ uint32_t dense_word;
  const uint32_t dense_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&dense_word) & 0xffffu);
  const uint32_t dense_value = 0xd6000000u;
  // Each repeated instruction remains a distinct static LDS site. The long
  // scalar tail places their ordinary appended bodies outside direct SOPP
  // range, forcing the shared dense dispatcher used by large E2E objects.
  asm volatile(".rept 1040\n\t" RJ_CONSAN_DENSE_THRESHOLD_STORE " %0, %1\n\t"
               ".endr\n\t" RJ_CONSAN_DENSE_THRESHOLD_WAIT ".rept 36000\n\t"
               "s_mov_b32 s8, s8\n\t"
               ".endr\n\t"
               :
               : "v"(dense_address), "v"(dense_value)
               : "s8", "memory");
  state->dense_value = load_lds(&dense_word);
}

__global__
__launch_bounds__(kThreadCount) void supercollider_dense_behavior_for_consan(DeviceState *state) {
  __shared__ uint32_t tile[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;

  if (wave == 0u)
    store_lds(&tile[lane], 0xd5000000u | lane);

  if (state->synchronize != 0u)
    __syncthreads();

  if (wave == 1u)
    state->observed[lane] = load_lds(&tile[lane]);
  state->control[tid] = 0xd7000000u | tid;
}

void run_supercollider_dense_threshold(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  supercollider_dense_sites_for_consan<<<1, 1>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  supercollider_dense_behavior_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  EXPECT_EQ(result.dense_value, 0xd6000000u);
  for (uint32_t tid = 0u; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0xd7000000u | tid) << "thread=" << tid;
  if (synchronize) {
    for (uint32_t lane = 0u; lane < kWaveSize; ++lane)
      EXPECT_EQ(result.observed[lane], 0xd5000000u | lane) << "lane=" << lane;
  }
}

TEST(ConSanDeviceSuperColliderDenseThresholdTest, Correct) {
  run_supercollider_dense_threshold(true);
}

TEST(ConSanDeviceSuperColliderDenseThresholdTest, Incorrect) {
  run_supercollider_dense_threshold(false);
}

} // namespace

#undef RJ_CONSAN_DENSE_THRESHOLD_STORE
#undef RJ_CONSAN_DENSE_THRESHOLD_LOAD
#undef RJ_CONSAN_DENSE_THRESHOLD_WAIT
