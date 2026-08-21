// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// A live SCC value crossing a dense-routed 16-bit LDS store. This shape is
/// distilled from hip-moi's D128 attention-pressure kernel, where a late
/// ds_write_b16 sits between a scalar comparison and its conditional branch.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kWaveSize = 64u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;
constexpr uint32_t kAccessCount = 18u;

struct DeviceState {
  uint32_t synchronize;
  uint16_t warmup[kThreadCount][kAccessCount];
  uint32_t branch_marker[kThreadCount];
  uint32_t observed[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void store_lds_b32(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  asm volatile("ds_write_b32 %0, %1\n\t" : : "v"(lds_address), "v"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_lds_b32(volatile uint32_t *address) {
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

__device__ __forceinline__ uint32_t load_lds_u16(volatile uint16_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0u;
  asm volatile("ds_read_u16 %0, %1\n\t"
               "s_waitcnt lgkmcnt(0)\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
  return value;
}

__global__ __launch_bounds__(kThreadCount) void cdna_b16_scc_route_for_consan(DeviceState *state) {
  __shared__ uint16_t warmup[kThreadCount][kAccessCount];
  __shared__ uint32_t tile[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t wave = tid / kWaveSize;
  const uint32_t warmup_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&warmup[tid][0]) & 0xffffu);
  const uint32_t value = 0xb000u | tid;
  const uint32_t synchronize = state->synchronize;

  uint32_t branch_marker = 0u;
  asm volatile("s_cmp_eq_u32 %3, %3\n\t"
               "ds_write_b16 %1, %2 offset:0\n\t"
               "ds_write_b16 %1, %2 offset:2\n\t"
               "ds_write_b16 %1, %2 offset:4\n\t"
               "ds_write_b16 %1, %2 offset:6\n\t"
               "ds_write_b16 %1, %2 offset:8\n\t"
               "ds_write_b16 %1, %2 offset:10\n\t"
               "ds_write_b16 %1, %2 offset:12\n\t"
               "ds_write_b16 %1, %2 offset:14\n\t"
               "ds_write_b16 %1, %2 offset:16\n\t"
               "ds_write_b16 %1, %2 offset:18\n\t"
               "ds_write_b16 %1, %2 offset:20\n\t"
               "ds_write_b16 %1, %2 offset:22\n\t"
               "ds_write_b16 %1, %2 offset:24\n\t"
               "ds_write_b16 %1, %2 offset:26\n\t"
               "ds_write_b16 %1, %2 offset:28\n\t"
               "ds_write_b16 %1, %2 offset:30\n\t"
               "ds_write_b16 %1, %2 offset:32\n\t"
               "ds_write_b16 %1, %2 offset:34\n\t"
               "s_cbranch_scc0 1f\n\t"
               "v_mov_b32 %0, 0xb1000001\n\t"
               "s_branch 2f\n\t"
               "1: v_mov_b32 %0, 0xb1000000\n\t"
               "2:\n\t"
               // Keep the accesses outside direct SOPP reach of their
               // appended probes, forcing the dense dispatcher path.
               ".rept 36000\n\t"
               "s_mov_b32 s20, s20\n\t"
               ".endr\n\t"
               : "=&v"(branch_marker)
               : "v"(warmup_address), "v"(value), "s"(synchronize)
               : "s20", "memory");

  state->branch_marker[tid] = branch_marker;
  for (uint32_t index = 0u; index < kAccessCount; ++index)
    state->warmup[tid][index] = static_cast<uint16_t>(load_lds_u16(&warmup[tid][index]));

  if (wave == 0u)
    store_lds_b32(&tile[lane], 0xb2000000u | lane);
  if (synchronize)
    __syncthreads();
  if (wave == 1u)
    state->observed[lane] = load_lds_b32(&tile[lane]);
  state->control[tid] = 0xb3000000u | tid;
}

void run_cdna_b16_scc_route(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  cdna_b16_scc_route_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0u; tid < kThreadCount; ++tid) {
    EXPECT_EQ(result.branch_marker[tid], 0xb1000001u) << "thread=" << tid;
    EXPECT_EQ(result.control[tid], 0xb3000000u | tid) << "thread=" << tid;
    for (uint32_t index = 0u; index < kAccessCount; ++index)
      EXPECT_EQ(result.warmup[tid][index], static_cast<uint16_t>(0xb000u | tid))
          << "thread=" << tid << " access=" << index;
  }
  if (synchronize) {
    for (uint32_t lane = 0u; lane < kWaveSize; ++lane)
      EXPECT_EQ(result.observed[lane], 0xb2000000u | lane) << "lane=" << lane;
  }
}

TEST(ConSanDeviceCdnaB16SccRouteTest, Correct) { run_cdna_b16_scc_route(true); }
TEST(ConSanDeviceCdnaB16SccRouteTest, Incorrect) { run_cdna_b16_scc_route(false); }

} // namespace
