// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Native 12-byte LDS tuple publication. The aliasing destination/address is
/// retained from framework-generated RDNA4/CDNA5 kernels: instrumentation must
/// snapshot the address before the B96 load overwrites the same VGPR.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;

struct DeviceState {
  uint32_t synchronize;
  uint32_t output[5];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void store_b96(uint32_t address) {
  constexpr uint32_t kValue0 = 0x13579bdfu;
  constexpr uint32_t kValue1 = 0x2468ace0u;
  constexpr uint32_t kValue2 = 0xa5a55a5au;
  asm volatile("v_mov_b32 v4, %0\n\t"
               "v_mov_b32 v5, %1\n\t"
               "v_mov_b32 v6, %2\n\t"
               "v_mov_b32 v7, %3\n\t"
               "ds_store_b96 v7, v[4:6]\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE
               :
               : "v"(kValue0), "v"(kValue1), "v"(kValue2), "v"(address)
               : "v4", "v5", "v6", "v7", "memory");
}

__device__ __forceinline__ uint3 load_b96_with_aliasing_address(uint32_t address) {
  uint32_t value0 = 0u;
  uint32_t value1 = 0u;
  uint32_t value2 = 0u;
  asm volatile("v_mov_b32 v4, %3\n\t"
               "ds_load_b96 v[4:6], v4\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               "v_mov_b32 %0, v4\n\t"
               "v_mov_b32 %1, v5\n\t"
               "v_mov_b32 %2, v6\n\t"
               : "=v"(value0), "=v"(value1), "=v"(value2)
               : "v"(address)
               : "v4", "v5", "v6", "memory");
  return make_uint3(value0, value1, value2);
}

__global__ __launch_bounds__(kThreadCount) void rdna_b96_boundary_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint32_t tuple_with_canaries[5];
  const uint32_t tid = threadIdx.x;
  if (tid == 0u) {
    tuple_with_canaries[0] = 0xfeed0001u;
    tuple_with_canaries[1] = 0x01020304u;
    tuple_with_canaries[2] = 0x11223344u;
    tuple_with_canaries[3] = 0x55667788u;
    tuple_with_canaries[4] = 0xfeed0002u;
  }
  __syncthreads();

  const uint32_t tuple_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&tuple_with_canaries[1]) & 0xffffu);
  if (tid == 0u)
    store_b96(tuple_address);
  if (state->synchronize != 0u)
    __syncthreads();
  if (tid == kWaveSize) {
    const uint3 value = load_b96_with_aliasing_address(tuple_address);
    state->output[0] = tuple_with_canaries[0];
    state->output[1] = value.x;
    state->output[2] = value.y;
    state->output[3] = value.z;
    state->output[4] = tuple_with_canaries[4];
  }
  state->control[tid] = 0xb9600000u | tid;
}

void run_rdna_b96_boundary(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  rdna_b96_boundary_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0xb9600000u | tid) << "thread=" << tid;
  if (!synchronize)
    return;
  constexpr uint32_t expected[] = {0xfeed0001u, 0x13579bdfu, 0x2468ace0u, 0xa5a55a5au, 0xfeed0002u};
  for (uint32_t i = 0; i < 5u; ++i)
    EXPECT_EQ(result.output[i], expected[i]) << "element=" << i;
}

TEST(ConSanDeviceRdnaB96BoundaryTest, Correct) { run_rdna_b96_boundary(true); }
TEST(ConSanDeviceRdnaB96BoundaryTest, Incorrect) { run_rdna_b96_boundary(false); }

} // namespace
