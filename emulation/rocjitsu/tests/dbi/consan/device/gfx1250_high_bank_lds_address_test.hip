// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// gfx1250 high-bank SRC0 addressing and tensor-to-LDS tail synchronization
/// distilled from the MXF4 TDM Tensile kernels. The guest DS instruction names
/// v286 through encoded v30 while the instrumentation scratch bank remains
/// below v256. The correct path also retains the workload's adjacent split
/// barriers separated by a DS wait. Record/Replay must preserve both the
/// physical address-bank distinction and the synchronization contract.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreadCount = 2u * kWaveSize;

struct DeviceState {
  uint32_t synchronize;
  uint32_t loaded[kWaveSize];
  uint32_t control[kThreadCount];
};

__device__ __forceinline__ void store_high_bank_address(uint32_t address, uint32_t value) {
  register uint32_t value_low asm("v20") = value;
  asm volatile("s_set_vgpr_msb 0x40\n\t"
               "v_mov_b32 v30, %0\n\t"
               "s_set_vgpr_msb 0x4000\n\t"
               "s_set_vgpr_msb 0x1\n\t"
               "ds_store_b32 v30, %1\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_set_vgpr_msb 0x100\n\t"
               :
               : "v"(address), "v"(value_low)
               : "memory");
}

__device__ __forceinline__ uint32_t load_high_bank_address(uint32_t address) {
  register uint32_t value_low asm("v21") = 0u;
  asm volatile("s_set_vgpr_msb 0x40\n\t"
               "v_mov_b32 v30, %1\n\t"
               "s_set_vgpr_msb 0x4000\n\t"
               "s_set_vgpr_msb 0x1\n\t"
               "ds_load_b32 %0, v30\n\t" RJ_CONSAN_DEVICE_LARGE_CAVE "s_wait_dscnt 0\n\t"
               "s_set_vgpr_msb 0x100\n\t"
               : "+v"(value_low)
               : "v"(address)
               : "memory");
  return value_low;
}

// Restricting occupancy to one wave per execution unit makes the code-object
// descriptor reserve enough physical VGPRs for the explicit v286 operand.
// The amdgpu_num_vgpr attribute remains an upper bound for register allocation.
__attribute__((amdgpu_num_vgpr(512), amdgpu_waves_per_eu(1, 1))) __global__
__launch_bounds__(kThreadCount) void gfx1250_high_bank_lds_address_for_consan(DeviceState *state) {
  __shared__ uint32_t staged[kWaveSize];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (kWaveSize - 1u);
  const uint32_t address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&staged[lane]) & 0xfffffu);

  if (tid < kWaveSize)
    store_high_bank_address(address, 0x12500000u | lane);
  if (state->synchronize != 0u) {
    __syncthreads();
    asm volatile("s_wait_dscnt 0" ::: "memory");
    __syncthreads();
  }
  if (tid >= kWaveSize)
    state->loaded[lane] = load_high_bank_address(address);
  state->control[tid] = 0x1250c000u | tid;
}

void run_gfx1250_high_bank_lds_address(bool synchronize) {
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  gfx1250_high_bank_lds_address_for_consan<<<1, kThreadCount>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < kThreadCount; ++tid)
    EXPECT_EQ(result.control[tid], 0x1250c000u | tid) << "thread=" << tid;
  if (!synchronize)
    return;
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    EXPECT_EQ(result.loaded[lane], 0x12500000u | lane) << "lane=" << lane;
}

TEST(ConSanDeviceGfx1250HighBankLdsAddressTest, Correct) {
  run_gfx1250_high_bank_lds_address(true);
}

TEST(ConSanDeviceGfx1250HighBankLdsAddressTest, Incorrect) {
  run_gfx1250_high_bank_lds_address(false);
}

} // namespace
