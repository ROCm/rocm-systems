// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_moi_rdna3_test.hip
/// @brief Compact physical gfx1100 fixture for all ConSan MOI engines.

#include <array>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string_view>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, 0) != hipSuccess ||
      !std::string_view(properties.gcnArchName).starts_with("gfx1100")) {
    std::cerr << "active HIP device is not gfx1100\n";
    return 1;
  }
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    const hipError_t error = (call);                                                               \
    ASSERT_EQ(error, hipSuccess) << "HIP error: " << hipGetErrorString(error);                     \
  } while (0)

#define RJ_CONSAN_RDNA3_NOP_CAVE                                                                   \
  ".rept 160\n\t"                                                                                  \
  "s_nop 0\n\t"                                                                                    \
  ".endr\n\t"

namespace {

__global__
__launch_bounds__(64) void moi_rdna3_barrier_ordered_accesses_for_instrumentation(uint32_t *out) {
  __shared__ volatile uint32_t lds[32];
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wave = threadIdx.x >> 5u;
  const uint32_t lds_address = lane * sizeof(uint32_t);
  const uint32_t value = lane + 100u + blockIdx.x + blockIdx.y + blockIdx.z;
  asm volatile("" : : "v"(lds) : "memory");
  if (wave == 0u) {
    asm volatile("ds_store_b32 %[address], %[value]\n\t" RJ_CONSAN_RDNA3_NOP_CAVE
                 "s_waitcnt lgkmcnt(0)\n\t"
                 :
                 : [address] "v"(lds_address), [value] "v"(value)
                 : "memory");
  }
  __syncthreads();
  if (wave == 1u) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %[loaded], %[address]\n\t" RJ_CONSAN_RDNA3_NOP_CAVE
                 "s_waitcnt lgkmcnt(0)\n\t"
                 : [loaded] "=v"(loaded)
                 : [address] "v"(lds_address)
                 : "memory");
    out[lane] = loaded;
  }
}

__global__
__launch_bounds__(64) void moi_rdna3_two_wave_racy_stores_for_instrumentation(uint32_t *out) {
  __shared__ volatile uint32_t lds[32];
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t lds_address = lane * sizeof(uint32_t);
  const uint32_t value = threadIdx.x + 1u;
  asm volatile("" : : "v"(lds) : "memory");
  asm volatile("ds_store_b32 %[address], %[value]\n\t" RJ_CONSAN_RDNA3_NOP_CAVE
               "s_waitcnt lgkmcnt(0)\n\t"
               :
               : [address] "v"(lds_address), [value] "v"(value)
               : "memory");
  out[threadIdx.x] = value;
}

TEST(ConSanMoiRdna3Test, BarrierOrderedAccessesPreserveOutput) {
  uint32_t *device_output = nullptr;
  HIP_ASSERT(hipMalloc(&device_output, 32u * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_output, 0, 32u * sizeof(uint32_t)));

  moi_rdna3_barrier_ordered_accesses_for_instrumentation<<<1, 64>>>(device_output);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, 32> output{};
  HIP_ASSERT(hipMemcpy(output.data(), device_output, output.size() * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_output));
  for (uint32_t lane = 0; lane < output.size(); ++lane)
    EXPECT_EQ(output[lane], lane + 100u) << "lane=" << lane;
}

TEST(ConSanMoiRdna3Test, TwoWaveRacyStoresComplete) {
  uint32_t *device_output = nullptr;
  HIP_ASSERT(hipMalloc(&device_output, 64u * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_output, 0, 64u * sizeof(uint32_t)));

  moi_rdna3_two_wave_racy_stores_for_instrumentation<<<1, 64>>>(device_output);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::array<uint32_t, 64> output{};
  HIP_ASSERT(hipMemcpy(output.data(), device_output, output.size() * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_output));
  for (uint32_t lane = 0; lane < output.size(); ++lane)
    EXPECT_EQ(output[lane], lane + 1u) << "lane=" << lane;
}

} // namespace
