// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_moi_gfx950_test.hip
/// @brief Focused native gfx950 ConSan MOI instrumentation tests.

#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

namespace {

__global__ __launch_bounds__(64) void gfx950_moi_single_lds_store(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  const uint32_t value = 0x95020000u | tid;
  asm volatile("ds_write_b32 %0, %1\n\t"
               ".rept 128\n\t"
               "s_nop 0\n\t"
               ".endr\n\t"
               :
               : "v"(address), "v"(value)
               : "memory");
  __syncthreads();
  if (tid == 0)
    out[0] = lds[0];
}

TEST(ConSanMoiGfx950Test, SingleLdsStoreCompletes) {
  uint32_t *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(device_out, 0, sizeof(uint32_t)));
  gfx950_moi_single_lds_store<<<1, 64>>>(device_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t out = 0;
  HIP_ASSERT(hipMemcpy(&out, device_out, sizeof(out), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));
  EXPECT_EQ(out & 0xffff0000u, 0x95020000u);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}
