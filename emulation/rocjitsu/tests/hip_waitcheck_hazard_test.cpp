// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_waitcheck_hazard_test.cpp
/// @brief HIP kernel with an intentional scalar-memory wait hazard.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

// Keep the producer and consumer in one inline-assembly block so LLVM cannot
// repair the deliberately missing lgkmcnt wait between them. The kernel is
// never allowed to execute in the fail-closed test.
__global__ void missing_smem_wait_kernel() {
  asm volatile("s_load_dword s4, s[0:1], 0\n\t"
               "s_mov_b32 s5, s4"
               :
               :
               : "s4", "s5", "memory");
}

TEST(HipWaitcheckHazardTest, FailClosedRejectsKernelCodeObject) {
  const hipError_t launch_status =
      hipLaunchKernel(reinterpret_cast<const void *>(missing_smem_wait_kernel), dim3(1), dim3(1),
                      nullptr, 0, nullptr);
  const hipError_t sync_status = hipDeviceSynchronize();
  EXPECT_TRUE(launch_status != hipSuccess || sync_status != hipSuccess)
      << "waitcheck fail-closed mode unexpectedly allowed the hazardous kernel to execute";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
