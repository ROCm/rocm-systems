// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_waitcheck_hazard_test.cpp
/// @brief HIP kernel with an intentional scalar-memory wait hazard.

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

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

std::filesystem::path g_hazard_fixture;

std::vector<char> read_hazard_fixture() {
  std::ifstream input(g_hazard_fixture, std::ios::binary);
  if (!input)
    return {};
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

TEST(HipWaitcheckHazardTest, FailClosedRejectsKernelCodeObject) {
  const auto image = read_hazard_fixture();
  ASSERT_FALSE(image.empty()) << g_hazard_fixture;
  hipModule_t module = nullptr;
  EXPECT_NE(hipModuleLoadData(&module, image.data()), hipSuccess)
      << "waitcheck fail-closed mode accepted the hazardous code object at module load";
}

TEST(HipWaitcheckHazardTest, ModuleLoadsWithoutChecking) {
  const auto image = read_hazard_fixture();
  ASSERT_FALSE(image.empty()) << g_hazard_fixture;
  hipModule_t module = nullptr;
  ASSERT_EQ(hipModuleLoadData(&module, image.data()), hipSuccess);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(hipModuleUnload(module), hipSuccess);
}

TEST(HipWaitcheckHazardTest, DispatchRejectsBeforeSubmission) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        (void)hipLaunchKernel(reinterpret_cast<const void *>(missing_smem_wait_kernel), dim3(1),
                              dim3(1), nullptr, 0, nullptr);
        (void)hipDeviceSynchronize();
        std::_Exit(0);
      },
      ::testing::KilledBySignal(SIGABRT),
      "refusing to dispatch kernel .*missing_smem_wait_kernel");
}

int main(int argc, char **argv) {
  g_hazard_fixture = std::filesystem::canonical("/proc/self/exe").parent_path() /
                     "hip_waitcheck_hazard_fixture.hsaco";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
