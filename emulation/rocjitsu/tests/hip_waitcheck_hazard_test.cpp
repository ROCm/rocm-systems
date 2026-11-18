// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_waitcheck_hazard_test.cpp
/// @brief HIP kernel with an intentional scalar-memory wait hazard.

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <fcntl.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <unistd.h>

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
std::filesystem::path g_clean_fixture;

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

TEST(HipWaitcheckHazardTest, SameSizeLoaderSubstitutionDoesNotReadTemporaryImage) {
  const auto image = read_hazard_fixture();
  ASSERT_FALSE(image.empty()) << g_hazard_fixture;
  const std::string option = "-substitute:1=" + g_hazard_fixture.string();
  ASSERT_EQ(setenv("LOADER_OPTIONS_APPEND", option.c_str(), 1), 0);
  hipModule_t module = nullptr;
  ASSERT_EQ(hipModuleLoadData(&module, image.data()), hipSuccess);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(hipModuleUnload(module), hipSuccess);
}

void check_clean_file_reader_load() {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
  hsa_agent_t gpu{};
  ASSERT_EQ(hsa_iterate_agents(
                [](hsa_agent_t agent, void *data) {
                  hsa_device_type_t type{};
                  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) ==
                          HSA_STATUS_SUCCESS &&
                      type == HSA_DEVICE_TYPE_GPU && static_cast<hsa_agent_t *>(data)->handle == 0)
                    *static_cast<hsa_agent_t *>(data) = agent;
                  return HSA_STATUS_SUCCESS;
                },
                &gpu),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(gpu.handle, 0u);

  const int file = ::open(g_clean_fixture.c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(file, 0) << g_clean_fixture;
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(hsa_code_object_reader_create_from_file(file, &reader), HSA_STATUS_SUCCESS);
  hsa_executable_t executable{};
  ASSERT_EQ(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &executable),
            HSA_STATUS_SUCCESS);
  hsa_loaded_code_object_t loaded{};
  EXPECT_EQ(hsa_executable_load_agent_code_object(executable, gpu, reader, nullptr, &loaded),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(loaded.handle, 0u);
  EXPECT_EQ(hsa_executable_freeze(executable, nullptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_executable_destroy(executable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_code_object_reader_destroy(reader), HSA_STATUS_SUCCESS);
  EXPECT_EQ(::close(file), 0);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(HipWaitcheckHazardTest, CleanFileReaderLoadsWithWaitChecking) {
  check_clean_file_reader_load();
}

TEST(HipWaitcheckHazardTest, CleanPathModuleLoadsWithWaitChecking) {
  hipModule_t module = nullptr;
  ASSERT_EQ(hipModuleLoad(&module, g_clean_fixture.c_str()), hipSuccess);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(hipModuleUnload(module), hipSuccess);
}

TEST(HipWaitcheckHazardTest, FileReaderSubstitutionIsUnavailable) {
  const std::string option = "-substitute:1=" + g_clean_fixture.string();
  ASSERT_EQ(setenv("LOADER_OPTIONS_APPEND", option.c_str(), 1), 0);
  check_clean_file_reader_load();
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
  const auto fixture_dir = std::filesystem::canonical("/proc/self/exe").parent_path();
  g_hazard_fixture = fixture_dir / "hip_waitcheck_hazard_fixture.hsaco";
  g_clean_fixture = fixture_dir / "hip_waitcheck_clean_fixture.hsaco";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
