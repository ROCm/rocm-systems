// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_init_test.cpp
/// @brief Verifies hsa_init() and GPU agent enumeration succeed through the
///        real ROCR runtime on the simulated GPU.
///
/// Requires LD_PRELOAD=librocjitsu.so.

#include <hsa/hsa.h>

#include <gtest/gtest.h>

class HsaTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    auto status = hsa_init();
    ASSERT_EQ(status, HSA_STATUS_SUCCESS) << "hsa_init failed: " << status;
  }
};

TEST_F(HsaTest, InitSucceeded) {
  SUCCEED(); // Init verified in SetUpTestSuite.
}

TEST_F(HsaTest, GpuAgentFound) {
  int gpu_count = 0;
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_GPU)
          ++*static_cast<int *>(data);
        return HSA_STATUS_SUCCESS;
      },
      &gpu_count);
  EXPECT_GE(gpu_count, 1) << "Expected at least one GPU agent";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  hsa_shut_down();
  return ret;
}
