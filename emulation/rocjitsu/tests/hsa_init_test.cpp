// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_init_test.cpp
/// @brief Verifies hsa_init() and GPU agent enumeration succeed through the
///        real ROCR runtime on the simulated GPU.
///
/// Requires LD_PRELOAD=librocjitsu.so.

#include <hsa/hsa.h>

#include <gtest/gtest.h>

#if defined(RJ_TEST_LEAK_SANITIZER)
#include <sanitizer/lsan_interface.h>
#endif

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

#if defined(RJ_TEST_LEAK_SANITIZER)
TEST_F(HsaTest, LiveSignalsRemainReachableForLeakSanitizer) {
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(0, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  // The signal handle points into driver-managed memory, which in turn owns
  // the host Signal object. Both remain live during this leak check.
  EXPECT_EQ(__lsan_do_recoverable_leak_check(), 0);
  ASSERT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(__lsan_do_recoverable_leak_check(), 0);
}
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  hsa_shut_down();
  return ret;
}
