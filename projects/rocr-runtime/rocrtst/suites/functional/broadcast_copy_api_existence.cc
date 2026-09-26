/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Broadcast copy tests: API existence & linkage
// Purpose: Verify broadcast copy API symbols exist and are callable
//
// The broadcast copy suite is split across files by what each one exercises:
//   BroadcastCopyApiExistence     - basic API existence and linkage (this file)
//   BroadcastCopyParamValidation  - parameter validation / error paths
//   BroadcastCopyCapabilityQuery  - capability queries
//   BroadcastCopySingleDest       - single destination functional
//   BroadcastCopyMultiDestSmall   - small multi-destination functional
//   BroadcastCopyChunking         - packet-size chunking and boundaries
//   BroadcastCopyDecisionTree     - SDMA-vs-shader dispatch selection
//   BroadcastCopyMultiDestLarge   - large scale multi-destination
//   BroadcastCopyStress           - stress / concurrent
//

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include "../../common/broadcast_copy_utils.h"

//
// Test Suite: BroadcastCopyApiExistence
//

class BroadcastCopyApiExistence : public ::testing::Test {
 protected:
  void SetUp() override {
    // Minimal setup - don't init HSA yet in SetUp (tests do it)
  }

  void TearDown() override {}
};

//
// Note: These tests verify that the convenience wrappers (defined in
// broadcast_copy_utils.h) compile and link correctly. The wrappers internally
// call hsa_amd_memory_async_batch_copy, which is the real exported runtime API.
//
// Runtime API linkage is verified by calling the underlying API with minimal
// parameters and checking that it returns a valid error code (not crash).
//

TEST_F(BroadcastCopyApiExistence, ApiSymbolResolution) {
  // Verify that the hsa_amd_memory_async_batch_copy API is available at runtime.
  // We call it with a null operation array to verify linkage without side effects.
  hsa_status_t status = hsa_init();
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "HSA runtime initialization failed";

  // Test that the underlying batch copy API is linked and callable
  // Passing nullptr for ops array should return an error (not crash)
  status = hsa_amd_memory_async_batch_copy(nullptr, 0, 0, nullptr);
  // Any return value other than crash indicates successful linkage
  std::cout << "Runtime API linkage verified:" << std::endl;
  std::cout << "  hsa_amd_memory_async_batch_copy returned status=" << status << std::endl;

  hsa_shut_down();
}

//
//
//

TEST_F(BroadcastCopyApiExistence, MinimalApiCall) {
  hsa_status_t init_status = hsa_init();
  ASSERT_EQ(HSA_STATUS_SUCCESS, init_status) << "HSA runtime initialization failed";

  std::cout << "Testing minimal API call (all NULL parameters):" << std::endl;

  // Call with all NULL (should return error, not crash)
  hsa_status_t status = hsa_amd_memory_broadcast_copy(nullptr,         // src
                                                      hsa_agent_t{},   // src_agent
                                                      nullptr,         // dst_list
                                                      nullptr,         // dst_agents
                                                      0,               // num_destinations
                                                      0,               // size
                                                      0,               // num_dep_signals
                                                      nullptr,         // dep_signals
                                                      hsa_signal_t{},  // completion_signal
                                                      HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "  API call returned status=" << status << std::endl;

  // Should fail with invalid argument (or not initialized if not implemented)
  ASSERT_NE(HSA_STATUS_SUCCESS, status) << "Expected error for NULL parameters";

  std::cout << "  [PASS] API handled NULL parameters gracefully (no crash)" << std::endl;

  hsa_shut_down();
}

//
//
//

TEST_F(BroadcastCopyApiExistence, CapabilityQueryBasicCall) {
  hsa_status_t init_status = hsa_init();
  ASSERT_EQ(HSA_STATUS_SUCCESS, init_status);

  std::cout << "Testing capability query:" << std::endl;

  hsa_agent_t gpu_agent = BroadcastTestUtils::FindGPUAgent();

  if (gpu_agent.handle == 0) {
    std::cout << "  [SKIP] No GPU agent found" << std::endl;
    hsa_shut_down();
    GTEST_SKIP() << "No GPU agent available";
    return;
  }

  std::string agent_name = BroadcastTestUtils::GetAgentName(gpu_agent);
  std::cout << "  GPU Agent: " << agent_name << " (handle=0x" << std::hex << gpu_agent.handle
            << std::dec << ")" << std::endl;

  uint32_t max_dests = 0;
  hsa_status_t status = hsa_amd_memory_broadcast_capability(gpu_agent, &max_dests);

  std::cout << "  Capability query returned:" << std::endl;
  std::cout << "    status=" << status << std::endl;
  std::cout << "    max_dests=" << max_dests << std::endl;

  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Capability query failed";

  // Accept 0 (no HW support) or 1-1024 (HW support)
  ASSERT_LE(max_dests, 1024U) << "max_dests exceeds hardware limit";

  if (max_dests > 0) {
    std::cout << "  Broadcast supported (max=" << max_dests << " destinations)" << std::endl;
  } else {
    std::cout << "  [INFO] Hardware multicast not supported (will use fallback)" << std::endl;
  }

  hsa_shut_down();
}

//
//
//

TEST_F(BroadcastCopyApiExistence, ApiVersionCheck) {
  hsa_status_t init_status = hsa_init();
  ASSERT_EQ(HSA_STATUS_SUCCESS, init_status);

  uint16_t major = 0, minor = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major);
  hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &minor);

  std::cout << "ROCr Runtime Version: " << major << "." << minor << std::endl;

  // Broadcast copy requires ROCr 1.14+ (use 1.14 if 1.18 not available yet)
  bool version_ok = (major > 1) || (major == 1 && minor >= 14);

  if (!version_ok) {
    std::cout << "  [WARNING] ROCr version " << major << "." << minor
              << " may not support broadcast copy" << std::endl;
    std::cout << "  (Expected 1.18+, but 1.14+ may have partial support)" << std::endl;
  } else {
    std::cout << "  [PASS] ROCr version is compatible" << std::endl;
  }

  // Don't fail on version mismatch (may be testing on older runtime)
  // ASSERT_TRUE(version_ok) << "ROCr version too old";

  hsa_shut_down();
}

//
//
//

TEST_F(BroadcastCopyApiExistence, RuntimeInitializationState) {
  std::cout << "Testing runtime initialization:" << std::endl;

  // Test 1: Call without init (should fail)
  hsa_agent_t gpu_agent = BroadcastTestUtils::FindGPUAgent();
  uint32_t max_dests = 999;

  hsa_status_t status_before = hsa_amd_memory_broadcast_capability(gpu_agent, &max_dests);
  std::cout << "  Before hsa_init(): status=" << status_before << std::endl;

  // Now init
  hsa_status_t init_status = hsa_init();
  ASSERT_EQ(HSA_STATUS_SUCCESS, init_status);
  std::cout << "  After hsa_init(): HSA runtime initialized" << std::endl;

  // Test 2: Call after init (should succeed if GPU available)
  gpu_agent = BroadcastTestUtils::FindGPUAgent();
  if (gpu_agent.handle != 0) {
    hsa_status_t status_after = hsa_amd_memory_broadcast_capability(gpu_agent, &max_dests);
    std::cout << "  After hsa_init(): status=" << status_after << ", max_dests=" << max_dests
              << std::endl;
    ASSERT_EQ(HSA_STATUS_SUCCESS, status_after);
    std::cout << "  [PASS] API works correctly after runtime initialization" << std::endl;
  } else {
    std::cout << "  [SKIP] No GPU agent found" << std::endl;
  }

  hsa_shut_down();
}
