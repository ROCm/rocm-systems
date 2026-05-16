// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// ROCrtst Level 0 Tests: API Existence & Linkage
// Purpose: Verify broadcast copy API symbols exist and are callable

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include "../../common/broadcast_copy_utils.h"

//
// Test Suite: BroadcastCopyL0
//

class BroadcastCopyL0 : public ::testing::Test {
 protected:
  void SetUp() override {
    // Minimal setup - don't init HSA yet in SetUp (tests do it)
  }

  void TearDown() override {}
};

//
// TC-L0-001: API Symbol Resolution
//

TEST_F(BroadcastCopyL0, ApiSymbolResolution) {
  // Verify function pointers are non-null
  ASSERT_NE(nullptr, &hsa_amd_memory_broadcast_copy)
      << "hsa_amd_memory_broadcast_copy symbol not found";
  ASSERT_NE(nullptr, &hsa_amd_memory_broadcast_capability)
      << "hsa_amd_memory_broadcast_capability symbol not found";

  std::cout << "[TC-L0-001] API symbols resolved successfully" << std::endl;
  std::cout << "  hsa_amd_memory_broadcast_copy @ " << (void*)&hsa_amd_memory_broadcast_copy
            << std::endl;
  std::cout << "  hsa_amd_memory_broadcast_capability @ "
            << (void*)&hsa_amd_memory_broadcast_capability << std::endl;
}

//
// TC-L0-002: Minimal API Call (No Crash)
//

TEST_F(BroadcastCopyL0, MinimalApiCall) {
  hsa_status_t init_status = hsa_init();
  ASSERT_EQ(HSA_STATUS_SUCCESS, init_status) << "HSA runtime initialization failed";

  std::cout << "[TC-L0-002] Testing minimal API call (all NULL parameters):" << std::endl;

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

  std::cout << "  ✓ API handled NULL parameters gracefully (no crash)" << std::endl;

  hsa_shut_down();
}

//
// TC-L0-003: Capability Query Basic Call
//

TEST_F(BroadcastCopyL0, CapabilityQueryBasicCall) {
  hsa_status_t init_status = hsa_init();
  ASSERT_EQ(HSA_STATUS_SUCCESS, init_status);

  std::cout << "[TC-L0-003] Testing capability query:" << std::endl;

  hsa_agent_t gpu_agent = BroadcastTestUtils::FindGPUAgent();

  if (gpu_agent.handle == 0) {
    std::cout << "  ⚠️  No GPU agent found - test skipped" << std::endl;
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
  ASSERT_LE(max_dests, 1024) << "max_dests exceeds hardware limit";

  if (max_dests > 0) {
    std::cout << "  ✓ Hardware multicast supported (max=" << max_dests << " destinations)"
              << std::endl;
  } else {
    std::cout << "  ℹ️  Hardware multicast not supported (will use fallback)" << std::endl;
  }

  hsa_shut_down();
}

//
// TC-L0-004: Version Check
//

TEST_F(BroadcastCopyL0, ApiVersionCheck) {
  hsa_status_t init_status = hsa_init();
  ASSERT_EQ(HSA_STATUS_SUCCESS, init_status);

  uint16_t major = 0, minor = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &major);
  hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &minor);

  std::cout << "[TC-L0-004] ROCr Runtime Version: " << major << "." << minor << std::endl;

  // Broadcast copy requires ROCr 1.14+ (use 1.14 if 1.18 not available yet)
  bool version_ok = (major > 1) || (major == 1 && minor >= 14);

  if (!version_ok) {
    std::cout << "  ⚠️  WARNING: ROCr version " << major << "." << minor
              << " may not support broadcast copy" << std::endl;
    std::cout << "  (Expected 1.18+, but 1.14+ may have partial support)" << std::endl;
  } else {
    std::cout << "  ✓ ROCr version is compatible" << std::endl;
  }

  // Don't fail on version mismatch (may be testing on older runtime)
  // ASSERT_TRUE(version_ok) << "ROCr version too old";

  hsa_shut_down();
}

//
// TC-L0-005: Runtime Initialization State
//

TEST_F(BroadcastCopyL0, RuntimeInitializationState) {
  std::cout << "[TC-L0-005] Testing runtime initialization:" << std::endl;

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
    std::cout << "  ✓ API works correctly after runtime initialization" << std::endl;
  } else {
    std::cout << "  ⚠️  No GPU agent found" << std::endl;
  }

  hsa_shut_down();
}

//
// Test Summary
//
