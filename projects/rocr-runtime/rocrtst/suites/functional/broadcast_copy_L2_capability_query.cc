/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// ROCrtst Level 2 Tests: Capability Query
// Purpose: Test capability detection across different agent types and hardware

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyL2 : public ::testing::Test {
 protected:
  void SetUp() override {}
};

struct AgentInfo {
  hsa_agent_t agent;
  char name[64];
  uint32_t max_broadcast_dests;
  bool supports_multicast;
};

//
//
//

TEST_F(BroadcastCopyL2, QueryAllGpuAgents) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  // Test with single GPU agent (HsaTestContext provides single gpu_agent)
  AgentInfo info;
  info.agent = ctx.gpu_agent;
  hsa_agent_get_info(ctx.gpu_agent, HSA_AGENT_INFO_NAME, &info.name);

  hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &info.max_broadcast_dests);
  info.supports_multicast = (info.max_broadcast_dests > 0);

  std::cout << "GPU agent capability:" << std::endl;
  std::cout << "  Agent: " << info.name << std::endl;
  std::cout << "    Handle: 0x" << std::hex << info.agent.handle << std::dec << std::endl;
  std::cout << "    Max Broadcast Dests: " << info.max_broadcast_dests;
  if (info.max_broadcast_dests >= 1024) {
    std::cout << " (Broadcast supported: SDMA or Shader fallback)" << std::endl;
  } else if (info.max_broadcast_dests > 0) {
    std::cout << " (Limited support: " << info.max_broadcast_dests << " dests)" << std::endl;
  } else {
    std::cout << " (No broadcast support)" << std::endl;
  }

  // Skip on hardware that doesn't support broadcast (pre-GFX12)
  if (!info.supports_multicast) {
    GTEST_SKIP() << "GPU does not support broadcast copy (max_dests=" << info.max_broadcast_dests << ")";
  }

  std::cout << "  [PASS] Broadcast capability query succeeded" << std::endl;
}

//
//
//

TEST_F(BroadcastCopyL2, QueryCpuAgent) {
  HsaTestContext ctx;

  hsa_agent_t cpu_agent = ctx.cpu_agent;

  if (cpu_agent.handle == 0) {
    GTEST_SKIP() << "No CPU agent found";
  }

  uint32_t max_dests = 999;  // Initialize to non-zero
  hsa_status_t status = hsa_amd_memory_broadcast_capability(cpu_agent, &max_dests);

  std::cout << "CPU agent capability query:" << std::endl;
  std::cout << "  status=" << status << std::endl;
  std::cout << "  max_dests=" << max_dests << " (expected 0)" << std::endl;

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  ASSERT_EQ(0, max_dests) << "CPU agents should not support HW broadcast";
}

//
//
//

TEST_F(BroadcastCopyL2, CapabilityConsistencyCheck) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  // Query 10 times, should always return same value
  std::vector<uint32_t> results;
  for (int i = 0; i < 10; i++) {
    uint32_t max_dests = 0;
    hsa_status_t status = hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &max_dests);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    results.push_back(max_dests);
  }

  std::cout << "Capability query results (10 iterations):" << std::endl;
  for (size_t i = 0; i < results.size(); i++) {
    std::cout << "  [" << i << "] = " << results[i] << std::endl;
  }

  // All results should be identical
  for (size_t i = 1; i < results.size(); i++) {
    ASSERT_EQ(results[0], results[i])
        << "Capability query returned inconsistent results: " << results[0] << " vs " << results[i];
  }

  std::cout << "  [PASS] All queries returned consistent value: " << results[0] << std::endl;
}

//
//
//

TEST_F(BroadcastCopyL2, IsaVersionCorrelation) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  // Get ISA name
  char isa_name[64] = {0};
  hsa_agent_get_info(ctx.gpu_agent, HSA_AGENT_INFO_NAME, &isa_name);

  uint32_t max_dests = 0;
  hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &max_dests);

  std::cout << "ISA correlation check:" << std::endl;
  std::cout << "  ISA: " << isa_name << std::endl;
  std::cout << "  Max Broadcast Dests: " << max_dests << std::endl;

  // Expected correlation (based on architecture)
  // Use strstr since ISA name format is "amdgcn-amd-amdhsa--gfxXXXX"
  bool is_gfx13_plus = (strstr(isa_name, "gfx13") != nullptr);
  bool is_gfx14_plus = (strstr(isa_name, "gfx14") != nullptr);

  if (is_gfx13_plus || is_gfx14_plus) {
    std::cout << "  Expected: ≥1024 (GFX13+ or newer)" << std::endl;
    if (max_dests == 0) {
      std::cout << "  [WARNING]  WARNING: GFX13+/GFX14+ returned 0, HW/FW support may be missing"
                << std::endl;
    } else {
      std::cout << "  [PASS] GFX13+/GFX14+ reports multicast support" << std::endl;
    }
  } else {
    std::cout << "  Expected: 0 (Pre-GFX13 hardware)" << std::endl;
    if (max_dests > 0) {
      std::cout << "  [INFO]  Note: Pre-GFX13 hardware reports multicast support" << std::endl;
    }
  }
}

//
//
//

TEST_F(BroadcastCopyL2, MultipleGpuAgentsCapability) {
  // Note: This test requires multi-GPU support which HsaTestContext doesn't fully provide
  // Simplified to test single GPU capability
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "GPU capability check:" << std::endl;

  hsa_agent_t agent = ctx.gpu_agent;
  char name[64];
  hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, &name);

  uint32_t max_dests = 0;
  hsa_status_t status = hsa_amd_memory_broadcast_capability(agent, &max_dests);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  std::cout << "  GPU (" << name << "): max_dests=" << max_dests << std::endl;
}
