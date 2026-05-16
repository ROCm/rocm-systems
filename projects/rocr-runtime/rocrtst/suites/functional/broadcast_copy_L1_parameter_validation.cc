// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// ROCrtst Level 1 Tests: Parameter Validation
// Purpose: Validate error handling for all invalid parameter combinations

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <vector>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyL1 : public ::testing::Test {
 protected:
  void SetUp() override {
    // Tests manage context individually
  }
};

//
// TC-L1-001: Null Source Pointer
//

TEST_F(BroadcastCopyL1, NullSourcePointer) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* dst = ctx.AllocateGPUBuffer(4096);
  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(nullptr,  // ❌ NULL src
                                    ctx.gpu_agent, dst_list, dst_agents, 1, 4096, 0, nullptr,
                                    hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "[TC-L1-001] Null src → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(dst);
}

//
// TC-L1-002: Null Destination List
//

TEST_F(BroadcastCopyL1, NullDestinationList) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  hsa_status_t status = hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent,
                                                      nullptr,  // ❌ NULL dst_list
                                                      dst_agents, 1, 4096, 0, nullptr,
                                                      hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "[TC-L1-002] Null dst_list → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
}

//
// TC-L1-003: Null Destination Agents
//

TEST_F(BroadcastCopyL1, NullDestinationAgents) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  void* dst = ctx.AllocateGPUBuffer(4096);
  void* dst_list[1] = {dst};

  hsa_status_t status = hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list,
                                                      nullptr,  // ❌ NULL dst_agents
                                                      1, 4096, 0, nullptr, hsa_signal_t{},
                                                      HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "[TC-L1-003] Null dst_agents → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
  ctx.Free(dst);
}

//
// TC-L1-004: Zero Destinations
//

TEST_F(BroadcastCopyL1, ZeroDestinations) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  void* dst = ctx.AllocateGPUBuffer(4096);
  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list, dst_agents,
                                    0,  // ❌ num_destinations=0
                                    4096, 0, nullptr, hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "[TC-L1-004] num_dests=0 → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
  ctx.Free(dst);
}

//
// TC-L1-005: Exceeds Max Destinations
//

TEST_F(BroadcastCopyL1, ExceedsMaxDestinations) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  // Query actual max
  uint32_t max_dests = 0;
  hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &max_dests);

  if (max_dests == 0) {
    GTEST_SKIP() << "Hardware multicast not supported, test not applicable";
  }

  void* src = ctx.AllocateGPUBuffer(4096);
  std::vector<void*> dst_list(max_dests + 1);
  std::vector<hsa_agent_t> dst_agents(max_dests + 1, ctx.gpu_agent);

  for (auto& dst : dst_list) {
    dst = ctx.AllocateGPUBuffer(4096);
  }

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list.data(), dst_agents.data(),
                                    max_dests + 1,  // ❌ Over limit
                                    4096, 0, nullptr, hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "[TC-L1-005] num_dests=" << (max_dests + 1) << " (max=" << max_dests
            << ") → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  for (auto dst : dst_list) ctx.Free(dst);
  ctx.Free(src);
}

//
// TC-L1-006: Zero Size
//

TEST_F(BroadcastCopyL1, ZeroSize) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  void* dst = ctx.AllocateGPUBuffer(4096);
  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list, dst_agents, 1,
                                    0,  // ❌ size=0
                                    0, nullptr, hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "[TC-L1-006] size=0 → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
  ctx.Free(dst);
}

//
// TC-L1-007: Invalid Source Agent
//

TEST_F(BroadcastCopyL1, InvalidSourceAgent) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  void* dst = ctx.AllocateGPUBuffer(4096);
  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  hsa_agent_t invalid_agent = {0xDEADBEEF};  // ❌ Invalid handle

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, invalid_agent, dst_list, dst_agents, 1, 4096, 0, nullptr,
                                    hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "[TC-L1-007] Invalid src_agent → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_AGENT, status);

  ctx.Free(src);
  ctx.Free(dst);
}

//
// TC-L1-008: Mismatched Signal Count
//

TEST_F(BroadcastCopyL1, MismatchedSignalCount) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  void* dst = ctx.AllocateGPUBuffer(4096);
  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list, dst_agents, 1, 4096,
                                    5,        // ❌ num_dep_signals=5
                                    nullptr,  // ❌ but dep_signals=NULL
                                    hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "[TC-L1-008] num_dep_signals=5, dep_signals=NULL → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
  ctx.Free(dst);
}

//
// TC-L1-009: Null Pointer in Destination List
//

TEST_F(BroadcastCopyL1, NullPointerInDestList) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  void* dst1 = ctx.AllocateGPUBuffer(4096);

  void* dst_list[3] = {dst1, nullptr, dst1};  // ❌ Middle element is NULL
  hsa_agent_t dst_agents[3] = {ctx.gpu_agent, ctx.gpu_agent, ctx.gpu_agent};

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list, dst_agents, 3, 4096, 0, nullptr,
                                    hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "[TC-L1-009] dst_list[1]=NULL → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
  ctx.Free(dst1);
}

//
// TC-L1-010: Capability Query with NULL Output
//

TEST_F(BroadcastCopyL1, CapabilityQueryNullOutput) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  hsa_status_t status =
      hsa_amd_memory_broadcast_capability(ctx.gpu_agent, nullptr  // ❌ NULL max_destinations
      );

  std::cout << "[TC-L1-010] Null max_destinations output → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);
}
