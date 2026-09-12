/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
//
//

TEST_F(BroadcastCopyL1, NullSourcePointer) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* dst = ctx.AllocateGPUBuffer(4096);
  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(nullptr,  // [FAIL] NULL src
                                    ctx.gpu_agent, dst_list, dst_agents, 1, 4096, 0, nullptr,
                                    hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "Null src → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(dst);
}

//
//
//

TEST_F(BroadcastCopyL1, NullDestinationList) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  hsa_status_t status = hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent,
                                                      nullptr,  // [FAIL] NULL dst_list
                                                      dst_agents, 1, 4096, 0, nullptr,
                                                      hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "Null dst_list → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
}

//
//
//

TEST_F(BroadcastCopyL1, NullDestinationAgents) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  void* dst = ctx.AllocateGPUBuffer(4096);
  void* dst_list[1] = {dst};

  hsa_status_t status = hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list,
                                                      nullptr,  // [FAIL] NULL dst_agents
                                                      1, 4096, 0, nullptr, hsa_signal_t{},
                                                      HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "Null dst_agents → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
  ctx.Free(dst);
}

//
//
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
                                    0,  // [FAIL] num_destinations=0
                                    4096, 0, nullptr, hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "num_dests=0 → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
  ctx.Free(dst);
}

//
//
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
                                    max_dests + 1,  // [FAIL] Over limit
                                    4096, 0, nullptr, hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "num_dests=" << (max_dests + 1) << " (max=" << max_dests
            << ") → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  for (auto dst : dst_list) ctx.Free(dst);
  ctx.Free(src);
}

//
//
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
                                    0,  // size=0 is a no-op, returns SUCCESS
                                    0, nullptr, hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "size=0 → status=" << status << std::endl;
  // size=0 is valid and acts as a no-op (consistent with standard memcpy behavior)
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  ctx.Free(src);
  ctx.Free(dst);
}

//
//
//

TEST_F(BroadcastCopyL1, InvalidSourceAgent) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  void* dst = ctx.AllocateGPUBuffer(4096);
  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  hsa_agent_t invalid_agent = {0xDEADBEEF};  // [FAIL] Invalid handle

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, invalid_agent, dst_list, dst_agents, 1, 4096, 0, nullptr,
                                    hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "Invalid src_agent → status=" << status << std::endl;
  // Runtime may return INVALID_ARGUMENT or INVALID_AGENT - both are acceptable
  ASSERT_TRUE(status == HSA_STATUS_ERROR_INVALID_AGENT ||
              status == HSA_STATUS_ERROR_INVALID_ARGUMENT)
      << "Expected INVALID_AGENT or INVALID_ARGUMENT, got " << status;

  ctx.Free(src);
  ctx.Free(dst);
}

//
//
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
                                    5,        // [FAIL] num_dep_signals=5
                                    nullptr,  // [FAIL] but dep_signals=NULL
                                    hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "num_dep_signals=5, dep_signals=NULL → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
  ctx.Free(dst);
}

//
//
//

TEST_F(BroadcastCopyL1, NullPointerInDestList) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* src = ctx.AllocateGPUBuffer(4096);
  void* dst1 = ctx.AllocateGPUBuffer(4096);

  void* dst_list[3] = {dst1, nullptr, dst1};  // [FAIL] Middle element is NULL
  hsa_agent_t dst_agents[3] = {ctx.gpu_agent, ctx.gpu_agent, ctx.gpu_agent};

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list, dst_agents, 3, 4096, 0, nullptr,
                                    hsa_signal_t{}, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "dst_list[1]=NULL → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  ctx.Free(src);
  ctx.Free(dst1);
}

//
//
//

TEST_F(BroadcastCopyL1, CapabilityQueryNullOutput) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  hsa_status_t status =
      hsa_amd_memory_broadcast_capability(ctx.gpu_agent, nullptr  // [FAIL] NULL max_destinations
      );

  std::cout << "Null max_destinations output → status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);
}
