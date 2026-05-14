/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/cuid.h"

CuidTest::CuidTest() : TestBase() {
  set_title("RocR Secondary CUID Test");
  set_description(
      "This test validates that ROCR returns secondary CUID values "
      "from /tmp/cuid file correctly when any user application queries for it "
      "using hsa_agent_get_info API.");
}

CuidTest::~CuidTest() {}

void CuidTest::SetUp() { TestBase::SetUp(); }

void CuidTest::Run() {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void CuidTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void CuidTest::DisplayResults() const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::DisplayResults();
}

void CuidTest::DisplayTestInfo() { 
    TestBase::DisplayTestInfo(); 
}

void CuidTest::ValidateGpuCuidTest() {
  // Find all GPU agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  // Test CUIDs for all GPUs found
  for (auto gpu : gpus) {
    uint8_t gpu_cuid[16];
    ASSERT_SUCCESS(hsa_agent_get_info(gpu, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CUID, &gpu_cuid));

    uint8_t zero_cuid[16] = {0};
    if (!memcmp(gpu_cuid, zero_cuid, sizeof(gpu_cuid))) {
      FAIL() << "GPU CUID should not be all zeros";
    }
    // Print CUID for debugging
    printf(
        "[Debug] GPU CUID: %02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
        gpu_cuid[0], gpu_cuid[1], gpu_cuid[2], gpu_cuid[3], gpu_cuid[4], gpu_cuid[5],
        gpu_cuid[6], gpu_cuid[7], gpu_cuid[8], gpu_cuid[9], gpu_cuid[10], gpu_cuid[11],
        gpu_cuid[12], gpu_cuid[13], gpu_cuid[14], gpu_cuid[15]);
  }
}