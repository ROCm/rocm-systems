/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/queue_modify.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

QueueModifyTest::QueueModifyTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("Queue Modify Test");
  set_description(
      "Exercises live queue reconfiguration (hsa_amd_queue_cu_set_mask / "
      "cu_get_mask and hsa_amd_queue_set_priority). In DRM user-queue mode "
      "these drive DrmDriver::ModifyQueue -> amdgpu_modify_userqueue "
      "(AMDGPU_USERQ_OP_MODIFY); the calls succeeding proves the modify ioctl "
      "works, and the CU-mask read-back confirms it took effect.");
}

QueueModifyTest::~QueueModifyTest(void) {
}

void QueueModifyTest::SetUp(void) {
  // A global CU mask (HSA_CU_MASK) would perturb the set/get round-trip; clear
  // it so the test controls the mask exactly.
  unsetenv("HSA_CU_MASK");
  unsetenv("HSA_CU_MASK_SKIP_INIT");

  TestBase::SetUp();  // checkPlatformFiltering() + hsa_init()
  if (isTestSkipped()) return;

  hsa_agent_t gpu_agent = {0};
  hsa_status_t err = hsa_iterate_agents(rocrtst::FindGPUDevice, &gpu_agent);
  if (err != HSA_STATUS_INFO_BREAK || gpu_agent.handle == 0) {
    fprintf(stdout, "[ SKIPPED ] No GPU agent visible\n");
    test_skipped_ = true;
    return;
  }
  set_gpu_device1(gpu_agent);
}

void QueueModifyTest::Run(void) {
  if (test_skipped_) return;
  TestBase::Run();
}

void QueueModifyTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void QueueModifyTest::DisplayResults(void) const {
  TestBase::DisplayResults();
}

void QueueModifyTest::Close(void) {
  TestBase::Close();
}

void QueueModifyTest::TestQueueModify(void) {
  if (test_skipped_) return;

  hsa_agent_t gpu_agent = *gpu_device1();
  ASSERT_NE(gpu_agent.handle, 0u);

  uint32_t cu_count = 0;
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_agent_get_info(
                gpu_agent,
                static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
                &cu_count));
  ASSERT_GT(cu_count, 0u);

  uint32_t queue_max = 0;
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max));
  ASSERT_GT(queue_max, 0u);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_queue_create(gpu_agent, queue_max, HSA_QUEUE_TYPE_MULTI, nullptr,
                             nullptr, 0, 0, &queue));
  ASSERT_NE(queue, nullptr);

  const uint32_t mask_bits = ((cu_count + 31u) / 32u) * 32u;  // multiple of 32
  const uint32_t words = mask_bits / 32u;

  // Build a WGP-valid subset mask: enable the first `enabled` CUs, with
  // `enabled` even so every enabled CU is part of a complete CU pair (GPUs
  // with WGPs require pairwise CU enablement).
  uint32_t enabled = (cu_count / 2u) & ~1u;
  if (enabled < 2u) enabled = (cu_count >= 2u) ? 2u : cu_count;

  std::vector<uint32_t> set_mask(words, 0);
  std::vector<uint32_t> got_mask(words, 0);
  for (uint32_t i = 0; i < enabled; ++i) set_mask[i / 32u] |= (1u << (i % 32u));

  // CU mask SET: in DRM mode this drives ModifyQueue -> amdgpu_modify_userqueue.
  // A failure here means the queue-modification ioctl path is broken.
  hsa_status_t st = hsa_amd_queue_cu_set_mask(queue, mask_bits, set_mask.data());
  ASSERT_EQ(HSA_STATUS_SUCCESS, st)
      << "hsa_amd_queue_cu_set_mask failed (queue modify path); status=" << st;

  // CU mask GET: confirm the read-back reports our requested CUs as enabled.
  // Read-back semantics differ by mode: DRM (amdgpu_modify_userqueue) reflects
  // the exact per-queue mask (get == set), whereas KFD's get_mask returns the
  // full physical CU mask. So require every requested CU to be enabled in the
  // read-back (a subset check that holds in both modes) rather than an exact
  // match. Deeper "disabled CUs are truly unused" verification is the domain of
  // the kernel-dispatch CU_Masking test.
  st = hsa_amd_queue_cu_get_mask(queue, mask_bits, got_mask.data());
  ASSERT_EQ(HSA_STATUS_SUCCESS, st) << "hsa_amd_queue_cu_get_mask failed; status=" << st;
  for (uint32_t i = 0; i < cu_count; ++i) {
    uint32_t sb = (set_mask[i / 32u] >> (i % 32u)) & 1u;
    if (sb) {
      uint32_t gb = (got_mask[i / 32u] >> (i % 32u)) & 1u;
      ASSERT_EQ(1u, gb) << "requested CU " << i << " not enabled in read-back mask";
    }
  }

  // Priority: in DRM mode this drives ModifyQueue with an updated priority.
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_queue_set_priority(queue, HSA_AMD_QUEUE_PRIORITY_HIGH));
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_queue_set_priority(queue, HSA_AMD_QUEUE_PRIORITY_LOW));
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_queue_set_priority(queue, HSA_AMD_QUEUE_PRIORITY_NORMAL));

  // Restore a full CU mask (all CUs enabled) - also a modify.
  std::vector<uint32_t> all_mask(words, 0);
  for (uint32_t i = 0; i < cu_count; ++i) all_mask[i / 32u] |= (1u << (i % 32u));
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_queue_cu_set_mask(queue, mask_bits, all_mask.data()));

  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_queue_destroy(queue));
}
