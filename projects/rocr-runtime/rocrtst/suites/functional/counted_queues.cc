/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/counted_queues.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa.h"
#include "common/base_rocr_utils.h"
#include "gtest/gtest.h"


CountedQueuesTest::CountedQueuesTest() : TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.

  set_title("RocR Counted Queues Test");
  set_description(
      "This test validates the behavior of Shared Counted Queues managed by the "
      "Counted Queue Manager in a scenario where both HIP and OpenMP utilize CP "
      "Queues, causing oversubscription and a subsequent performance degradation.");
}

CountedQueuesTest::~CountedQueuesTest() {}

void CountedQueuesTest::SetUp() { TestBase::SetUp(); }

void CountedQueuesTest::Run() {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void CountedQueuesTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void CountedQueuesTest::DisplayResults() const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::DisplayResults();
}

void CountedQueuesTest::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

void CountedQueuesTest::CountedQueueBasicApiTest() {
  hsa_status_t status;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  hsa_queue_t* queue = nullptr;
  status = hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL, nullptr, nullptr, 0, &queue);

  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  // Query counted queue and check internal reference count
  uint32_t use_count = 0;
  status = hsa_amd_counted_queue_get_info(queue, HSA_QUEUE_INFO_USE_COUNT, &use_count);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(use_count, 1);  // should be 4 after acquire

  // Release the queue
  status = hsa_amd_counted_queue_release(queue);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);
}

void CountedQueuesTest::CountedQueues_SamePriority_MaxLimitTest() {
  hsa_status_t status;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // export GPU_MAX_HW_QUEUES=2
  hsa_queue_t *q1 = nullptr, *q2 = nullptr, *q3 = nullptr, *q4 = nullptr;
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &q1),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &q2),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &q3),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &q4),
            HSA_STATUS_SUCCESS);

  // Get HW queue ids of all queues
  uint32_t hwid1 = 0, hwid2 = 0, hwid3 = 0, hwid4 = 0;
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q1, HSA_QUEUE_INFO_HW_ID, &hwid1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q3, HSA_QUEUE_INFO_HW_ID, &hwid3), HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_amd_counted_queue_get_info(q2, HSA_QUEUE_INFO_HW_ID, &hwid2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q4, HSA_QUEUE_INFO_HW_ID, &hwid4), HSA_STATUS_SUCCESS);

  // Third queue should reuse first HW queue
  EXPECT_EQ(hwid1, hwid3);
  EXPECT_NE(q1, q3);  // unique handles for the same HW queue

  // Fourth queue should get 2nd HW queue (least used at this point)
  EXPECT_EQ(hwid2, hwid4);
  EXPECT_NE(q2, q4);

  // Check how many times the first and second queues have been shared
  uint32_t use_count1 = 0, use_count2 = 0;
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q1, HSA_QUEUE_INFO_USE_COUNT, &use_count1),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q2, HSA_QUEUE_INFO_USE_COUNT, &use_count2),
            HSA_STATUS_SUCCESS);

  EXPECT_EQ(use_count1, 2);
  EXPECT_EQ(use_count2, 2);

  // Release the third and fourth queues
  EXPECT_EQ(hsa_amd_counted_queue_release(q3), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_release(q4), HSA_STATUS_SUCCESS);

  // Check the use counts and hwids of remaining queues; should be two different HW queues with ref
  // counts of 1 for each
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q1, HSA_QUEUE_INFO_USE_COUNT, &use_count1),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q2, HSA_QUEUE_INFO_USE_COUNT, &use_count2),
            HSA_STATUS_SUCCESS);

  EXPECT_EQ(use_count1, 1);
  EXPECT_EQ(use_count2, 1);

  uint32_t id1 = 0, id2 = 0;
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q1, HSA_QUEUE_INFO_HW_ID, &id1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q2, HSA_QUEUE_INFO_HW_ID, &id2), HSA_STATUS_SUCCESS);
  EXPECT_NE(id1, id2);  // should be two different hw ids now

  // Release the two queues
  EXPECT_EQ(hsa_amd_counted_queue_release(q1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_release(q2), HSA_STATUS_SUCCESS);

  int32_t c1 = UINT32_MAX, c2 = UINT32_MAX, c3 = UINT32_MAX, c4 = UINT32_MAX;
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q1, HSA_QUEUE_INFO_USE_COUNT, &c1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q2, HSA_QUEUE_INFO_USE_COUNT, &c2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q3, HSA_QUEUE_INFO_USE_COUNT, &c3), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(q4, HSA_QUEUE_INFO_USE_COUNT, &c4), HSA_STATUS_SUCCESS);

  EXPECT_EQ(c1, -1);
  EXPECT_EQ(c2, -1);
  EXPECT_EQ(c3, -1);
  EXPECT_EQ(c4, -1);

  // Acquire another queue again and see if we get the existing HW queue or a new one with id > 2
  hsa_queue_t* new_queue = nullptr;
  uint32_t new_hw_id = 0, refCount = 0;
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &new_queue),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(new_queue, HSA_QUEUE_INFO_HW_ID, &new_hw_id),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(new_queue, HSA_QUEUE_INFO_USE_COUNT, &refCount),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(new_hw_id, 1);
  EXPECT_EQ(refCount, 1);

  // Release this queue
  EXPECT_EQ(hsa_amd_counted_queue_release(new_queue), HSA_STATUS_SUCCESS);
}

void CountedQueuesTest::InvalidArgsTest() {
  hsa_status_t status;
  hsa_queue_t* q = nullptr;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Invalid queue pointer
  status = hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Invalid priority
  const hsa_amd_queue_priority_t invalid_priority = static_cast<hsa_amd_queue_priority_t>(999); 
  status = hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, invalid_priority,
                                          nullptr, nullptr, 0, &q);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Support multi producer queues only
  status = hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_SINGLE, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &q);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_QUEUE_CREATION);

  // Check release API params
  hsa_queue_t* queue = nullptr;
  status = hsa_amd_counted_queue_release(queue);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Invalid queue handle
  hsa_queue_t inv_queue = {};
  status = hsa_amd_counted_queue_release(&inv_queue);
  EXPECT_EQ(status, HSA_STATUS_ERROR);
}

void CountedQueuesTest::CountedQueuesAllPrioritiesLimitTest() {
  hsa_status_t status;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Set GPU_MAX_HW_QUEUES=2 in shell before running this test
  // Each priority level should have its own limit (LOW, NORMAL, HIGH)

  // Acquire 2 queues per priority (total 6 queues)
  hsa_queue_t *low1 = nullptr, *low2 = nullptr, *low3 = nullptr;
  hsa_queue_t *normal1 = nullptr, *normal2 = nullptr, *normal3 = nullptr;
  hsa_queue_t *high1 = nullptr, *high2 = nullptr, *high3 = nullptr;

  // Low Priority 
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &low1),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &low2),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &low3),
            HSA_STATUS_SUCCESS);  // should reuse low1

  // Normal Priority
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL,
                                          nullptr, nullptr, 0, &normal1),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL,
                                          nullptr, nullptr, 0, &normal2),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL,
                                          nullptr, nullptr, 0, &normal3),
            HSA_STATUS_SUCCESS);  // should reuse normal1

  // High Priority
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_HIGH,
                                          nullptr, nullptr, 0, &high1),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_HIGH,
                                          nullptr, nullptr, 0, &high2),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_HIGH,
                                          nullptr, nullptr, 0, &high3),
            HSA_STATUS_SUCCESS);

  // Verify reuse and independence per priority
  uint32_t low_id1 = 0, low_id2 = 0, low_id3 = 0;
  uint32_t norm_id1 = 0, norm_id2 = 0, norm_id3 = 0;
  uint32_t high_id1 = 0, high_id2 = 0, high_id3 = 0;

  EXPECT_EQ(hsa_amd_counted_queue_get_info(low1, HSA_QUEUE_INFO_HW_ID, &low_id1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(low2, HSA_QUEUE_INFO_HW_ID, &low_id2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(low3, HSA_QUEUE_INFO_HW_ID, &low_id3), HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_amd_counted_queue_get_info(normal1, HSA_QUEUE_INFO_HW_ID, &norm_id1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(normal2, HSA_QUEUE_INFO_HW_ID, &norm_id2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(normal3, HSA_QUEUE_INFO_HW_ID, &norm_id3), HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_amd_counted_queue_get_info(high1, HSA_QUEUE_INFO_HW_ID, &high_id1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(high2, HSA_QUEUE_INFO_HW_ID, &high_id2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(high3, HSA_QUEUE_INFO_HW_ID, &high_id3), HSA_STATUS_SUCCESS);

  // Within same priority: max 2 unique HW queues
  EXPECT_NE(low_id1, low_id2);
  EXPECT_TRUE(low_id3 == low_id1);

  EXPECT_NE(norm_id1, norm_id2);
  EXPECT_TRUE(norm_id3 == norm_id1);

  EXPECT_NE(high_id1, high_id2);
  EXPECT_TRUE(high_id3 == high_id1);

  // Ensure different queues are used across priorities
  EXPECT_NE(low_id1, norm_id1);
  EXPECT_NE(norm_id1, high_id1);
  EXPECT_NE(low_id1, high_id1);

  // Verify use counts of first two HW queues
  uint32_t low_use1 = 0, low_use2 = 0, low_use3;
  uint32_t norm_use1 = 0, norm_use2 = 0, norm_use3 = 0;
  uint32_t high_use1 = 0, high_use2 = 0, high_use3 = 0;

  EXPECT_EQ(hsa_amd_counted_queue_get_info(low1, HSA_QUEUE_INFO_USE_COUNT, &low_use1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(low2, HSA_QUEUE_INFO_USE_COUNT, &low_use2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(low3, HSA_QUEUE_INFO_USE_COUNT, &low_use3), HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_amd_counted_queue_get_info(normal1, HSA_QUEUE_INFO_USE_COUNT, &norm_use1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(normal2, HSA_QUEUE_INFO_USE_COUNT, &norm_use2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(normal3, HSA_QUEUE_INFO_USE_COUNT, &norm_use3), HSA_STATUS_SUCCESS);
  
  EXPECT_EQ(hsa_amd_counted_queue_get_info(high1, HSA_QUEUE_INFO_USE_COUNT, &high_use1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(high2, HSA_QUEUE_INFO_USE_COUNT, &high_use2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_get_info(high3, HSA_QUEUE_INFO_USE_COUNT, &high_use3), HSA_STATUS_SUCCESS);

  EXPECT_EQ(low_use1, 2);
  EXPECT_EQ(low_use2, 1);
  EXPECT_TRUE(low_use1 == low_use3); // same HW queues, same ref count

  EXPECT_EQ(norm_use1, 2);
  EXPECT_EQ(norm_use2, 1);
  EXPECT_TRUE(norm_use1 == norm_use3); 

  EXPECT_EQ(high_use1, 2);
  EXPECT_EQ(high_use2, 1);
  EXPECT_TRUE(high_use1 == high_use3); 

  // Release all queues
  EXPECT_EQ(hsa_amd_counted_queue_release(low1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_release(low2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_release(low3), HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_amd_counted_queue_release(normal1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_release(normal2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_release(normal3), HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_amd_counted_queue_release(high1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_release(high2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_counted_queue_release(high3), HSA_STATUS_SUCCESS);
}

void CountedQueuesTest::CountedQueuesSetPriorityNackTest() {
  hsa_status_t status;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Create a counted queue
  hsa_queue_t* queue = nullptr;
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &queue),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(queue, nullptr);

  // Try to set priority on this queue; should fail
  status = hsa_amd_queue_set_priority(queue, HSA_AMD_QUEUE_PRIORITY_HIGH);
  EXPECT_TRUE(status != HSA_STATUS_SUCCESS);

  // release queue
  EXPECT_EQ(hsa_amd_counted_queue_release(queue), HSA_STATUS_SUCCESS);
}

void CountedQueuesTest::CountedQueuesSetCUMaskNackTest() {
  hsa_status_t status;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Create a counted queue
  hsa_queue_t* queue = nullptr;
  EXPECT_EQ(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &queue),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(queue, nullptr);

  // Attempt to set CU mask on counted queue; should fail
  uint32_t cu_mask[32] = {0};  // dummy mask
  status = hsa_amd_queue_cu_set_mask(queue, 1, cu_mask);
  EXPECT_TRUE(status != HSA_STATUS_SUCCESS);

  // release queue
  EXPECT_EQ(hsa_amd_counted_queue_release(queue), HSA_STATUS_SUCCESS);
}