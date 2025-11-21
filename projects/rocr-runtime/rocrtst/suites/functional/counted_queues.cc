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
#include "common/os.h"
#include <thread>
#include <mutex>
#include <atomic>

static bool VerifyResult(uint32_t* ar, size_t sz) {
  for (size_t i = 0; i < sz; ++i) {
    if (i * i != ar[i]) {
      return false;
    }
  }
  return true;
}

CountedQueuesTest::CountedQueuesTest() : TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.

  set_title("RocR Counted Queues Test");
  set_description(
      "This test validates the behavior of Shared Counted Queues managed by the "
      "Counted Queue Manager in a scenario where different libraries use CP "
      "Queues and it avoids oversubscription and a subsequent performance degradation.");
}

CountedQueuesTest::~CountedQueuesTest() {}

void CountedQueuesTest::SetUp() {
  // Set environment variable to limit max number of HW queues based on test case
  const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  if (test_info) {
    std::string test_name = test_info->name();
    if (test_name == "Counted_Queue_Multithreaded_Dispatch_Test") {
      // This test will try to create 1 CP queue with multiple threads
      // All of the user apps will share the same queue
      rocrtst::SetEnv("GPU_MAX_HW_QUEUES", "1");
    } else {
      rocrtst::SetEnv("GPU_MAX_HW_QUEUES", "2");
    }
  }
  TestBase::SetUp();
}

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
  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  hsa_queue_t* queue = nullptr;
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL, nullptr, nullptr, 0, &queue));
  ASSERT_NE(queue, nullptr);

  // Query counted queue and check internal reference count
  int32_t use_count = 0;
  ASSERT_SUCCESS(hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_USE_COUNT, &use_count));
  EXPECT_EQ(use_count, 1);  // should be 1 after acquire

  // Release the queue
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(queue));

  // Check that ref count is back to 0 after release
  hsa_status_t status;
  status = hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_USE_COUNT, &use_count);
  ASSERT_EQ(status, HSA_STATUS_ERROR_INVALID_ARGUMENT);
}

void CountedQueuesTest::CountedQueues_SamePriority_MaxLimitTest() {
  hsa_status_t status;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  hsa_queue_t *q1 = nullptr, *q2 = nullptr, *q3 = nullptr, *q4 = nullptr;
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &q1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &q2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &q3));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &q4));

  // Get HW queue ids of all queues
  uint32_t hwid1 = 0, hwid2 = 0, hwid3 = 0, hwid4 = 0;
  ASSERT_SUCCESS(hsa_amd_queue_get_info(q1, HSA_QUEUE_INFO_HW_ID, &hwid1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(q3, HSA_QUEUE_INFO_HW_ID, &hwid3));

  ASSERT_SUCCESS(hsa_amd_queue_get_info(q2, HSA_QUEUE_INFO_HW_ID, &hwid2));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(q4, HSA_QUEUE_INFO_HW_ID, &hwid4));

  // Third queue should reuse first HW queue
  EXPECT_EQ(hwid1, hwid3);
  EXPECT_NE(q1, q3);  // unique handles for the same HW queue

  // Fourth queue should get 2nd HW queue (least used at this point)
  EXPECT_EQ(hwid2, hwid4);
  EXPECT_NE(q2, q4);

  // Check how many times the first and second queues have been shared
  uint32_t use_count1 = 0, use_count2 = 0;
  ASSERT_SUCCESS(hsa_amd_queue_get_info(q1, HSA_QUEUE_INFO_USE_COUNT, &use_count1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(q2, HSA_QUEUE_INFO_USE_COUNT, &use_count2));

  EXPECT_EQ(use_count1, 2);
  EXPECT_EQ(use_count2, 2);

  // Release the third and fourth queues
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(q3));
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(q4));

  // Check the use counts and hwids of remaining queues; should be two different HW queues with ref
  // counts of 1 for each
  ASSERT_SUCCESS(hsa_amd_queue_get_info(q1, HSA_QUEUE_INFO_USE_COUNT, &use_count1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(q2, HSA_QUEUE_INFO_USE_COUNT, &use_count2));

  EXPECT_EQ(use_count1, 1);
  EXPECT_EQ(use_count2, 1);

  uint32_t id1 = 0, id2 = 0;
  ASSERT_SUCCESS(hsa_amd_queue_get_info(q1, HSA_QUEUE_INFO_HW_ID, &id1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(q2, HSA_QUEUE_INFO_HW_ID, &id2));
  EXPECT_NE(id1, id2);  // should be two different hw ids now

  // Release the two queues
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(q1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(q2));

  // Now all queues have been released; getting use count should return invalid arg error
  int32_t c1 = UINT32_MAX, c2 = UINT32_MAX, c3 = UINT32_MAX, c4 = UINT32_MAX;
  EXPECT_EQ(hsa_amd_queue_get_info(q1, HSA_QUEUE_INFO_USE_COUNT, &c1),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(hsa_amd_queue_get_info(q2, HSA_QUEUE_INFO_USE_COUNT, &c2),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(hsa_amd_queue_get_info(q3, HSA_QUEUE_INFO_USE_COUNT, &c3),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(hsa_amd_queue_get_info(q4, HSA_QUEUE_INFO_USE_COUNT, &c4),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Acquire another queue again and see if we get the existing HW queue or a new one with id > 2
  hsa_queue_t* new_queue = nullptr;
  uint32_t new_hw_id = 0, refCount = 0;
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &new_queue));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(new_queue, HSA_QUEUE_INFO_HW_ID, &new_hw_id));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(new_queue, HSA_QUEUE_INFO_USE_COUNT, &refCount));

  EXPECT_EQ(new_hw_id, 1);
  EXPECT_EQ(refCount, 1);

  // Release this queue
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(new_queue));
}

void CountedQueuesTest::InvalidArgsTest() {
  hsa_status_t status;
  hsa_queue_t* q = nullptr;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  // Invalid queue pointer
  status = hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                         nullptr, nullptr, 0, nullptr);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Invalid priority
  const hsa_amd_queue_priority_t invalid_priority = static_cast<hsa_amd_queue_priority_t>(999);
  status = hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, invalid_priority, nullptr,
                                         nullptr, 0, &q);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Support multi producer queues only
  status = hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_SINGLE, HSA_AMD_QUEUE_PRIORITY_LOW,
                                         nullptr, nullptr, 0, &q);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_QUEUE_CREATION);

  // Check release API params
  hsa_queue_t* queue = nullptr;
  status = hsa_amd_counted_queue_release(queue);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_ARGUMENT);
}

void CountedQueuesTest::CountedQueuesAllPrioritiesLimitTest() {
  hsa_status_t status;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  // Acquire 2 queues per priority (total 6 queues)
  hsa_queue_t *low1 = nullptr, *low2 = nullptr, *low3 = nullptr;
  hsa_queue_t *normal1 = nullptr, *normal2 = nullptr, *normal3 = nullptr;
  hsa_queue_t *high1 = nullptr, *high2 = nullptr, *high3 = nullptr;

  // Low Priority
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &low1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &low2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &low3));  // should reuse low1

  // Normal Priority
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL,
                                    nullptr, nullptr, 0, &normal1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL,
                                    nullptr, nullptr, 0, &normal2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL,
                                    nullptr, nullptr, 0, &normal3));  // should reuse normal1

  // High Priority
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI,
                                          HSA_AMD_QUEUE_PRIORITY_HIGH, nullptr, nullptr, 0, &high1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI,
                                          HSA_AMD_QUEUE_PRIORITY_HIGH, nullptr, nullptr, 0, &high2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI,
                                          HSA_AMD_QUEUE_PRIORITY_HIGH, nullptr, nullptr, 0, &high3));

  // Verify reuse and independence per priority
  uint32_t low_id1 = 0, low_id2 = 0, low_id3 = 0;
  uint32_t norm_id1 = 0, norm_id2 = 0, norm_id3 = 0;
  uint32_t high_id1 = 0, high_id2 = 0, high_id3 = 0;

  ASSERT_SUCCESS(hsa_amd_queue_get_info(low1, HSA_QUEUE_INFO_HW_ID, &low_id1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(low2, HSA_QUEUE_INFO_HW_ID, &low_id2));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(low3, HSA_QUEUE_INFO_HW_ID, &low_id3));

  ASSERT_SUCCESS(hsa_amd_queue_get_info(normal1, HSA_QUEUE_INFO_HW_ID, &norm_id1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(normal2, HSA_QUEUE_INFO_HW_ID, &norm_id2));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(normal3, HSA_QUEUE_INFO_HW_ID, &norm_id3));

  ASSERT_SUCCESS(hsa_amd_queue_get_info(high1, HSA_QUEUE_INFO_HW_ID, &high_id1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(high2, HSA_QUEUE_INFO_HW_ID, &high_id2));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(high3, HSA_QUEUE_INFO_HW_ID, &high_id3));

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

  ASSERT_SUCCESS(hsa_amd_queue_get_info(low1, HSA_QUEUE_INFO_USE_COUNT, &low_use1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(low2, HSA_QUEUE_INFO_USE_COUNT, &low_use2));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(low3, HSA_QUEUE_INFO_USE_COUNT, &low_use3));

  ASSERT_SUCCESS(hsa_amd_queue_get_info(normal1, HSA_QUEUE_INFO_USE_COUNT, &norm_use1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(normal2, HSA_QUEUE_INFO_USE_COUNT, &norm_use2));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(normal3, HSA_QUEUE_INFO_USE_COUNT, &norm_use3));

  ASSERT_SUCCESS(hsa_amd_queue_get_info(high1, HSA_QUEUE_INFO_USE_COUNT, &high_use1));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(high2, HSA_QUEUE_INFO_USE_COUNT, &high_use2));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(high3, HSA_QUEUE_INFO_USE_COUNT, &high_use3));

  EXPECT_EQ(low_use1, 2);
  EXPECT_EQ(low_use2, 1);
  EXPECT_TRUE(low_use1 == low_use3);  // same HW queues, same ref count

  EXPECT_EQ(norm_use1, 2);
  EXPECT_EQ(norm_use2, 1);
  EXPECT_TRUE(norm_use1 == norm_use3);

  EXPECT_EQ(high_use1, 2);
  EXPECT_EQ(high_use2, 1);
  EXPECT_TRUE(high_use1 == high_use3);

  // Release all queues
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(low1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(low2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(low3));

  ASSERT_SUCCESS(hsa_amd_counted_queue_release(normal1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(normal2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(normal3));

  ASSERT_SUCCESS(hsa_amd_counted_queue_release(high1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(high2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(high3));
}

void CountedQueuesTest::CountedQueuesSetPriorityNackTest() {
  hsa_status_t status;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  // Create a counted queue
  hsa_queue_t* queue = nullptr;
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &queue));
  EXPECT_NE(queue, nullptr);

  // Try to set priority on this queue; should fail
  status = hsa_amd_queue_set_priority(queue, HSA_AMD_QUEUE_PRIORITY_HIGH);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_QUEUE);

  // release queue
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(queue));
}

void CountedQueuesTest::CountedQueuesSetCUMaskNackTest() {
  hsa_status_t status;

  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  // Create a counted queue
  hsa_queue_t* queue = nullptr;
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &queue));
  EXPECT_NE(queue, nullptr);

  // Attempt to set CU mask on counted queue; should fail
  uint32_t cu_mask[32] = {0};  // dummy mask
  status = hsa_amd_queue_cu_set_mask(queue, 1, cu_mask);
  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_QUEUE);

  // release queue
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(queue));
}

void CountedQueuesTest::CountedQueuesDispatchTest() {
  hsa_status_t status;

  // Set CPU and GPU agents
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));

  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  // get gpu device
  hsa_agent_t gpu_dev = *gpu_device1();

  // set kernel file name
  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");

  // Create a counted queue
  hsa_queue_t* queue = nullptr;
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpu_dev, HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW,
                                          nullptr, nullptr, 0, &queue));
  EXPECT_NE(queue, nullptr);
  set_main_queue(queue);

  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, &gpu_dev));
  ASSERT_SUCCESS(rocrtst::InitializeAQLPacket(this, &aql()));

  // Allocate buffers for source and destination and add values into the source buffer
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0,
                                        reinterpret_cast<void**>(&src_buffer_)));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, src_buffer_));

  for (uint32_t i = 0; i < 256; ++i) {
    reinterpret_cast<uint32_t*>(src_buffer_)[i] = i;
  }

  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0,
                                        reinterpret_cast<void**>(&dst_buffer_)));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, dst_buffer_));

  struct __attribute__((aligned(16))) local_args_t {
    uint32_t* dstArray;
    uint32_t* srcArray;
    uint32_t size;
    uint32_t pad;
    uint64_t global_offset_x;
    uint64_t global_offset_y;
    uint64_t global_offset_z;
    uint64_t printf_buffer;
    uint64_t default_queue;
    uint64_t completion_action;
  } local_args;

  local_args.dstArray = reinterpret_cast<uint32_t*>(dst_buffer_);
  local_args.srcArray = reinterpret_cast<uint32_t*>(src_buffer_);
  local_args.size = 256;
  local_args.global_offset_x = 0;
  local_args.global_offset_y = 0;
  local_args.global_offset_z = 0;
  local_args.printf_buffer = 0;
  local_args.default_queue = 0;
  local_args.completion_action = 0;

  ASSERT_SUCCESS(rocrtst::AllocAndSetKernArgs(this, &local_args, sizeof(local_args)));

  aql().workgroup_size_x = 256;
  aql().grid_size_x = 256;

  int it = num_iteration() * 5;

  hsa_kernel_dispatch_packet_t* queue_aql_packet;
  uint64_t index;

  for (int i = 0; i < it; i++) {
    queue_aql_packet = WriteAQLToQueue(this, &index);
    ASSERT_EQ(queue_aql_packet, reinterpret_cast<hsa_kernel_dispatch_packet_t*>(main_queue()->base_address) + index);

    // Prepare the AQL packet header
    uint32_t aql_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    __atomic_store_n(reinterpret_cast<uint32_t*>(queue_aql_packet),
                     aql_header | (aql().setup << 16), __ATOMIC_RELEASE);

    // Store the new index into the doorbell signal indicating new work
    hsa_signal_store_screlease(main_queue()->doorbell_signal, index);

    // Wait till kernel finished
    while (hsa_signal_wait_scacquire(aql().completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                     (uint64_t)-1, HSA_WAIT_STATE_ACTIVE)) {
    }

    // reset completion signal to 1 before next iteration
    hsa_signal_store_screlease(aql().completion_signal, 1);

    // verify that dest buffer has squared values of every value from source buffer
    ASSERT_TRUE(VerifyResult(reinterpret_cast<uint32_t*>(dst_buffer_), 256));
  }

  // Get the use count; should be 1
  int32_t count = 0;
  ASSERT_SUCCESS(hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_USE_COUNT, &count));
  EXPECT_EQ(count, 1);

  // Release the counted queue
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(queue));

  // Get use count info after release; should return invalid arg error since queue has been released
  status = hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_USE_COUNT, &count);
  ASSERT_EQ(status, HSA_STATUS_ERROR_INVALID_ARGUMENT);
}

void CountedQueuesTest::CountedQueuesMultithreadedDispatchTest() {
  hsa_status_t status;

  // Common setup
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  // Load kernel
  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, gpu_device1()));

  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  // Shared source buffer (read-only)
  void* shared_src_buffer = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &shared_src_buffer));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, shared_src_buffer));

  // Initialize source data
  for (uint32_t i = 0; i < 256; ++i) {
    reinterpret_cast<uint32_t*>(shared_src_buffer)[i] = i;
  }

  // Structures for validation later on
  std::mutex hwIdsMutex;
  std::vector<uint32_t> allHwIds;
  std::atomic<int32_t> maxUseCount{0};

  auto func = [&](int thread_id) {
    // local dest buffer for each user application
    void* local_dst_buffer = nullptr;
    ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &local_dst_buffer));
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, local_dst_buffer));

    // Local completion signal for every user application
    hsa_signal_t local_signal;
    ASSERT_SUCCESS(hsa_signal_create(1, 0, nullptr, &local_signal));

    // Get a counted queue
    hsa_queue_t* queue = nullptr;
    ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(*gpu_device1(), HSA_QUEUE_TYPE_MULTI,
                                           HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0, &queue));
    EXPECT_NE(queue, nullptr);

    if (queue == nullptr) {
      hsa_signal_destroy(local_signal);
      hsa_amd_memory_pool_free(local_dst_buffer);
      return;
    }

    // Store query results for later analysis
    int32_t localUseCount = 0;
    uint32_t localHwId = 0;

    ASSERT_SUCCESS(hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_USE_COUNT, &localUseCount));
    ASSERT_SUCCESS(hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_HW_ID, &localHwId));

    // Update use_count if it is larger than previous value 
    int expected = maxUseCount.load();
    while (localUseCount > expected &&
           !maxUseCount.compare_exchange_weak(expected, localUseCount)) {
    }

    // Store hw id for validation later on
    {
      std::lock_guard<std::mutex> lock(hwIdsMutex);
      allHwIds.push_back(localHwId);
    }

    struct __attribute__((aligned(16))) local_args_t {
      uint32_t* dstArray;
      uint32_t* srcArray;
      uint32_t size;
      uint32_t pad;
      uint64_t global_offset_x;
      uint64_t global_offset_y;
      uint64_t global_offset_z;
      uint64_t printf_buffer;
      uint64_t default_queue;
      uint64_t completion_action;
    } local_args;

    local_args.dstArray = reinterpret_cast<uint32_t*>(local_dst_buffer);
    local_args.srcArray = reinterpret_cast<uint32_t*>(shared_src_buffer);
    local_args.size = 256;
    local_args.global_offset_x = 0;
    local_args.global_offset_y = 0;
    local_args.global_offset_z = 0;
    local_args.printf_buffer = 0;
    local_args.default_queue = 0;
    local_args.completion_action = 0;

    void* kernarg_address = nullptr;
    ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(kern_arg_pool(), sizeof(local_args), 0, &kernarg_address));
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, kernarg_address));

    memcpy(kernarg_address, &local_args, sizeof(local_args));

    // Dispatch loop
    int it = num_iteration() * 5;
    const uint32_t queue_mask = queue->size - 1;

    for (int i = 0; i < it; i++) {
      // Reserve a slot in the queue
      uint64_t index = hsa_queue_add_write_index_relaxed(queue, 1);

      // Get pointer to the reserved packet slot and validate address
      hsa_kernel_dispatch_packet_t* queue_aql_packet = &(
          reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address))[index & queue_mask]; 
      ASSERT_EQ(queue_aql_packet, reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address) + index);

      // Fill packet fields
      queue_aql_packet->setup = 1;
      queue_aql_packet->workgroup_size_x = 256;
      queue_aql_packet->workgroup_size_y = 1;
      queue_aql_packet->workgroup_size_z = 1;
      queue_aql_packet->grid_size_x = 256;
      queue_aql_packet->grid_size_y = 1;
      queue_aql_packet->grid_size_z = 1;
      queue_aql_packet->private_segment_size = 0;
      queue_aql_packet->group_segment_size = 0;
      queue_aql_packet->kernel_object = kernel_object();
      queue_aql_packet->kernarg_address = kernarg_address;
      queue_aql_packet->completion_signal = local_signal;

      // Write header for packet
      uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
      header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
      header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
      __atomic_store_n(reinterpret_cast<uint16_t*>(&queue_aql_packet->header), header,
                       __ATOMIC_RELEASE);

      // Ring doorbell to notify GPU
      hsa_signal_store_screlease(queue->doorbell_signal, index);

      // Wait for completion signal to be less than 1
      while (hsa_signal_wait_scacquire(local_signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                       HSA_WAIT_STATE_ACTIVE)) {
      }

      // Reset signal for next iteration
      hsa_signal_store_screlease(local_signal, 1);

      ASSERT_TRUE(VerifyResult(reinterpret_cast<uint32_t*>(local_dst_buffer), 256));
    }

    // Cleanup
    hsa_amd_memory_pool_free(kernarg_address);
    hsa_signal_destroy(local_signal);
    hsa_amd_memory_pool_free(local_dst_buffer);

    // Release the counted queue
    ASSERT_SUCCESS(hsa_amd_counted_queue_release(queue));
  };

  constexpr int kThreads = 2;
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; i++) {
    threads.emplace_back(func, i);
  }

  for (auto& th : threads) {
    th.join();
  }

  // With GPU_MAX_HW_QUEUES=1, all threads should share the same HW queue
  // Check if largest useCount is same as the number of user apps accessing queues
  EXPECT_EQ(maxUseCount.load(), kThreads);

  // All HW IDs should be the same (only 1 HW queue created)
  EXPECT_EQ(allHwIds.size(), static_cast<size_t>(kThreads));
  for (size_t i = 1; i < allHwIds.size(); i++) {
    EXPECT_EQ(allHwIds[i], allHwIds[0]);
  }

  hsa_amd_memory_pool_free(shared_src_buffer);
}