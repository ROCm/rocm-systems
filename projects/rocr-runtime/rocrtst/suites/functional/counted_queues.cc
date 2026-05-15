/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <algorithm>

#include "suites/functional/counted_queues.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa.h"
#include "common/base_rocr_utils.h"
#include "gtest/gtest.h"
#include "common/os.h"


static bool VerifyResult(uint32_t* ar, size_t sz) {
  for (size_t i = 0; i < sz; ++i) {
    if (i * i != ar[i]) {
      return false;
    }
  }
  return true;
}

CountedQueuesTest::CountedQueuesTest() : TestBase() {
  set_title("RocR Counted Queues Test");
  set_description(
      "This test validates the behavior of Shared Counted Queues managed by the "
      "Counted Queue Manager in a scenario where different libraries use CP "
      "Queues and it avoids oversubscription and a subsequent performance degradation.");
}

CountedQueuesTest::~CountedQueuesTest() {}

void CountedQueuesTest::SetUp() {
  const std::string kDefaultLimit = "2";
  static const std::unordered_map<std::string, std::string> kQueueLimits = {
      {"Counted_Queue_Multithreaded_Dispatch_Test", "1"},
      {"Counted_Queue_Overflow_And_Wraparound_Test", "1"},
      {"Counted_Queue_Same_Priority_Max_Limit_Test", "4"}};

  const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  if (test_info) {
    const std::string test_name = test_info->name();

    // Find the current test's required limit from map and set the env var
    // Set default HW queue limit if not found in map 
    auto it = kQueueLimits.find(test_name);
    const std::string& limit = (it != kQueueLimits.end()) ? it->second : kDefaultLimit;
    rocrtst::SetEnv("GPU_MAX_HW_QUEUES", limit.c_str());
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

  // Find all GPU agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));
  ASSERT_FALSE(gpus.empty());

  const int NUM_QUEUES = 50;
  const int MAX_HW_QUEUES = std::stoi(rocrtst::GetEnv("GPU_MAX_HW_QUEUES"));

  std::vector<hsa_queue_t*> queues(NUM_QUEUES, nullptr);
  std::vector<uint32_t> hw_ids(NUM_QUEUES, 0);

  // Acquire NUM_QUEUES counted queues
  for (int i = 0; i < NUM_QUEUES; i++) {
    ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI,
                                                 HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0,
                                                 &queues[i]));
    ASSERT_NE(queues[i], nullptr);
  }

  // Query HW IDs
  for (int i = 0; i < NUM_QUEUES; i++) {
    ASSERT_SUCCESS(hsa_amd_queue_get_info(queues[i], HSA_QUEUE_INFO_HW_ID, &hw_ids[i]));
  }

  // Sort and remove duplicate HW IDs
  std::sort(hw_ids.begin(), hw_ids.end());
  auto it = std::unique(hw_ids.begin(), hw_ids.end());
  hw_ids.resize(std::distance(hw_ids.begin(), it));

  // Ensure hardware queue count matches MAX_HW_QUEUES
  ASSERT_EQ(hw_ids.size(), MAX_HW_QUEUES);

  // Verify even distribution of logical queues over HW queues
  // Map HW ID -> use count
  std::unordered_map<uint32_t, uint32_t> use_counts;

  for (auto* q : queues) {
    uint32_t hwid = 0, count = 0;
    ASSERT_SUCCESS(hsa_amd_queue_get_info(q, HSA_QUEUE_INFO_HW_ID, &hwid));
    ASSERT_SUCCESS(hsa_amd_queue_get_info(q, HSA_QUEUE_INFO_USE_COUNT, &count));
    use_counts[hwid] = count;  // overwrites but counts are per-hw, same across queues
  }

  // Gather all use-counts for fairness check
  std::vector<uint32_t> dist;
  for (auto& kv : use_counts) {
    dist.push_back(kv.second);
  }

  ASSERT_EQ(dist.size(), MAX_HW_QUEUES);

  // Fair distribution: difference should not exceed 1
  auto [min_it, max_it] = std::minmax_element(dist.begin(), dist.end());
  EXPECT_LE(*max_it - *min_it, 1);

  // Release queues
  for (auto* q : queues) {
    ASSERT_SUCCESS(hsa_amd_counted_queue_release(q));
  }

  // After release, querying use-count should return invalid argument
  for (auto* q : queues) {
    uint32_t tmp = 0;
    EXPECT_EQ(hsa_amd_queue_get_info(q, HSA_QUEUE_INFO_USE_COUNT, &tmp),
              HSA_STATUS_ERROR_INVALID_ARGUMENT);
  }
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
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0, &low1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0, &low2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI,
                                               HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0,
                                               &low3));  // should reuse low1

  // Normal Priority
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL, nullptr, nullptr, 0, &normal1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_NORMAL, nullptr, nullptr, 0, &normal2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(gpus[0], HSA_QUEUE_TYPE_MULTI,
                                               HSA_AMD_QUEUE_PRIORITY_NORMAL, nullptr, nullptr, 0,
                                               &normal3));  // should reuse normal1

  // High Priority
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_HIGH, nullptr, nullptr, 0, &high1));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_HIGH, nullptr, nullptr, 0, &high2));
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_HIGH, nullptr, nullptr, 0, &high3));

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
  uint32_t low_use1 = 0, low_use2 = 0, low_use3 = 0;
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
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0, &queue));
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
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(
      gpus[0], HSA_QUEUE_TYPE_MULTI, HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0, &queue));
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

  // Common setup
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  // Load kernel
  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, gpu_device1()));

  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  // Allocate source buffer
  void* src_buffer = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &src_buffer));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, src_buffer));

  // Initialize source data
  for (uint32_t i = 0; i < 256; ++i) {
    reinterpret_cast<uint32_t*>(src_buffer)[i] = i;
  }

  // Allocate destination buffer
  void* dst_buffer = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &dst_buffer));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, dst_buffer));

  // Create completion signal
  hsa_signal_t completion_signal;
  ASSERT_SUCCESS(hsa_signal_create(1, 0, nullptr, &completion_signal));

  // Get a counted queue
  hsa_queue_t* queue = nullptr;
  ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(*gpu_device1(), HSA_QUEUE_TYPE_MULTI,
                                               HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0,
                                               &queue));
  EXPECT_NE(queue, nullptr);

  // Query queue info
  int32_t use_count = 0;
  uint32_t hw_id = 0;
  ASSERT_SUCCESS(hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_USE_COUNT, &use_count));
  ASSERT_SUCCESS(hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_HW_ID, &hw_id));
  EXPECT_EQ(use_count, 1);

  // Prepare kernel arguments
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

  local_args.dstArray = reinterpret_cast<uint32_t*>(dst_buffer);
  local_args.srcArray = reinterpret_cast<uint32_t*>(src_buffer);
  local_args.size = 256;
  local_args.global_offset_x = 0;
  local_args.global_offset_y = 0;
  local_args.global_offset_z = 0;
  local_args.printf_buffer = 0;
  local_args.default_queue = 0;
  local_args.completion_action = 0;

  // Allocate kernel arguments
  void* kernarg_address = nullptr;
  ASSERT_SUCCESS(
      hsa_amd_memory_pool_allocate(kern_arg_pool(), sizeof(local_args), 0, &kernarg_address));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, kernarg_address));
  memcpy(kernarg_address, &local_args, sizeof(local_args));

  // Dispatch loop
  int it = num_iteration() * 5;
  const uint32_t queue_mask = queue->size - 1;

  for (int i = 0; i < it; i++) {
    // Reserve a slot in the queue
    uint64_t index = hsa_queue_add_write_index_relaxed(queue, 1);

    // Get pointer to the reserved packet slot
    hsa_kernel_dispatch_packet_t* queue_aql_packet =
        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address))[index & queue_mask];

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
    queue_aql_packet->completion_signal = completion_signal;

    // Write header for packet
    uint32_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    __atomic_store_n(reinterpret_cast<uint16_t*>(&queue_aql_packet->header), header,
                     __ATOMIC_RELEASE);

    // Ring doorbell to notify GPU
    hsa_signal_store_screlease(queue->doorbell_signal, index);

    // Wait for completion signal
    while (hsa_signal_wait_scacquire(completion_signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                     HSA_WAIT_STATE_ACTIVE)) {
    }

    // Reset signal for next iteration
    hsa_signal_store_screlease(completion_signal, 1);

    // Verify results
    ASSERT_TRUE(VerifyResult(reinterpret_cast<uint32_t*>(dst_buffer), 256));
  }

  // Verify use count before release
  ASSERT_SUCCESS(hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_USE_COUNT, &use_count));
  EXPECT_EQ(use_count, 1);

  // Release the counted queue
  ASSERT_SUCCESS(hsa_amd_counted_queue_release(queue));

  // Verify queue info returns error after release
  status = hsa_amd_queue_get_info(queue, HSA_QUEUE_INFO_USE_COUNT, &use_count);
  ASSERT_EQ(status, HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Cleanup
  ASSERT_SUCCESS(hsa_amd_memory_pool_free(kernarg_address));
  ASSERT_SUCCESS(hsa_signal_destroy(completion_signal));
  ASSERT_SUCCESS(hsa_amd_memory_pool_free(src_buffer));
  ASSERT_SUCCESS(hsa_amd_memory_pool_free(dst_buffer));
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
  ASSERT_SUCCESS(
      hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &shared_src_buffer));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, shared_src_buffer));

  // Initialize source data
  for (uint32_t i = 0; i < 256; ++i) {
    reinterpret_cast<uint32_t*>(shared_src_buffer)[i] = i;
  }

  // Structures for validation later on
  std::mutex hwIdsMutex;
  std::vector<uint32_t> allHwIds;
  std::atomic<int32_t> maxUseCount{0};

  auto func = [&]() {
    // local dest buffer for each user application
    void* local_dst_buffer = nullptr;
    ASSERT_SUCCESS(
        hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &local_dst_buffer));
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, local_dst_buffer));

    // Local completion signal for every user application
    hsa_signal_t local_signal;
    ASSERT_SUCCESS(hsa_signal_create(1, 0, nullptr, &local_signal));

    // Get a counted queue
    hsa_queue_t* queue = nullptr;
    ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(*gpu_device1(), HSA_QUEUE_TYPE_MULTI,
                                                 HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0,
                                                 &queue));
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
    ASSERT_SUCCESS(
        hsa_amd_memory_pool_allocate(kern_arg_pool(), sizeof(local_args), 0, &kernarg_address));
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
      ASSERT_EQ(queue_aql_packet,
                reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address) + (index & queue_mask));

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
      uint32_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
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
    threads.emplace_back(func);
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

void CountedQueuesTest::CountedQueuesOverflowWrapAroundTest() {
  hsa_status_t status;

  // Common setup
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  // Load kernel
  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, gpu_device1()));

  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  void* shared_src_buffer = nullptr;
  ASSERT_SUCCESS(
      hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &shared_src_buffer));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, shared_src_buffer));

  for (uint32_t i = 0; i < 256; ++i) {
    reinterpret_cast<uint32_t*>(shared_src_buffer)[i] = i;
  }

  // To verify that after the queue has been used up, next index wraps around
  std::atomic<uint64_t> maxIndexSeen{0};
  std::atomic<uint32_t> countedQueueSize{0};

  auto func = [&]() {
    // local dest buffer for each user application
    void* local_dst_buffer = nullptr;
    ASSERT_SUCCESS(
        hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &local_dst_buffer));
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, local_dst_buffer));

    // Local completion signal for every user application
    hsa_signal_t local_signal;
    ASSERT_SUCCESS(hsa_signal_create(1, 0, nullptr, &local_signal));

    // Get a counted queue
    hsa_queue_t* queue = nullptr;
    ASSERT_SUCCESS(hsa_amd_counted_queue_acquire(*gpu_device1(), HSA_QUEUE_TYPE_MULTI,
                                                 HSA_AMD_QUEUE_PRIORITY_LOW, nullptr, nullptr, 0,
                                                 &queue));
    EXPECT_NE(queue, nullptr);

    if (queue == nullptr) {
      hsa_signal_destroy(local_signal);
      hsa_amd_memory_pool_free(local_dst_buffer);
      return;
    }

    uint32_t queue_size = queue->size;           // should be 16384
    const uint32_t queue_mask = queue_size - 1;  // used for index wraparound

    countedQueueSize.store(queue_size);

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
    ASSERT_SUCCESS(
        hsa_amd_memory_pool_allocate(kern_arg_pool(), sizeof(local_args), 0, &kernarg_address));
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, kernarg_address));

    memcpy(kernarg_address, &local_args, sizeof(local_args));

    // Dispatch more packets than queue size to force overflow and ensure that indices wrap around
    int it = queue_size + 5;

    for (int i = 0; i < it; i++) {
      // Reserve a slot in the queue
      uint64_t index = hsa_queue_add_write_index_relaxed(queue, 1);

      uint64_t curr_max = maxIndexSeen.load();
      while (index > curr_max && !maxIndexSeen.compare_exchange_weak(curr_max, index)) {
      }

      // Get pointer to the reserved packet slot using wraparound masking
      uint64_t wrapped_index = index & queue_mask;
      hsa_kernel_dispatch_packet_t* queue_aql_packet =
          &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address))[wrapped_index];

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
      uint32_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
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

      // Verify results are still correct after wraparound
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
    threads.emplace_back(func);
  }

  for (auto& th : threads) {
    th.join();
  }

  // Verify value of max seen index based on counted queue size
  uint64_t maxId = maxIndexSeen.load();
  EXPECT_EQ(maxId, (countedQueueSize.load() + 5) * kThreads - 1);

  hsa_amd_memory_pool_free(shared_src_buffer);
}