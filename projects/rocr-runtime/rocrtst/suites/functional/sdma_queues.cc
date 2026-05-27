/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */


#include "suites/functional/sdma_queues.h"
#include "hsa/hsa_ext_amd.h"
#include "common/base_rocr_utils.h"
#include <chrono>
#include <thread>
#include <unordered_set>

SdmaQueuesTest::SdmaQueuesTest() : TestBase() {
  set_title("RocR - User SDMA Queues Test");
  set_description(
      "This test validates the behavior of SDMA queues created by user applications through the "
      "hsa_amd_sdma_queue_create API, ensuring they are properly registered, can be queried for "
      "information and used to submit packets.");
}

SdmaQueuesTest::~SdmaQueuesTest() {}

void SdmaQueuesTest::SetUp() {
  TestBase::SetUp();
  if (test_skipped_) return;
}

void SdmaQueuesTest::Run() {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void SdmaQueuesTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void SdmaQueuesTest::DisplayResults() const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::DisplayResults();
}

void SdmaQueuesTest::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

void SdmaQueuesTest::CreateDestroy() {
  // Find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  // Find all cpu agents
  std::vector<hsa_agent_t> cpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus));

  ASSERT_NE(gpus.size(), 0);
  ASSERT_NE(cpus.size(), 0);
  hsa_agent_t cpu_agent = cpus[0];

  for (auto cpu : cpus) {
    hsa_amd_sdma_queue_t queue;
    ASSERT_EQ(HSA_STATUS_ERROR_INVALID_AGENT,
              hsa_amd_sdma_queue_create(cpu, 0, (hsa_amd_sdma_engine_id)0, 0, &queue));
  }

  for (const auto& gpu : gpus) {
    // Get preferred SDMA engine for H2D copy
    uint32_t preferred = 0;
    ASSERT_SUCCESS(hsa_amd_memory_get_preferred_copy_engine(gpu, cpus[0], &preferred));
    if (preferred == 0) {
      preferred = HSA_AMD_SDMA_ENGINE_0;
    }

    uint32_t flags = HSA_AMD_SDMA_QUEUE_FLAG_USE_ENGINE_ID;
    hsa_amd_sdma_queue_t queue;
    ASSERT_SUCCESS(
        hsa_amd_sdma_queue_create(gpu, flags, (hsa_amd_sdma_engine_id_t)preferred, 0, &queue));
    EXPECT_NE(queue.handle, 0u);

    // Query ring buffer's address, size, read and write pointers
    hsa_amd_sdma_queue_resource_t queue_info = {};
    ASSERT_SUCCESS(
        hsa_amd_sdma_queue_get_info(queue, HSA_AMD_SDMA_QUEUE_INFO_RESOURCE, &queue_info));
    EXPECT_NE(queue_info.ring_base, nullptr);
    EXPECT_GT(queue_info.ring_size, 0);
    EXPECT_NE(queue_info.read_ptr, nullptr);
    EXPECT_NE(queue_info.write_ptr, nullptr);
    EXPECT_NE(queue_info.doorbell, nullptr);

    // Check if this is created with the correct gpu agent
    hsa_agent_t agent;
    ASSERT_SUCCESS(hsa_amd_sdma_queue_get_info(queue, HSA_AMD_SDMA_QUEUE_INFO_AGENT, &agent));
    EXPECT_EQ(agent.handle, gpu.handle);
    // Destroy the queue
    ASSERT_SUCCESS(hsa_amd_sdma_queue_destroy(queue));
  }
}

void SdmaQueuesTest::SubmitLinearCopy() {
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));
  ASSERT_GT(gpus.size(), 0);

  std::vector<hsa_agent_t> cpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus));
  ASSERT_GT(cpus.size(), 0);
  hsa_agent_t cpu_agent = cpus[0];

  for (const auto& gpu : gpus) {
    hsa_amd_memory_pool_t sys_pool;
    ASSERT_SUCCESS(
        hsa_amd_agent_iterate_memory_pools(cpu_agent, rocrtst::GetGlobalMemoryPool, &sys_pool));

    // Create SDMA queue
    hsa_amd_sdma_queue_t queue;
    ASSERT_SUCCESS(hsa_amd_sdma_queue_create(gpu, 0, (hsa_amd_sdma_engine_id_t)0, 0, &queue));

    // Query queue properties
    hsa_amd_sdma_queue_resource_t queue_info = {};
    ASSERT_SUCCESS(
        hsa_amd_sdma_queue_get_info(queue, HSA_AMD_SDMA_QUEUE_INFO_RESOURCE, &queue_info));
    EXPECT_NE(queue_info.ring_base, nullptr);
    EXPECT_GT(queue_info.ring_size, 0);
    EXPECT_NE(queue_info.read_ptr, nullptr);
    EXPECT_NE(queue_info.write_ptr, nullptr);
    EXPECT_NE(queue_info.doorbell, nullptr);

    size_t min_submission = 0;
    ASSERT_SUCCESS(hsa_amd_sdma_queue_get_info(queue, HSA_AMD_SDMA_QUEUE_INFO_MIN_SUBMISSION_SIZE,
                                               &min_submission));

    // Allocate src and dest buffers
    const size_t kCopySize = 4096;
    void* src_buf = nullptr;
    void* dst_buf = nullptr;
    ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(sys_pool, kCopySize, 0, &src_buf));
    ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(sys_pool, kCopySize, 0, &dst_buf));

    // Grant CPU and GPU access to both buffers
    hsa_agent_t access_agents[] = {gpu, cpu_agent};
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, access_agents, nullptr, src_buf));
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, access_agents, nullptr, dst_buf));

    auto BuildCopyPacket = [&](uint32_t (&pkt)[7]) {
      memset(pkt, 0, sizeof(pkt));
      pkt[0] = 1 | (0 << 8);
      pkt[1] = (kCopySize - 1) & 0x003FFFFF;
      pkt[2] = 0;
      pkt[3] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src_buf));
      pkt[4] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src_buf) >> 32);
      pkt[5] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst_buf));
      pkt[6] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst_buf) >> 32);
    };

    const uint32_t kPacketDwords = 7;
    const size_t kPacketSize = kPacketDwords * sizeof(uint32_t);
    size_t submission_size = std::max(kPacketSize, min_submission);
    ASSERT_LE(submission_size, queue_info.ring_size);

    uint8_t* ring = static_cast<uint8_t*>(queue_info.ring_base);

    auto SubmitAndVerify = [&](uint32_t pattern, bool use_api_doorbell) {
      // Init src buffer with pattern, dest buffer with zeroes
      ASSERT_SUCCESS(hsa_amd_memory_fill(src_buf, pattern, kCopySize / sizeof(uint32_t)));
      memset(dst_buf, 0, kCopySize);

      uint32_t packet[kPacketDwords];
      BuildCopyPacket(packet);

      uint64_t current_wptr = *queue_info.write_ptr;
      uint64_t wptr_offset = current_wptr % queue_info.ring_size;
      memset(ring + wptr_offset, 0, submission_size);
      memcpy(ring + wptr_offset, packet, kPacketSize);
      std::atomic_thread_fence(std::memory_order_release);

      uint64_t new_wptr = current_wptr + submission_size;

      if (use_api_doorbell) {
        ASSERT_SUCCESS(hsa_amd_sdma_queue_ring_doorbell(queue, new_wptr));
      } else {
        *queue_info.write_ptr = new_wptr;
        std::atomic_thread_fence(std::memory_order_release);
        *queue_info.doorbell = new_wptr;
      }

      // Wait for SDMA engine to consume the packet and verify the values in dest buffer
      auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(5);
      while (*queue_info.read_ptr < new_wptr) {
        if (std::chrono::high_resolution_clock::now() > deadline) {
          FAIL() << "Timed out waiting for SDMA copy to complete";
          return;
        }
        std::this_thread::yield();
      }
      std::atomic_thread_fence(std::memory_order_acquire);

      // validate values in dest buffer
      uint32_t* dst_words = reinterpret_cast<uint32_t*>(dst_buf);
      for (size_t i = 0; i < kCopySize / sizeof(uint32_t); i++) {
        EXPECT_EQ(dst_words[i], pattern);
      }
    };

    SubmitAndVerify(2, true);

#if defined(__linux__)
    // Direct MMIO path (Linux only)
    SubmitAndVerify(1, false);
#endif

    // Cleanup
    ASSERT_SUCCESS(hsa_amd_memory_pool_free(src_buf));
    ASSERT_SUCCESS(hsa_amd_memory_pool_free(dst_buf));
    ASSERT_SUCCESS(hsa_amd_sdma_queue_destroy(queue));
  }
}

void SdmaQueuesTest::ExclusiveQueueResources() {
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));
  ASSERT_GT(gpus.size(), 0);

  std::vector<hsa_agent_t> cpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus));
  ASSERT_GT(cpus.size(), 0);
  hsa_agent_t cpu_agent = cpus[0];

  for (const auto& gpu : gpus) {
    constexpr size_t kNumQueues = 4;

    struct QueueInfo {
      hsa_amd_sdma_queue_t queue;
      hsa_amd_sdma_queue_resource_t res;
      uint64_t queue_id;
    };

    // Sets to track uniqueness of sdma queues created
    std::unordered_set<uint64_t> seen_handles;
    std::unordered_set<uint64_t> seen_queue_ids;
    std::unordered_set<void*> seen_ring_bases;
    std::unordered_set<volatile uint64_t*> seen_read_ptrs;
    std::unordered_set<volatile uint64_t*> seen_write_ptrs;
    std::unordered_set<volatile uint64_t*> seen_doorbells;

    // Store queue info for cleanup and async_copy test.
    std::vector<QueueInfo> queues(kNumQueues);

    // Create multiple SDMA queues
    for (size_t i = 0; i < kNumQueues; i++) {
      ASSERT_SUCCESS(
          hsa_amd_sdma_queue_create(gpu, 0, (hsa_amd_sdma_engine_id_t)0, 0, &queues[i].queue));
      ASSERT_NE(queues[i].queue.handle, 0);
      ASSERT_SUCCESS(hsa_amd_sdma_queue_get_info(queues[i].queue, HSA_AMD_SDMA_QUEUE_INFO_RESOURCE,
                                                 &queues[i].res));
      ASSERT_NE(queues[i].res.ring_base, nullptr);
      ASSERT_GT(queues[i].res.ring_size, 0u);
      ASSERT_NE(queues[i].res.read_ptr, nullptr);
      ASSERT_NE(queues[i].res.write_ptr, nullptr);
      ASSERT_NE(queues[i].res.doorbell, nullptr);

      ASSERT_SUCCESS(hsa_amd_sdma_queue_get_info(queues[i].queue, HSA_AMD_SDMA_QUEUE_INFO_QUEUE_ID,
                                                 &queues[i].queue_id));
      ASSERT_TRUE(seen_handles.insert(queues[i].queue.handle).second);
      ASSERT_TRUE(seen_queue_ids.insert(queues[i].queue_id).second);
      ASSERT_TRUE(seen_ring_bases.insert(queues[i].res.ring_base).second);
      ASSERT_TRUE(seen_read_ptrs.insert(queues[i].res.read_ptr).second);
      ASSERT_TRUE(seen_write_ptrs.insert(queues[i].res.write_ptr).second);
      ASSERT_TRUE(seen_doorbells.insert(queues[i].res.doorbell).second);
    }

    /* While all user SDMA queues are held, verify that hsa_amd_memory_async_copy works
    and it does not use the ones exposed to application user */
    hsa_amd_memory_pool_t sys_pool;
    ASSERT_SUCCESS(
        hsa_amd_agent_iterate_memory_pools(cpu_agent, rocrtst::GetGlobalMemoryPool, &sys_pool));

    const size_t kCopySize = 4096;
    const uint32_t kPattern = 1;
    void* src_buf = nullptr;
    void* dst_buf = nullptr;
    ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(sys_pool, kCopySize, 0, &src_buf));
    ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(sys_pool, kCopySize, 0, &dst_buf));

    hsa_agent_t access_agents[] = {gpu, cpu_agent};
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, access_agents, nullptr, src_buf));
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, access_agents, nullptr, dst_buf));

    // init src with 1s and dest with 0s
    ASSERT_SUCCESS(hsa_amd_memory_fill(src_buf, kPattern, (kCopySize / sizeof(uint32_t))));
    memset(dst_buf, 0, kCopySize);

    // Create a completion signal for async copy
    hsa_signal_t completion_signal;
    ASSERT_SUCCESS(hsa_signal_create(1, 0, nullptr, &completion_signal));

    // Perform async copy while user SDMA queues are held
    ASSERT_SUCCESS(hsa_amd_memory_async_copy(dst_buf, cpu_agent, src_buf, gpu, kCopySize, 0,
                                             nullptr, completion_signal));

    // Wait for completion of async copy
    ASSERT_SUCCESS(hsa_signal_wait_scacquire(completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                             (uint64_t)-1, HSA_WAIT_STATE_ACTIVE));

    // Verify copied data
    uint32_t* dst_words = reinterpret_cast<uint32_t*>(dst_buf);
    for (size_t i = 0; i < (kCopySize / sizeof(uint32_t)); i++) {
      EXPECT_EQ(dst_words[i], kPattern);
    }

    // Cleanup
    ASSERT_SUCCESS(hsa_signal_destroy(completion_signal));
    ASSERT_SUCCESS(hsa_amd_memory_pool_free(src_buf));
    ASSERT_SUCCESS(hsa_amd_memory_pool_free(dst_buf));
    for (size_t i = 0; i < kNumQueues; i++) {
      ASSERT_SUCCESS(hsa_amd_sdma_queue_destroy(queues[i].queue));
    }
  }
}