/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */


#include "suites/functional/sdma_queues.h"
#include "hsa/hsa_ext_amd.h"
#include "common/base_rocr_utils.h"

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
              hsa_amd_sdma_queue_create(cpu, 0, (hsa_amd_sdma_engine_id)0, &queue));
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
        hsa_amd_sdma_queue_create(gpu, flags, (hsa_amd_sdma_engine_id_t)preferred, &queue));
    EXPECT_NE(queue.handle, 0u);

    // Query ring buffer's address, size, read and write pointers
    hsa_amd_sdma_queue_resource_t queue_info = {};
    ASSERT_SUCCESS(hsa_amd_sdma_queue_get_info(queue, HSA_AMD_SDMA_QUEUE_INFO_RESOURCE, &queue_info));
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
    ASSERT_SUCCESS(hsa_amd_agent_iterate_memory_pools(cpu_agent, rocrtst::GetGlobalMemoryPool, &sys_pool));

    // Create SDMA queue
    hsa_amd_sdma_queue_t queue;
    ASSERT_SUCCESS(hsa_amd_sdma_queue_create(gpu, 0, (hsa_amd_sdma_engine_id_t)0, &queue));

    // Query queue properties
    hsa_amd_sdma_queue_resource_t queue_info = {};
    ASSERT_SUCCESS(hsa_amd_sdma_queue_get_info(queue, HSA_AMD_SDMA_QUEUE_INFO_RESOURCE, &queue_info));
    EXPECT_NE(queue_info.ring_base, nullptr);
    EXPECT_GT(queue_info.ring_size, 0);
    EXPECT_NE(queue_info.read_ptr, nullptr);
    EXPECT_NE(queue_info.write_ptr, nullptr);
    EXPECT_NE(queue_info.doorbell, nullptr);

    uint32_t min_submission = 0;
    ASSERT_SUCCESS(hsa_amd_sdma_queue_get_info(queue, HSA_AMD_SDMA_QUEUE_INFO_MIN_SUBMISSION_SIZE,
                                               &min_submission));

    // Allocate src and dest buffers
    const size_t kCopySize = 4096;
    const uint32_t kPattern = 1;
    void* src_buf = nullptr;
    void* dst_buf = nullptr;
    ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(sys_pool, kCopySize, 0, &src_buf));
    ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(sys_pool, kCopySize, 0, &dst_buf));

    // Grant CPU and GPU access to both buffers
    hsa_agent_t access_agents[] = {gpu, cpu_agent};
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, access_agents, nullptr, src_buf));
    ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, access_agents, nullptr, dst_buf));

    // Initialize src with ones and dest buffer with zeroes
    ASSERT_SUCCESS(hsa_amd_memory_fill(src_buf, kPattern, kCopySize / sizeof(uint32_t)));
    memset(dst_buf, 0, kCopySize);
    EXPECT_EQ(reinterpret_cast<uint32_t*>(src_buf)[0], kPattern);
    EXPECT_EQ(reinterpret_cast<uint32_t*>(dst_buf)[0], 0);

    // Build SDMA_PKT_COPY_LINEAR packet
    const uint32_t kPacketDwords = 7;
    const uint32_t kPacketSize = kPacketDwords * sizeof(uint32_t);
    uint32_t packet[kPacketDwords];
    memset(packet, 0, sizeof(packet));

    // header (op=1 for copy, sub_op=0 for linear)
    packet[0] = 1 | (0 << 8);
    packet[1] = (kCopySize - 1) & 0x003FFFFF;
    packet[2] = 0;

    // src address
    packet[3] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src_buf));
    packet[4] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src_buf) >> 32);

    // dest address
    packet[5] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst_buf));
    packet[6] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst_buf) >> 32);

    // Calculate total submission size. Pad with NOP if less than min_submission size
    uint32_t submission_size = kPacketSize;
    if (submission_size < min_submission) {
      submission_size = min_submission;
    }
    ASSERT_LE(submission_size, queue_info.ring_size);

    // Write packet into ring at current wptr offset
    uint64_t current_wptr = *queue_info.write_ptr;
    uint64_t wptr_offset = current_wptr % queue_info.ring_size;
    uint8_t* ring = static_cast<uint8_t*>(queue_info.ring_base);

    // Copy actual packet
    memset(ring + wptr_offset, 0, submission_size);
    memcpy(ring + wptr_offset, packet, kPacketSize);

    // Memory barrier before updating wptr
    std::atomic_thread_fence(std::memory_order_release);

    // Update write pointer
    uint64_t new_wptr = current_wptr + submission_size;
    *queue_info.write_ptr = new_wptr;

    std::atomic_thread_fence(std::memory_order_release);

    // Ring doorbell
    *queue_info.doorbell = new_wptr;

    // Wait for HW to consume the packet
    // read_ptr should catch up with new write_ptr after packet is processed
    const uint32_t kTimeoutMs = 5000;
    const uint32_t kPollIntervalUs = 100;
    uint32_t elapsed_us = 0;
    while (*queue_info.read_ptr != new_wptr && elapsed_us < kTimeoutMs * 1000) {
      usleep(kPollIntervalUs);
      elapsed_us += kPollIntervalUs;
    }
    EXPECT_EQ(*queue_info.read_ptr, new_wptr);

    // Verify destination buffer contains the source pattern
    uint32_t* dst_words = reinterpret_cast<uint32_t*>(dst_buf);
    for (size_t i = 0; i < (kCopySize/sizeof(uint32_t)); i++) {
      EXPECT_EQ(dst_words[i], kPattern);
    }

    // Cleanup
    ASSERT_SUCCESS(hsa_amd_memory_pool_free(src_buf));
    ASSERT_SUCCESS(hsa_amd_memory_pool_free(dst_buf));
    ASSERT_SUCCESS(hsa_amd_sdma_queue_destroy(queue));
  }
}