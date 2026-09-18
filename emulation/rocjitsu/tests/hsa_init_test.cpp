// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_init_test.cpp
/// @brief Verifies hsa_init() and GPU agent enumeration succeed through the
///        real ROCR runtime on the simulated GPU.
///
/// Requires LD_PRELOAD=librocjitsu.so.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <thread>

class HsaTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    auto status = hsa_init();
    ASSERT_EQ(status, HSA_STATUS_SUCCESS) << "hsa_init failed: " << status;
  }
};

TEST_F(HsaTest, InitSucceeded) {
  SUCCEED(); // Init verified in SetUpTestSuite.
}

TEST_F(HsaTest, GpuAgentFound) {
  int gpu_count = 0;
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_GPU)
          ++*static_cast<int *>(data);
        return HSA_STATUS_SUCCESS;
      },
      &gpu_count);
  EXPECT_GE(gpu_count, 1) << "Expected at least one GPU agent";
}

TEST_F(HsaTest, SdmaPublishesCompletePackets) {
#if defined(HSA_AMD_QUEUE_CREATE_DESC_VERSION)
  constexpr uint32_t kSdmaOpWrite = 2;
  constexpr uint32_t kSdmaOpFence = 5;
  hsa_agent_t gpu{};
  ASSERT_EQ(hsa_iterate_agents(
                [](hsa_agent_t agent, void *data) -> hsa_status_t {
                  hsa_device_type_t type;
                  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
                  if (status == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU)
                    *static_cast<hsa_agent_t *>(data) = agent;
                  return status;
                },
                &gpu),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(gpu.handle, 0u);
  hsa_amd_queue_create_desc_t desc{};
  desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  desc.engine_type = HSA_AMD_QUEUE_ENGINE_SDMA;
  desc.queue_size_bytes = 4096;
  desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  desc.engine.sdma.sdma_engine_id = HSA_AMD_SDMA_ENGINE_ID_ANY;
  ASSERT_EQ(hsa_amd_queue_create(gpu, &desc, 1), HSA_STATUS_SUCCESS);
  std::unique_ptr<hsa_queue_t, decltype(&hsa_queue_destroy)> queue(desc.queue, hsa_queue_destroy);

  alignas(8) std::array<uint32_t, 2> values{};
  uint64_t cursor = 0;
  for (uint32_t iteration = 1; iteration <= 1024; ++iteration) {
    SCOPED_TRACE(iteration);
    // Alternate opcode, destination, and data while reusing the ring. The
    // launcher runs the CP and its doorbell monitor concurrently with us.
    const size_t destination = iteration % values.size();
    const uint64_t address = reinterpret_cast<uint64_t>(&values[destination]);
    auto *packet = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(queue->base_address) +
                                                cursor % queue->size);
    std::fill_n(packet, 8, 0u); // Pad each submission to 32 bytes with NOPs.
    packet[0] = iteration % 2 ? kSdmaOpFence : kSdmaOpWrite;
    packet[1] = static_cast<uint32_t>(address);
    packet[2] = static_cast<uint32_t>(address >> 32);
    packet[iteration % 2 ? 3 : 4] = iteration;
    cursor += 8 * sizeof(uint32_t);
    hsa_queue_store_write_index_screlease(queue.get(), cursor);
    // This dispatches through ROCr's SdmaQueue::StoreRelease, which publishes
    // queue_wptr_ with release semantics and rings a volatile doorbell.
    hsa_signal_store_screlease(queue->doorbell_signal, cursor);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (hsa_queue_load_read_index_scacquire(queue.get()) < cursor &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    ASSERT_EQ(hsa_queue_load_read_index_scacquire(queue.get()), cursor);
    EXPECT_EQ(values[destination], iteration);
    EXPECT_EQ(values[1 - destination], iteration - 1);
  }
#else
  GTEST_SKIP() << "ROCr headers do not expose direct SDMA queue creation";
#endif
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  hsa_shut_down();
  return ret;
}
