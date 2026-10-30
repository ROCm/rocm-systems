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
    std::ranges::fill_n(packet, 8, 0u); // Pad each submission to 32 bytes with NOPs.
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

TEST_F(HsaTest, SdmaSwapExchangesBuffersAndCompletes) {
#if defined(HSA_AMD_MEMORY_COPY_OP_VERSION)
  hsa_agent_t gpu{};
  ASSERT_EQ(hsa_iterate_agents(
                [](hsa_agent_t agent, void *data) -> hsa_status_t {
                  hsa_device_type_t type;
                  const hsa_status_t status =
                      hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
                  if (status == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU)
                    *static_cast<hsa_agent_t *>(data) = agent;
                  return status;
                },
                &gpu),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(gpu.handle, 0u);
  hsa_amd_memory_pool_t pool{};
  ASSERT_EQ(hsa_amd_agent_iterate_memory_pools(
                gpu,
                [](hsa_amd_memory_pool_t candidate, void *data) -> hsa_status_t {
                  hsa_amd_segment_t segment;
                  hsa_status_t status = hsa_amd_memory_pool_get_info(
                      candidate, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
                  if (status != HSA_STATUS_SUCCESS || segment != HSA_AMD_SEGMENT_GLOBAL)
                    return status;
                  bool allocatable = false;
                  status = hsa_amd_memory_pool_get_info(
                      candidate, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allocatable);
                  if (status == HSA_STATUS_SUCCESS && allocatable)
                    *static_cast<hsa_amd_memory_pool_t *>(data) = candidate;
                  return status;
                },
                &pool),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(pool.handle, 0u);

  // Cross two 4096-byte exchange chunks and include a short final chunk.
  // Distinct bytes in each chunk detect reuse of an earlier chunk's contents.
  constexpr size_t kBytes = 8256;
  std::array<uint8_t, kBytes> original_a;
  std::array<uint8_t, kBytes> original_b;
  for (size_t index = 0; index < kBytes; ++index) {
    original_a[index] = static_cast<uint8_t>((index * 3 + 1) ^ (index >> 8));
    original_b[index] = static_cast<uint8_t>((index * 7 + 5) ^ (index >> 8));
  }
  void *allocation = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(pool, kBytes, 0, &allocation), HSA_STATUS_SUCCESS);
  std::unique_ptr<void, decltype(&hsa_amd_memory_pool_free)> buffer_a(allocation,
                                                                      hsa_amd_memory_pool_free);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(pool, kBytes, 0, &allocation), HSA_STATUS_SUCCESS);
  std::unique_ptr<void, decltype(&hsa_amd_memory_pool_free)> buffer_b(allocation,
                                                                      hsa_amd_memory_pool_free);
  ASSERT_EQ(hsa_memory_copy(buffer_a.get(), original_a.data(), kBytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(buffer_b.get(), original_b.data(), kBytes), HSA_STATUS_SUCCESS);
  hsa_signal_t completion{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &completion), HSA_STATUS_SUCCESS);
  hsa_amd_memory_copy_op_t operation{};
  operation.version = HSA_AMD_MEMORY_COPY_OP_VERSION;
  operation.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP;
  operation.src = buffer_a.get();
  operation.dst = buffer_b.get();
  operation.src_agent = gpu;
  operation.dst_agent = gpu;
  operation.src_size = kBytes;
  operation.dst_size = kBytes;
  operation.completion_signal = completion;
  ASSERT_EQ(hsa_amd_memory_async_batch_copy(&operation, 1, 0, nullptr), HSA_STATUS_SUCCESS);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (hsa_signal_load_scacquire(completion) != 0 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  ASSERT_EQ(hsa_signal_load_scacquire(completion), 0) << "SDMA swap completion remains pending";
  std::array<uint8_t, kBytes> actual_a;
  std::array<uint8_t, kBytes> actual_b;
  ASSERT_EQ(hsa_memory_copy(actual_a.data(), buffer_a.get(), kBytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_memory_copy(actual_b.data(), buffer_b.get(), kBytes), HSA_STATUS_SUCCESS);
  EXPECT_EQ(actual_a, original_b);
  EXPECT_EQ(actual_b, original_a);
  EXPECT_EQ(hsa_signal_destroy(completion), HSA_STATUS_SUCCESS);
#else
  GTEST_SKIP() << "ROCr headers do not expose batch memory copies";
#endif
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  hsa_shut_down();
  return ret;
}
