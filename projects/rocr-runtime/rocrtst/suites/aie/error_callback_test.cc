/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <atomic>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "aie_test_env.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

using aie_test::dispatch_error;

// Shared fixture: AieTestBase initializes the HSA runtime per test and tears it down again in
// TearDown, including when the test skips; this fixture resolves the first AIE agent on top of
// that. Unlike the dispatch fixtures, it skips rather than fails when there is no NPU: these tests
// need no kernel artifacts, so the binary is expected to run anywhere.
class ErrorCallback : public aie_test::AieTestBase {
 protected:
  hsa_agent_t agent_{};

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(AieTestBase::SetUp());
    if (aie_agents.empty()) {
      GTEST_SKIP() << "No AIE device found; skipping test";
    }
    agent_ = aie_agents.front();
  }
};

// Creating a queue with an error callback must succeed, and the callback must not fire while no
// error has occurred.
TEST_F(ErrorCallback, QueueCreateWithCallback) {
  uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(agent_, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);

  dispatch_error capture;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(aie_test::create_queue_with_error_callback(agent_, min_queue_size, &capture, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  EXPECT_FALSE(capture.invoked.load(std::memory_order_acquire));

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// Submitting a packet that references an unregistered (invalid) buffer must be rejected by the
// driver and reported through the per-queue error callback.
TEST_F(ErrorCallback, InvalidDispatchInvokesCallback) {
  uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(agent_, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);

  dispatch_error capture;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(aie_test::create_queue_with_error_callback(agent_, min_queue_size, &capture, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  // Build a dispatch packet whose PDI address is not a buffer registered with the driver, so the
  // command is rejected at submission time (before reaching the device).
  auto* ring = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(queue->base_address);
  const uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);

  hsa_amd_aie_kernel_dispatch_packet_t pkt{};
  pkt.header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
               (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
               (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
  pkt.count = 24;
  pkt.pdi_addr = reinterpret_cast<void*>(0x1000);  // deliberately not a registered BO
  pkt.num_kernargs = 0;
  pkt.kernarg_address = nullptr;
  ring[wr_idx % queue->size] = pkt;

  // Ringing the doorbell submits the packet synchronously on this thread for KMQ queues, so the
  // callback has fired by the time the store returns.
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  EXPECT_TRUE(capture.invoked.load(std::memory_order_acquire));
  EXPECT_NE(capture.status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(capture.source, queue);

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}
