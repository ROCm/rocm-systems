/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <atomic>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

namespace {

// ---------------------------------------------------------------------------
// Agent discovery
// ---------------------------------------------------------------------------

template <hsa_device_type_t DeviceType>
hsa_status_t discover_agents(hsa_agent_t agent, void* data) {
  if (!data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t device_type = {};
  const auto status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (device_type == DeviceType) {
    static_cast<std::vector<hsa_agent_t>*>(data)->push_back(agent);
  }

  return HSA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Error-callback capture
// ---------------------------------------------------------------------------

struct error_capture {
  std::atomic<bool> invoked{false};
  hsa_status_t status{HSA_STATUS_SUCCESS};
  hsa_queue_t* source{nullptr};
};

void error_callback(hsa_status_t status, hsa_queue_t* source, void* data) {
  auto& capture = *static_cast<error_capture*>(data);
  capture.status = status;
  capture.source = source;
  capture.invoked.store(true, std::memory_order_release);
}

// Returns the first AIE agent; sets @p found to false if there is no AIE device.
hsa_agent_t first_aie_agent(bool& found) {
  std::vector<hsa_agent_t> aie_agents;
  found = (hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents) ==
           HSA_STATUS_SUCCESS) &&
          !aie_agents.empty();
  return found ? aie_agents.front() : hsa_agent_t{};
}

}  // namespace

// Shared fixture: initializes the HSA runtime once for the suite (always tearing it down, even when
// individual tests skip) and resolves the first AIE agent per test, skipping when no NPU is present.
class ErrorCallback : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    initialized_ = true;
  }

  static void TearDownTestSuite() {
    if (initialized_) {
      EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
      initialized_ = false;
    }
  }

  void SetUp() override {
    bool found = false;
    agent_ = first_aie_agent(found);
    if (!found) {
      GTEST_SKIP() << "No AIE device found; skipping test";
    }
  }

  hsa_agent_t agent_{};
  static bool initialized_;
};

bool ErrorCallback::initialized_ = false;

// Creating a queue with an error callback must succeed, and the callback must not fire while no
// error has occurred.
TEST_F(ErrorCallback, QueueCreateWithCallback) {
  uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(agent_, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);

  error_capture capture;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(agent_, min_queue_size, HSA_QUEUE_TYPE_SINGLE, error_callback, &capture,
                             0, 0, &queue),
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

  error_capture capture;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(agent_, min_queue_size, HSA_QUEUE_TYPE_SINGLE, error_callback, &capture,
                             0, 0, &queue),
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
