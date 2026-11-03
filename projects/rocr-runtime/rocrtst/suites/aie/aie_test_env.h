/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Setup shared by more than one AIE test binary: the runtime and agent fixture, agent discovery,
// and the queue-error-callback plumbing.
//
// Only what more than one file uses belongs here. Pool discovery differs between test_dispatch.cc
// and test_memory.cc, and each fixture beyond the common base is specific to its binary, so those
// live with the tests that use them.

#ifndef ROCRTST_SUITES_AIE_AIE_TEST_ENV_H_
#define ROCRTST_SUITES_AIE_AIE_TEST_ENV_H_

#include <atomic>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

namespace aie_test {

// ---------------------------------------------------------------------------
// Agent discovery
// ---------------------------------------------------------------------------

/**
 * @brief HSA agent iteration callback: appends every agent of type DeviceType to the
 * std::vector<hsa_agent_t>* stored in `data`.
 *
 * @tparam DeviceType device type to collect
 * @param agent agent being visited
 * @param data std::vector<hsa_agent_t>* to append a match to
 * @return the status of the HSA call that failed, or HSA_STATUS_SUCCESS
 */
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
    auto* const agents = static_cast<std::vector<hsa_agent_t>*>(data);
    agents->push_back(agent);
  }

  return HSA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

// Runtime lifetime and AIE agent discovery, per test. Derive from this and add whatever else the
// binary needs; the pools each one wants are not the same.
//
// What to do when there is no NPU is left to the derived fixture: the dispatch tests treat a
// missing agent as a failure, while ErrorCallback and the memory tests, which need no kernel
// artifacts and so are built everywhere, skip.
//
// TearDown shuts the runtime down even when SetUp or the test failed partway, which the
// hand-written per-test versions did not: an ASSERT in the middle of a test returned before its
// hsa_shut_down() and leaked the runtime for the rest of the binary.
class AieTestBase : public ::testing::Test {
 protected:
  std::vector<hsa_agent_t> aie_agents;

  void SetUp() override {
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    initialized_ = true;
    ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
              HSA_STATUS_SUCCESS);
  }

  void TearDown() override {
    if (initialized_) {
      initialized_ = false;
      EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
    }
  }

 private:
  bool initialized_ = false;
};

// ---------------------------------------------------------------------------
// Queue error reporting
// ---------------------------------------------------------------------------

/** @brief Captures the arguments of a queue error callback. */
struct dispatch_error {
  std::atomic<bool> invoked{false};
  hsa_status_t status{HSA_STATUS_SUCCESS};
  hsa_queue_t* source{nullptr};

  /**
   * @brief HSA queue error callback that records its arguments into the dispatch_error passed as
   * @p data.
   *
   * @param status error reported by the queue
   * @param source queue the error came from
   * @param data dispatch_error* to record into
   */
  static void callback(hsa_status_t status, hsa_queue_t* source, void* data) {
    auto& self = *static_cast<dispatch_error*>(data);
    self.status = status;
    self.source = source;
    self.invoked.store(true, std::memory_order_release);
  }
};

/**
 * @brief Creates a queue whose errors land in @p err.
 *
 * @param agent agent to create the queue on
 * @param size queue size in packets
 * @param err dispatch_error to report errors into
 * @param queue receives the created queue
 * @return the status of hsa_queue_create
 */
[[nodiscard]] inline hsa_status_t create_queue_with_error_callback(hsa_agent_t agent,
                                                                   std::uint32_t size,
                                                                   dispatch_error* err,
                                                                   hsa_queue_t** queue) {
  return hsa_queue_create(agent, size, HSA_QUEUE_TYPE_SINGLE, dispatch_error::callback, err, 0, 0,
                          queue);
}

}  // namespace aie_test

#endif  // ROCRTST_SUITES_AIE_AIE_TEST_ENV_H_
