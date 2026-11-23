/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_NEGATIVE_SIGNAL_EXHAUSTION_ERROR_HANDLER_H_
#define ROCRTST_SUITES_NEGATIVE_SIGNAL_EXHAUSTION_ERROR_HANDLER_H_

#include <sys/resource.h>

#include <atomic>
#include <cstdint>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "suites/test_common/test_base.h"

// Verifies that GPU exceptions reported by KFD are still delivered to the system event handler
// after the process has used up all of its KFD events.
class SignalExhaustionErrorHandlerTest : public TestBase {
 public:
  SignalExhaustionErrorHandlerTest();

  // @Brief: Destructor for the SignalExhaustionErrorHandlerTest class
  virtual ~SignalExhaustionErrorHandlerTest();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrieve the resource
  virtual void Close();

  // @Brief: Display results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  // @Brief: Allocate more signals than KFD has events, subscribe to the system
  // event handler, submit an invalid AQL dispatch packet and verify that the
  // resulting exception is reported to the system event handler.
  void TestSystemErrorHandlerAfterSignalExhaustion(void);

 private:
  // State shared with the system event handler. It is a member (not a local) because the handler
  // stays registered until hsa_shut_down() and is called one last time with the shutdown event.
  struct CallbackData {
    std::atomic<bool> fault_reported{false};
    std::atomic<uint64_t> fault_address{0};
    std::atomic<uint64_t> fault_agent{0};
  };

  static hsa_status_t SystemEventCallback(const hsa_amd_event_t* event, void* data);

  void RunOnAgent(hsa_agent_t gpu_agent);

  struct rlimit rlimit_;  // value of rlimit before test starts
  CallbackData cb_data_;
  bool system_handler_registered_;
};

#endif  // ROCRTST_SUITES_NEGATIVE_SIGNAL_EXHAUSTION_ERROR_HANDLER_H_
