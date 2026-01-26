/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_COUNTED_QUEUES_H
#define ROCRTST_SUITES_FUNCTIONAL_COUNTED_QUEUES_H


#include "suites/test_common/test_base.h"

class CountedQueuesTest : public TestBase {
 public:
  explicit CountedQueuesTest();

  // @Brief: Destructor for test case of CountedQueuesTest
  virtual ~CountedQueuesTest();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrive the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  /// @brief Basic API test to acquire, query and release 1 HW queue
  void CountedQueueBasicApiTest();

  /// @brief This test verifies that when many queues of the same priority are created, 
  // they are evenly distributed across the limited set of hardware queues, reuse those 
  // hardware queues correctly, and report proper use-counts before and after release
  void CountedQueues_SamePriority_MaxLimitTest();

  // @brief Test to verify HSA status codes for incorrect arguments sent via API
  void InvalidArgsTest();

  // @brief Test to verify that counted queues across all priorities each reuse only 
  // their own priority-specific hardware queues
  void CountedQueuesAllPrioritiesLimitTest();

  /// @brief Test to verify hsa_amd_queue_set_priority() does not work on counted queues
  void CountedQueuesSetPriorityNackTest();

  /// @brief Test to verify hsa_amd_queue_cu_set_mask() does not work on counted queues
  void CountedQueuesSetCUMaskNackTest();

  /// @brief Test to verify that a counted queue correctly supports kernel dispatches end-to-end
  void CountedQueuesDispatchTest();

  /// @brief Test to verify kernel dispatches onto shared queues from multiple user apps even when they
  // all share the same HW queue 
  void CountedQueuesMultithreadedDispatchTest();

  /// @brief Test to verify ring buffer wrap around when more than queue_size number of 
  // AQL packets are enqueued
  void CountedQueuesOverflowWrapAroundTest();
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_COUNTED_QUEUES_H