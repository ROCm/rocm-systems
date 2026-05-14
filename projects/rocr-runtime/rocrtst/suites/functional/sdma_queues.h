/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */


#ifndef ROCRTST_SUITES_FUNCTIONAL_SDMA_QUEUES_H
#define ROCRTST_SUITES_FUNCTIONAL_SDMA_QUEUES_H

#include "suites/test_common/test_base.h"

class SdmaQueuesTest : public TestBase {
 public:
  explicit SdmaQueuesTest();

  // @Brief: Destructor for test case of SdmaQueuesTest
  virtual ~SdmaQueuesTest();

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

  // @Brief: Test SDMA queue create, query info, and destroy on available GPU agents
  void CreateDestroy();

  // @Brief: Test linear copy submission via ring_doorbell API and direct MMIO (Linux only)
  void SubmitLinearCopy();

  /* @Brief Creates multiple user SDMA queues and verifies:
    1. Each queue is unique (ring_base, rptr, wptr, doorbell, queue_id)
    2. If the user calls hsa_amd_memory_async_copy, SDMA queues already
       owned by the user are not reused. */
  void ExclusiveQueueResources();
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SDMA_QUEUES_H