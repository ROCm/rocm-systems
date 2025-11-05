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

  void CountedQueueBasicApiTest();
  void CountedQueues_SamePriority_MaxLimitTest();
  void InvalidArgsTest();
  void CountedQueuesAllPrioritiesLimitTest();
  void CountedQueuesSetPriorityNackTest();
  void CountedQueuesSetCUMaskNackTest();
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_COUNTED_QUEUES_H