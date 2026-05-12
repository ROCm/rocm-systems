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

  // @Brief: Destructor for test case of CountedQueuesTest
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

  // @brief Basic API test to create SDMA queues, query information and destroy on available GPU agents
  void CreateDestroy();

  // @brief Test packet submission to direct sdma queues
  void SubmitLinearCopy();
};

#endif // ROCRTST_SUITES_FUNCTIONAL_SDMA_QUEUES_H