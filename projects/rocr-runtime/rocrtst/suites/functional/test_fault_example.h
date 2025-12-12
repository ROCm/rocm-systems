/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_TEST_FAULT_EXAMPLE_H_
#define ROCRTST_SUITES_FUNCTIONAL_TEST_FAULT_EXAMPLE_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class TestFaultExample : public TestBase {
 public:
  TestFaultExample();

  // @Brief: Destructor for test case of TestFaultExample
  virtual ~TestFaultExample();

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
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_TEST_FAULT_EXAMPLE_H_
