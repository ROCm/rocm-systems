// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_GPU_BUSY_READ_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_GPU_BUSY_READ_H_

#include "test_base.h"

class TestGPUBusyRead : public TestBase {
 public:
  TestGPUBusyRead();

  // @Brief: Destructor for test case of TestGPUBusyRead
  virtual ~TestGPUBusyRead();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrieve the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_GPU_BUSY_READ_H_
