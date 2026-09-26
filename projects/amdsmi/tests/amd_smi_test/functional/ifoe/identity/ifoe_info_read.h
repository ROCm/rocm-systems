// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_IFOE_INFO_READ_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_IFOE_INFO_READ_H_

#include "test_base.h"

class TestIfoeInfoRead : public TestBase {
 public:
  TestIfoeInfoRead();

  // @Brief: Destructor for test case of TestIfoeInfoRead
  virtual ~TestIfoeInfoRead();

  // @Brief: Setup the environment for measurement
  void SetUp() override;

  // @Brief: Core measurement execution
  void Run() override;

  // @Brief: Clean up and retrieve the resource
  void Close() override;

  // @Brief: Display  results
  void DisplayResults() const override;

  // @Brief: Display information about what this test does
  void DisplayTestInfo() override;
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_IFOE_INFO_READ_H_
