// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMDSMI_TEST_FUNCTIONAL_FAN_READ_WRITE_H_
#define TESTS_AMDSMI_TEST_FUNCTIONAL_FAN_READ_WRITE_H_

#include "test_base.h"

class TestFanReadWrite : public TestBase {
 public:
  TestFanReadWrite();

  // @Brief: Destructor for test case of TestFanReadWrite
  virtual ~TestFanReadWrite();

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

#endif  // TESTS_AMDSMI_TEST_FUNCTIONAL_FAN_READ_WRITE_H_
