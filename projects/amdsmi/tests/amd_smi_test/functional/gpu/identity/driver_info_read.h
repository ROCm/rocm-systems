// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_GPU_IDENTITY_DRIVER_INFO_READ_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_GPU_IDENTITY_DRIVER_INFO_READ_H_

#include "test_base.h"

class TestDriverInfoRead : public TestBase {
 public:
  TestDriverInfoRead();
  virtual ~TestDriverInfoRead();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_GPU_IDENTITY_DRIVER_INFO_READ_H_
