// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_GPU_IDENTITY_DEVICE_CUID_READ_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_GPU_IDENTITY_DEVICE_CUID_READ_H_

#include "test_base.h"

class TestDeviceCuidRead : public TestBase {
 public:
  TestDeviceCuidRead();
  virtual ~TestDeviceCuidRead();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_GPU_IDENTITY_DEVICE_CUID_READ_H_
