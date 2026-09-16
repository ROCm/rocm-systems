// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_FABRIC_WRITE_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_FABRIC_WRITE_H_

#include "test_base.h"

class TestFabricWrite : public TestBase {
 public:
  TestFabricWrite();
  virtual ~TestFabricWrite();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo();
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_FABRIC_WRITE_H_
