// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_PCI_READ_WRITE_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_PCI_READ_WRITE_H_

#include "test_base.h"

class TestPciReadWrite : public TestBase {
 public:
  TestPciReadWrite();

  // @Brief: Destructor for test case of TestPciReadWrite
  virtual ~TestPciReadWrite();

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

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_PCI_READ_WRITE_H_
