// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_UNIT_PCI_UTIL_TEST_H_
#define CUID_TEST_UNIT_PCI_UTIL_TEST_H_

#include "test_base.h"

// Pins the interpretation of PCI configuration-space bytes. These values feed
// straight into the primary CUID, so getting them wrong silently changes every
// identifier a machine issues.
class TestPciConfigDecode : public TestBase {
 public:
  TestPciConfigDecode();
  void SetUp() override;
  void Run() override;
  void DisplayTestInfo() override;
  void DisplayResults() const override;
  void Close() override;
};

#endif  // CUID_TEST_UNIT_PCI_UTIL_TEST_H_
