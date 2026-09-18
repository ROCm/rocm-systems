// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_UNIT_UTILITIES_TEST_H_
#define CUID_TEST_UNIT_UTILITIES_TEST_H_

#include "test_base.h"

class TestUtilities : public TestBase {
 public:
  TestUtilities();
  void SetUp() override;
  void Run() override;
  void DisplayTestInfo() override;
  void DisplayResults() const override;
  void Close() override;
};

#endif  // CUID_TEST_UNIT_UTILITIES_TEST_H_
