// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_UNIT_ID_STRING_TEST_H_
#define CUID_TEST_UNIT_ID_STRING_TEST_H_

#include "test_base.h"

class TestIdString : public TestBase {
 public:
  TestIdString();
  void SetUp() override;
  void Run() override;
  void DisplayTestInfo() override;
  void DisplayResults() const override;
  void Close() override;
};

#endif  // CUID_TEST_UNIT_ID_STRING_TEST_H_
