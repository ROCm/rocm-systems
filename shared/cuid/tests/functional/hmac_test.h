// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_FUNCTIONAL_HMAC_TEST_H_
#define CUID_TEST_FUNCTIONAL_HMAC_TEST_H_

#include "test_base.h"

class TestHMAC : public TestBase {
 public:
  TestHMAC();
  void SetUp() override;
  void Run() override;
};

#endif  // CUID_TEST_FUNCTIONAL_HMAC_TEST_H_
