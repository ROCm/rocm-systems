// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_FUNCTIONAL_DEVICE_QUERY_TEST_H_
#define CUID_TEST_FUNCTIONAL_DEVICE_QUERY_TEST_H_

#include "test_base.h"

class TestDeviceQuery : public TestBase {
 public:
  TestDeviceQuery();
  void Run() override;
};

#endif  // CUID_TEST_FUNCTIONAL_DEVICE_QUERY_TEST_H_
