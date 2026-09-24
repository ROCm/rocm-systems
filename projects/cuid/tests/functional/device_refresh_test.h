// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_FUNCTIONAL_DEVICE_REFRESH_TEST_H_
#define CUID_TEST_FUNCTIONAL_DEVICE_REFRESH_TEST_H_

#include "test_base.h"

class TestDeviceRefresh : public TestBase {
 public:
  TestDeviceRefresh();
  // No device enumeration needed before refresh — SetUp left as default.
  void Run() override;
};

#endif  // CUID_TEST_FUNCTIONAL_DEVICE_REFRESH_TEST_H_
