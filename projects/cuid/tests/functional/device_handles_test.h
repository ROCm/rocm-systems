// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_FUNCTIONAL_DEVICE_HANDLES_TEST_H_
#define CUID_TEST_FUNCTIONAL_DEVICE_HANDLES_TEST_H_

#include "test_base.h"

class TestGetAllHandles : public TestBase {
 public:
  TestGetAllHandles();
  void Run() override;
};

class TestGetHandleByBDF : public TestBase {
 public:
  TestGetHandleByBDF();
  void Run() override;
};

class TestGetHandleByDevPath : public TestBase {
 public:
  TestGetHandleByDevPath();
  void Run() override;
};

class TestGetHandleByFD : public TestBase {
 public:
  TestGetHandleByFD();
  void Run() override;
};

#endif  // CUID_TEST_FUNCTIONAL_DEVICE_HANDLES_TEST_H_
