// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_UNIT_CONCURRENCY_TEST_H_
#define CUID_TEST_UNIT_CONCURRENCY_TEST_H_

#include "test_base.h"

// Drives the public C API from several threads at once so that
// CuidDeviceManager's shared state is exercised concurrently. Without this the
// suite is single-threaded and ThreadSanitizer has nothing to observe.
class TestConcurrentApi : public TestBase {
 public:
  TestConcurrentApi();
  void SetUp() override;
  void Run() override;
  void DisplayTestInfo() override;
  void DisplayResults() const override;
  void Close() override;
};

#endif  // CUID_TEST_UNIT_CONCURRENCY_TEST_H_
