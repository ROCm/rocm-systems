// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_FUNCTIONAL_REVERSE_LOOKUP_TEST_H_
#define CUID_TEST_FUNCTIONAL_REVERSE_LOOKUP_TEST_H_

#include "test_base.h"

// Each reverse-lookup test is a separate class so failures are reported
// per-field rather than stopping at the first failing device.

class TestReverseSerialNumber : public TestBase {
 public:
  TestReverseSerialNumber();
  void Run() override;
};

class TestReverseVendorId : public TestBase {
 public:
  TestReverseVendorId();
  void Run() override;
};

class TestReverseDeviceId : public TestBase {
 public:
  TestReverseDeviceId();
  void Run() override;
};

class TestReverseRevisionId : public TestBase {
 public:
  TestReverseRevisionId();
  void Run() override;
};

class TestReverseUnitId : public TestBase {
 public:
  TestReverseUnitId();
  void Run() override;
};

class TestReverseDeviceType : public TestBase {
 public:
  TestReverseDeviceType();
  void Run() override;
};

#endif  // CUID_TEST_FUNCTIONAL_REVERSE_LOOKUP_TEST_H_
