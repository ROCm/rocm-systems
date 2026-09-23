// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_UNIT_GIM_UTIL_TEST_H_
#define CUID_TEST_UNIT_GIM_UTIL_TEST_H_

#include "test_base.h"

// Availability and absent-driver behavior of the GIM ioctl client. Runs on any
// host: when the GIM device node is absent the client must report UNSUPPORTED,
// and when present the absent-driver assertions are skipped.
class TestGimClientAvailability : public TestBase {
 public:
  TestGimClientAvailability();
  void SetUp() override;
  void Run() override;
};

// Pure parsing of ASIC serial hex strings; no device required.
class TestGimParseAsicSerial : public TestBase {
 public:
  TestGimParseAsicSerial();
  void SetUp() override;
  void Run() override;
};

// Formatting of packed GIM BDF values into canonical PCI form; no device
// required.
class TestGimFormatBdf : public TestBase {
 public:
  TestGimFormatBdf();
  void SetUp() override;
  void Run() override;
};

// End-to-end enumeration against a live GIM driver. Requires root and the GIM
// device node; asserts every device has a unique, canonical BDF and a unique,
// parseable ASIC serial. Guards against wire-ABI regressions (a wrong struct
// size or handle width yields zero/garbage devices, duplicate BDFs, or
// duplicate serials that would collapse GPUs into one CUID).
class TestGimDeviceEnumeration : public TestBase {
 public:
  TestGimDeviceEnumeration();
  void SetUp() override;
  void Run() override;
};

#endif  // CUID_TEST_UNIT_GIM_UTIL_TEST_H_
