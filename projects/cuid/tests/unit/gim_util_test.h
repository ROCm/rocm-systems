/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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

#endif  // CUID_TEST_UNIT_GIM_UTIL_TEST_H_
