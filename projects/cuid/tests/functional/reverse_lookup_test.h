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
