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

#ifndef CUID_TEST_FUNCTIONAL_SYSFS_CUID_TEST_H_
#define CUID_TEST_FUNCTIONAL_SYSFS_CUID_TEST_H_

#include "test_base.h"

// Verify that the primary CUID read directly from the cuid_primary sysfs file
// matches the value returned by AMDCUID_QUERY_PRIMARY_CUID for each amdgpu
// device.  Requires root (the library returns a temporary primary CUID for
// non-root callers, which would not match the sysfs value).
class TestSysfsReadPrimaryCuid : public TestBase {
 public:
  TestSysfsReadPrimaryCuid();
  void Run() override;
};

// Verify that the derived CUID read directly from the cuid_secondary sysfs
// file matches the value returned by AMDCUID_QUERY_DERIVED_CUID for each
// amdgpu device.  Requires root so that the library uses the real HMAC key
// when computing the derived CUID, which must match the driver-generated value.
class TestSysfsReadSecondaryCuid : public TestBase {
 public:
  TestSysfsReadSecondaryCuid();
  void Run() override;
};

#endif  // CUID_TEST_FUNCTIONAL_SYSFS_CUID_TEST_H_
