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

#include "unit/version_read_test.h"

#include <gtest/gtest.h>

#include <cstdio>

TestVersionRead::TestVersionRead() {
  SetTitle("Version Read");
  SetDescription(
      "Verify amdcuid_get_library_version and "
      "amdcuid_library_version_to_string return consistent values.");
}

// No device enumeration needed for this unit test.
void TestVersionRead::SetUp() {}

void TestVersionRead::Run() {
  uint32_t major = 0, minor = 0, patch = 0;
  amdcuid_get_library_version(&major, &minor, &patch);

  EXPECT_EQ(major, AMDCUID_LIB_VERSION_MAJOR);
  EXPECT_EQ(minor, AMDCUID_LIB_VERSION_MINOR);
  EXPECT_EQ(patch, AMDCUID_LIB_VERSION_PATCH);

  IF_VERB(1) { printf("  Library version: %u.%u.%u\n", major, minor, patch); }

  const char* version_str = amdcuid_library_version_to_string();
  EXPECT_NE(version_str, nullptr);

  char expected[16];
  snprintf(expected, sizeof(expected), "%u.%u.%u", AMDCUID_LIB_VERSION_MAJOR,
           AMDCUID_LIB_VERSION_MINOR, AMDCUID_LIB_VERSION_PATCH);
  EXPECT_STREQ(version_str, expected);

  IF_VERB(1) { printf("  Version string:  %s\n", version_str); }
}

void TestVersionRead::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestVersionRead::DisplayResults() const { TestBase::DisplayResults(); }
void TestVersionRead::Close() {}
