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

#include "unit/status_string_test.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

TestStatusString::TestStatusString() {
  SetTitle("Status String");
  SetDescription(
      "Verify amdcuid_status_to_string returns a non-empty string for every "
      "known status code.");
}

void TestStatusString::SetUp() {}

void TestStatusString::Run() {
  static const amdcuid_status_t kStatuses[] = {
      AMDCUID_STATUS_SUCCESS,
      AMDCUID_STATUS_FILE_NOT_FOUND,
      AMDCUID_STATUS_DEVICE_NOT_FOUND,
      AMDCUID_STATUS_INVALID_ARGUMENT,
      AMDCUID_STATUS_PERMISSION_DENIED,
      AMDCUID_STATUS_UNSUPPORTED,
      AMDCUID_STATUS_WRONG_DEVICE_TYPE,
      AMDCUID_STATUS_INSUFFICIENT_SIZE,
      AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND,
      AMDCUID_STATUS_KEY_ERROR,
      AMDCUID_STATUS_HMAC_ERROR,
      AMDCUID_STATUS_FILE_ERROR,
      AMDCUID_STATUS_INVALID_FORMAT,
      AMDCUID_STATUS_PCI_ERROR,
      AMDCUID_STATUS_SMBIOS_ERROR,
      AMDCUID_STATUS_ACPI_ERROR,
      AMDCUID_STATUS_CPUINFO_ERROR,
      AMDCUID_STATUS_IPC_ERROR,
  };

  for (auto status : kStatuses) {
    const char* str = amdcuid_status_to_string(status);
    EXPECT_NE(str, nullptr) << "status code " << status << " returned nullptr";
    EXPECT_GT(strlen(str), 0u) << "status code " << status << " returned empty string";
    IF_VERB(2) { printf("  [%2d] %s\n", status, str); }
  }
}

void TestStatusString::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestStatusString::DisplayResults() const { TestBase::DisplayResults(); }
void TestStatusString::Close() {}
