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

#include "test_base.h"

#include <gtest/gtest.h>

#include <cstdio>

TestBase::TestBase() = default;

void TestBase::SetUp() {
  uint32_t count = 0;
  amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);

  if (status == AMDCUID_STATUS_UNSUPPORTED) {
    device_handles_.clear();
    return;
  }
  if (status != AMDCUID_STATUS_INSUFFICIENT_SIZE && status != AMDCUID_STATUS_SUCCESS) {
    IF_VERB(1) {
      printf("  SetUp: amdcuid_get_all_handles (count query) returned %s\n",
             amdcuid_status_to_string(status));
    }
    setup_failed_ = true;
    return;
  }

  device_handles_.resize(count);
  status = amdcuid_get_all_handles(device_handles_.data(), &count);

  if (status != AMDCUID_STATUS_SUCCESS) {
    IF_VERB(1) {
      printf("  SetUp: amdcuid_get_all_handles returned %s\n", amdcuid_status_to_string(status));
    }
    setup_failed_ = true;
    device_handles_.clear();
    return;
  }

  device_handles_.resize(count);

  IF_VERB(1) { printf("  SetUp: found %u device(s)\n", count); }
}

void TestBase::Close() {}

void TestBase::DisplayTestInfo() {
  IF_VERB(1) {
    printf("\n** Test: %s\n", title_.c_str());
    if (!description_.empty()) {
      printf("   %s\n", description_.c_str());
    }
  }
}

void TestBase::DisplayResults() const {
  IF_VERB(1) { printf("** %s: done\n", title_.c_str()); }
}

void RunGenericTest(TestBase* test) {
  test->DisplayTestInfo();
  test->SetUp();
  if (!test->SetupFailed()) {
    test->Run();
  }
  test->DisplayResults();
  test->Close();
}
