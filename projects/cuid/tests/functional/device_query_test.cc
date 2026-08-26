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

#include "functional/device_query_test.h"

#include <gtest/gtest.h>

#include <cstdio>

TestDeviceQuery::TestDeviceQuery() {
  SetTitle("Device Query");
  SetDescription(
      "Verify amdcuid_query_device_property returns SUCCESS and the correct "
      "output size when querying AMDCUID_QUERY_DEVICE_TYPE for each handle.");
}

void TestDeviceQuery::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping device query test.";
  }

  for (size_t i = 0; i < device_handles_.size(); ++i) {
    amdcuid_device_type_t device_type;
    uint32_t length = sizeof(device_type);
    amdcuid_status_t status = amdcuid_query_device_property(
        device_handles_[i], AMDCUID_QUERY_DEVICE_TYPE, &device_type, &length);

    CHK_ERR_ASRT(status);
    EXPECT_EQ(length, sizeof(device_type));

    IF_VERB(1) { printf("  Device [%zu] type: %u\n", i, static_cast<unsigned>(device_type)); }
  }
}
