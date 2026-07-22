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

#include "functional/device_refresh_test.h"

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>

TestDeviceRefresh::TestDeviceRefresh() {
  SetTitle("Device Refresh");
  SetDescription(
      "Verify amdcuid_refresh succeeds at any privilege level, that the "
      "unprivileged CUID file is always written, that the privileged file is "
      "additionally written when running as root, and that device handles "
      "remain obtainable after a refresh.");
}

void TestDeviceRefresh::Run() {
  bool is_root = (geteuid() == 0);

  amdcuid_status_t status = amdcuid_refresh();
  EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

  IF_VERB(1) {
    printf("  amdcuid_refresh status: %s (running as %s)\n", amdcuid_status_to_string(status),
           is_root ? "root" : "non-root");
  }

  // The unprivileged CUID file must exist after any successful refresh.
  struct stat st;
  EXPECT_EQ(stat("/tmp/cuid", &st), 0)
      << "Unprivileged CUID file /tmp/cuid not found after refresh";

  IF_VERB(1) { printf("  /tmp/cuid present: yes\n"); }

  // The privileged CUID file is only written when running as root.
  if (is_root) {
    EXPECT_EQ(stat("/tmp/priv_cuid", &st), 0)
        << "Privileged CUID file /tmp/priv_cuid not found after root refresh";
    IF_VERB(1) { printf("  /tmp/priv_cuid present: yes\n"); }
  }

  // Device handles must still be obtainable after the refresh repopulates the
  // internal registry.
  uint32_t count = 0;
  status = amdcuid_get_all_handles(nullptr, &count);
  if (status == AMDCUID_STATUS_UNSUPPORTED) {
    GTEST_SKIP() << "No supported devices found after refresh";
  }
  ASSERT_TRUE(status == AMDCUID_STATUS_INSUFFICIENT_SIZE || status == AMDCUID_STATUS_SUCCESS);
  ASSERT_GT(count, 0u) << "No devices found after refresh";

  std::vector<amdcuid_id_t> handles(count);
  status = amdcuid_get_all_handles(handles.data(), &count);
  EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

  IF_VERB(1) { printf("  Handles after refresh: %u\n", count); }
}
