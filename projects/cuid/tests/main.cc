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

#include "test_common.h"

// Unit tests (no root or device required)
#include "unit/file_lock_test.h"
#include "unit/id_string_test.h"
#include "unit/status_string_test.h"
#include "unit/utilities_test.h"
#include "unit/version_read_test.h"

// Functional tests (device or root required)
#include <gtest/gtest.h>
#include <unistd.h>

#include "functional/device_handles_test.h"
#include "functional/device_query_test.h"
#include "functional/device_refresh_test.h"
#include "functional/hmac_test.h"
#include "functional/reverse_lookup_test.h"

// =============================================================================
// cuidtstUnprivileged — tests that run without root
// =============================================================================

TEST(cuidtstUnprivileged, VersionRead) {
  TestVersionRead tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, StatusString) {
  TestStatusString tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, IdString) {
  TestIdString tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, Utilities) {
  TestUtilities tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, FileLockBasic) {
  TestFileLockBasic tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, FileLockRAII) {
  TestFileLockRAII tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, FileLockMultipleShared) {
  TestFileLockMultipleShared tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, FileLockExclusiveBlocks) {
  TestFileLockExclusiveBlocks tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, FileLockTimeout) {
  TestFileLockTimeout tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, FileLockTimeoutSpecialCases) {
  TestFileLockTimeoutSpecialCases tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, GetAllHandles) {
  TestGetAllHandles tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, GetHandleByBDF) {
  TestGetHandleByBDF tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, GetHandleByDevPath) {
  TestGetHandleByDevPath tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, GetHandleByFD) {
  TestGetHandleByFD tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, DeviceQuery) {
  TestDeviceQuery tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, DeviceRefresh) {
  TestDeviceRefresh tst;
  RunGenericTest(&tst);
}

// =============================================================================
// cuidtstPrivileged — tests that require root
// =============================================================================

TEST(cuidtstPrivileged, HMAC) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Requires root; run with sudo to enable.";
  }
  TestHMAC tst;
  RunGenericTest(&tst);
}

TEST(cuidtstPrivileged, ReverseSerialNumber) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Requires root; run with sudo to enable.";
  }
  TestReverseSerialNumber tst;
  RunGenericTest(&tst);
}

TEST(cuidtstPrivileged, ReverseVendorId) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Requires root; run with sudo to enable.";
  }
  TestReverseVendorId tst;
  RunGenericTest(&tst);
}

TEST(cuidtstPrivileged, ReverseDeviceId) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Requires root; run with sudo to enable.";
  }
  TestReverseDeviceId tst;
  RunGenericTest(&tst);
}

TEST(cuidtstPrivileged, ReverseRevisionId) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Requires root; run with sudo to enable.";
  }
  TestReverseRevisionId tst;
  RunGenericTest(&tst);
}

TEST(cuidtstPrivileged, ReverseUnitId) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Requires root; run with sudo to enable.";
  }
  TestReverseUnitId tst;
  RunGenericTest(&tst);
}

TEST(cuidtstPrivileged, ReverseDeviceType) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Requires root; run with sudo to enable.";
  }
  TestReverseDeviceType tst;
  RunGenericTest(&tst);
}

// =============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ProcessCmdline(&sCUIDGlvalues, argc, argv);
  return RUN_ALL_TESTS();
}
