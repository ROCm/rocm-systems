// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_common.h"

// Unit tests (no root or device required)
#include "unit/acpi_parser_test.h"
#include "unit/concurrency_test.h"
#include "unit/cuid_gpu_test.h"
#include "unit/file_lock_test.h"
#include "unit/gim_util_test.h"
#include "unit/id_string_test.h"
#include "unit/pci_util_test.h"
#include "unit/sha256_test.h"
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
#include "src/gim_util.h"

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

TEST(cuidtstUnprivileged, Sha256Kat) {
  TestSha256Kat tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, HmacSha256Kat) {
  TestHmacSha256Kat tst;
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

TEST(cuidtstUnprivileged, GimClientAvailability) {
  TestGimClientAvailability tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, GimParseAsicSerial) {
  TestGimParseAsicSerial tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, GimFormatBdf) {
  TestGimFormatBdf tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, PciConfigDecode) {
  TestPciConfigDecode tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, ConcurrentApi) {
  TestConcurrentApi tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, CuidGpuRenderNode) {
  TestCuidGpuRenderNode tst;
  RunGenericTest(&tst);
}

TEST(cuidtstUnprivileged, AcpiMadtParse) {
  TestAcpiMadtParse tst;
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

TEST(cuidtstPrivileged, GimDeviceEnumeration) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "Requires root; run with sudo to enable.";
  }
  if (!cuid::gim::GimClient::is_available()) {
    GTEST_SKIP() << "GIM device node not present; skipping.";
  }
  TestGimDeviceEnumeration tst;
  RunGenericTest(&tst);
}

// =============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ProcessCmdline(&sCUIDGlvalues, argc, argv);
  return RUN_ALL_TESTS();
}
