// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "functional/device_refresh_test.h"

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>

#include "src/cuid_util.h"

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
  //
  // Asked for by accessor, never by literal path: the store moved off /tmp with
  // the local-privilege-escalation fix, and a test naming the old path passes
  // on a stale /tmp/cuid without the refresh having written anything.
  const std::string& unpriv_path = CuidUtilities::cuid_file();
  const std::string& priv_path = CuidUtilities::priv_cuid_file();

  struct stat st;
  EXPECT_EQ(stat(unpriv_path.c_str(), &st), 0)
      << "Unprivileged CUID file " << unpriv_path << " not found after refresh";

  IF_VERB(1) { printf("  %s present: yes\n", unpriv_path.c_str()); }

  // The privileged CUID file is only written when running as root.
  if (is_root) {
    EXPECT_EQ(stat(priv_path.c_str(), &st), 0)
        << "Privileged CUID file " << priv_path << " not found after root refresh";
    IF_VERB(1) { printf("  %s present: yes\n", priv_path.c_str()); }
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
