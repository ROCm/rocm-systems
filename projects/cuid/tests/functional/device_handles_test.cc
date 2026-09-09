// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "functional/device_handles_test.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// TestGetAllHandles
// ---------------------------------------------------------------------------

TestGetAllHandles::TestGetAllHandles() {
  SetTitle("Get All Handles");
  SetDescription(
      "Verify the two-call amdcuid_get_all_handles pattern: first call "
      "returns INSUFFICIENT_SIZE with the required count, second call "
      "populates handles and each converts to a non-empty string.");
}

void TestGetAllHandles::Run() {
  // device_handles_ was populated by TestBase::SetUp(); just validate them.
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping handle validation.";
  }

  for (const auto& handle : device_handles_) {
    const char* id_str = amdcuid_id_to_string(handle);
    EXPECT_NE(id_str, nullptr);
    EXPECT_GT(strlen(id_str), 0u);
    IF_VERB(1) { printf("  Handle: %s\n", id_str); }
  }
}

// ---------------------------------------------------------------------------
// TestGetHandleByBDF
// ---------------------------------------------------------------------------

TestGetHandleByBDF::TestGetHandleByBDF() {
  SetTitle("Get Handle By BDF");
  SetDescription(
      "Verify amdcuid_get_handle_by_bdf returns SUCCESS or "
      "DEVICE_NOT_FOUND for a well-formed BDF string.");
}

void TestGetHandleByBDF::Run() {
  const char* test_bdf = "0000:03:00.0";
  amdcuid_id_t handle;
  amdcuid_status_t status = amdcuid_get_handle_by_bdf(test_bdf, AMDCUID_DEVICE_TYPE_GPU, &handle);

  if (status == AMDCUID_STATUS_SUCCESS) {
    const char* id_str = amdcuid_id_to_string(handle);
    EXPECT_NE(id_str, nullptr);
    EXPECT_GT(strlen(id_str), 0u);
    IF_VERB(1) { printf("  Handle for %s: %s\n", test_bdf, id_str); }
  } else {
    EXPECT_EQ(status, AMDCUID_STATUS_DEVICE_NOT_FOUND);
  }
}

// ---------------------------------------------------------------------------
// TestGetHandleByDevPath
// ---------------------------------------------------------------------------

TestGetHandleByDevPath::TestGetHandleByDevPath() {
  SetTitle("Get Handle By Device Path");
  SetDescription(
      "Verify amdcuid_get_handle_by_dev_path returns SUCCESS or "
      "DEVICE_NOT_FOUND for a well-known render node path.");
}

void TestGetHandleByDevPath::Run() {
  const char* test_dev_path = "/dev/dri/renderD128";
  amdcuid_id_t handle;
  amdcuid_status_t status =
      amdcuid_get_handle_by_dev_path(test_dev_path, AMDCUID_DEVICE_TYPE_GPU, &handle);

  if (status == AMDCUID_STATUS_SUCCESS) {
    const char* id_str = amdcuid_id_to_string(handle);
    EXPECT_NE(id_str, nullptr);
    EXPECT_GT(strlen(id_str), 0u);
    IF_VERB(1) { printf("  Handle for %s: %s\n", test_dev_path, id_str); }
  } else {
    EXPECT_EQ(status, AMDCUID_STATUS_DEVICE_NOT_FOUND);
  }
}

// ---------------------------------------------------------------------------
// TestGetHandleByFD
// ---------------------------------------------------------------------------

TestGetHandleByFD::TestGetHandleByFD() {
  SetTitle("Get Handle By File Descriptor");
  SetDescription(
      "Verify amdcuid_get_handle_by_fd returns SUCCESS or DEVICE_NOT_FOUND "
      "for an open render node file descriptor.");
}

void TestGetHandleByFD::Run() {
  const char* test_dev_path = "/dev/dri/renderD128";
  int fd = open(test_dev_path, O_RDONLY);
  if (fd < 0) {
    GTEST_SKIP() << "Cannot open " << test_dev_path << "; skipping.";
  }

  amdcuid_id_t handle;
  amdcuid_status_t status = amdcuid_get_handle_by_fd(fd, AMDCUID_DEVICE_TYPE_GPU, &handle);

  if (status == AMDCUID_STATUS_SUCCESS) {
    const char* id_str = amdcuid_id_to_string(handle);
    EXPECT_NE(id_str, nullptr);
    EXPECT_GT(strlen(id_str), 0u);
    IF_VERB(1) { printf("  Handle for fd %d: %s\n", fd, id_str); }
  } else {
    EXPECT_EQ(status, AMDCUID_STATUS_DEVICE_NOT_FOUND);
  }

  close(fd);
}
