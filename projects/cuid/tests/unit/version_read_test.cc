// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
