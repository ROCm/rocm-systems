// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "unit/id_string_test.h"

#include <gtest/gtest.h>

#include <cstdio>

TestIdString::TestIdString() {
  SetTitle("ID String");
  SetDescription(
      "Verify amdcuid_id_to_string formats a known ID into the expected "
      "UUID string representation.");
}

void TestIdString::SetUp() {}

void TestIdString::Run() {
  amdcuid_id_t test_id = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8,
                          0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x0};
  const char* id_str = amdcuid_id_to_string(test_id);
  EXPECT_NE(id_str, nullptr);
  EXPECT_STREQ(id_str, "01020304-0506-0708-090a-0b0c0d0e0f00");

  IF_VERB(1) { printf("  ID string: %s\n", id_str); }
}

void TestIdString::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestIdString::DisplayResults() const { TestBase::DisplayResults(); }
void TestIdString::Close() {}
