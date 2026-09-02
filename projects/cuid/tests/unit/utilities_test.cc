// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "unit/utilities_test.h"

#include <gtest/gtest.h>

#include "src/cuid_util.h"

TestUtilities::TestUtilities() {
  SetTitle("Utilities");
  SetDescription(
      "Verify CuidUtilities::remove_UUIDv8_bits correctly strips UUIDv8 "
      "overhead and handles null pointer inputs safely.");
}

void TestUtilities::SetUp() {}

void TestUtilities::Run() {
  // Roundtrip: verify that remove_UUIDv8_bits recovers the expected raw bits
  // from a canned UUIDv8-encoded amdcuid_id_t.
  {
    amdcuid_id_t id = {{0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0x8C, 0xDE, 0xBC, 0x48, 0xD1, 0x59,
                        0xE2, 0x6A, 0xF3, 0x7B}};
    const uint8_t expected[16] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                                  0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xC0};
    uint8_t out[16] = {0};
    CuidUtilities::remove_UUIDv8_bits(&id, out);

    for (int i = 0; i < 16; ++i) {
      EXPECT_EQ(out[i], expected[i]) << "Mismatch at byte " << i;
    }
  }

  // Null safety: neither call should crash or modify the output buffer.
  {
    amdcuid_id_t id = {{0}};
    uint8_t out[16] = {0xFF};

    CuidUtilities::remove_UUIDv8_bits(nullptr, out);
    EXPECT_EQ(out[0], 0xFF);

    CuidUtilities::remove_UUIDv8_bits(&id, nullptr);
  }
}

void TestUtilities::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestUtilities::DisplayResults() const { TestBase::DisplayResults(); }
void TestUtilities::Close() {}
