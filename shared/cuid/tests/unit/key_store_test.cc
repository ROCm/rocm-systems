// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The AmdCuidKey variable as efivarfs presents it, and the keys
// amdcuid_set_hash_key() refuses.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstring>

#include "include/amd_cuid.h"
#include "src/hmac.h"

namespace {

constexpr size_t kVariableLen = 4 + kKeyVariablePayloadLen;

void Variable(uint8_t out[kVariableLen], uint8_t version, uint8_t flags, uint8_t reserved) {
  out[0] = 0x07;
  out[1] = out[2] = out[3] = 0;
  out[4] = version;
  out[5] = flags;
  out[6] = reserved;
  out[7] = 0;
  for (size_t i = 0; i < key_length; ++i) out[8 + i] = static_cast<uint8_t>(0xa0 + i);
}

}  // namespace

TEST(cuidtstUnprivileged, KeyVariableParses) {
  uint8_t data[kVariableLen];
  uint8_t key[key_length];
  bool provisioned = true;

  Variable(data, 1, 0, 0);
  ASSERT_EQ(CuidUtilities::parse_key_variable(data, sizeof(data), key, provisioned),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_FALSE(provisioned);
  EXPECT_EQ(std::memcmp(key, data + 8, key_length), 0);

  Variable(data, 1, 1, 0);
  ASSERT_EQ(CuidUtilities::parse_key_variable(data, sizeof(data), key, provisioned),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_TRUE(provisioned);
}

TEST(cuidtstUnprivileged, MalformedKeyVariableIsRefused) {
  uint8_t data[kVariableLen + 1] = {};
  uint8_t key[key_length];
  bool provisioned = false;
  const struct {
    uint8_t version, flags, reserved;
  } bad[] = {{0, 0, 0}, {2, 0, 0}, {1, 2, 0}, {1, 0x80, 0}, {1, 0, 1}};
  for (const auto& b : bad) {
    Variable(data, b.version, b.flags, b.reserved);
    EXPECT_EQ(CuidUtilities::parse_key_variable(data, kVariableLen, key, provisioned),
              AMDCUID_STATUS_KEY_ERROR)
        << int(b.version) << "/" << int(b.flags) << "/" << int(b.reserved);
  }
  Variable(data, 1, 0, 0);
  data[7] = 1;
  EXPECT_EQ(CuidUtilities::parse_key_variable(data, kVariableLen, key, provisioned),
            AMDCUID_STATUS_KEY_ERROR);
  Variable(data, 1, 0, 0);
  EXPECT_EQ(CuidUtilities::parse_key_variable(data, kVariableLen - 1, key, provisioned),
            AMDCUID_STATUS_KEY_ERROR);
  EXPECT_EQ(CuidUtilities::parse_key_variable(data, kVariableLen + 1, key, provisioned),
            AMDCUID_STATUS_KEY_ERROR);
  EXPECT_EQ(CuidUtilities::parse_key_variable(data, 4 + key_length, key, provisioned),
            AMDCUID_STATUS_KEY_ERROR);
}

TEST(cuidtstUnprivileged, BuiltKeyVariableIsProvisioned) {
  uint8_t key[key_length];
  for (size_t i = 0; i < key_length; ++i) key[i] = static_cast<uint8_t>(0x40 + 3 * i);
  uint8_t data[kVariableLen];
  CuidUtilities::build_key_variable(key, data);
  const uint8_t header[8] = {0x07, 0, 0, 0, 1, 1, 0, 0};
  EXPECT_EQ(std::memcmp(data, header, sizeof(header)), 0);

  uint8_t parsed[key_length];
  bool provisioned = false;
  ASSERT_EQ(CuidUtilities::parse_key_variable(data, sizeof(data), parsed, provisioned),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_TRUE(provisioned);
  EXPECT_EQ(std::memcmp(parsed, key, key_length), 0);
}

TEST(cuidtstUnprivileged, PublicAndTrivialKeysAreRejected) {
  uint8_t key[key_length];
  for (const uint8_t fill : {0x00, 0x41, 0xff}) {
    std::memset(key, fill, sizeof(key));
    EXPECT_TRUE(CuidUtilities::is_rejected_key(key)) << int(fill);
  }
  for (const char* constant : {"AMD-CUID-DEFAULT-SEED-v1", "AMD-CUID-TEMP-KEY-v1"}) {
    std::memset(key, 0, sizeof(key));
    std::memcpy(key, constant, std::strlen(constant));
    EXPECT_TRUE(CuidUtilities::is_rejected_key(key)) << constant;
  }
  for (size_t i = 0; i < key_length; ++i) key[i] = static_cast<uint8_t>(0xA5 ^ i);
  EXPECT_TRUE(CuidUtilities::is_rejected_key(key));
  for (size_t i = 0; i < key_length; ++i) key[i] = static_cast<uint8_t>(i);
  EXPECT_TRUE(CuidUtilities::is_rejected_key(key));

  key[0] = 0x80;
  EXPECT_FALSE(CuidUtilities::is_rejected_key(key));
  std::memset(key, 0, sizeof(key));
  std::memcpy(key, "AMD-CUID-TEMP-KEY-v1", 20);
  key[31] = 1;
  EXPECT_FALSE(CuidUtilities::is_rejected_key(key));
}

// Root is never exercised here: a key this suite passes would reach firmware.
TEST(cuidtstUnprivileged, SetHashKeyNeedsRoot) {
  if (geteuid() == 0) GTEST_SKIP() << "only an ordinary user can call it safely";
  uint8_t key[key_length];
  for (size_t i = 0; i < key_length; ++i) key[i] = static_cast<uint8_t>(0x40 + 3 * i);
  EXPECT_EQ(amdcuid_set_hash_key(key), AMDCUID_STATUS_PERMISSION_DENIED);
}
