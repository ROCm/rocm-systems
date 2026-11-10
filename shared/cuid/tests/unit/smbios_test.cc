// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The SMBIOS structure-table fallback for the System UUID, on crafted tables.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "src/cuid_platform.h"
#include "src/cuid_util.h"
#include "src/smbios_util.h"

namespace {

const uint8_t kRawUuid[16] = {0x33, 0x22, 0x11, 0x00, 0x55, 0x44, 0x77, 0x66,
                              0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};

std::vector<uint8_t> BiosInformation() {
  std::vector<uint8_t> s(0x18, 0);
  s[1] = 0x18;
  for (const char c : std::string("Vendor")) s.push_back(static_cast<uint8_t>(c));
  s.insert(s.end(), {0, 0});
  return s;
}

// A Type 1 structure whose formatted area is `formatted` octets long.
std::vector<uint8_t> SystemInformation(uint8_t formatted) {
  std::vector<uint8_t> s = {1, formatted, 0x01, 0x00, 1, 2, 0, 0};
  s.insert(s.end(), kRawUuid, kRawUuid + 16);
  s.insert(s.end(), {6, 0, 0});
  s.resize(formatted, 0);
  for (const char c : std::string("StringOne")) s.push_back(static_cast<uint8_t>(c));
  s.push_back(0);
  for (const char c : std::string("StringTwo")) s.push_back(static_cast<uint8_t>(c));
  s.insert(s.end(), {0, 0});
  return s;
}

std::vector<uint8_t> Table(std::initializer_list<std::vector<uint8_t>> structures) {
  std::vector<uint8_t> table;
  for (const auto& s : structures) table.insert(table.end(), s.begin(), s.end());
  table.insert(table.end(), {127, 4, 0x02, 0x00, 0, 0});
  return table;
}

std::string Parse(const std::vector<uint8_t>& table, uint16_t version, amdcuid_status_t want) {
  uint8_t uuid[16] = {};
  EXPECT_EQ(SmbiosUtil::parse_system_uuid(table.data(), table.size(), version, uuid), want);
  amdcuid_id_t id{};
  std::memcpy(id.bytes, uuid, 16);
  return CuidUtilities::get_cuid_as_string(&id);
}

}  // namespace

// From SMBIOS 2.6 the first three fields are little-endian, before it they
// are stored as written: Linux's product_uuid shows the table that way.
TEST(cuidtstUnprivileged, SmbiosUuidByteOrderFollowsTheVersion) {
  const auto table = Table({BiosInformation(), SystemInformation(0x1b)});
  EXPECT_EQ(Parse(table, 0x0300, AMDCUID_STATUS_SUCCESS), "00112233-4455-6677-8899-aabbccddeeff");
  EXPECT_EQ(Parse(table, 0x0206, AMDCUID_STATUS_SUCCESS), "00112233-4455-6677-8899-aabbccddeeff");
  EXPECT_EQ(Parse(table, 0x0204, AMDCUID_STATUS_SUCCESS), "33221100-5544-7766-8899-aabbccddeeff");
  // The shortest structure that holds the field.
  EXPECT_EQ(Parse(Table({SystemInformation(0x18)}), 0x0300, AMDCUID_STATUS_SUCCESS),
            "00112233-4455-6677-8899-aabbccddeeff");
}

// A Type 1 structure without the UUID field, or a table cut short, has no UUID
// rather than one made of whatever octets are there.
TEST(cuidtstUnprivileged, SmbiosWithoutAWholeUuidHasNone) {
  uint8_t uuid[16];
  Parse(Table({BiosInformation(), SystemInformation(0x08)}), 0x0300, AMDCUID_STATUS_SMBIOS_ERROR);
  auto table = Table({BiosInformation(), SystemInformation(0x1b)});
  table.resize(BiosInformation().size() + 0x8 + 2);
  Parse(table, 0x0300, AMDCUID_STATUS_SMBIOS_ERROR);
  Parse(Table({BiosInformation()}), 0x0300, AMDCUID_STATUS_SMBIOS_ERROR);
  // A whole UUID in a structure whose strings the table cuts off: Linux does
  // not read that structure.
  auto cut_strings = Table({SystemInformation(0x1b)});
  cut_strings.resize(0x1b + 4);
  Parse(cut_strings, 0x0300, AMDCUID_STATUS_SMBIOS_ERROR);
  const std::vector<uint8_t> bad_length = {1, 2, 0, 0, 0, 0};
  Parse(bad_length, 0x0300, AMDCUID_STATUS_SMBIOS_ERROR);
  EXPECT_EQ(SmbiosUtil::parse_system_uuid(nullptr, 0, 0x0300, uuid),
            AMDCUID_STATUS_INVALID_ARGUMENT);
}

TEST(cuidtstUnprivileged, SmbiosEntryPointVersion) {
  uint16_t version = 0;
  const uint8_t sm2[0x1f] = {'_', 'S', 'M', '_', 0, 0x1f, 2, 4};
  EXPECT_EQ(SmbiosUtil::parse_entry_point_version(sm2, sizeof(sm2), version),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(version, 0x0204);
  const uint8_t sm3[0x18] = {'_', 'S', 'M', '3', '_', 0, 0x18, 3, 6};
  EXPECT_EQ(SmbiosUtil::parse_entry_point_version(sm3, sizeof(sm3), version),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(version, 0x0306);
  const uint8_t old[0x1f] = {'_', 'S', 'M', '_', 0, 0x1f, 2, 0};
  EXPECT_EQ(SmbiosUtil::parse_entry_point_version(old, sizeof(old), version),
            AMDCUID_STATUS_UNSUPPORTED);
  const uint8_t junk[8] = {'_', 'D', 'M', 'I', '_', 0, 0, 0};
  EXPECT_EQ(SmbiosUtil::parse_entry_point_version(junk, sizeof(junk), version),
            AMDCUID_STATUS_SMBIOS_ERROR);
  EXPECT_EQ(SmbiosUtil::parse_entry_point_version(sm2, 4, version), AMDCUID_STATUS_SMBIOS_ERROR);
}

// Only all-zero and all-ones mean the firmware has no UUID; one that merely
// contains 00 and FF octets is a real UUID.
TEST(cuidtstUnprivileged, PlatformUuidAbsentOnlyWhenAllZeroOrAllOnes) {
  uint8_t uuid[16];
  std::memset(uuid, 0x00, sizeof(uuid));
  EXPECT_FALSE(CuidPlatform::is_present_uuid(uuid));
  std::memset(uuid, 0xFF, sizeof(uuid));
  EXPECT_FALSE(CuidPlatform::is_present_uuid(uuid));
  for (int i = 0; i < 16; ++i) uuid[i] = (i % 2 == 0) ? 0x00 : 0xFF;
  EXPECT_TRUE(CuidPlatform::is_present_uuid(uuid));
  for (int i = 0; i < 16; ++i) uuid[i] = (i % 2 == 0) ? 0xFF : 0x00;
  EXPECT_TRUE(CuidPlatform::is_present_uuid(uuid));
  std::memset(uuid, 0x00, sizeof(uuid));
  uuid[15] = 0xFF;
  EXPECT_TRUE(CuidPlatform::is_present_uuid(uuid));
  std::memset(uuid, 0xFF, sizeof(uuid));
  uuid[0] = 0x00;
  EXPECT_TRUE(CuidPlatform::is_present_uuid(uuid));
  std::memset(uuid, 0x00, sizeof(uuid));
  uuid[7] = 0x01;
  EXPECT_TRUE(CuidPlatform::is_present_uuid(uuid));
}
