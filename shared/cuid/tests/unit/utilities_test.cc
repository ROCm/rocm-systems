// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "unit/utilities_test.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "src/cuid_util.h"
#include "src/hmac.h"

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
    // The last raw octet carries payload bits 120:121 in its low two bits. This
    // vector previously ended 0xC0, from the withdrawn convention that read
    // those bits into payload 126:127, which is padding.
    const uint8_t expected[16] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                                  0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0x03};
    uint8_t out[16] = {0};
    CuidUtilities::remove_UUIDv8_bits(&id, out);

    for (int i = 0; i < 16; ++i) {
      EXPECT_EQ(out[i], expected[i]) << "Mismatch at byte " << i;
    }

    amdcuid_id_t reframed = {{0}};
    CuidUtilities::add_UUIDv8_bits(out, &reframed);
    for (int i = 0; i < 16; ++i) {
      EXPECT_EQ(reframed.bytes[i], id.bytes[i]) << "Round-trip mismatch at byte " << i;
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

// A Platform CUID adopted verbatim from firmware is not a constructed UUIDv8,
// so none of its payload fields may be decoded. Reading them anyway reported a
// Platform as an NPU, or as auxiliary, depending on what the firmware wrote.
TEST(cuidtstUnprivileged, AdoptedFirmwareUuidIsNotConstructed) {
  struct {
    const char* uuid;
    bool constructed;
  } cases[] = {
      // Real-world shaped SMBIOS system UUIDs: versions 3, 0 and 4.
      {"4c4c4544-0037-3010-8055-b4c04f434332", false},
      {"03000200-0400-0500-0006-000700080009", false},
      {"aabbccdd-eeff-4011-9222-3344556677ff", false},
      // A CUID this library constructed: conformance vector P-1.
      {"d4abaad3-9b34-8c50-9800-028dcc084200", true},
  };

  for (const auto& c : cases) {
    amdcuid_id_t id{};
    ASSERT_EQ(CuidUtilities::uuid_string_to_uint8(c.uuid, id.bytes), AMDCUID_STATUS_SUCCESS)
        << c.uuid;
    EXPECT_EQ(CuidUtilities::is_constructed(&id), c.constructed) << c.uuid;
  }
}

// Payload bits 118:121 straddle the last two octets, and the primary and
// derived packers both write them. They had drifted into payload 126:127, which
// is padding. Nothing caught it because the derived packer's field is reserved
// and always zero, so the placement is exercised here with every value the
// field can one day carry.
TEST(cuidtstUnprivileged, ComponentTypeBitsLandOutsideThePadding) {
  for (uint8_t value = 0; value < 16; ++value) {
    uint8_t raw[16] = {0};
    CuidUtilities::pack_component_type_bits(value, raw);

    // Payload 118:119 in raw[14] bits 6:7, payload 120:121 in raw[15] bits 0:1.
    EXPECT_EQ((raw[14] >> 6) & 0x3, value & 0x3) << "value " << static_cast<int>(value);
    EXPECT_EQ(raw[15] & 0x3, (value >> 2) & 0x3) << "value " << static_cast<int>(value);

    // Payload 122:127 is padding and must stay zero.
    EXPECT_EQ(raw[15] & 0xFC, 0) << "value " << static_cast<int>(value);

    // remove_UUIDv8_bits() is the only reader of these bits, so a packer that
    // disagrees with it reduces the field modulo 4.
    amdcuid_id_t framed = {{0}};
    CuidUtilities::add_UUIDv8_bits(raw, &framed);
    uint8_t recovered[16] = {0};
    CuidUtilities::remove_UUIDv8_bits(&framed, recovered);
    const uint8_t field =
        static_cast<uint8_t>(((recovered[14] >> 6) & 0x3) | ((recovered[15] & 0x3) << 2));
    EXPECT_EQ(field, value) << "round trip lost bits for " << static_cast<int>(value);
  }

  // Every other bit of the payload is left alone: the field is OR'd into an
  // octet that already carries UnitID part 2 and the Auxiliary Value Identifier.
  {
    uint8_t raw[16] = {0};
    raw[14] = 0x3F;  // payload 112:117 all set
    CuidUtilities::pack_component_type_bits(0x4, raw);
    EXPECT_EQ(raw[14] & 0x3F, 0x3F);
    EXPECT_EQ(raw[15], 0x01);
  }
}

// A derived CUID and the primary it was derived from are framed by the same
// reader, so a derived packer that writes payload 118:121 anywhere else
// produces a value whose reserved field cannot be read back.
TEST(cuidtstUnprivileged, DerivedAndPrimaryAgreeOnTheLastOctets) {
  amdcuid_primary_id primary = {};
  ASSERT_EQ(CuidUtilities::generate_primary_cuid(0xD3AAABD406C5349Bull, 0x1234, 0xC1, 0x73A1,
                                                 0x1002, AMDCUID_DEVICE_TYPE_NPU, &primary, false),
            AMDCUID_STATUS_SUCCESS);

  uint8_t key[32] = {0};
  for (size_t i = 0; i < sizeof(key); ++i) key[i] = static_cast<uint8_t>(i);
  cuid_hmac hmac(key);

  amdcuid_derived_id derived = {};
  ASSERT_EQ(CuidUtilities::generate_derived_cuid(&primary, &derived, &hmac),
            AMDCUID_STATUS_SUCCESS);

  // Component Type 0x4 (NPU) survives in the primary: it needs both halves.
  EXPECT_EQ(((primary.raw_bits[14] >> 6) & 0x3) | ((primary.raw_bits[15] & 0x3) << 2), 0x4);

  // Payload 122:127 is padding in both. The derived packer used to write the
  // reserved field's low half here.
  EXPECT_EQ(primary.raw_bits[15] & 0xFC, 0);
  EXPECT_EQ(derived.raw_bits[15] & 0xFC, 0) << "derived payload wrote into the UUID padding";
}

// The HMAC message for an adopted identifier is the firmware UUID itself, all
// sixteen octets. De-framing it first drops six bits, so two platforms whose
// system UUIDs differ only in their version and variant bits would collide.
TEST(cuidtstUnprivileged, DeframingAnAdoptedUuidWouldLoseBits) {
  amdcuid_id_t id{};
  ASSERT_EQ(CuidUtilities::uuid_string_to_uint8("aabbccdd-eeff-4011-9222-3344556677ff", id.bytes),
            AMDCUID_STATUS_SUCCESS);

  uint8_t raw[16];
  CuidUtilities::remove_UUIDv8_bits(&id, raw);

  amdcuid_id_t reframed{};
  CuidUtilities::add_UUIDv8_bits(raw, &reframed);

  // Not a round trip: this is why the adopted path must not de-frame.
  EXPECT_NE(0, std::memcmp(id.bytes, reframed.bytes, sizeof(id.bytes)));
}

// A machine-id is 32 hex digits and never all zero (machine-id(5)). Anything
// else, including the all-zero ID of an unfinished image and systemd's
// "uninitialized" placeholder, is no machine-id rather than a K_app every
// such clone would share.
TEST(cuidtstUnprivileged, MachineIdMustBeWellFormedAndNonZero) {
  uint8_t id[16];
  ASSERT_TRUE(CuidUtilities::parse_machine_id("0123456789abcdef0123456789ABCDEF", id));
  EXPECT_EQ(id[0], 0x01);
  EXPECT_EQ(id[15], 0xEF);
  for (const char* bad :
       {"00000000000000000000000000000000", "uninitialized", "0123456789abcdef0123456789abcdef0",
        "0123456789abcdef0123456789abcde", "0123456789abcdef0123456789abcdef garbage",
        "0123456789abcdef0123456789abcdeg", ""}) {
    EXPECT_FALSE(CuidUtilities::parse_machine_id(bad, id)) << '"' << bad << '"';
  }
}

// /var/lib/dbus/machine-id stands in only for an /etc/machine-id that does not
// exist. One that exists but is empty, "uninitialized" or all zero belongs to
// an image whose identity is not set yet, and a D-Bus ID shipped beside it is
// shared by every clone of that image.
TEST(cuidtstUnprivileged, InvalidEtcMachineIdIsNotReplacedByDbus) {
  char tmpl[] = "/tmp/cuid_machine_id_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  ASSERT_NE(dir, nullptr);
  const std::string etc = std::string(dir) + "/etc-machine-id";
  const std::string dbus = std::string(dir) + "/dbus-machine-id";
  const auto write = [](const std::string& path, const char* text) {
    std::ofstream(path, std::ios::trunc) << text;
  };
  uint8_t id[16];

  write(dbus, "fedcba9876543210fedcba9876543210\n");
  EXPECT_TRUE(CuidUtilities::read_machine_id(etc, dbus, id)) << "no /etc/machine-id";
  EXPECT_EQ(id[0], 0xFE);

  write(etc, "0123456789abcdef0123456789abcdef\n");
  EXPECT_TRUE(CuidUtilities::read_machine_id(etc, dbus, id));
  EXPECT_EQ(id[0], 0x01);

  for (const char* invalid : {"", "uninitialized\n", "00000000000000000000000000000000\n"}) {
    write(etc, invalid);
    EXPECT_FALSE(CuidUtilities::read_machine_id(etc, dbus, id)) << '"' << invalid << '"';
  }

  unlink(etc.c_str());
  unlink(dbus.c_str());
  EXPECT_FALSE(CuidUtilities::read_machine_id(etc, dbus, id));
  rmdir(dir);
}
