// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include "rocjitsu/vm/amdgpu/pci/ip_discovery_profile.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/// @brief A device with the blocks a driver needs before it will go further.
rocjitsu::IpDiscoverySpec gfx1250_spec() { return rocjitsu::gfx1250_discovery_spec(); }

/// @brief Build a table, insisting it could be built at all.
///
/// @details Failure throws rather than recording a non-fatal expectation,
/// because every caller reads the returned table by offset: a build that failed
/// would return an empty one and the test would index past its end rather than
/// stop at the real problem. ASSERT_ is unavailable here, since this returns a
/// value.
std::vector<std::byte> build(const rocjitsu::IpDiscoverySpec &spec) {
  const rocjitsu::IpDiscoveryBuild built = rocjitsu::build_ip_discovery_table(spec);
  if (!built.ok()) {
    throw std::runtime_error("cannot build a discovery table: " + built.problem);
  }
  return built.table;
}

uint16_t read16(const std::vector<std::byte> &table, std::size_t at) {
  return static_cast<uint16_t>(std::to_integer<uint8_t>(table[at]) |
                               (std::to_integer<uint8_t>(table[at + 1]) << 8));
}

void write16(std::vector<std::byte> &table, std::size_t at, uint16_t value) {
  table[at] = static_cast<std::byte>(value & 0xff);
  table[at + 1] = static_cast<std::byte>((value >> 8) & 0xff);
}

/// @brief Offset of the block list, as the table list records it.
std::size_t ip_table_offset(const std::vector<std::byte> &table) { return read16(table, 12); }

uint16_t byte_sum(const std::vector<std::byte> &table, std::size_t at, std::size_t length) {
  uint16_t sum = 0;
  for (std::size_t i = at; i < at + length; ++i) {
    sum = static_cast<uint16_t>(sum + std::to_integer<uint8_t>(table[i]));
  }
  return sum;
}

/// @brief Restore the block list's checksum after tampering with its contents.
///
/// @details Without this a test that corrupts something structural inside the
/// block list is rejected for a failed checksum instead, which looks like a
/// pass and proves nothing about the check it meant to exercise.
void reseal_block_list(std::vector<std::byte> &table) {
  const std::size_t ip_table = ip_table_offset(table);
  write16(table, 12 + 2, byte_sum(table, ip_table, read16(table, ip_table + 6)));
}

/// @brief Restore the binary checksum, which covers everything after it.
void reseal_binary(std::vector<std::byte> &table) {
  write16(table, 8, byte_sum(table, 10, table.size() - 10));
}

TEST(IpDiscoveryTable, BuildsATableTheDriverWouldAccept) {
  const std::vector<std::byte> table = build(gfx1250_spec());

  const rocjitsu::IpDiscoveryValidation checked = rocjitsu::validate_ip_discovery_table(table);
  EXPECT_TRUE(checked.valid) << checked.problem;
}

// Register bases are segments, and the driver picks between them with each
// register's `_BASE_IDX` (`SOC15_REG_OFFSET` indexes this very list). A block
// publishing one base therefore answers only for its index-0 registers, and
// gfx1250 puts more registers at index 1 than at index 0. Nothing reports the
// mismatch: the access resolves against the base's zero high dword and lands at
// a raw offset, which reads as a broken register model rather than a short
// table. This is what left GMC initializing against a memory size of zero.
TEST(IpDiscoveryProfile, PublishesEverySegmentTheDriverIndexesInto) {
  // Exact counts, not a floor: these are what a real GFX12 part publishes, and
  // the failure that matters is a list losing its *tail*. NBIO reaches
  // _BASE_IDX 5, so a truncation to two entries would break it while sailing
  // past any lower bound.
  const std::map<rocjitsu::IpHardwareId, std::size_t> expected = {
      {rocjitsu::IpHardwareId::Gc, 4},    {rocjitsu::IpHardwareId::Mp0, 15},
      {rocjitsu::IpHardwareId::Mp1, 15},  {rocjitsu::IpHardwareId::OssSys, 2},
      {rocjitsu::IpHardwareId::Nbif, 6},  {rocjitsu::IpHardwareId::Hdp, 2},
      {rocjitsu::IpHardwareId::MmHub, 2}, {rocjitsu::IpHardwareId::AtHub, 2},
      {rocjitsu::IpHardwareId::Sdma0, 4},
  };

  const std::vector<rocjitsu::IpBlock> &blocks = gfx1250_spec().blocks;

  // The whole contract, not a property of whatever happens to be present. A
  // per-present-block loop still passes if a block disappears, is duplicated,
  // or changes version -- exactly the edits that silently change which driver
  // support the guest binds and where it resolves that block's registers.
  //
  // Every base below is what a real GPU of this family published through
  // /sys/.../ip_discovery/die/0/<ip>/<inst>/base_addr. NBIF segment 0 really is
  // zero: its registers start at raw offset 0.
  struct Expected {
    rocjitsu::IpHardwareId id;
    uint16_t major;
    uint16_t minor;
    uint16_t revision;
    std::vector<uint64_t> bases;
  };
  const std::vector<Expected> contract = {
      {rocjitsu::IpHardwareId::Gc, 12, 1, 0, {0x00001260, 0x0000A000, 0x0001C000, 0x02402C00}},
      {rocjitsu::IpHardwareId::Mp0,
       15,
       0,
       8,
       {0x00016000, 0x00016200, 0x0001CE00, 0x00DC0000, 0x00E00000, 0x00E40000, 0x00E80000,
        0x00EC0000, 0x00F00000, 0x02400400, 0x0243FC00, 0x0244D400, 0x03200000, 0x03240000,
        0x03280000}},
      {rocjitsu::IpHardwareId::Mp1,
       15,
       0,
       8,
       {0x00016000, 0x00016200, 0x0001CE00, 0x00DC0000, 0x00E00000, 0x00E40000, 0x00E80000,
        0x00EC0000, 0x00F00000, 0x02400400, 0x0243FC00, 0x0244D400, 0x03200000, 0x03240000,
        0x03280000}},
      {rocjitsu::IpHardwareId::OssSys, 7, 1, 0, {0x000010A0, 0x0240A000}},
      {rocjitsu::IpHardwareId::Nbif,
       7,
       11,
       0,
       {0x00000000, 0x00000014, 0x00000D20, 0x00010400, 0x0241B000, 0x04040000}},
      // 7.0.0 deliberately: the driver's HDP switch has no arm for 7.1.0, which
      // would leave hdp.funcs null and the flush path silently doing nothing.
      {rocjitsu::IpHardwareId::Hdp, 7, 0, 0, {0x00000F20, 0x0240A400}},
      {rocjitsu::IpHardwareId::MmHub, 4, 1, 0, {0x0001A000, 0x02408800}},
      {rocjitsu::IpHardwareId::AtHub, 4, 1, 0, {0x00000C00, 0x02408C00}},
      // GC's segments, which this family shares between the two.
      {rocjitsu::IpHardwareId::Sdma0, 7, 1, 0, {0x00001260, 0x0000A000, 0x0001C000, 0x02402C00}},
  };

  ASSERT_EQ(blocks.size(), contract.size())
      << "the profile publishes a different set of blocks than this contract names";
  for (std::size_t i = 0; i < contract.size(); ++i) {
    const rocjitsu::IpBlock &block = blocks[i];
    const Expected &want = contract[i];
    EXPECT_EQ(block.hardware_id, want.id) << "block " << i;
    EXPECT_EQ(block.instance, 0u) << "block " << i << ": one instance of each is modelled";
    EXPECT_EQ(block.major, want.major) << "block " << i;
    EXPECT_EQ(block.minor, want.minor) << "block " << i;
    EXPECT_EQ(block.revision, want.revision) << "block " << i;
    EXPECT_EQ(block.register_bases, want.bases)
        << "block " << i << ": a changed base resolves this block's registers elsewhere";
  }
}

TEST(IpDiscoveryTable, StartsWithTheSignatureTheDriverLooksFor) {
  const std::vector<std::byte> table = build(gfx1250_spec());

  ASSERT_GE(table.size(), 4u);
  const auto byte_at = [&table](std::size_t index) {
    return std::to_integer<unsigned>(table[index]);
  };
  // 0x28211407, little endian, which the driver reports as an invalid signature
  // when it is absent.
  EXPECT_EQ(byte_at(0), 0x07u);
  EXPECT_EQ(byte_at(1), 0x14u);
  EXPECT_EQ(byte_at(2), 0x21u);
  EXPECT_EQ(byte_at(3), 0x28u);
}

// The driver checksums the binary and the block list separately, so a table
// corrupted anywhere has to be caught here rather than in a guest's dmesg.
TEST(IpDiscoveryTable, RejectsACorruptedBlockList) {
  std::vector<std::byte> table = build(gfx1250_spec());
  ASSERT_TRUE(rocjitsu::validate_ip_discovery_table(table).valid);

  table[table.size() - 1] = static_cast<std::byte>(std::to_integer<uint8_t>(table.back()) ^ 0xff);

  EXPECT_FALSE(rocjitsu::validate_ip_discovery_table(table).valid);
}

TEST(IpDiscoveryTable, RejectsATableWithTheWrongSignature) {
  std::vector<std::byte> table = build(gfx1250_spec());
  table[0] = std::byte{0};

  const rocjitsu::IpDiscoveryValidation checked = rocjitsu::validate_ip_discovery_table(table);

  EXPECT_FALSE(checked.valid);
  EXPECT_NE(checked.problem.find("signature"), std::string::npos);
}

TEST(IpDiscoveryTable, RejectsATruncatedTable) {
  std::vector<std::byte> table = build(gfx1250_spec());
  table.resize(table.size() - 8);

  EXPECT_FALSE(rocjitsu::validate_ip_discovery_table(table).valid);
}

// Every length inside the table is the thing under test, so none of them may
// bound a read of it. Each of these once walked off the end of the buffer.
TEST(IpDiscoveryTable, RejectsInputTooShortToHoldAHeader) {
  for (std::size_t size = 0; size < 60; ++size) {
    const std::vector<std::byte> table(size, std::byte{0});
    EXPECT_FALSE(rocjitsu::validate_ip_discovery_table(table).valid) << "at " << size << " bytes";
  }
}

TEST(IpDiscoveryTable, RejectsABlockListShorterThanItsOwnHeader) {
  std::vector<std::byte> table = build(gfx1250_spec());
  // Point the block list one byte before the end, so the range check on its
  // recorded size passes while its fixed header does not fit.
  write16(table, 12, static_cast<uint16_t>(table.size() - 1));
  write16(table, 12 + 4, 1);
  reseal_binary(table);

  EXPECT_FALSE(rocjitsu::validate_ip_discovery_table(table).valid);
}

TEST(IpDiscoveryTable, RejectsMoreDiesThanTheHeaderHasRoomFor) {
  std::vector<std::byte> table = build(gfx1250_spec());
  write16(table, ip_table_offset(table) + 12, 17);
  reseal_block_list(table);
  reseal_binary(table);

  const rocjitsu::IpDiscoveryValidation checked = rocjitsu::validate_ip_discovery_table(table);
  EXPECT_FALSE(checked.valid);
  EXPECT_NE(checked.problem.find("dies"), std::string::npos);
}

TEST(IpDiscoveryTable, RejectsADiePointedPastTheEnd) {
  std::vector<std::byte> table = build(gfx1250_spec());
  write16(table, ip_table_offset(table) + 14 + 2, static_cast<uint16_t>(table.size() - 2));
  reseal_block_list(table);
  reseal_binary(table);

  EXPECT_FALSE(rocjitsu::validate_ip_discovery_table(table).valid);
}

// The bug that shipped: a die offset measured from the table rather than from
// the binary still lands inside the table, so only reading what it points at
// catches it.
TEST(IpDiscoveryTable, RejectsADieOffsetMeasuredFromTheWrongPlace) {
  std::vector<std::byte> table = build(gfx1250_spec());
  const std::size_t ip_table = ip_table_offset(table);
  const uint16_t correct = read16(table, ip_table + 14 + 2);
  write16(table, ip_table + 14 + 2, static_cast<uint16_t>(correct - ip_table));
  reseal_block_list(table);
  reseal_binary(table);

  EXPECT_FALSE(rocjitsu::validate_ip_discovery_table(table).valid);
}

TEST(IpDiscoveryTable, RejectsDisagreeingEmbeddedAndOuterSizes) {
  std::vector<std::byte> table = build(gfx1250_spec());
  const std::size_t ip_table = ip_table_offset(table);
  write16(table, ip_table + 6, static_cast<uint16_t>(read16(table, ip_table + 6) - 8));
  reseal_binary(table);

  const rocjitsu::IpDiscoveryValidation checked = rocjitsu::validate_ip_discovery_table(table);
  EXPECT_FALSE(checked.valid);
  EXPECT_NE(checked.problem.find("bytes"), std::string::npos);
}

TEST(IpDiscoveryTable, RejectsBlockRecordsThatRunPastTheEnd) {
  std::vector<std::byte> table = build(gfx1250_spec());
  const std::size_t ip_table = ip_table_offset(table);
  const uint16_t die = read16(table, ip_table + 14 + 2);
  // Claim far more blocks than the records behind the die can supply.
  write16(table, die + 2, 4096);
  reseal_block_list(table);
  reseal_binary(table);

  EXPECT_FALSE(rocjitsu::validate_ip_discovery_table(table).valid);
}

TEST(IpDiscoveryTable, RejectsATableWithNoGraphicsBlock) {
  rocjitsu::IpDiscoverySpec spec;
  spec.blocks.push_back({.hardware_id = rocjitsu::IpHardwareId::Mp0,
                         .instance = 0,
                         .major = 15,
                         .minor = 0,
                         .revision = 8,
                         .register_bases = {0x00016000}});

  const rocjitsu::IpDiscoveryValidation checked =
      rocjitsu::validate_ip_discovery_table(build(spec));

  EXPECT_FALSE(checked.valid);
  EXPECT_NE(checked.problem.find("graphics"), std::string::npos);
}

TEST(IpDiscoveryTable, RejectsACorruptedHarvestTable) {
  std::vector<std::byte> table = build(gfx1250_spec());
  const uint16_t harvest = read16(table, 12 + 2 * 8);
  ASSERT_NE(harvest, 0u);
  table[harvest] = static_cast<std::byte>(std::to_integer<uint8_t>(table[harvest]) ^ 0xff);
  reseal_binary(table);

  const rocjitsu::IpDiscoveryValidation checked = rocjitsu::validate_ip_discovery_table(table);
  EXPECT_FALSE(checked.valid);
  EXPECT_NE(checked.problem.find("harvest"), std::string::npos);
}

// Walks to the block count the way the driver does, from the start of the
// binary rather than from the table holding the die list. Written out here
// instead of asking the validator, because a table whose die offsets are
// measured from the wrong place still validates against any checker that makes
// the same mistake — which is how this shipped as "9 blocks" to a driver that
// read zero of them and refused the device with no diagnostic.
TEST(IpDiscoveryTable, PutsItsDieWhereTheDriverLooksForIt) {
  const rocjitsu::IpDiscoverySpec spec = gfx1250_spec();
  const std::vector<std::byte> table = build(spec);

  const unsigned ip_table = read16(table, 12);           // table_list[0].offset
  const unsigned die = read16(table, ip_table + 14 + 2); // die_info[0].die_offset

  ASSERT_LT(die + 4u, table.size());
  EXPECT_EQ(read16(table, die), 0u) << "die 0 does not identify itself as die 0";
  EXPECT_EQ(read16(table, die + 2), spec.blocks.size());
}

TEST(IpDiscoveryTable, DescribesEveryBlockItWasGiven) {
  rocjitsu::IpDiscoverySpec spec = gfx1250_spec();
  const std::vector<std::byte> smaller = build(spec);

  spec.blocks.push_back({.hardware_id = rocjitsu::IpHardwareId::Sdma0,
                         .instance = 1,
                         .major = 7,
                         .minor = 1,
                         .revision = 0,
                         .register_bases = {0x00001260}});
  const std::vector<std::byte> larger = build(spec);

  EXPECT_GT(larger.size(), smaller.size());
  EXPECT_TRUE(rocjitsu::validate_ip_discovery_table(larger).valid);
}

// The format records a base count in one byte, so a block with more bases than
// that would serialize a count meaning something else entirely.
TEST(IpDiscoveryTable, RefusesToBuildABlockWithMoreBasesThanTheFormatHolds) {
  rocjitsu::IpDiscoverySpec spec = gfx1250_spec();
  spec.blocks.push_back({.hardware_id = rocjitsu::IpHardwareId::Hdp,
                         .instance = 1,
                         .major = 7,
                         .minor = 1,
                         .revision = 0,
                         .register_bases = std::vector<uint64_t>(256, 0x1000)});

  const rocjitsu::IpDiscoveryBuild built = rocjitsu::build_ip_discovery_table(spec);

  EXPECT_FALSE(built.ok());
  EXPECT_TRUE(built.table.empty());
}

// The advertised graphics instance count becomes the guest's XCC mask, so it is
// a deliberate choice rather than an incidental one.
TEST(IpDiscoveryProfile, AdvertisesTheRequestedGraphicsInstances) {
  const rocjitsu::IpDiscoverySpec spec =
      rocjitsu::gfx1250_discovery_spec({.graphics_instances = 4});

  int graphics = 0;
  for (const rocjitsu::IpBlock &block : spec.blocks) {
    if (block.hardware_id == rocjitsu::IpHardwareId::Gc) {
      ++graphics;
    }
  }

  EXPECT_EQ(graphics, 4);
  EXPECT_TRUE(rocjitsu::validate_ip_discovery_table(build(spec)).valid);
}

} // namespace
