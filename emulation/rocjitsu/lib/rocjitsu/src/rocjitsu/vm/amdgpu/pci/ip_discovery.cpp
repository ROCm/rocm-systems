// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <numeric>
#include <optional>
#include <span>

namespace rocjitsu {
namespace {

/// @brief Marks the start of a discovery binary.
constexpr uint32_t kBinarySignature = 0x28211407;

/// @brief Marks the start of the table listing the device's blocks.
constexpr uint32_t kTableSignature = 0x53445049;

/// @brief Tables the binary header can point at.
constexpr std::size_t kTableCount = 6;

/// @brief Index of the block list within the table list.
constexpr std::size_t kIpDiscoveryTableIndex = 0;

/// @brief Index of the table saying which blocks are disabled in this part.
constexpr std::size_t kHarvestTableIndex = 2;

/// @brief Marks the start of the harvest table.
constexpr uint32_t kHarvestSignature = 0x56524148;

/// @brief Entries the harvest table always carries, used or not.
constexpr std::size_t kHarvestEntries = 32;

/// @brief Size of the harvest table: a signature, a version, and the entries.
constexpr std::size_t kHarvestTableSize = 8 + kHarvestEntries * 4;

/// @brief Table version that says base addresses are 64 bits wide.
constexpr uint16_t kIpDiscoveryVersion = 4;

/// @brief Dies the header can describe. The field is fixed size whether or not
/// they are used.
constexpr std::size_t kMaxDies = 16;

/// @brief Instances per block the driver has room for, its HWIP_MAX_INSTANCE.
constexpr uint16_t kMaxInstances = 48;

/// @brief One past the largest hardware id the driver keeps, its HW_ID_MAX.
constexpr uint16_t kMaxHardwareId = 300;

/// @brief Offsets within the binary header, which the driver reads as a packed
/// structure. Spelled out rather than mirrored as a C++ struct, because the
/// generator's job is to produce exactly these bytes.
constexpr std::size_t kBinaryChecksumOffset = 8;
constexpr std::size_t kBinarySizeOffset = 10;
constexpr std::size_t kTableListOffset = 12;
constexpr std::size_t kTableInfoSize = 8;
constexpr std::size_t kBinaryHeaderSize = kTableListOffset + kTableCount * kTableInfoSize;

/// @brief Offsets within the table that lists the blocks.
constexpr std::size_t kTableSizeOffset = 6;
constexpr std::size_t kTableNumDiesOffset = 12;
constexpr std::size_t kTableDieInfoOffset = 14;
constexpr std::size_t kTableHeaderSize = kTableDieInfoOffset + kMaxDies * 4 + 2;

void put16(std::vector<std::byte> &into, std::size_t at, uint16_t value) {
  into[at] = static_cast<std::byte>(value & 0xff);
  into[at + 1] = static_cast<std::byte>((value >> 8) & 0xff);
}

void put32(std::vector<std::byte> &into, std::size_t at, uint32_t value) {
  put16(into, at, static_cast<uint16_t>(value & 0xffff));
  put16(into, at + 2, static_cast<uint16_t>((value >> 16) & 0xffff));
}

/// @brief The driver's checksum: a plain sum of bytes, truncated to 16 bits.
uint16_t byte_sum(std::span<const std::byte> bytes) {
  return std::accumulate(bytes.begin(), bytes.end(), uint16_t{0},
                         [](uint16_t total, std::byte value) {
                           return static_cast<uint16_t>(total + std::to_integer<uint8_t>(value));
                         });
}

/// @brief Little-endian reads that refuse to run off the end of the input.
///
/// @details A table under validation may be arbitrary bytes, and every length
/// written inside it is exactly what we are trying to check, so nothing inside
/// the table is trusted to bound a read of the table.
class Reader {
public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  /// @returns Whether @p length bytes at @p at lie inside the input.
  [[nodiscard]] bool covers(std::size_t at, std::size_t length) const {
    return at <= bytes_.size() && length <= bytes_.size() - at;
  }

  [[nodiscard]] std::optional<uint8_t> u8(std::size_t at) const {
    if (!covers(at, 1)) {
      return std::nullopt;
    }
    return std::to_integer<uint8_t>(bytes_[at]);
  }

  [[nodiscard]] std::optional<uint16_t> u16(std::size_t at) const {
    if (!covers(at, 2)) {
      return std::nullopt;
    }
    return static_cast<uint16_t>(std::to_integer<uint8_t>(bytes_[at]) |
                                 (std::to_integer<uint8_t>(bytes_[at + 1]) << 8));
  }

  [[nodiscard]] std::optional<uint32_t> u32(std::size_t at) const {
    const std::optional<uint16_t> low = u16(at);
    const std::optional<uint16_t> high = u16(at + 2);
    if (!low || !high) {
      return std::nullopt;
    }
    return static_cast<uint32_t>(*low) | (static_cast<uint32_t>(*high) << 16);
  }

  /// @returns The byte sum of a range, or nothing if it is not all present.
  [[nodiscard]] std::optional<uint16_t> checksum(std::size_t at, std::size_t length) const {
    if (!covers(at, length)) {
      return std::nullopt;
    }
    return byte_sum(bytes_.subspan(at, length));
  }

private:
  std::span<const std::byte> bytes_;
};

/// @brief Largest value each narrow field of the format can carry.
constexpr std::size_t kMaxBasesPerBlock = 0xff;
constexpr std::size_t kMaxBlocksPerDie = 0xffff;
constexpr std::size_t kMaxTableBytes = 0xffff;

} // namespace

IpDiscoveryBuild build_ip_discovery_table(const IpDiscoverySpec &spec) {
  if (spec.blocks.size() > kMaxBlocksPerDie) {
    return {.table = {},
            .problem = std::format("{} blocks, more than the {} a die can list", spec.blocks.size(),
                                   kMaxBlocksPerDie)};
  }
  for (const IpBlock &block : spec.blocks) {
    if (block.register_bases.size() > kMaxBasesPerBlock) {
      return {.table = {},
              .problem = std::format("a block declares {} register bases, more than the {} its "
                                     "count field can express",
                                     block.register_bases.size(), kMaxBasesPerBlock)};
    }
  }

  // One die carrying every block. Real parts spread blocks across dies, but the
  // driver reads instance numbers rather than dies to tell copies apart, and a
  // single die keeps the generated table's shape obvious.
  std::vector<std::byte> ip_table(kTableHeaderSize + 4, std::byte{0});
  put32(ip_table, 0, kTableSignature);
  put16(ip_table, 4, kIpDiscoveryVersion);
  put32(ip_table, 8, 0);
  put16(ip_table, kTableNumDiesOffset, 1);
  put16(ip_table, kTableDieInfoOffset, 0); // die id
  // A die's offset is measured from the start of the binary, not from the table
  // that holds it, even though that table's own offset is what led the driver
  // here. Getting this wrong points the driver at the binary header, where it
  // reads a die of zero blocks and rejects the device with no diagnostic.
  put16(ip_table, kTableDieInfoOffset + 2,
        static_cast<uint16_t>(kBinaryHeaderSize + kTableHeaderSize));
  // Base addresses are 64 bits wide, which the flag immediately after the die
  // list declares.
  ip_table[kTableHeaderSize - 2] = std::byte{1};

  put16(ip_table, kTableHeaderSize, 0); // die id
  put16(ip_table, kTableHeaderSize + 2, static_cast<uint16_t>(spec.blocks.size()));

  for (const IpBlock &block : spec.blocks) {
    const std::size_t at = ip_table.size();
    ip_table.resize(at + 8 + block.register_bases.size() * sizeof(uint64_t), std::byte{0});
    put16(ip_table, at, static_cast<uint16_t>(block.hardware_id));
    ip_table[at + 2] = static_cast<std::byte>(block.instance);
    ip_table[at + 3] = static_cast<std::byte>(block.register_bases.size());
    ip_table[at + 4] = static_cast<std::byte>(block.major);
    ip_table[at + 5] = static_cast<std::byte>(block.minor);
    ip_table[at + 6] = static_cast<std::byte>(block.revision);
    ip_table[at + 7] = std::byte{0}; // sub-revision and variant
    for (std::size_t i = 0; i < block.register_bases.size(); ++i) {
      const uint64_t base = block.register_bases[i];
      put32(ip_table, at + 8 + i * sizeof(uint64_t), static_cast<uint32_t>(base));
      put32(ip_table, at + 12 + i * sizeof(uint64_t), static_cast<uint32_t>(base >> 32));
    }
  }
  if (ip_table.size() > kMaxTableBytes) {
    return {.table = {},
            .problem = std::format("the block list is {} bytes, more than the {} its "
                                   "size field can express",
                                   ip_table.size(), kMaxTableBytes)};
  }
  put16(ip_table, kTableSizeOffset, static_cast<uint16_t>(ip_table.size()));

  std::vector<std::byte> binary(kBinaryHeaderSize, std::byte{0});
  put32(binary, 0, kBinarySignature);
  put16(binary, 4, 1); // version major
  put16(binary, 6, 0); // version minor

  // Which parts of the device are fused off. Everything present is described as
  // working. Omitting the table would not be fatal — the driver logs "invalid
  // harvest table offset" and continues with no harvest counts — but every real
  // part publishes one, and the log line reads like a defect.
  std::vector<std::byte> harvest_table(kHarvestTableSize, std::byte{0});
  put32(harvest_table, 0, kHarvestSignature);
  put32(harvest_table, 4, 0);

  const auto describe_table = [&binary](std::size_t index, std::size_t offset,
                                        std::span<const std::byte> contents) {
    const std::size_t at = kTableListOffset + index * kTableInfoSize;
    put16(binary, at, static_cast<uint16_t>(offset));
    put16(binary, at + 2, byte_sum(contents));
    put16(binary, at + 4, static_cast<uint16_t>(contents.size()));
  };

  describe_table(kIpDiscoveryTableIndex, kBinaryHeaderSize, ip_table);
  describe_table(kHarvestTableIndex, kBinaryHeaderSize + ip_table.size(), harvest_table);

  binary.insert(binary.end(), ip_table.begin(), ip_table.end());
  binary.insert(binary.end(), harvest_table.begin(), harvest_table.end());
  if (binary.size() > kMaxTableBytes) {
    return {.table = {},
            .problem = std::format("the binary is {} bytes, more than the {} its size "
                                   "field can express",
                                   binary.size(), kMaxTableBytes)};
  }
  put16(binary, kBinarySizeOffset, static_cast<uint16_t>(binary.size()));

  // The binary's own checksum covers everything after the checksum field.
  const std::span<const std::byte> after_checksum(binary.data() + kBinaryChecksumOffset + 2,
                                                  binary.size() - kBinaryChecksumOffset - 2);
  put16(binary, kBinaryChecksumOffset, byte_sum(after_checksum));
  return {.table = std::move(binary), .problem = {}};
}

IpDiscoveryValidation validate_ip_discovery_table(std::span<const std::byte> table) {
  const auto reject = [](std::string problem) {
    return IpDiscoveryValidation{.valid = false, .problem = std::move(problem)};
  };
  const Reader in(table);

  const std::optional<uint32_t> signature = in.u32(0);
  if (!signature) {
    return reject("shorter than a binary header");
  }
  if (*signature != kBinarySignature) {
    return reject(
        std::format("binary signature is {:#x}, expected {:#x}", *signature, kBinarySignature));
  }
  if (!in.covers(0, kBinaryHeaderSize)) {
    return reject("shorter than a binary header");
  }

  // amdgpu_discovery_get_table_info accepts 0, 1 and 2, but its `case 2` reads
  // a binary_header_v2, whose table list sits somewhere else entirely. Versions
  // 0 and 1 share the one layout this reader assumes, so both are accepted and
  // only 2 is refused.
  const uint16_t version_major = in.u16(4).value();
  if (version_major > 1) {
    return reject(std::format("binary version {} puts its table list somewhere this reader does "
                              "not look; only 0 and 1 share the layout it assumes",
                              version_major));
  }

  const uint16_t declared_size = in.u16(kBinarySizeOffset).value();
  if (declared_size != table.size()) {
    return reject(std::format("declares {} bytes but is {}", declared_size, table.size()));
  }

  const uint16_t declared_checksum = in.u16(kBinaryChecksumOffset).value();
  const uint16_t actual_checksum =
      in.checksum(kBinaryChecksumOffset + 2, table.size() - kBinaryChecksumOffset - 2).value();
  if (declared_checksum != actual_checksum) {
    return reject(std::format("binary checksum is {:#x}, computed {:#x}", declared_checksum,
                              actual_checksum));
  }

  // Each table_info is offset, checksum, size. The driver checksums the harvest
  // table over its own fixed size rather than the size recorded here, so that is
  // the length checked. A zero offset is *not* rejected: the driver guards both
  // consumers on it (`check_table && offset` at amdgpu_discovery.c:641, and an
  // early return at :836-840 that logs "invalid harvest table offset" and
  // carries on with no harvest counts). Rejecting it here would refuse a table
  // the driver accepts, which is the one thing this function must not do.
  const std::size_t harvest_info = kTableListOffset + kHarvestTableIndex * kTableInfoSize;
  const uint16_t harvest_offset = in.u16(harvest_info).value();
  const uint16_t harvest_checksum = in.u16(harvest_info + 2).value();
  if (harvest_offset != 0) {
    if (!in.covers(harvest_offset, kHarvestTableSize)) {
      return reject("the harvest table runs past the end of the binary");
    }
    if (in.u32(harvest_offset).value() != kHarvestSignature) {
      return reject("the harvest table has the wrong signature");
    }
    if (in.checksum(harvest_offset, kHarvestTableSize).value() != harvest_checksum) {
      return reject("the harvest table checksum does not match");
    }
  }

  const std::size_t ip_info = kTableListOffset + kIpDiscoveryTableIndex * kTableInfoSize;
  const uint16_t ip_offset = in.u16(ip_info).value();
  const uint16_t ip_checksum = in.u16(ip_info + 2).value();
  const uint16_t ip_size = in.u16(ip_info + 4).value();
  if (ip_offset == 0 || ip_size == 0) {
    return reject("describes no block list");
  }
  if (!in.covers(ip_offset, kTableHeaderSize)) {
    return reject("the block list is too short to hold its own header");
  }
  if (in.u32(ip_offset).value() != kTableSignature) {
    return reject("the block list has the wrong signature");
  }

  // The size recorded in the table list and the one inside the block list are
  // both used: the driver checksums against the embedded one and bounds nothing
  // against the outer one, so a disagreement means one of the two consumers is
  // reading a different table than the other.
  const uint16_t embedded_size = in.u16(ip_offset + kTableSizeOffset).value();
  if (embedded_size != ip_size) {
    return reject(std::format("the block list says it is {} bytes but the table list says {}",
                              embedded_size, ip_size));
  }
  if (!in.covers(ip_offset, embedded_size)) {
    return reject("the block list runs past the end of the binary");
  }
  if (in.checksum(ip_offset, embedded_size).value() != ip_checksum) {
    return reject("the block list checksum does not match");
  }

  const uint16_t dies = in.u16(ip_offset + kTableNumDiesOffset).value();
  if (dies == 0) {
    return reject("describes no dies");
  }
  if (dies > kMaxDies) {
    return reject(
        std::format("describes {} dies, more than the {} the header has room for", dies, kMaxDies));
  }
  const bool bases_are_64_bit = (in.u8(ip_offset + kTableHeaderSize - 2).value() & 1) != 0;
  const std::size_t base_width = bases_are_64_bit ? sizeof(uint64_t) : sizeof(uint32_t);

  // Walk every record the way the driver does, from the start of the binary. A
  // die offset measured from the wrong place still lands inside the table, so
  // this checks by reading what it points at rather than by arithmetic.
  bool has_graphics = false;
  for (uint16_t die = 0; die < dies; ++die) {
    const uint16_t die_offset = in.u16(ip_offset + kTableDieInfoOffset + die * 4 + 2).value();
    if (!in.covers(die_offset, 4)) {
      return reject(std::format("die {} sits past the end of the binary", die));
    }
    const uint16_t die_id = in.u16(die_offset).value();
    if (die_id != die) {
      return reject(std::format("die {} identifies itself as {}, which means its offset points at "
                                "the wrong place",
                                die, die_id));
    }
    const uint16_t blocks = in.u16(die_offset + 2).value();
    if (blocks == 0) {
      return reject(std::format("die {} lists no blocks", die));
    }

    std::size_t at = static_cast<std::size_t>(die_offset) + 4;
    for (uint16_t block = 0; block < blocks; ++block) {
      if (!in.covers(at, 8)) {
        return reject(std::format("die {} block {} runs past the end of the binary", die, block));
      }
      const uint16_t hardware_id = in.u16(at).value();
      const uint8_t instance = in.u8(at + 2).value();
      const uint8_t bases = in.u8(at + 3).value();
      // The driver drops these records rather than refusing the table
      // (amdgpu_discovery_validate_ip, then `goto next_ip`), so a table
      // containing them parses into a guest quietly missing blocks. Refusing
      // here is the difference between a named fault and a silent one.
      if (instance >= kMaxInstances) {
        return reject(std::format("die {} block {} is instance {}, past the {} the driver keeps",
                                  die, block, instance, kMaxInstances));
      }
      if (hardware_id >= kMaxHardwareId) {
        return reject(std::format("die {} block {} has hardware id {}, past the {} the driver "
                                  "keeps",
                                  die, block, hardware_id, kMaxHardwareId));
      }
      if (!in.covers(at + 8, bases * base_width)) {
        return reject(std::format("die {} block {} declares {} register bases that run past the "
                                  "end of the binary",
                                  die, block, bases));
      }
      if (hardware_id == static_cast<uint16_t>(IpHardwareId::Gc)) {
        has_graphics = true;
      }
      at += 8 + bases * base_width;
    }
  }

  // A table listing no graphics block leaves the driver with no recognized GC
  // version, and it refuses the device in a switch whose default returns
  // -EINVAL without printing anything.
  if (!has_graphics) {
    return reject("lists no graphics block, which the driver refuses with no diagnostic");
  }

  return {.valid = true, .problem = {}};
}

} // namespace rocjitsu
