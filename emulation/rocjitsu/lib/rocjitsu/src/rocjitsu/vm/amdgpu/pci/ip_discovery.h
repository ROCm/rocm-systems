// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file ip_discovery.h
/// @brief Builds the table a guest driver reads to learn what the GPU contains.
///
/// @details An AMD GPU does not tell the driver what it is through its PCI IDs.
/// The driver binds on the class code and then reads a table out of the device
/// that lists every IP block present, its version, and where its registers live.
/// Almost nothing else can happen first: the table is what decides which drivers
/// for which blocks are even instantiated.
///
/// So an emulated GPU has to publish one. Its contents are a design decision
/// rather than a transcription, because omitting a block is the cleanest way to
/// say "this device does not have one" and keeps the driver from initialising
/// something the model cannot answer.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief How much of the table the driver will read, its DISCOVERY_TMR_SIZE.
///
/// @details Both paths that deliver a table are bounded by this: the device
/// stores only this much at the top of memory, and the driver refuses a larger
/// file outright ("ip discovery firmware too large"). It belongs to the format
/// rather than to either consumer, so both check against the same number.
constexpr std::size_t kDiscoveryTableBytes = 10 * 1024;

/// @brief Hardware identifiers, as the driver's tables spell them.
///
/// @details Values come from the driver's own soc15_hw_ip.h. Only the blocks a
/// generated table can currently describe are listed.
enum class IpHardwareId : uint16_t {
  Mp1 = 1,     ///< System management, which owns power and clocks.
  Gc = 11,     ///< Graphics and compute; its instance count is the XCC mask.
  MmHub = 34,  ///< Memory hub.
  AtHub = 35,  ///< Address translation hub.
  OssSys = 40, ///< Interrupt handling, among other system blocks.
  Hdp = 41,    ///< Host data path.
  Sdma0 = 42,  ///< System DMA.
  Nbif = 108,  ///< North bridge interface, the bus side of the device.
  Mp0 = 255    ///< Security processor, which the driver requires to be present.
};

/// @brief One IP block as the table describes it.
struct IpBlock {
  IpHardwareId hardware_id = IpHardwareId::Gc; ///< Which block this is.
  uint8_t instance = 0;                        ///< Which copy of it.
  uint8_t major = 0;                           ///< Version, as IP_VERSION spells it.
  uint8_t minor = 0;
  uint8_t revision = 0;
  std::vector<uint64_t> register_bases; ///< Where its registers start.
};

/// @brief Everything a generated table describes.
struct IpDiscoverySpec {
  std::vector<IpBlock> blocks; ///< The device's blocks, in any order.
};

/// @brief A serialized table, or the reason there is not one.
///
/// @details Building can fail, because the format's fields are narrower than
/// the spec's: a block count, a table size, or a base-address count that does
/// not fit would truncate into a table the driver misparses rather than
/// rejects. Saying so is better than emitting bytes that mean something else.
struct IpDiscoveryBuild {
  std::vector<std::byte> table; ///< The table, when it could be built.
  std::string problem;          ///< Why it could not be, when it could not.

  /// @returns Whether a table was produced.
  [[nodiscard]] bool ok() const { return problem.empty(); }
};

/// @brief Serialize @p spec into the binary the driver parses.
/// @param[in] spec The blocks to describe.
/// @returns The table, or why the spec cannot be expressed in the format.
[[nodiscard]] IpDiscoveryBuild build_ip_discovery_table(const IpDiscoverySpec &spec);

/// @brief Why a table would be rejected, if it would be.
///
/// @details Mirrors the checks the driver performs before it will read a table,
/// so a generated one can be validated here rather than by booting a guest and
/// reading a failure out of dmesg.
struct IpDiscoveryValidation {
  bool valid = false;  ///< Whether the driver would accept the table.
  std::string problem; ///< What is wrong with it, when it would not.
};

/// @brief Check that the driver would accept a table *and use every record in it*.
///
/// @details Covers what `amdgpu_discovery_init` verifies before it will use a
/// table — signatures and checksums for the binary, the block list, and the
/// harvest table — and then walks the structures the way
/// `amdgpu_discovery_reg_base_init` walks them. That second half is the part
/// that matters: a table can satisfy every checksum and still describe a device
/// the driver refuses, and it refuses with a bare `-EINVAL` and no diagnostic.
/// A die pointed at the wrong place, a block count that disagrees with the
/// records behind it, and a missing graphics block all fail that way.
///
/// It is deliberately stricter than the driver in one direction: records the
/// driver would silently *drop* rather than refuse — an instance or hardware id
/// past the range it keeps — are reported here as problems. A table full of
/// dropped records parses into a guest quietly missing blocks, which is worse
/// to debug than a refusal. So a rejection means "the driver would not use this
/// table as written", not always "the driver would fail to parse it". Callers
/// that gate device construction on this are choosing that stricter contract.
///
/// Every read is bounds-checked against @p table rather than trusting the
/// lengths written inside it, because the input may be arbitrary bytes.
///
/// @param[in] table The serialized table.
/// @returns Whether the driver would accept it, and why not if it would not.
[[nodiscard]] IpDiscoveryValidation validate_ip_discovery_table(std::span<const std::byte> table);

} // namespace rocjitsu
