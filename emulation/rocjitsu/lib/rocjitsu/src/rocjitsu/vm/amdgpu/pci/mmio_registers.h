// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mmio_registers.h
/// @brief Register offsets the driver uses before it knows what device it has.
///
/// @details Almost every AMD GPU register lives at an offset the driver only
/// learns from the IP discovery table, but a handful come first: they are how
/// the driver finds that table in the first place. The kernel defines them
/// locally rather than in a per-ASIC header, noting that they are the same
/// across all SOCs, so they are the one set of offsets an emulated device can
/// answer before it has an identity.
///
/// Offsets are dword indices, as the driver's register accessors take them; the
/// byte offset within the aperture is four times the index.

#pragma once

#include <cstdint>

namespace rocjitsu {

/// @brief Register index, in dwords, as the driver's accessors take it.
enum class MmioRegister : uint32_t {
  MmIndex = 0x0,                ///< Selects a register for the indirect window.
  MmData = 0x1,                 ///< Reads or writes the register selected above.
  MmIndexHi = 0x6,              ///< High half of the indirect window selector.
  DriverScratch0 = 0x94,        ///< Low half of a discovery table offset, when nonzero.
  DriverScratch1 = 0x95,        ///< High half of that offset.
  DriverScratch2 = 0x96,        ///< Discovery table size; zero means it is not published here.
  RccConfigMemsize = 0xde3,     ///< Video memory size in megabytes.
  RccIovFuncIdentifier = 0xde5, ///< Whether this function is virtualized.
  Mp0SmnC2pmsg33 = 0x16061,     ///< Firmware init status; bit 31 means initialization finished.
  IpDiscoveryVersion = 0x16a00  ///< Version of the discovery table format.
};

/// @brief Bytes in one register.
///
/// @details Every register in this aperture is 32 bits wide, and the driver's
/// accessors index them in dwords. So this is one number wearing two hats: the
/// width of a register, and the scale between a dword index and a byte offset.
/// They are the same number because they are the same fact, which is why there
/// is one constant rather than two.
inline constexpr uint64_t kRegisterBytes = sizeof(uint32_t);

/// @brief Byte offset of a dword register index within the aperture.
///
/// @details Register offsets arrive as dword indices from two places -- the
/// enumeration above and the segment lists the discovery table publishes -- and
/// every one of them has to be scaled before it can address the aperture. This
/// names that scaling, because doing it inline is how an index gets used as an
/// address by accident.
///
/// Deliberately does not range-check: callers work in different address spaces
/// and against different limits, and a check here would be the wrong one in most
/// of them. Overflow of the multiply is the caller's to prevent, which for
/// segment addresses is what @c segment_within_aperture exists to do.
///
/// @param[in] dword_index Register index, in dwords.
/// @returns Its byte offset within the aperture.
[[nodiscard]] constexpr uint64_t byte_offset_of_dword(uint64_t dword_index) {
  return dword_index * kRegisterBytes;
}

/// @brief Dword index holding a byte offset.
/// @param[in] byte_offset Byte offset within the aperture. Assumed aligned;
///                        an unaligned offset names the register containing it.
/// @returns The index of the register at that offset.
[[nodiscard]] constexpr uint64_t dword_index_of(uint64_t byte_offset) {
  return byte_offset / kRegisterBytes;
}

/// @brief Byte offset of a register within the aperture.
/// @param[in] reg The register.
/// @returns Its byte offset.
[[nodiscard]] constexpr uint64_t byte_offset_of(MmioRegister reg) {
  return byte_offset_of_dword(static_cast<uint64_t>(reg));
}

/// @brief Bit the driver polls in @ref MmioRegister::Mp0SmnC2pmsg33.
///
/// @details The driver waits up to two seconds for this, because on real
/// hardware the firmware may still be starting. An emulated device has nothing
/// to wait for and reports ready immediately.
inline constexpr uint32_t kFirmwareInitDoneBit = 0x80000000;

/// @brief Value reported by @ref MmioRegister::IpDiscoveryVersion.
///
/// @details Zero, which is what a device publishing its discovery table at the
/// top of video memory reports: the register distinguishes formats that place
/// the table elsewhere, and this device uses the memory-resident one. It is
/// defined rather than left absent because the register sits inside the aperture
/// the driver reads before discovery, so an undefined one would read as a
/// register this device does not model.
inline constexpr uint32_t kIpDiscoveryVersion = 0;

} // namespace rocjitsu
