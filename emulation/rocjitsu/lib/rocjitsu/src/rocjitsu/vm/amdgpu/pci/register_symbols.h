// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_symbols.h
/// @brief Names for the MMIO registers a guest driver touches.
///
/// @details Bringing an unmodified kernel driver up against an emulated device
/// is a burndown: boot, see which register the driver blocked on, model it,
/// repeat. A raw byte offset makes that loop slow, so every diagnostic the
/// device emits is symbolized through this table.
///
/// Offsets are recorded per BAR because the same numeric offset means different
/// things in different apertures. The driver addresses registers by dword index
/// (that is what its RREG32 takes), so @ref rocjitsu::RegisterSymbols::add_dword
/// exists to let a table be transcribed from driver headers without hand
/// converting every entry.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace rocjitsu {

/// @brief A lookup from a BAR-relative byte offset to a register name.
class RegisterSymbols {
public:
  /// @brief Name the register at a BAR-relative byte offset.
  /// @param[in] bar BAR index the register lives in.
  /// @param[in] byte_offset Byte offset of the register within that BAR.
  /// @param[in] name Register name, as spelled by the driver headers.
  void add(int bar, uint64_t byte_offset, std::string name);

  /// @brief Name a register given its dword index, as driver headers spell it.
  /// @param[in] bar BAR index the register lives in.
  /// @param[in] dword_index Register index in dwords, as passed to RREG32.
  /// @param[in] name Register name, as spelled by the driver headers.
  void add_dword(int bar, uint64_t dword_index, std::string name);

  /// @brief Look up the name of the register at a BAR-relative byte offset.
  /// @param[in] bar BAR index the access targeted.
  /// @param[in] byte_offset Byte offset of the access within that BAR.
  /// @returns The register name, or an empty view if the offset has no name.
  [[nodiscard]] std::string_view lookup(int bar, uint64_t byte_offset) const;

  /// @brief Return the number of named registers.
  [[nodiscard]] std::size_t size() const { return names_.size(); }

private:
  static uint64_t make_key(int bar, uint64_t byte_offset);

  std::unordered_map<uint64_t, std::string> names_;
};

/// @brief Add the registers the driver reads before IP discovery runs.
///
/// @param[in,out] symbols Table to add the names to.
/// @param[in] bar BAR index carrying the register aperture.
///
/// @details These are the handful of registers amdgpu touches before it knows
/// anything about the device, so they are the first thing an emulated device
/// must answer and the first thing worth naming in a trace. The driver defines
/// them locally in amdgpu_discovery.c with the comment that they are consistent
/// across all SOCs, rather than in a per-ASIC header.
void add_pre_discovery_symbols(RegisterSymbols &symbols, int bar);

} // namespace rocjitsu
