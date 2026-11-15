// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file location.h
/// @brief Architectural locations tracked by the gfx950 P0 SALU differential model.
///
/// @details One location is one independently writable 32-bit (or 1-bit) piece of
/// scalar state. VCC and EXEC are split into their LO/HI halves because SALU code
/// writes them independently (`s_mov_b32 vcc_hi, ...`); a whole-register singleton
/// would make a half write look like a kill. EXECZ/VCCZ are not locations: the
/// emulator derives them from EXEC/VCC on read, so they cannot be made inconsistent.
#pragma once

#include <bitset>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rocjitsu::isa_diff {

/// Addressable SGPRs on gfx950 (s0..s101). Selectors 102..105 name FLAT_SCRATCH
/// and XNACK_MASK, which are not ordinary SGPRs and are outside P0.
inline constexpr unsigned kNumSgprs = 102;

enum Loc : unsigned {
  kLocVccLo = kNumSgprs,
  kLocVccHi,
  kLocExecLo,
  kLocExecHi,
  kLocM0,
  kLocScc,
  kNumLocs,
};

using LocSet = std::bitset<kNumLocs>;

/// @brief Every tracked location.
[[nodiscard]] LocSet all_locs();

/// @brief Canonical name: "s7", "vcc_lo", "vcc_hi", "exec_lo", "exec_hi", "m0", "scc".
[[nodiscard]] std::string loc_name(unsigned loc);

/// @brief Inverse of loc_name().
[[nodiscard]] std::optional<unsigned> parse_loc(std::string_view name);

/// @brief Comma-separated canonical names in location order.
[[nodiscard]] std::string format_locs(const LocSet &locs);

/// @brief Parse format_locs() output. Returns nullopt on an unknown name.
[[nodiscard]] std::optional<LocSet> parse_locs(std::string_view text);

/// @brief Value mask of a location: 1 for SCC, all 32 bits otherwise.
[[nodiscard]] constexpr uint32_t loc_value_mask(unsigned loc) {
  return loc == kLocScc ? 1u : 0xffffffffu;
}

} // namespace rocjitsu::isa_diff
