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

/// Vector registers v0..v255 (VALU support). A VGPR is one location for analysis
/// (liveness, effects, contracts); its 64 lanes are separate values in MachineState.
inline constexpr unsigned kNumVgprs = 256;
inline constexpr unsigned kWaveLanes = 64;
using VSet = std::bitset<kNumVgprs>;

/// MODE at dispatch without a kernel descriptor (assembler defaults): round to nearest,
/// f32 denormals flushed, f16/f64 denormals kept, DX10_CLAMP and IEEE on.
inline constexpr uint32_t kDefaultInitialMode = (3u << 6) | (1u << 8) | (1u << 9);

/// "v0,v3" in register order; empty for none.
[[nodiscard]] std::string format_vgprs(const VSet &v);
[[nodiscard]] std::optional<VSet> parse_vgprs(std::string_view text);
/// Highest set VGPR + 1, or 0 when none.
[[nodiscard]] unsigned vgpr_extent(const VSet &v);

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
