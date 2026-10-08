// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vgpr_msb.h
/// @brief AMDGPU VGPR high-bank mode helpers shared by model and execution code.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_VGPR_MSB_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_VGPR_MSB_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {

/// @brief Logical VALU operand role selected by VGPR_MSB and GPR_IDX fields.
enum class VgprMsbRole : uint8_t {
  None,
  Src0,
  Src1,
  Src2,
  Dst,
};

constexpr uint32_t VGPR_MSB_MODE_SHIFT = 12;
constexpr uint32_t VGPR_MSB_MODE_MASK = 0xffu << VGPR_MSB_MODE_SHIFT;
constexpr uint16_t MODE_HWREG = 1;
constexpr size_t VGPR_MSB_ROLE_COUNT = 4;

/// @brief Known bank for each role in S_SET_VGPR_MSB field order.
using VgprMsbBanks = std::array<std::optional<uint8_t>, VGPR_MSB_ROLE_COUNT>;

/// @brief Return the packed S_SET_VGPR_MSB field index for an operand role.
///
/// A missing role is deliberately unknown rather than bank zero: explicit
/// operands must declare how MODE selects their physical register.
[[nodiscard]] constexpr std::optional<size_t> vgpr_msb_role_index(VgprMsbRole role) {
  switch (role) {
  case VgprMsbRole::Src0:
    return 0;
  case VgprMsbRole::Src1:
    return 1;
  case VgprMsbRole::Src2:
    return 2;
  case VgprMsbRole::Dst:
    return 3;
  case VgprMsbRole::None:
    return std::nullopt;
  }
  return std::nullopt;
}

/// @brief Resolve one operand role from a packed S_SET_VGPR_MSB value.
[[nodiscard]] constexpr std::optional<uint8_t> vgpr_msb_bank_for_role(std::optional<uint8_t> value,
                                                                      VgprMsbRole role) {
  const auto index = vgpr_msb_role_index(role);
  if (!value || !index)
    return std::nullopt;
  return static_cast<uint8_t>((*value >> (2 * *index)) & 0x3u);
}

/// @brief Decoded fields of an S_SETREG* HWREG immediate.
struct HwregSlice {
  uint16_t id;
  uint16_t begin;
  uint16_t width;
};

/// @brief Decode HWREG id[5:0], offset[10:6], and size-1[15:11].
[[nodiscard]] constexpr HwregSlice decode_vgpr_msb_hwreg(uint16_t hwreg) {
  return HwregSlice{.id = static_cast<uint16_t>(hwreg & 0x3f),
                    .begin = static_cast<uint16_t>((hwreg >> 6) & 0x1f),
                    .width = static_cast<uint16_t>(((hwreg >> 11) & 0x1f) + 1)};
}

/// @brief Apply one right-justified S_SETREG value to MODE.VGPR_MSB fields.
///
/// @details Dynamic writes pass std::nullopt and make only overlapping bank
/// fields unknown. Immediate writes replace the requested bit slice while
/// preserving every disjoint field and any known untouched bit. S_SETREG
/// sources are right-justified to the requested HWREG slice, matching the
/// generic HWREG insertion used by the execution model.
inline void apply_vgpr_msb_mode_write(VgprMsbBanks &banks, uint16_t hwreg,
                                      std::optional<uint32_t> value) {
  constexpr std::array<uint8_t, VGPR_MSB_ROLE_COUNT> mode_bit_offset = {14, 16, 18, 12};
  const HwregSlice slice = decode_vgpr_msb_hwreg(hwreg);
  const uint16_t begin = slice.begin;
  if (slice.id != MODE_HWREG || begin >= 32)
    return;
  const uint16_t width = std::min<uint16_t>(slice.width, static_cast<uint16_t>(32 - begin));
  const uint16_t end = static_cast<uint16_t>(begin + width);

  for (size_t role = 0; role < banks.size(); ++role) {
    const uint16_t field_begin = mode_bit_offset[role];
    const uint16_t field_end = static_cast<uint16_t>(field_begin + 2);
    if (begin >= field_end || field_begin >= end)
      continue;
    if (!value) {
      banks[role] = std::nullopt;
      continue;
    }

    const uint16_t overlap_begin = std::max(begin, field_begin);
    const uint16_t overlap_end = std::min(end, field_end);
    if (overlap_begin == field_begin && overlap_end == field_end) {
      const uint8_t source_bit = static_cast<uint8_t>(field_begin - begin);
      banks[role] = static_cast<uint8_t>((*value >> source_bit) & 0x3u);
      continue;
    }
    if (!banks[role])
      continue;
    uint8_t bank = *banks[role];
    for (uint16_t mode_bit = overlap_begin; mode_bit < overlap_end; ++mode_bit) {
      const uint8_t field_bit = static_cast<uint8_t>(mode_bit - field_begin);
      const uint8_t source_bit = static_cast<uint8_t>(mode_bit - begin);
      const uint8_t bit = static_cast<uint8_t>((*value >> source_bit) & 1u);
      bank = static_cast<uint8_t>((bank & ~(uint8_t{1} << field_bit)) | (bit << field_bit));
    }
    banks[role] = bank;
  }
}

/// @brief Expand a packed S_SET_VGPR_MSB byte into independently-known roles.
[[nodiscard]] inline VgprMsbBanks unpack_vgpr_msb_banks(std::optional<uint8_t> value) {
  VgprMsbBanks banks;
  banks.fill(std::nullopt);
  if (!value)
    return banks;
  for (size_t role = 0; role < banks.size(); ++role)
    banks[role] = static_cast<uint8_t>((*value >> (2 * role)) & 0x3u);
  return banks;
}

/// @brief Pack independently-known roles, or return unknown if any role is unknown.
[[nodiscard]] inline std::optional<uint8_t> pack_vgpr_msb_banks(const VgprMsbBanks &banks) {
  uint8_t value = 0;
  for (size_t role = 0; role < banks.size(); ++role) {
    if (!banks[role])
      return std::nullopt;
    value = static_cast<uint8_t>(value | (*banks[role] << (2 * role)));
  }
  return value;
}

/// @brief Convert S_SET_VGPR_MSB layout to MODE[19:12] layout.
///
/// S_SET_VGPR_MSB packs src0,src1,src2,dst in that order. MODE stores
/// dst,src0,src1,src2, so this is a byte rotate left by one two-bit field.
constexpr uint8_t set_vgpr_msb_to_mode_layout(uint8_t value) {
  return static_cast<uint8_t>(((value << 2) | (value >> 6)) & 0xffu);
}

/// @brief Convert MODE[19:12] layout to S_SET_VGPR_MSB layout.
constexpr uint8_t mode_layout_to_set_vgpr_msb(uint8_t value) {
  return static_cast<uint8_t>(((value >> 2) | (value << 6)) & 0xffu);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_VGPR_MSB_H_
