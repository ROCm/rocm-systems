// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vgpr_msb.h
/// @brief AMDGPU VGPR high-bank mode helpers.

#ifndef ROCJITSU_VM_AMDGPU_VGPR_MSB_H_
#define ROCJITSU_VM_AMDGPU_VGPR_MSB_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief Operand role selected by the two-bit fields in S_SET_VGPR_MSB.
enum class VgprMsbRole : uint8_t {
  None,
  Src0,
  Src1,
  Src2,
  Dst,
};

constexpr uint32_t VGPR_MSB_MODE_SHIFT = 12;
constexpr uint32_t VGPR_MSB_MODE_MASK = 0xffu << VGPR_MSB_MODE_SHIFT;

/// @brief Return the S_SET_VGPR_MSB mode that is effective after the instruction.
///
/// On gfx1250, SIMM16[7:0] is the architectural mode written by normal
/// S_SET_VGPR_MSB execution. SIMM16[15:8] carries the previous mode for trap
/// fixup when a VALU instruction is followed by S_SET_VGPR_MSB.
constexpr uint8_t s_set_vgpr_msb_new_mode(uint16_t simm16) {
  return static_cast<uint8_t>(simm16 & 0xffu);
}

/// @brief Return the previous-mode snapshot carried in S_SET_VGPR_MSB SIMM16[15:8].
constexpr uint8_t s_set_vgpr_msb_previous_mode(uint16_t simm16) {
  return static_cast<uint8_t>((simm16 >> 8) & 0xffu);
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

/// @brief Return the previous-mode snapshot in MODE[19:12] layout.
constexpr uint8_t s_set_vgpr_msb_previous_mode_layout(uint16_t simm16) {
  return set_vgpr_msb_to_mode_layout(s_set_vgpr_msb_previous_mode(simm16));
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_VGPR_MSB_H_
