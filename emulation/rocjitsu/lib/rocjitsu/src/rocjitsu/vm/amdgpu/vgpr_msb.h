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

#endif // ROCJITSU_VM_AMDGPU_VGPR_MSB_H_
