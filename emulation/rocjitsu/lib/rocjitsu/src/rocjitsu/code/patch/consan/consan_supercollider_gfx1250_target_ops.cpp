// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_gfx1250_target_ops.cpp
/// @brief gfx1250 recipes consumed by SuperCollider.

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops_internal.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"

namespace rocjitsu::consan_sc_target_detail {

std::optional<std::array<uint32_t, 3>>
retarget_gfx1250_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst) {
  words[1] = (words[1] & ~0xffu) | vdst;
  return words;
}

std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits,
                                   uint16_t vdst) {
  uint16_t load_op = 0;
  switch (width_bits) {
  case 8:
    load_op = cdna5::kFlatLoadU8Vflat;
    break;
  case 16:
    load_op = cdna5::kFlatLoadU16Vflat;
    break;
  case 32:
    load_op = cdna5::kFlatLoadB32Vflat;
    break;
  case 64:
    load_op = cdna5::kFlatLoadB64Vflat;
    break;
  case 128:
    load_op = cdna5::kFlatLoadB128Vflat;
    break;
  default:
    return std::nullopt;
  }
  constexpr uint32_t kFlatOpMask = 0xffu << 14u;
  constexpr uint32_t kFlatVsrcMask = 0xffu << 23u;
  words[0] = (words[0] & ~kFlatOpMask) | (static_cast<uint32_t>(load_op) << 14u);
  words[1] = (words[1] & ~0xffu & ~kFlatVsrcMask) | vdst;
  return words;
}

} // namespace rocjitsu::consan_sc_target_detail
