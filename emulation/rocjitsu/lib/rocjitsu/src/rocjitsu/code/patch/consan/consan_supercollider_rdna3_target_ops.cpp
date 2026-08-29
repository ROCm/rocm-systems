// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_rdna3_target_ops.cpp
/// @brief RDNA3 recipes consumed by SuperCollider.

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops_internal.h"

#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"

namespace rocjitsu::consan_sc_target_detail {

std::optional<std::array<uint32_t, 3>>
build_rdna3_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits,
                                 uint16_t vdst) {
  uint16_t load_op = 0;
  switch (width_bits) {
  case 8:
    load_op = rdna3::kFlatLoadU8Flat;
    break;
  case 16:
    load_op = rdna3::kFlatLoadU16Flat;
    break;
  case 32:
    load_op = rdna3::kFlatLoadB32Flat;
    break;
  case 64:
    load_op = rdna3::kFlatLoadB64Flat;
    break;
  case 128:
    load_op = rdna3::kFlatLoadB128Flat;
    break;
  default:
    return std::nullopt;
  }
  constexpr uint32_t kFlatOpMask = 0x7fu << 18u;
  constexpr uint32_t kFlatDataMask = 0xffu << 8u;
  words[0] = (words[0] & ~kFlatOpMask) | (static_cast<uint32_t>(load_op) << 18u);
  words[1] = (words[1] & ~kFlatDataMask & 0x00ffffffu) | (static_cast<uint32_t>(vdst) << 24u);
  return words;
}

} // namespace rocjitsu::consan_sc_target_detail
