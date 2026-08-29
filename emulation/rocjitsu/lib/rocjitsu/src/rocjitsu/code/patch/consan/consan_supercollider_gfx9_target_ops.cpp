// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_gfx9_target_ops.cpp
/// @brief GFX9 CDNA3/CDNA4 recipes consumed by SuperCollider.

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops_internal.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"

namespace rocjitsu::consan_sc_target_detail {

std::optional<uint32_t> build_gfx9_ds_load_word0(const ConSanAccessLoweringForm &form,
                                                 uint32_t original_word0) {
  uint32_t base = 0;
  if (form.kind == ConSanAccessLoweringFormKind::NativeTwoRange) {
    switch (original_word0 & 0xFFFF0000u) {
    case 0xD81C0000u:
      base = 0xD86E0000u;
      break;
    case 0xD81E0000u:
      base = 0xD8700000u;
      break;
    case 0xD89C0000u:
      base = 0xD8EE0000u;
      break;
    case 0xD89E0000u:
      base = 0xD8F00000u;
      break;
    default:
      return std::nullopt;
    }
  } else {
    switch (form.element_width_bits) {
    case 8:
      base = 0xD8E80000u;
      break;
    case 16:
      base = 0xD8780000u;
      break;
    case 32:
      base = 0xD86C0000u;
      break;
    case 64:
      base = 0xD8EC0000u;
      break;
    case 96:
      base = 0xD9FC0000u;
      break;
    case 128:
      base = 0xD9FE0000u;
      break;
    default:
      return std::nullopt;
    }
  }
  constexpr uint32_t kDsOffsetMask = 0x0000FFFFu;
  return base | (original_word0 & kDsOffsetMask);
}

std::optional<std::array<uint32_t, 2>>
build_gfx9_cdna_accvgpr_read_b32(uint16_t dst_vgpr, uint16_t src_accvgpr,
                                 ConSanEncodingFamily family) {
  if (dst_vgpr > 255u || src_accvgpr > 255u)
    return std::nullopt;
  if (family == ConSanEncodingFamily::Gfx9Cdna3) {
    return cdna3::build_vop3p(cdna3::kVAccvgprReadVop3p,
                              {.vdst = static_cast<uint8_t>(dst_vgpr),
                               .op_sel_hi_2 = 1u,
                               .src0 = static_cast<uint16_t>(256u + src_accvgpr),
                               .op_sel_hi = 3u});
  }
  if (family == ConSanEncodingFamily::Gfx9Cdna4) {
    return cdna4::build_vop3p(cdna4::kVAccvgprReadVop3p,
                              {.vdst = static_cast<uint8_t>(dst_vgpr),
                               .op_sel_hi_2 = 1u,
                               .src0 = static_cast<uint16_t>(256u + src_accvgpr),
                               .op_sel_hi = 3u});
  }
  return std::nullopt;
}

std::optional<std::array<uint32_t, 3>>
build_gfx9_cdna_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits,
                                     uint16_t vdst) {
  uint32_t load_op = 0;
  switch (width_bits) {
  case 8:
    load_op = cdna4::kFlatLoadUbyteFlat;
    break;
  case 16:
    load_op = cdna4::kFlatLoadUshortFlat;
    break;
  case 32:
    load_op = cdna4::kFlatLoadDwordFlat;
    break;
  case 64:
    load_op = cdna4::kFlatLoadDwordx2Flat;
    break;
  case 128:
    load_op = cdna4::kFlatLoadDwordx4Flat;
    break;
  default:
    return std::nullopt;
  }
  constexpr uint32_t kFlatOpMask = 0x7fu << 18u;
  constexpr uint32_t kFlatDataMask = 0xffu << 8u;
  words[0] = (words[0] & ~kFlatOpMask) | (load_op << 18u);
  words[1] = (words[1] & ~kFlatDataMask & 0x00FFFFFFu) | (static_cast<uint32_t>(vdst) << 24u);
  return words;
}

} // namespace rocjitsu::consan_sc_target_detail
