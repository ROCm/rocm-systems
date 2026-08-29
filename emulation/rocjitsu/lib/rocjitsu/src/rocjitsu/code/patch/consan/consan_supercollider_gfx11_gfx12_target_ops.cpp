// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_gfx11_gfx12_target_ops.cpp
/// @brief Family-shared GFX11/GFX12 recipes consumed by SuperCollider.

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops_internal.h"

namespace rocjitsu::consan_sc_target_detail {

std::optional<uint32_t> build_gfx11_gfx12_ds_load_word0(const ConSanAccessLoweringForm &form,
                                                        uint32_t original_word0) {
  uint32_t base = 0;
  if (form.kind == ConSanAccessLoweringFormKind::NativeTwoRange) {
    switch (original_word0 & 0xFFFF0000u) {
    case 0xD8380000u:
      base = 0xD8DC0000u;
      break;
    case 0xD83C0000u:
      base = 0xD8E00000u;
      break;
    case 0xD9380000u:
      base = 0xD9DC0000u;
      break;
    case 0xD93C0000u:
      base = 0xD9E00000u;
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
      base = 0xD8F00000u;
      break;
    case 32:
      base = 0xD8D80000u;
      break;
    case 64:
      base = 0xD9D80000u;
      break;
    case 96:
      base = 0xDBF80000u;
      break;
    case 128:
      base = 0xDBFC0000u;
      break;
    default:
      return std::nullopt;
    }
  }
  constexpr uint32_t kDsOffsetMask = 0x0000FFFFu;
  return base | (original_word0 & kDsOffsetMask);
}

std::optional<std::array<uint32_t, 3>>
retarget_classic_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst) {
  words[1] = (words[1] & 0x00FFFFFFu) | (static_cast<uint32_t>(vdst) << 24u);
  return words;
}

} // namespace rocjitsu::consan_sc_target_detail
