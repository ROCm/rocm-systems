// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops.h"

#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <climits>

namespace rocjitsu {

std::optional<uint32_t> consan_sc_build_wait_dscnt(uint16_t count, rj_code_arch_t arch) {
  // Every current SuperCollider consumer requests the completed state. Keep
  // nonzero counts out of the semantic interface until an engine needs them.
  if (count != 0)
    return std::nullopt;
  return instrumentation::build_s_wait_lds0(arch);
}

std::optional<uint32_t> consan_sc_delay_instruction_word_count(const ConSanOptions &options,
                                                               std::vector<std::string> &errors,
                                                               std::string_view context) {
  if (options.delay_nops == 0)
    return 0u;

  switch (options.delay_mode) {
  case ConSanDelayMode::Nop:
    return options.delay_nops;
  case ConSanDelayMode::Sleep:
    if (options.delay_nops > UINT16_MAX) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return std::nullopt;
    }
    return 1u;
  case ConSanDelayMode::SleepVar:
    if (options.delay_var_ssrc > 255) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return std::nullopt;
    }
    return 1u;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      consan_delay_mode_name(options.delay_mode) + "'");
  return std::nullopt;
}

bool consan_sc_append_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                  const ConSanOptions &options, std::vector<std::string> &errors,
                                  std::string_view context) {
  if (options.delay_nops == 0)
    return true;

  switch (options.delay_mode) {
  case ConSanDelayMode::Nop:
    for (uint32_t i = 0; i < options.delay_nops; ++i)
      words.push_back(build_s_nop(0, arch));
    return true;
  case ConSanDelayMode::Sleep:
    if (options.delay_nops > UINT16_MAX) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return false;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(options.delay_nops), arch));
    return true;
  case ConSanDelayMode::SleepVar:
    if (options.delay_var_ssrc > 255) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return false;
    }
    words.push_back(build_s_sleep_var(options.delay_var_ssrc, arch));
    return true;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      consan_delay_mode_name(options.delay_mode) + "'");
  return false;
}

std::optional<uint32_t> consan_sc_build_v_cmp_ne_u32(uint16_t src0, uint16_t vsrc1,
                                                     rj_code_arch_t arch) {
  if (src0 > 255 || vsrc1 > 255)
    return std::nullopt;
  return instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(src0), vsrc1, arch);
}

std::optional<uint32_t> consan_sc_build_v_cmp_ne_u16(uint16_t src0, uint16_t vsrc1,
                                                     rj_code_arch_t arch) {
  if (src0 > 255 || vsrc1 > 255)
    return std::nullopt;
  return instrumentation::build_v_cmp_ne_u16_vcc(vector_source_vgpr(src0), vsrc1, arch);
}

std::optional<uint32_t> consan_sc_build_ds_load_word0(const ConSanAccessLoweringForm &form,
                                                      uint32_t word0, rj_code_arch_t arch) {
  const bool two_address = form.kind == ConSanAccessLoweringFormKind::NativeTwoRange;
  const uint32_t width_bits = form.element_width_bits;
  uint32_t base = 0;
  if (consan_uses_gfx11_or_gfx12_encoding(arch)) {
    if (two_address) {
      switch (word0 & 0xFFFF0000u) {
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
      constexpr uint32_t kDsOffsetMask = 0x0000FFFFu;
      return base | (word0 & kDsOffsetMask);
    }
    switch (width_bits) {
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
  } else if (consan_uses_gfx9_cdna_encoding(arch) && !two_address) {
    switch (width_bits) {
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
  } else if (consan_uses_gfx9_cdna_encoding(arch) && two_address) {
    switch (word0 & 0xFFFF0000u) {
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
    return std::nullopt;
  }
  constexpr uint32_t kDsOffsetMask = 0x0000FFFFu;
  return base | (word0 & kDsOffsetMask);
}

uint32_t consan_sc_build_ds_load_word1(uint16_t addr_vgpr, uint16_t dst_vgpr) {
  return static_cast<uint32_t>(addr_vgpr) | (static_cast<uint32_t>(dst_vgpr) << 24u);
}

std::optional<ConSanScTwoAddressLdsByteOffsets>
consan_sc_two_address_lds_byte_offsets(const ConSanAccessLoweringForm &form, uint32_t word0) {
  if (form.kind != ConSanAccessLoweringFormKind::NativeTwoRange ||
      form.encoded_offset_scale_bytes == 0u)
    return std::nullopt;
  return ConSanScTwoAddressLdsByteOffsets{
      .first = (word0 & 0xffu) * form.encoded_offset_scale_bytes,
      .second = ((word0 >> 8u) & 0xffu) * form.encoded_offset_scale_bytes,
  };
}

} // namespace rocjitsu
