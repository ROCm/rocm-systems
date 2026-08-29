// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops.h"

#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"

#include <climits>
#include <cstring>

namespace rocjitsu {
namespace {

[[nodiscard]] std::optional<uint8_t> flat_load_op_for_width(uint32_t width_bits) {
  switch (width_bits) {
  case 8:
    return rdna4::kFlatLoadU8Vflat;
  case 16:
    return rdna4::kFlatLoadU16Vflat;
  case 32:
    return rdna4::kFlatLoadB32Vflat;
  case 64:
    return rdna4::kFlatLoadB64Vflat;
  case 128:
    return rdna4::kFlatLoadB128Vflat;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_rdna4_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits,
                                 uint16_t vdst) {
  const auto load_op = flat_load_op_for_width(width_bits);
  if (!load_op)
    return std::nullopt;
  rdna4::VflatMachineInst inst{};
  std::memcpy(&inst, words.data(), sizeof(inst));
  inst.op = *load_op;
  inst.vdst = vdst;
  inst.vsrc = 0;
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

} // namespace

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

std::optional<std::array<uint32_t, 2>>
consan_sc_build_cdna_accvgpr_read_b32(uint16_t dst_vgpr, uint16_t src_accvgpr,
                                      const ConSanTargetProfile &target) {
  if ((target.encoding_family != ConSanEncodingFamily::Gfx9Cdna3 &&
       target.encoding_family != ConSanEncodingFamily::Gfx9Cdna4) ||
      dst_vgpr > 255u || src_accvgpr > 255u) {
    return std::nullopt;
  }
  if (target.encoding_family == ConSanEncodingFamily::Gfx9Cdna3) {
    return cdna3::build_vop3p(cdna3::kVAccvgprReadVop3p,
                              {.vdst = static_cast<uint8_t>(dst_vgpr),
                               .op_sel_hi_2 = 1u,
                               .src0 = static_cast<uint16_t>(256u + src_accvgpr),
                               .op_sel_hi = 3u});
  }
  return cdna4::build_vop3p(cdna4::kVAccvgprReadVop3p,
                            {.vdst = static_cast<uint8_t>(dst_vgpr),
                             .op_sel_hi_2 = 1u,
                             .src0 = static_cast<uint16_t>(256u + src_accvgpr),
                             .op_sel_hi = 3u});
}

std::optional<std::vector<uint32_t>> consan_sc_build_split_single_address_lds_pair(
    ConSanScTwoAddressLdsByteOffsets offsets, uint16_t element_dwords, uint16_t address_vgpr,
    uint16_t first_data_vgpr, uint16_t second_data_vgpr, uint16_t adjusted_address_vgpr, bool load,
    const ConSanTargetProfile &target) {
  if (address_vgpr > 255u || adjusted_address_vgpr > 255u || first_data_vgpr > 255u ||
      second_data_vgpr > 255u ||
      (element_dwords == 2u && (first_data_vgpr > 254u || second_data_vgpr > 254u)) ||
      (element_dwords != 1u && element_dwords != 2u)) {
    return std::nullopt;
  }
  const uint16_t op = load ? (element_dwords == 1u ? cdna5::kDsLoadB32Vds : cdna5::kDsLoadB64Vds)
                           : (element_dwords == 1u ? cdna5::kDsStoreB32Vds : cdna5::kDsStoreB64Vds);
  const uint32_t expected_words = 4u + 2u * (static_cast<uint32_t>(offsets.first > UINT16_MAX) +
                                             static_cast<uint32_t>(offsets.second > UINT16_MAX));
  std::vector<uint32_t> words;
  words.reserve(expected_words);
  const auto append_access = [&](uint32_t byte_offset, uint16_t data_vgpr) -> bool {
    uint16_t effective_address_vgpr = address_vgpr;
    uint16_t immediate = 0;
    if (byte_offset > UINT16_MAX) {
      const auto adjust = instrumentation::build_v_add_u32_literal(
          adjusted_address_vgpr, adjusted_address_vgpr, byte_offset, address_vgpr, target.arch);
      if (!adjust)
        return false;
      words.insert(words.end(), adjust->begin(), adjust->end());
      effective_address_vgpr = adjusted_address_vgpr;
    } else {
      immediate = static_cast<uint16_t>(byte_offset);
    }
    cdna5::VdsBuilderFields fields{
        .offset0 = static_cast<uint8_t>(immediate),
        .offset1 = static_cast<uint8_t>(immediate >> 8u),
        .addr = static_cast<uint8_t>(effective_address_vgpr),
    };
    if (load)
      fields.vdst = static_cast<uint8_t>(data_vgpr);
    else
      fields.data0 = static_cast<uint8_t>(data_vgpr);
    const auto access = cdna5::build_vds(op, fields);
    words.insert(words.end(), access.begin(), access.end());
    return true;
  };
  if (!append_access(offsets.first, first_data_vgpr) ||
      !append_access(offsets.second, second_data_vgpr) || words.size() != expected_words) {
    return std::nullopt;
  }
  return words;
}

std::optional<std::array<uint32_t, 3>> retarget_flat_load_vdst(std::array<uint32_t, 3> words,
                                                               uint16_t vdst, rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch) || consan_uses_gfx11_encoding(arch)) {
    words[1] = (words[1] & 0x00FFFFFFu) | (static_cast<uint32_t>(vdst) << 24u);
    return words;
  }
  if (!consan_uses_gfx12_encoding(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
    words[1] = (words[1] & ~0xffu) | vdst;
    return words;
  }
  rdna4::VflatMachineInst inst{};
  std::memcpy(&inst, words.data(), sizeof(inst));
  inst.vdst = vdst;
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

std::optional<std::array<uint32_t, 3>>
build_flat_load_from_flat_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst,
                                rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return build_rdna4_flat_load_from_store(words, width_bits, vdst);
  if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
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
    constexpr uint32_t kRdna3FlatOpMask = 0x7fu << 18u;
    constexpr uint32_t kRdna3FlatDataMask = 0xffu << 8u;
    words[0] = (words[0] & ~kRdna3FlatOpMask) | (static_cast<uint32_t>(load_op) << 18u);
    words[1] =
        (words[1] & ~kRdna3FlatDataMask & 0x00ffffffu) | (static_cast<uint32_t>(vdst) << 24u);
    return words;
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
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
    constexpr uint32_t kGfx1250FlatOpMask = 0xffu << 14u;
    constexpr uint32_t kGfx1250FlatVsrcMask = 0xffu << 23u;
    words[0] = (words[0] & ~kGfx1250FlatOpMask) | (static_cast<uint32_t>(load_op) << 14u);
    words[1] = (words[1] & ~0xffu & ~kGfx1250FlatVsrcMask) | vdst;
    return words;
  }
  if (!consan_uses_gfx9_cdna_encoding(arch))
    return std::nullopt;
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
  constexpr uint32_t kCdnaFlatOpMask = 0x7fu << 18u;
  words[0] = (words[0] & ~kCdnaFlatOpMask) | (load_op << 18u);
  constexpr uint32_t kCdnaFlatDataMask = 0xffu << 8u;
  words[1] = (words[1] & ~kCdnaFlatDataMask & 0x00FFFFFFu) | (static_cast<uint32_t>(vdst) << 24u);
  return words;
}

} // namespace rocjitsu
