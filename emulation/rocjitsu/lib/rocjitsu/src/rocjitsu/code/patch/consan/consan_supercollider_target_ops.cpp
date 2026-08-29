// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops.h"

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops_internal.h"
#include "rocjitsu/code/patch/consan/consan_target_lds_ops.h"
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
  if (consan_uses_gfx11_or_gfx12_encoding(arch))
    return consan_sc_target_detail::build_gfx11_gfx12_ds_load_word0(form, word0);
  if (consan_uses_gfx9_cdna_encoding(arch))
    return consan_sc_target_detail::build_gfx9_ds_load_word0(form, word0);
  return std::nullopt;
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
  return consan_sc_target_detail::build_gfx9_cdna_accvgpr_read_b32(dst_vgpr, src_accvgpr,
                                                                   target.encoding_family);
}

std::optional<std::vector<uint32_t>> consan_sc_build_split_single_address_lds_pair(
    ConSanScTwoAddressLdsByteOffsets offsets, uint16_t element_dwords, uint16_t address_vgpr,
    uint16_t first_data_vgpr, uint16_t second_data_vgpr, uint16_t adjusted_address_vgpr, bool load,
    const ConSanTargetProfile &target) {
  if (!target.requires_split_two_address_lds_relocation)
    return std::nullopt;
  return consan_build_split_two_address_lds_pair({.first_byte_offset = offsets.first,
                                                  .second_byte_offset = offsets.second,
                                                  .element_dwords = element_dwords,
                                                  .address_vgpr = address_vgpr,
                                                  .first_data_vgpr = first_data_vgpr,
                                                  .second_data_vgpr = second_data_vgpr,
                                                  .adjusted_address_vgpr = adjusted_address_vgpr,
                                                  .load = load},
                                                 target.arch);
}

std::optional<std::array<uint32_t, 3>> retarget_flat_load_vdst(std::array<uint32_t, 3> words,
                                                               uint16_t vdst, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  if (target->encoding_family == ConSanEncodingFamily::Gfx9Cdna3 ||
      target->encoding_family == ConSanEncodingFamily::Gfx9Cdna4 ||
      target->encoding_family == ConSanEncodingFamily::Gfx11) {
    return consan_sc_target_detail::retarget_classic_flat_load_vdst(words, vdst);
  }
  if (target->architecture_family == ConSanArchitectureFamily::Cdna)
    return consan_sc_target_detail::retarget_gfx1250_flat_load_vdst(words, vdst);
  return consan_sc_target_detail::retarget_rdna4_flat_load_vdst(words, vdst);
}

std::optional<std::array<uint32_t, 3>>
build_flat_load_from_flat_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst,
                                rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  switch (target->encoding_family) {
  case ConSanEncodingFamily::Gfx9Cdna3:
  case ConSanEncodingFamily::Gfx9Cdna4:
    return consan_sc_target_detail::build_gfx9_cdna_flat_load_from_store(words, width_bits, vdst);
  case ConSanEncodingFamily::Gfx11:
    return consan_sc_target_detail::build_rdna3_flat_load_from_store(words, width_bits, vdst);
  case ConSanEncodingFamily::Gfx12:
    return target->architecture_family == ConSanArchitectureFamily::Cdna
               ? consan_sc_target_detail::build_gfx1250_flat_load_from_store(words, width_bits,
                                                                             vdst)
               : consan_sc_target_detail::build_rdna4_flat_load_from_store(words, width_bits, vdst);
  }
  return std::nullopt;
}

} // namespace rocjitsu
