// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/targets/consan_supercollider_target_ops.h"

#include "rocjitsu/code/patch/consan/targets/consan_supercollider_target_ops_internal.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_lds_ops.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <climits>

namespace rocjitsu {

std::optional<uint32_t> consan_sc_build_guest_flat_completion_wait(rj_code_arch_t arch) {
  return consan_arch_is_cdna3_or_cdna4(arch) ? instrumentation::build_s_wait_flat_load0(arch)
                                             : instrumentation::build_s_wait_lds0(arch);
}

std::optional<std::vector<uint32_t>>
consan_sc_build_delay_words(rj_code_arch_t arch, const ConSanRequest &request,
                            std::vector<std::string> &errors, std::string_view context) {
  std::vector<uint32_t> words;
  if (request.delay_nops == 0)
    return words;

  switch (request.delay_mode) {
  case ConSanDelayMode::Nop:
    for (uint32_t i = 0; i < request.delay_nops; ++i)
      words.push_back(build_s_nop(0, arch));
    return words;
  case ConSanDelayMode::Sleep:
    if (request.delay_nops > UINT16_MAX) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return std::nullopt;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(request.delay_nops), arch));
    return words;
  case ConSanDelayMode::SleepVar:
    if (request.delay_var_ssrc > 255) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return std::nullopt;
    }
    words.push_back(build_s_sleep_var(request.delay_var_ssrc, arch));
    return words;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      consan_delay_mode_name(request.delay_mode) + "'");
  return std::nullopt;
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
  if (consan_arch_is_rdna3_rdna4_or_cdna5(arch))
    return consan_sc_target_detail::build_rdna3_rdna4_cdna5_ds_load_word0(form, word0);
  if (consan_arch_is_cdna3_or_cdna4(arch))
    return consan_sc_target_detail::build_cdna3_cdna4_ds_load_word0(form, word0);
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
  return consan_sc_target_detail::build_cdna3_cdna4_accvgpr_read_b32(dst_vgpr, src_accvgpr,
                                                                     target.arch);
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
  if (consan_arch_is_cdna3_or_cdna4(arch) || consan_arch_is_rdna3(arch)) {
    return consan_sc_target_detail::retarget_cdna3_cdna4_rdna3_flat_load_vdst(words, vdst);
  }
  if (consan_arch_is_cdna5(arch))
    return consan_sc_target_detail::retarget_cdna5_flat_load_vdst(words, vdst);
  return consan_arch_is_rdna4_or_cdna5(arch)
             ? consan_sc_target_detail::retarget_rdna4_flat_load_vdst(words, vdst)
             : std::nullopt;
}

std::optional<std::array<uint32_t, 3>>
build_flat_load_from_flat_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst,
                                rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  switch (target->arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return consan_sc_target_detail::build_cdna3_cdna4_flat_load_from_store(words, width_bits, vdst);
  case ROCJITSU_CODE_ARCH_RDNA3:
    return consan_sc_target_detail::build_rdna3_flat_load_from_store(words, width_bits, vdst);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return consan_sc_target_detail::build_rdna4_flat_load_from_store(words, width_bits, vdst);
  case ROCJITSU_CODE_ARCH_CDNA5:
    return consan_sc_target_detail::build_cdna5_flat_load_from_store(words, width_bits, vdst);
  default:
    return std::nullopt;
  }
}

} // namespace rocjitsu
