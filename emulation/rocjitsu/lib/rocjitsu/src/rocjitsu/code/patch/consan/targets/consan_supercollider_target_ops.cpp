// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/targets/consan_supercollider_target_ops.h"

#include "rocjitsu/code/patch/consan/targets/consan_supercollider_target_ops_internal.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_lds_ops.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <bit>
#include <climits>

namespace rocjitsu::consan {

std::optional<uint32_t> supercollider_build_guest_flat_completion_wait(rj_code_arch_t arch) {
  return arch_is_cdna3_or_cdna4(arch) ? instrumentation::build_s_wait_flat_load0(arch)
                                      : instrumentation::build_s_wait_lds0(arch);
}

std::optional<std::vector<uint32_t>>
supercollider_build_delay_words(const TargetProfile &target, const Request &request,
                                uint16_t temporary_sgpr, std::vector<std::string> &errors,
                                std::string_view context) {
  const rj_code_arch_t arch = target.arch;
  std::vector<uint32_t> words;
  if (request.supercollider_delay_nops == 0)
    return words;

  switch (request.supercollider_delay_mode) {
  case SuperColliderDelayMode::Nop:
    for (uint32_t i = 0; i < request.supercollider_delay_nops; ++i)
      words.push_back(build_s_nop(0, arch));
    return words;
  case SuperColliderDelayMode::Sleep:
    if (request.supercollider_delay_nops > UINT16_MAX) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return std::nullopt;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(request.supercollider_delay_nops), arch));
    return words;
  case SuperColliderDelayMode::SleepWave: {
    const uint32_t maximum = request.supercollider_delay_nops;
    if (arch != ROCJITSU_CODE_ARCH_RDNA4 || maximum > 127u || !std::has_single_bit(maximum + 1u) ||
        temporary_sgpr >= REGISTER_SET_ALLOCATABLE_SGPRS) {
      errors.emplace_back(
          std::string(context) +
          " sleep_wave requires RDNA4, scalar scratch, and maximum 1/3/7/15/31/63/127");
      return std::nullopt;
    }
    // The VCC-save scratch is not live yet. Read a bounded resident-wave
    // identity without changing SCC, EXEC or VCC; no extra register allocation.
    const auto &identity = target.resident_wave_identity;
    const auto width = static_cast<uint8_t>(std::bit_width(maximum));
    // Include the SIMD bits at the top of this identity. Neighboring waves
    // can occupy different SIMDs with the same low resident-wave slot number.
    const auto offset = static_cast<uint8_t>(identity.bit_offset + identity.bit_width - width);
    const auto hwreg = build_hwreg_imm(identity.hwreg_id, offset, width);
    const auto read =
        hwreg ? instrumentation::build_s_getreg_b32(temporary_sgpr, *hwreg, arch) : std::nullopt;
    const auto wait = instrumentation::build_salu_dependency_delay(arch);
    if (!read || !wait) {
      errors.emplace_back(std::string(context) + " could not encode resident-wave delay");
      return std::nullopt;
    }
    words.push_back(*read);
    words.push_back(*wait);
    words.push_back(build_s_sleep_var(temporary_sgpr, arch));
    return words;
  }
  case SuperColliderDelayMode::SleepVar:
    if (request.supercollider_delay_var_ssrc > 255) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return std::nullopt;
    }
    words.push_back(build_s_sleep_var(request.supercollider_delay_var_ssrc, arch));
    return words;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      delay_mode_name(request.supercollider_delay_mode) + "'");
  return std::nullopt;
}

std::optional<uint32_t> supercollider_build_v_cmp_ne_u32(uint16_t src0, uint16_t vsrc1,
                                                         rj_code_arch_t arch) {
  if (src0 > 255 || vsrc1 > 255)
    return std::nullopt;
  return instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(src0), vsrc1, arch);
}

std::optional<uint32_t> supercollider_build_v_cmp_ne_u16(uint16_t src0, uint16_t vsrc1,
                                                         rj_code_arch_t arch) {
  if (src0 > 255 || vsrc1 > 255)
    return std::nullopt;
  return instrumentation::build_v_cmp_ne_u16_vcc(vector_source_vgpr(src0), vsrc1, arch);
}

std::optional<uint32_t> supercollider_build_ds_load_word0(const AccessLoweringForm &form,
                                                          uint32_t word0, rj_code_arch_t arch) {
  if (arch_is_rdna3(arch) || arch_is_rdna4_or_cdna5(arch))
    return supercollider_target_detail::build_rdna3_rdna4_cdna5_ds_load_word0(form, word0);
  if (arch_is_cdna3_or_cdna4(arch))
    return supercollider_target_detail::build_cdna3_cdna4_ds_load_word0(form, word0);
  return std::nullopt;
}

uint32_t supercollider_build_ds_load_word1(uint16_t addr_vgpr, uint16_t dst_vgpr) {
  return static_cast<uint32_t>(addr_vgpr) | (static_cast<uint32_t>(dst_vgpr) << 24u);
}

std::optional<SuperColliderDirectToLdsTransfer> supercollider_build_direct_to_lds_transfer(
    std::array<uint32_t, 2> original_words, uint32_t width_bits, uint16_t address_vgpr,
    uint16_t payload_vgpr, uint16_t readback_vgpr, rj_code_arch_t arch) {
  if (!arch_is_cdna3_or_cdna4(arch))
    return std::nullopt;
  return supercollider_target_detail::build_cdna3_cdna4_direct_to_lds_transfer(
      original_words, width_bits, address_vgpr, payload_vgpr, readback_vgpr, arch);
}

std::optional<SuperColliderTwoAddressLdsByteOffsets>
supercollider_two_address_lds_byte_offsets(const AccessLoweringForm &form, uint32_t word0) {
  if (form.kind != AccessLoweringFormKind::NativeTwoRange || form.encoded_offset_scale_bytes == 0u)
    return std::nullopt;
  return SuperColliderTwoAddressLdsByteOffsets{
      .first = (word0 & 0xffu) * form.encoded_offset_scale_bytes,
      .second = ((word0 >> 8u) & 0xffu) * form.encoded_offset_scale_bytes,
  };
}

std::optional<std::array<uint32_t, 2>>
supercollider_build_cdna_accvgpr_read_b32(uint16_t dst_vgpr, uint16_t src_accvgpr,
                                          const TargetProfile &target) {
  return supercollider_target_detail::build_cdna3_cdna4_accvgpr_read_b32(dst_vgpr, src_accvgpr,
                                                                         target.arch);
}

std::optional<std::vector<uint32_t>> supercollider_build_split_single_address_lds_pair(
    SuperColliderTwoAddressLdsByteOffsets offsets, uint16_t element_dwords, uint16_t address_vgpr,
    uint16_t first_data_vgpr, uint16_t second_data_vgpr, uint16_t adjusted_address_vgpr, bool load,
    const TargetProfile &target) {
  if (!target.requires_split_two_address_lds_relocation)
    return std::nullopt;
  return build_split_two_address_lds_pair({.first_byte_offset = offsets.first,
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
  const TargetProfile *target = target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  if (arch_is_cdna3_or_cdna4(arch) || arch_is_rdna3(arch)) {
    return supercollider_target_detail::retarget_cdna3_cdna4_rdna3_flat_load_vdst(words, vdst);
  }
  if (arch_is_cdna5(arch))
    return supercollider_target_detail::retarget_cdna5_flat_load_vdst(words, vdst);
  return arch_is_rdna4_or_cdna5(arch)
             ? supercollider_target_detail::retarget_rdna4_flat_load_vdst(words, vdst)
             : std::nullopt;
}

std::optional<std::array<uint32_t, 3>>
build_flat_load_from_flat_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst,
                                rj_code_arch_t arch) {
  const TargetProfile *target = target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  switch (target->arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return supercollider_target_detail::build_cdna3_cdna4_flat_load_from_store(words, width_bits,
                                                                               vdst);
  case ROCJITSU_CODE_ARCH_RDNA3:
    return supercollider_target_detail::build_rdna3_flat_load_from_store(words, width_bits, vdst);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return supercollider_target_detail::build_rdna4_flat_load_from_store(words, width_bits, vdst);
  case ROCJITSU_CODE_ARCH_CDNA5:
    return supercollider_target_detail::build_cdna5_flat_load_from_store(words, width_bits, vdst);
  default:
    return std::nullopt;
  }
}

} // namespace rocjitsu::consan
