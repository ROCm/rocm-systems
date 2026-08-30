// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops_internal.h"

namespace rocjitsu {

std::optional<ConSanScratchComponentEncoding>
decode_consan_scratch_component_encoding(std::span<const uint8_t> instruction,
                                         rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch))
    return consan_program_analysis_target_detail::decode_gfx9_cdna_scratch_component(instruction);
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return consan_program_analysis_target_detail::decode_gfx1201_scratch_component(instruction);
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return consan_program_analysis_target_detail::decode_gfx1250_scratch_component(instruction);
  return std::nullopt;
}

std::optional<ConSanPrivateComponentEncoding>
decode_consan_private_component_encoding(std::span<const uint8_t> instruction,
                                         rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch))
    return consan_program_analysis_target_detail::decode_gfx9_cdna_private_component(instruction);
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return consan_program_analysis_target_detail::decode_gfx1201_private_component(instruction);
  return std::nullopt;
}

std::optional<ConSanLaneTransferEncoding>
decode_consan_lane_transfer_encoding(std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch))
    return consan_program_analysis_target_detail::decode_gfx9_cdna_lane_transfer(instruction);
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return consan_program_analysis_target_detail::decode_gfx1201_lane_transfer(instruction);
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return consan_program_analysis_target_detail::decode_gfx1250_lane_transfer(instruction);
  return std::nullopt;
}

std::optional<ConSanAccvgprTransferEncoding>
decode_consan_accvgpr_transfer_index(std::span<const uint8_t> instruction, rj_code_arch_t arch,
                                     bool write_accumulator) {
  if (consan_uses_gfx9_cdna_encoding(arch)) {
    return consan_program_analysis_target_detail::decode_gfx9_cdna_accvgpr_transfer(
        instruction, write_accumulator);
  }
  return std::nullopt;
}

ConSanVectorMemoryDecode decode_consan_flat_memory_encoding(std::span<const uint8_t> instruction,
                                                            rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch))
    return consan_program_analysis_target_detail::decode_gfx9_cdna_flat_memory(instruction);
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return consan_program_analysis_target_detail::decode_gfx1100_flat_memory(instruction);
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return consan_program_analysis_target_detail::decode_gfx1201_flat_memory(instruction);
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return consan_program_analysis_target_detail::decode_gfx1250_flat_memory(instruction);
  return {};
}

ConSanVectorMemoryDecode decode_consan_global_memory_encoding(std::span<const uint8_t> instruction,
                                                              rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch))
    return consan_program_analysis_target_detail::decode_gfx9_cdna_global_memory(instruction);
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return consan_program_analysis_target_detail::decode_gfx1100_global_memory(instruction);
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return consan_program_analysis_target_detail::decode_gfx1201_global_memory(instruction);
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return consan_program_analysis_target_detail::decode_gfx1250_global_memory(instruction);
  return {};
}

ConSanBufferMemoryDecode decode_consan_buffer_memory_encoding(std::span<const uint8_t> instruction,
                                                              rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return consan_program_analysis_target_detail::decode_gfx1250_buffer_memory(instruction);
  return {};
}

std::optional<ConSanDirectLdsTransferEncoding> decode_consan_direct_lds_transfer_encoding(
    std::string_view mnemonic, std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch)) {
    return consan_program_analysis_target_detail::decode_gfx9_cdna_direct_lds_transfer(mnemonic,
                                                                                       instruction);
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
    return consan_program_analysis_target_detail::decode_gfx1250_direct_lds_transfer(mnemonic,
                                                                                     instruction);
  }
  return std::nullopt;
}

bool decode_consan_atomic_site_encoding(ConSanAtomicSite &site, std::string_view mnemonic,
                                        std::span<const uint8_t> instruction, rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch)) {
    return consan_program_analysis_target_detail::decode_gfx9_cdna_atomic_site(site, mnemonic,
                                                                               instruction);
  }
  if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
    return consan_program_analysis_target_detail::decode_gfx1100_atomic_site(site, mnemonic,
                                                                             instruction);
  }
  if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
    return consan_program_analysis_target_detail::decode_gfx1201_atomic_site(site, mnemonic,
                                                                             instruction);
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
    return consan_program_analysis_target_detail::decode_gfx1250_atomic_site(site, mnemonic,
                                                                             instruction);
  }
  return false;
}

} // namespace rocjitsu
