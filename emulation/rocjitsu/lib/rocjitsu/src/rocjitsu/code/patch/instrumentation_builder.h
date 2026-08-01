// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instrumentation_builder.h
/// @brief Architecture-neutral instruction builders used by DBI instrumentation.
///
/// Target-specific encodings remain in their architecture-named headers.  This
/// facade is the only layer from which an instrumentation engine should select
/// between those backends.

#pragma once

#include "rocjitsu/code/patch/cdna3_instrumentation_builder.h"
#include "rocjitsu/code/patch/cdna4_instrumentation_builder.h"
#include "rocjitsu/code/patch/gfx1250_instrumentation_builder.h"
#include "rocjitsu/code/patch/rdna3_instrumentation_builder.h"
#include "rocjitsu/code/patch/rdna4_instrumentation_builder.h"

namespace rocjitsu::instrumentation {

[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_set_vgpr_msb(uint16_t mode,
                                                                            rj_code_arch_t arch) {
  return build_gfx1250_s_set_vgpr_msb(mode, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_set_vgpr_msb_transition(uint8_t previous_mode, uint8_t new_mode, rj_code_arch_t arch) {
  return build_gfx1250_s_set_vgpr_msb_transition(previous_mode, new_mode, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_call_i64(uint16_t sdst, int16_t simm16, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250)
    return std::nullopt;
  return build_gfx1250_s_call_i64(sdst, simm16, arch);
}

[[nodiscard]] inline constexpr bool is_cdna_family_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] inline constexpr bool is_rdna_family_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || is_rdna4_family_arch(arch);
}

[[nodiscard]] inline constexpr bool is_admitted_arch(rj_code_arch_t arch) {
  return is_cdna_family_arch(arch) || is_rdna_family_arch(arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_getreg_b32(uint16_t sdst, uint16_t hwreg, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_getreg_b32(sdst, hwreg, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_getreg_b32(sdst, hwreg, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_s_getreg_b32(sdst, hwreg, arch);
  return rocjitsu::build_s_getreg_b32(sdst, hwreg, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_mov_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_mov_b64(sdst, ssrc0, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_mov_b64(sdst, ssrc0, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_mov_b64(sdst, ssrc0, arch)
                                          : rocjitsu::build_s_mov_b64(sdst, ssrc0, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_and_saveexec_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_and_saveexec_b64(sdst, ssrc0, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_and_saveexec_b64(sdst, ssrc0, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_and_saveexec_b64(sdst, ssrc0, arch)
                                          : rocjitsu::build_s_and_saveexec_b64(sdst, ssrc0, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_andn2_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_andn2_b64(sdst, ssrc0, ssrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_andn2_b64(sdst, ssrc0, ssrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? build_cdna4_s_andn2_b64(sdst, ssrc0, ssrc1, arch)
             : rocjitsu::build_s_and_not1_b64(sdst, ssrc0, ssrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_and_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_and_b64(sdst, ssrc0, ssrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_and_b64(sdst, ssrc0, ssrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_and_b64(sdst, ssrc0, ssrc1, arch)
                                          : rocjitsu::build_s_and_b64(sdst, ssrc0, ssrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_bcnt1_i32_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_bcnt1_i32_b64(sdst, ssrc0, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_bcnt1_i32_b64(sdst, ssrc0, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_bcnt1_i32_b64(sdst, ssrc0, arch)
                                          : rocjitsu::build_s_bcnt1_i32_b64(sdst, ssrc0, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_xor_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_xor_b64(sdst, ssrc0, ssrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_xor_b64(sdst, ssrc0, ssrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_xor_b64(sdst, ssrc0, ssrc1, arch)
                                          : rocjitsu::build_s_xor_b64(sdst, ssrc0, ssrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_sub_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_sub_u32(sdst, ssrc0, ssrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_sub_u32(sdst, ssrc0, ssrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_sub_u32(sdst, ssrc0, ssrc1, arch)
                                          : rocjitsu::build_s_sub_u32(sdst, ssrc0, ssrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_cselect_b32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_cselect_b32(sdst, ssrc0, ssrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_cselect_b32(sdst, ssrc0, ssrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_cselect_b32(sdst, ssrc0, ssrc1, arch)
                                          : rocjitsu::build_s_cselect_b32(sdst, ssrc0, ssrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_cmp_lg_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_cmp_lg_u32(ssrc0, ssrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_cmp_lg_u32(ssrc0, ssrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_cmp_lg_u32(ssrc0, ssrc1, arch)
                                          : rocjitsu::build_s_cmp_lg_u32(ssrc0, ssrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_cmp_eq_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_cmp_eq_u32(ssrc0, ssrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_cmp_eq_u32(ssrc0, ssrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_cmp_eq_u32(ssrc0, ssrc1, arch)
                                          : rocjitsu::build_s_cmp_eq_u32(ssrc0, ssrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_s_load_dword(uint16_t sdst, uint16_t sbase, uint32_t byte_offset, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_load_dword(sdst, sbase, byte_offset, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_load_dword(sdst, sbase, byte_offset, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_s_load_dword(sdst, sbase, byte_offset, arch);
  return std::nullopt;
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_scc0(int16_t offset_dwords,
                                                                            rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_cbranch(rdna3::kSCbranchScc0Sopp, offset_dwords, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_cbranch(cdna3::kSCbranchScc0Sopp, offset_dwords, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_cbranch_scc0(offset_dwords, arch)
                                          : rocjitsu::build_s_cbranch_scc0(offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_scc1(int16_t offset_dwords,
                                                                            rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_cbranch(rdna3::kSCbranchScc1Sopp, offset_dwords, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_cbranch(cdna3::kSCbranchScc1Sopp, offset_dwords, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_cbranch_scc1(offset_dwords, arch)
                                          : rocjitsu::build_s_cbranch_scc1(offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_vccz(int16_t offset_dwords,
                                                                            rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_cbranch(rdna3::kSCbranchVcczSopp, offset_dwords, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_cbranch(cdna3::kSCbranchVcczSopp, offset_dwords, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_cbranch_vccz(offset_dwords, arch)
                                          : rocjitsu::build_s_cbranch_vccz(offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_vccnz(int16_t offset_dwords,
                                                                             rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_cbranch(rdna3::kSCbranchVccnzSopp, offset_dwords, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_cbranch(cdna3::kSCbranchVccnzSopp, offset_dwords, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_cbranch_vccnz(offset_dwords, arch)
                                          : rocjitsu::build_s_cbranch_vccnz(offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_execz(int16_t offset_dwords,
                                                                             rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_cbranch(rdna3::kSCbranchExeczSopp, offset_dwords, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_cbranch(cdna3::kSCbranchExeczSopp, offset_dwords, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_cbranch_execz(offset_dwords, arch)
                                          : rocjitsu::build_s_cbranch_execz(offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_execnz(int16_t offset_dwords,
                                                                              rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_cbranch(rdna3::kSCbranchExecnzSopp, offset_dwords, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_cbranch(cdna3::kSCbranchExecnzSopp, offset_dwords, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_cbranch_execnz(offset_dwords, arch)
                                          : rocjitsu::build_s_cbranch_execnz(offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_lshrrev_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_admitted_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_lshrrev_b32(vdst, src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_lshrrev_b32(vdst, src0, vsrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? build_cdna4_v_lshrrev_b32(vdst, src0, vsrc1, arch)
             : rocjitsu::build_v_lshrrev_b32_e32(vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_lshlrev_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_admitted_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_lshlrev_b32(vdst, src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_lshlrev_b32(vdst, src0, vsrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? build_cdna4_v_lshlrev_b32(vdst, src0, vsrc1, arch)
             : rocjitsu::build_v_lshlrev_b32_e32(vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_and_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_admitted_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_and_b32(vdst, src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_and_b32(vdst, src0, vsrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_v_and_b32(vdst, src0, vsrc1, arch)
                                          : rocjitsu::build_v_and_b32_e32(vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_min_u32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_admitted_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_min_u32(vdst, src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_min_u32(vdst, src0, vsrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_v_min_u32(vdst, src0, vsrc1, arch)
                                          : rocjitsu::build_v_min_u32_e32(vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_xor_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_admitted_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_xor_b32(vdst, src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_xor_b32(vdst, src0, vsrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_v_xor_b32(vdst, src0, vsrc1, arch)
                                          : rocjitsu::build_v_xor_b32_e32(vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_eq_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_cmp_eq_u32_vcc(src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_cmp_eq_u32_vcc(src0, vsrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_v_cmp_eq_u32_vcc(src0, vsrc1, arch)
                                          : rocjitsu::build_v_cmp_eq_u32_e32_vcc(src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_ne_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_cmp_ne_u32_vcc(src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_cmp_ne_u32_vcc(src0, vsrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_v_cmp_ne_u32_vcc(src0, vsrc1, arch)
                                          : rocjitsu::build_v_cmp_ne_u32_e32_vcc(src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_ne_u16_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_cmp_ne_u16_vcc(src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_cmp_ne_u16_vcc(src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_v_cmp_ne_u16_vcc(src0, vsrc1, arch);
  return rocjitsu::build_v_cmp_ne_u16_e32_vcc(src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_gt_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_cmp_gt_u32_vcc(src0, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_cmp_gt_u32_vcc(src0, vsrc1, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_v_cmp_gt_u32_vcc(src0, vsrc1, arch)
                                          : rocjitsu::build_v_cmp_gt_u32_e32_vcc(src0, vsrc1, arch);
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_cmp_gt_u32_literal_vcc(uint32_t literal, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
    const auto compare = build_cdna3_v_cmp_gt_u32_vcc(kVopLiteralSource, vsrc1, arch);
    if (!compare)
      return std::nullopt;
    return std::vector<uint32_t>{*compare, literal};
  }
  if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
    const auto compare = build_rdna3_v_cmp_gt_u32_vcc(kVopLiteralSource, vsrc1, arch);
    if (!compare)
      return std::nullopt;
    return std::vector<uint32_t>{*compare, literal};
  }
  const auto words = rocjitsu::build_v_cmp_gt_u32_e32_vcc_literal(literal, vsrc1, arch);
  if (!words)
    return std::nullopt;
  return std::vector<uint32_t>(words->begin(), words->end());
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_readfirstlane_b32(uint16_t sdst, uint16_t vsrc, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_readfirstlane_b32(sdst, vsrc, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_readfirstlane_b32(sdst, vsrc, arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_v_readfirstlane_b32(sdst, vsrc, arch)
                                          : rocjitsu::build_v_readfirstlane_b32(sdst, vsrc, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_writelane_b32(uint16_t vdst, uint16_t ssrc, uint16_t lane, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_writelane_b32(vdst, ssrc, lane, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_writelane_b32(vdst, ssrc, lane, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_v_writelane_b32(vdst, ssrc, lane, arch);
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return build_rdna4_v_writelane_b32(vdst, ssrc, lane, arch);
  return arch == ROCJITSU_CODE_ARCH_GFX1250 ? build_gfx1250_v_writelane_b32(vdst, ssrc, lane, arch)
                                            : std::nullopt;
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_readlane_b32(uint16_t sdst, uint16_t vsrc, uint16_t lane, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_readlane_b32(sdst, vsrc, lane, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_readlane_b32(sdst, vsrc, lane, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_v_readlane_b32(sdst, vsrc, lane, arch);
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return build_rdna4_v_readlane_b32(sdst, vsrc, lane, arch);
  return arch == ROCJITSU_CODE_ARCH_GFX1250 ? build_gfx1250_v_readlane_b32(sdst, vsrc, lane, arch)
                                            : std::nullopt;
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_flat_load0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_wait_vmcnt_lgkmcnt0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_wait_vmcnt_lgkmcnt0(arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_flat0(arch)
                                          : rocjitsu::build_s_wait_loadcnt_dscnt0(arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_flat_store0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_wait_vmcnt_lgkmcnt0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_wait_vmcnt_lgkmcnt0(arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_flat0(arch)
                                          : rocjitsu::build_s_wait_storecnt_dscnt0(arch);
}

// Instrumentation-owned global memory never aliases LDS. Keep these waits
// separate from build_s_wait_flat_* so report-buffer traffic does not
// unnecessarily drain outstanding LDS operations on architectures with
// independent counters.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_global_load0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_wait_vmcnt0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_wait_vmcnt0(arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                          : rocjitsu::build_s_wait_loadcnt0(arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_global_store0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_wait_vmcnt0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_wait_vmcnt0(arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                          : rocjitsu::build_s_wait_storecnt0(arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_wait_lds0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_wait_lgkmcnt0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_s_wait_lds0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_wait_lgkmcnt0(arch);
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return pack_sopp(rdna4::kSWaitDscntSopp, 0);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_scalar_load0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_wait_lgkmcnt0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_s_wait_scalar_load0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_wait_lgkmcnt0(arch);
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return pack_sopp(rdna4::kSWaitKmcntSopp, 0);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_flat_load_lds0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_wait_vmcnt_lgkmcnt0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_wait_vmcnt_lgkmcnt0(arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_flat0(arch)
                                          : rocjitsu::build_s_wait_loadcnt_dscnt0(arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_flat_store_lds0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_wait_vmcnt_lgkmcnt0(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_s_wait_vmcnt_lgkmcnt0(arch);
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_flat0(arch)
                                          : rocjitsu::build_s_wait_storecnt_dscnt0(arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_salu_dependency_delay(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_salu_dependency_delay(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_s_nop(0, arch);
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return rocjitsu::build_s_delay_alu(kDelayAluSaluDep1, arch);
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return rocjitsu::build_s_delay_alu(kDelayAluSaluDep1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_indirect_pc0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_salu_dependency_delay(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_s_nop(0, arch);
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return rocjitsu::build_s_delay_alu(kDelayAluSaluDep1, arch);
  return rocjitsu::build_s_wait_alu_sa_sdst0(arch);
}

/// @brief Separate a SALU-produced SGPR from a direct VALU operand consumer.
///
/// gfx12 exposes this dependency through SA_SDST. CDNA targets do not have the
/// gfx12 depctr instruction, so use the target's documented scalar-to-vector
/// scheduling separation. This helper is deliberately limited to direct VALU
/// operands such as `v_mov` and `v_mbcnt`; it is not a generic VMEM-address or
/// lane-index dependency primitive.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_salu_to_valu_dependency_wait(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_salu_dependency_delay(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_s_nop(0, arch);
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return rocjitsu::build_s_delay_alu(kDelayAluSaluDep1, arch);
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return rocjitsu::build_s_wait_alu_sa_sdst0(arch);
}

/// @brief Separate a fixed-lane VALU transfer from its direct scalar consumer.
///
/// gfx12 exposes this dependency through VA_SDST. CDNA targets do not have the
/// gfx12 depctr instruction, so use the target's documented vector-to-scalar
/// scheduling separation. Callers use this after `v_readlane_b32`; it is not a
/// general vector-memory completion wait.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_valu_to_salu_dependency_wait(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_salu_dependency_delay(arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_s_nop(0, arch);
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return rocjitsu::build_s_delay_alu(kDelayAluSaluDep1, arch);
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return rocjitsu::build_s_wait_alu_va_sdst0(arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_alu_va_sdst0(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return rocjitsu::build_s_delay_alu(kDelayAluSaluDep1, arch);
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return rocjitsu::build_s_wait_alu_va_sdst0(arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_trap(uint16_t simm16,
                                                                    rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_s_trap(simm16, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_s_trap(simm16, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return rocjitsu::build_s_trap(arch, simm16);
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, rdna4::kSTrapSopp, simm16);
}

template <size_t N>
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
copy_words(const std::optional<std::array<uint32_t, N>> &words) {
  if (!words)
    return std::nullopt;
  return std::vector<uint32_t>(words->begin(), words->end());
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_mov_b32_literal(uint16_t vdst, uint32_t literal, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_v_mov_b32_literal(vdst, literal, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_v_mov_b32_literal(vdst, literal, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_v_mov_b32_literal(vdst, literal, arch))
             : copy_words(rocjitsu::build_v_mov_b32_e64_literal(vdst, literal, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_and_b32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_admitted_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_vop2_literal(cdna3::kVAndB32Vop2, vdst, literal, vsrc1, arch));
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_vop2_literal(rdna3::kVAndB32Vop2, vdst, literal, vsrc1, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_v_and_b32_literal(vdst, literal, vsrc1, arch))
             : copy_words(rocjitsu::build_v_and_b32_e32_literal(vdst, literal, vsrc1, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_min_u32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_admitted_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_vop2_literal(cdna3::kVMinU32Vop2, vdst, literal, vsrc1, arch));
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_vop2_literal(rdna3::kVMinU32Vop2, vdst, literal, vsrc1, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_v_min_u32_literal(vdst, literal, vsrc1, arch))
             : copy_words(rocjitsu::build_v_min_u32_e32_literal(vdst, literal, vsrc1, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_add_u32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_admitted_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
    const auto word = build_cdna3_v_add_u32(vdst, src0, vsrc1, arch);
    if (!word)
      return std::nullopt;
    return std::vector<uint32_t>{*word};
  }
  if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
    const auto word = build_rdna3_v_add_u32(vdst, src0, vsrc1, arch);
    if (!word)
      return std::nullopt;
    return std::vector<uint32_t>{*word};
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return copy_words(build_cdna4_v_add_u32(vdst, src0, vsrc1, arch));
  const auto word = rocjitsu::build_v_add_nc_u32_e32(vdst, src0, vsrc1, arch);
  if (!word)
    return std::nullopt;
  return std::vector<uint32_t>{*word};
}

/// @brief Add a literal using an explicit materialization VGPR when required.
///
/// CDNA4 materializes the literal in @p literal_vgpr, which is clobbered and
/// must not alias @p vsrc1. Targets with a native literal form ignore
/// @p literal_vgpr and leave it untouched. Callers should still provide a
/// proven-dead register so the same recipe remains valid across targets.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_add_u32_literal(uint16_t vdst, uint16_t literal_vgpr, uint32_t literal, uint16_t vsrc1,
                        rj_code_arch_t arch) {
  if (!is_admitted_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_vop2_literal(cdna3::kVAddU32Vop2, vdst, literal, vsrc1, arch));
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_vop2_literal(rdna3::kVAddNcU32Vop2, vdst, literal, vsrc1, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? build_cdna4_v_add_u32_literal(vdst, literal_vgpr, literal, vsrc1, arch)
             : copy_words(rocjitsu::build_v_add_nc_u32_e32_literal(vdst, literal, vsrc1, arch));
}

/// @brief Add a literal using the destination as any required materialization VGPR.
///
/// CDNA4 deliberately rejects an in-place result through this compact overload.
/// Call the overload with an explicit, proven-dead materialization VGPR when
/// @p vdst aliases @p vsrc1.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_add_u32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_v_add_u32_literal(vdst, vdst, literal, vsrc1, arch);
}

/// @brief Multiply a VGPR by a literal using an explicit materialization VGPR
/// when required by the target ISA.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_mul_lo_u32_literal(uint16_t vdst, uint16_t literal_vgpr, uint32_t literal, uint16_t vsrc1,
                           rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_v_mul_lo_u32_literal(vdst, literal, vsrc1, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_mul_lo_u32_literal(vdst, literal_vgpr, literal, vsrc1, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_v_mul_lo_u32_literal(vdst, literal_vgpr, literal, vsrc1, arch);
  return copy_words(rocjitsu::build_v_mul_lo_u32_vop3_literal(vdst, literal, vsrc1, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_mbcnt_lo_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_v_mbcnt_lo_u32_b32(vdst, src0, src1, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_v_mbcnt_lo_u32_b32(vdst, src0, src1, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return copy_words(build_cdna4_v_mbcnt_lo_u32_b32(vdst, src0, src1, arch));
  return copy_words(rocjitsu::build_v_mbcnt_lo_u32_b32(vdst, src0, src1, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_mbcnt_hi_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_v_mbcnt_hi_u32_b32(vdst, src0, src1, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_v_mbcnt_hi_u32_b32(vdst, src0, src1, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return copy_words(build_cdna4_v_mbcnt_hi_u32_b32(vdst, src0, src1, arch));
  return copy_words(rocjitsu::build_v_mbcnt_hi_u32_b32(vdst, src0, src1, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_add_u64_vgpr_offset(uint16_t address_vgpr, uint16_t offset_vgpr, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_add_u64_vgpr_offset(address_vgpr, offset_vgpr, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_add_u64_vgpr_offset(address_vgpr, offset_vgpr, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_v_add_u64_vgpr_offset(address_vgpr, offset_vgpr, arch);
  return copy_words(rocjitsu::build_v_add_u64_vgpr_offset(address_vgpr, offset_vgpr, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_add_u64_literal(uint16_t address_vgpr, uint64_t literal, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_add_u64_literal(address_vgpr, literal, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_add_u64_literal(address_vgpr, literal, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_v_add_u64_literal(address_vgpr, literal, arch);
  return rocjitsu::build_v_add_u64_literal(address_vgpr, literal, arch);
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_add_u64_signed_vgpr_offset(uint16_t address_vgpr, uint16_t offset_vgpr, uint16_t sign_vgpr,
                                   rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_add_u64_signed_vgpr_offset(address_vgpr, offset_vgpr, sign_vgpr, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_add_u64_signed_vgpr_offset(address_vgpr, offset_vgpr, sign_vgpr, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_v_add_u64_signed_vgpr_offset(address_vgpr, offset_vgpr, sign_vgpr, arch);
  return std::nullopt;
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_v_add_u64_signed_i24(uint16_t address_vgpr, int32_t displacement, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return build_rdna3_v_add_u64_signed_i24(address_vgpr, displacement, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return build_cdna3_v_add_u64_signed_i24(address_vgpr, displacement, arch);
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return build_cdna4_v_add_u64_signed_i24(address_vgpr, displacement, arch);
  return copy_words(rocjitsu::build_v_add_u64_signed_i24(address_vgpr, displacement, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_flat_store_b32(uint16_t vaddr, uint16_t vsrc, rj_code_arch_t arch, uint32_t byte_offset = 0) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
    if (byte_offset > UINT16_MAX)
      return std::nullopt;
    return copy_words(
        build_rdna3_flat_store_b32(vaddr, vsrc, static_cast<uint16_t>(byte_offset), arch));
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
    if (byte_offset > UINT16_MAX)
      return std::nullopt;
    return copy_words(
        build_cdna3_flat_store_b32(vaddr, vsrc, static_cast<uint16_t>(byte_offset), arch));
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    if (byte_offset > UINT16_MAX)
      return std::nullopt;
    return copy_words(
        build_cdna4_flat_store_b32(vaddr, vsrc, static_cast<uint16_t>(byte_offset), arch));
  }
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return copy_words(build_gfx1250_flat_store_b32(vaddr, vsrc, byte_offset, arch));
  return copy_words(rocjitsu::build_flat_store_b32_vaddr_vsrc(vaddr, vsrc, arch, byte_offset));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_flat_load_b32(uint16_t vaddr, uint16_t vdst, rj_code_arch_t arch, uint32_t byte_offset = 0) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
    if (byte_offset > UINT16_MAX)
      return std::nullopt;
    return copy_words(
        build_rdna3_flat_load_b32(vaddr, vdst, static_cast<uint16_t>(byte_offset), arch));
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
    if (byte_offset > UINT16_MAX)
      return std::nullopt;
    return copy_words(
        build_cdna3_flat_load_b32(vaddr, vdst, static_cast<uint16_t>(byte_offset), arch));
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    if (byte_offset > UINT16_MAX)
      return std::nullopt;
    return copy_words(
        build_cdna4_flat_load_b32(vaddr, vdst, static_cast<uint16_t>(byte_offset), arch));
  }
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return copy_words(build_gfx1250_flat_load_b32(vaddr, vdst, byte_offset, arch));
  return copy_words(rocjitsu::build_flat_load_b32_vaddr_vdst(vaddr, vdst, arch, byte_offset));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_private_store_b32(uint16_t vsrc, uint32_t byte_offset, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_address_free_scratch_store_b32(vsrc, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_address_free_scratch_store_b32(vsrc, byte_offset, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_address_free_scratch_store_b32(vsrc, byte_offset, arch))
             : copy_words(rocjitsu::build_address_free_scratch_store_b32(vsrc, byte_offset, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_private_load_b32(uint16_t vdst, uint32_t byte_offset, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_address_free_scratch_load_b32(vdst, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_address_free_scratch_load_b32(vdst, byte_offset, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_address_free_scratch_load_b32(vdst, byte_offset, arch))
             : copy_words(rocjitsu::build_address_free_scratch_load_b32(vdst, byte_offset, arch));
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_private_load0(rj_code_arch_t arch) {
  return build_s_wait_global_load0(arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_private_store0(rj_code_arch_t arch) {
  return build_s_wait_global_store0(arch);
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_ds_store_b32(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_ds_store_b32(vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_ds_store_b32(vaddr, vdata, byte_offset, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_ds_store_b32(vaddr, vdata, byte_offset, arch))
             : copy_words(rocjitsu::build_ds_store_b32(vaddr, vdata, byte_offset, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_ds_store_b64(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_ds_store_b64(vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_ds_store_b64(vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return copy_words(rocjitsu::build_ds_store_b64(vaddr, vdata, byte_offset, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_ds_store_b128(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_ds_store_b128(vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_ds_store_b128(vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return copy_words(rocjitsu::build_ds_store_b128(vaddr, vdata, byte_offset, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_ds_storexchg_rtn_b64(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                           rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_ds_storexchg_rtn_b64(vdst, vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_ds_storexchg_rtn_b64(vdst, vaddr, vdata, byte_offset, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_ds_storexchg_rtn_b64(vdst, vaddr, vdata, byte_offset, arch))
             : copy_words(
                   rocjitsu::build_ds_storexchg_rtn_b64(vdst, vaddr, vdata, byte_offset, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_ds_storexchg_rtn_b32(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                           rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_ds_storexchg_rtn_b32(vdst, vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_ds_storexchg_rtn_b32(vdst, vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return copy_words(rocjitsu::build_ds_storexchg_rtn_b32(vdst, vaddr, vdata, byte_offset, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_ds_or_rtn_b32(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                    rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_ds_or_rtn_b32(vdst, vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_ds_or_rtn_b32(vdst, vaddr, vdata, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return copy_words(rocjitsu::build_ds_or_rtn_b32(vdst, vaddr, vdata, byte_offset, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_ds_load_b32(uint16_t vdst, uint16_t vaddr, uint8_t byte_offset, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(build_rdna3_ds_load_b32(vdst, vaddr, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(build_cdna3_ds_load_b32(vdst, vaddr, byte_offset, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return copy_words(rocjitsu::build_ds_load_b32(vdst, vaddr, byte_offset, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_flat_atomic_add_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                          uint8_t scope, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(
        build_rdna3_flat_atomic_add_u32(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return copy_words(
        build_gfx1250_flat_atomic_add_u32(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(
        build_cdna3_flat_atomic_add_u32(vaddr, vsrc, vdst, return_old_value, scope, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_flat_atomic_add_u32(vaddr, vsrc, vdst, return_old_value,
                                                          scope, arch))
             : copy_words(rocjitsu::build_flat_atomic_add_u32_vaddr_vsrc_vdst(
                   vaddr, vsrc, vdst, return_old_value, scope, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_flat_atomic_or_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                         uint8_t scope, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(
        build_rdna3_flat_atomic_or_u32(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return copy_words(
        build_gfx1250_flat_atomic_or_u32(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(
        build_cdna3_flat_atomic_or_u32(vaddr, vsrc, vdst, return_old_value, scope, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(
                   build_cdna4_flat_atomic_or_u32(vaddr, vsrc, vdst, return_old_value, scope, arch))
             : copy_words(rocjitsu::build_flat_atomic_or_u32_vaddr_vsrc_vdst(
                   vaddr, vsrc, vdst, return_old_value, scope, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_flat_atomic_cmpswap_b32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                              uint8_t scope, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(
        build_rdna3_flat_atomic_cmpswap_b32(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return copy_words(
        build_gfx1250_flat_atomic_cmpswap_b32(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(
        build_cdna3_flat_atomic_cmpswap_b32(vaddr, vsrc, vdst, return_old_value, scope, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_flat_atomic_cmpswap_b32(vaddr, vsrc, vdst, return_old_value,
                                                              scope, arch))
             : copy_words(rocjitsu::build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
                   vaddr, vsrc, vdst, return_old_value, scope, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_flat_atomic_cmpswap_b64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                              uint8_t scope, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(
        build_rdna3_flat_atomic_cmpswap_b64(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return copy_words(
        build_gfx1250_flat_atomic_cmpswap_b64(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(
        build_cdna3_flat_atomic_cmpswap_b64(vaddr, vsrc, vdst, return_old_value, scope, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_flat_atomic_cmpswap_b64(vaddr, vsrc, vdst, return_old_value,
                                                              scope, arch))
             : copy_words(rocjitsu::build_flat_atomic_cmpswap_b64_vaddr_vsrc_vdst(
                   vaddr, vsrc, vdst, return_old_value, scope, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_flat_atomic_swap_b64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                           uint8_t scope, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(
        build_rdna3_flat_atomic_swap_b64(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return copy_words(
        build_gfx1250_flat_atomic_swap_b64(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(
        build_cdna3_flat_atomic_swap_b64(vaddr, vsrc, vdst, return_old_value, scope, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_flat_atomic_swap_b64(vaddr, vsrc, vdst, return_old_value,
                                                           scope, arch))
             : copy_words(rocjitsu::build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
                   vaddr, vsrc, vdst, return_old_value, scope, arch));
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_flat_atomic_add_u64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                          uint8_t scope, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3)
    return copy_words(
        build_rdna3_flat_atomic_add_u64(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return copy_words(
        build_gfx1250_flat_atomic_add_u64(vaddr, vsrc, vdst, return_old_value, scope, arch));
  if (arch == ROCJITSU_CODE_ARCH_CDNA3)
    return copy_words(
        build_cdna3_flat_atomic_add_u64(vaddr, vsrc, vdst, return_old_value, scope, arch));
  return arch == ROCJITSU_CODE_ARCH_CDNA4
             ? copy_words(build_cdna4_flat_atomic_add_u64(vaddr, vsrc, vdst, return_old_value,
                                                          scope, arch))
             : copy_words(rocjitsu::build_flat_atomic_add_u64_vaddr_vsrc_vdst(
                   vaddr, vsrc, vdst, return_old_value, scope, arch));
}

/// @brief Build a workgroup barrier without an added memory drain.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_workgroup_barrier_only(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
    const auto barrier = build_rdna3_s_barrier(arch);
    if (!barrier)
      return std::nullopt;
    return std::vector<uint32_t>{*barrier};
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
    const auto barrier = build_cdna3_s_barrier(arch);
    if (!barrier)
      return std::nullopt;
    return std::vector<uint32_t>{*barrier};
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    const auto barrier = build_cdna4_s_barrier(arch);
    if (!barrier)
      return std::nullopt;
    return std::vector<uint32_t>{*barrier};
  }
  const auto signal = rocjitsu::build_s_barrier_signal_all(arch);
  const auto barrier_wait = rocjitsu::build_s_barrier_wait_all(arch);
  if (!signal || !barrier_wait)
    return std::nullopt;
  return std::vector<uint32_t>{*signal, *barrier_wait};
}

/// @brief Build a conservative memory drain followed by a workgroup barrier.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_workgroup_barrier(rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA4)
    return copy_words(build_cdna4_s_barrier_with_memory_wait(arch));
  const auto wait = build_s_wait_flat_load_lds0(arch);
  const auto barrier = build_workgroup_barrier_only(arch);
  if (!wait || !barrier)
    return std::nullopt;
  std::vector<uint32_t> words{*wait};
  words.insert(words.end(), barrier->begin(), barrier->end());
  return words;
}

} // namespace rocjitsu::instrumentation
