// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna4_instrumentation_builder.h
/// @brief CDNA4 instruction encoders used by DBI instrumentation.

#pragma once

#include "rocjitsu/code/patch/instruction_builder.h"

namespace rocjitsu {

/// @brief Encode CDNA4 `s_mov_b64 s[sdst:sdst+1], s[ssrc0:ssrc0+1]`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_mov_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  // 255 is the literal marker; this one-word helper cannot append its payload.
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 126 || ssrc0 > 254)
    return std::nullopt;
  return build_sop1_encoding(arch, cdna4::kSMovB64Sop1, sdst, ssrc0);
}

/// @brief Encode CDNA4 `s_and_saveexec_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_and_saveexec_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  // 255 is the literal marker; this one-word helper cannot append its payload.
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 126 || ssrc0 > 254)
    return std::nullopt;
  return build_sop1_encoding(arch, cdna4::kSAndSaveExecB64Sop1, sdst, ssrc0);
}

/// @brief Encode CDNA4 `s_andn2_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_andn2_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  return build_sop2_encoding(arch, cdna4::kSAndn2B64Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA4 `s_and_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_and_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  return build_sop2_encoding(arch, cdna4::kSAndB64Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA4 `s_bcnt1_i32_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_bcnt1_i32_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 127 || ssrc0 > 254)
    return std::nullopt;
  return build_sop1_encoding(arch, cdna4::kSBcnt1I32B64Sop1, sdst, ssrc0);
}

/// @brief Encode CDNA4 `s_xor_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_xor_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  return build_sop2_encoding(arch, cdna4::kSXorB64Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA4 `s_sub_u32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_sub_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  constexpr uint8_t kCdna4SSubU32Sop2 = 1;
  return build_sop2_encoding(arch, kCdna4SSubU32Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA4 `s_add_u32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_add_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  constexpr uint8_t kCdna4SAddU32Sop2 = 0;
  return build_sop2_encoding(arch, kCdna4SAddU32Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA4 `s_cselect_b32` for SCC capture.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cselect_b32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_s_cselect_b32(sdst, ssrc0, ssrc1, arch);
}

/// @brief Encode CDNA4 `s_cmp_lg_u32` for SCC restoration.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cmp_lg_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_s_cmp_lg_u32(ssrc0, ssrc1, arch);
}

/// @brief Encode CDNA4 `s_cmp_eq_u32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cmp_eq_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sopc_encoding(arch, cdna4::kSCmpEqU32Sopc, ssrc0, ssrc1);
}

/// @brief Encode one of CDNA4's scalar conditional branches.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cbranch(uint16_t op, int16_t offset_dwords, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return build_sopp_encoding(arch, op, static_cast<uint16_t>(offset_dwords));
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cbranch_scc0(int16_t offset_dwords, rj_code_arch_t arch) {
  return build_cdna4_s_cbranch(cdna4::kSCbranchScc0Sopp, offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cbranch_scc1(int16_t offset_dwords, rj_code_arch_t arch) {
  return build_cdna4_s_cbranch(cdna4::kSCbranchScc1Sopp, offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cbranch_vccz(int16_t offset_dwords, rj_code_arch_t arch) {
  return build_cdna4_s_cbranch(cdna4::kSCbranchVcczSopp, offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cbranch_vccnz(int16_t offset_dwords, rj_code_arch_t arch) {
  return build_cdna4_s_cbranch(cdna4::kSCbranchVccnzSopp, offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cbranch_execz(int16_t offset_dwords, rj_code_arch_t arch) {
  return build_cdna4_s_cbranch(cdna4::kSCbranchExeczSopp, offset_dwords, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_cbranch_execnz(int16_t offset_dwords, rj_code_arch_t arch) {
  return build_cdna4_s_cbranch(cdna4::kSCbranchExecnzSopp, offset_dwords, arch);
}

/// @brief Encode the CDNA4 VOP2 arithmetic used by instrumentation.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_vop2(uint16_t op, uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna4::build_vop2(
      op,
      {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1), .vdst = static_cast<uint8_t>(vdst)})[0];
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_lshrrev_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_cdna4_vop2(cdna4::kVLshrrevB32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_lshlrev_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_cdna4_vop2(cdna4::kVLshlrevB32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_and_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_cdna4_vop2(cdna4::kVAndB32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_xor_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_cdna4_vop2(cdna4::kVXorB32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_v_min_u32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1,
                              rj_code_arch_t arch) {
  const auto word = build_cdna4_vop2(cdna4::kVMinU32Vop2, vdst, kVopLiteralSource, vsrc1, arch);
  if (!word)
    return std::nullopt;
  return std::array<uint32_t, 2>{*word, literal};
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_v_and_b32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1,
                              rj_code_arch_t arch) {
  const auto word = build_cdna4_vop2(cdna4::kVAndB32Vop2, vdst, kVopLiteralSource, vsrc1, arch);
  if (!word)
    return std::nullopt;
  return std::array<uint32_t, 2>{*word, literal};
}

/// @brief Encode VCC-preserving CDNA4 `v_add3_u32 vdst, src0, vsrc1, 0`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_v_add_u32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna4::build_vop3(cdna4::kVAdd3U32Vop3, {.vdst = static_cast<uint8_t>(vdst),
                                                  .src0 = src0,
                                                  .src1 = vector_source_vgpr(vsrc1),
                                                  .src2 = scalar_positive_inline_u32(0)});
}

/// @brief Materialize a literal in a VGPR with CDNA4's E32 literal form.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_v_mov_b32_literal(uint16_t vdst, uint32_t literal, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 255)
    return std::nullopt;
  return std::array<uint32_t, 2>{
      cdna4::build_vop1(cdna4::kVMovB32Vop1,
                        {.src0 = kVopLiteralSource, .vdst = static_cast<uint8_t>(vdst)})[0],
      literal};
}

/// @brief Encode a literal add without using CDNA4's forbidden VOP3 literal source.
///
/// The destination is written with the literal first, then read alongside
/// @p vsrc1 by `v_add3_u32`. Therefore the destination must not alias @p vsrc1.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna4_v_add_u32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1,
                              rj_code_arch_t arch) {
  if (vdst == vsrc1)
    return std::nullopt;
  const auto materialize = build_cdna4_v_mov_b32_literal(vdst, literal, arch);
  const auto add = build_cdna4_v_add_u32(vdst, vector_source_vgpr(vdst), vsrc1, arch);
  if (!materialize || !add)
    return std::nullopt;
  return std::vector<uint32_t>{(*materialize)[0], (*materialize)[1], (*add)[0], (*add)[1]};
}

/// @brief Encode CDNA4 `v_mad_u32_u24`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_v_mad_u32_u24(uint16_t vdst, uint16_t src0, uint16_t vsrc1, uint16_t vsrc2,
                          rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 255 || src0 > 511 || vsrc1 > 255 || vsrc2 > 255)
    return std::nullopt;
  return cdna4::build_vop3(cdna4::kVMadU32U24Vop3, {.vdst = static_cast<uint8_t>(vdst),
                                                    .src0 = src0,
                                                    .src1 = vector_source_vgpr(vsrc1),
                                                    .src2 = vector_source_vgpr(vsrc2)});
}

/// @brief Encode a CDNA4 multiply by a materialized literal.
///
/// CDNA4 rejects literal operands on `v_mul_lo_u32`, so @p literal_vgpr is an
/// explicit scratch requirement. It may alias @p vdst, but must not alias the
/// value in @p vsrc1 that the multiply consumes.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna4_v_mul_lo_u32_literal(uint16_t vdst, uint16_t literal_vgpr, uint32_t literal,
                                 uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 255 || literal_vgpr > 255 || vsrc1 > 255 ||
      literal_vgpr == vsrc1)
    return std::nullopt;
  const auto materialize = build_cdna4_v_mov_b32_literal(literal_vgpr, literal, arch);
  if (!materialize)
    return std::nullopt;
  const auto multiply =
      cdna4::build_vop3(cdna4::kVMulLoU32Vop3, {.vdst = static_cast<uint8_t>(vdst),
                                                .src0 = vector_source_vgpr(literal_vgpr),
                                                .src1 = vector_source_vgpr(vsrc1)});
  return std::vector<uint32_t>{(*materialize)[0], (*materialize)[1], multiply[0], multiply[1]};
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_readfirstlane_b32(uint16_t sdst, uint16_t vsrc, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 127 || vsrc > 255)
    return std::nullopt;
  return cdna4::build_vop1(cdna4::kVReadfirstlaneB32Vop1, {.src0 = vector_source_vgpr(vsrc),
                                                           .vdst = static_cast<uint8_t>(sdst)})[0];
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_v_mbcnt_lo_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return cdna4::build_vop3(cdna4::kVMbcntLoU32B32Vop3,
                           {.vdst = static_cast<uint8_t>(vdst), .src0 = src0, .src1 = src1});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_v_mbcnt_hi_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return cdna4::build_vop3(cdna4::kVMbcntHiU32B32Vop3,
                           {.vdst = static_cast<uint8_t>(vdst), .src0 = src0, .src1 = src1});
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_cmp_u32_vcc(uint16_t op, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna4::build_vopc(op, {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_cmp_eq_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_cdna4_v_cmp_u32_vcc(cdna4::kVCmpEqU32Vopc, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_cmp_ne_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_cdna4_v_cmp_u32_vcc(cdna4::kVCmpNeU32Vopc, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_cmp_ne_u16_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_cdna4_v_cmp_u32_vcc(cdna4::kVCmpNeU16Vopc, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_v_cmp_gt_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_cdna4_v_cmp_u32_vcc(cdna4::kVCmpGtU32Vopc, src0, vsrc1, arch);
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna4_v_add_u64_vgpr_offset(uint16_t address_vgpr, uint16_t offset_vgpr,
                                  rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || address_vgpr >= 255 || offset_vgpr > 255)
    return std::nullopt;
  return std::vector<uint32_t>{
      *build_cdna4_vop2(cdna4::kVAddCoU32Vop2, address_vgpr, vector_source_vgpr(offset_vgpr),
                        address_vgpr, arch),
      *build_cdna4_vop2(cdna4::kVAddcCoU32Vop2, static_cast<uint16_t>(address_vgpr + 1u),
                        scalar_positive_inline_u32(0), static_cast<uint16_t>(address_vgpr + 1u),
                        arch)};
}

/// @brief Add one sign-extended 32-bit VGPR offset to a 64-bit CDNA4 address pair.
/// @details @p sign_vgpr is scratch distinct from the address pair and offset.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna4_v_add_u64_signed_vgpr_offset(uint16_t address_vgpr, uint16_t offset_vgpr,
                                         uint16_t sign_vgpr, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || address_vgpr >= 255u || offset_vgpr > 255u ||
      sign_vgpr > 255u || sign_vgpr == offset_vgpr || sign_vgpr == address_vgpr ||
      sign_vgpr == address_vgpr + 1u)
    return std::nullopt;
  return std::vector<uint32_t>{
      *build_cdna4_vop2(cdna4::kVAshrrevI32Vop2, sign_vgpr, scalar_positive_inline_u32(31u),
                        offset_vgpr, arch),
      *build_cdna4_vop2(cdna4::kVAddCoU32Vop2, address_vgpr, vector_source_vgpr(offset_vgpr),
                        address_vgpr, arch),
      *build_cdna4_vop2(cdna4::kVAddcCoU32Vop2, static_cast<uint16_t>(address_vgpr + 1u),
                        vector_source_vgpr(sign_vgpr), static_cast<uint16_t>(address_vgpr + 1u),
                        arch)};
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna4_v_add_u64_signed_i24(uint16_t address_vgpr, int32_t displacement, rj_code_arch_t arch) {
  constexpr int32_t kSigned24Min = -(1 << 23);
  constexpr int32_t kSigned24Max = (1 << 23) - 1;
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || address_vgpr >= 255 || displacement < kSigned24Min ||
      displacement > kSigned24Max)
    return std::nullopt;
  uint16_t low_src = kVopLiteralSource;
  if (displacement >= 0 && displacement <= 64)
    low_src = scalar_positive_inline_u32(static_cast<uint16_t>(displacement));
  else if (displacement >= -16 && displacement < 0)
    low_src = static_cast<uint16_t>(192 - displacement);
  const uint16_t high_src = displacement < 0 ? 193u : scalar_positive_inline_u32(0);
  std::vector<uint32_t> words{
      *build_cdna4_vop2(cdna4::kVAddCoU32Vop2, address_vgpr, low_src, address_vgpr, arch)};
  if (low_src == kVopLiteralSource)
    words.push_back(static_cast<uint32_t>(displacement));
  words.push_back(*build_cdna4_vop2(cdna4::kVAddcCoU32Vop2,
                                    static_cast<uint16_t>(address_vgpr + 1u), high_src,
                                    static_cast<uint16_t>(address_vgpr + 1u), arch));
  return words;
}

/// @brief Encode CDNA4 `s_load_dword sdst, s[sbase:sbase+1], byte_offset`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_s_load_dword(uint16_t sdst, uint16_t sbase, uint32_t byte_offset, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || sdst > 101 || sbase > 100 || sbase % 2u != 0 ||
      byte_offset % sizeof(uint32_t) != 0 || byte_offset > 0xfffffu)
    return std::nullopt;
  return cdna4::build_smem(cdna4::kSLoadDwordSmem, {.sbase = static_cast<uint8_t>(sbase / 2u),
                                                    .sdata = static_cast<uint8_t>(sdst),
                                                    .imm = 1,
                                                    .offset = byte_offset});
}

/// @brief Encode CDNA4 `flat_store_dword v[vaddr:vaddr+1], vsrc`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_flat_store_b32(uint16_t vaddr, uint16_t vsrc, uint16_t byte_offset,
                           rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vaddr > 254 || vaddr % 2u != 0 || vsrc > 255 ||
      byte_offset > 0xfffu)
    return std::nullopt;
  return cdna4::build_flat(cdna4::kFlatStoreDwordFlat, {.offset = byte_offset,
                                                        .addr = static_cast<uint8_t>(vaddr),
                                                        .data = static_cast<uint8_t>(vsrc)});
}

/// @brief Encode CDNA4 `flat_load_dword vdst, v[vaddr:vaddr+1]`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_flat_load_b32(uint16_t vaddr, uint16_t vdst, uint16_t byte_offset,
                          rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vaddr > 254 || vaddr % 2u != 0 || vdst > 255 ||
      byte_offset > 0xfffu)
    return std::nullopt;
  return cdna4::build_flat(cdna4::kFlatLoadDwordFlat, {.offset = byte_offset,
                                                       .addr = static_cast<uint8_t>(vaddr),
                                                       .vdst = static_cast<uint8_t>(vdst)});
}

/// @brief Encode CDNA4 `ds_write_b32`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_ds_store_b32(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return cdna4::build_ds(cdna4::kDsWriteB32Ds, {.offset0 = byte_offset,
                                                .addr = static_cast<uint8_t>(vaddr),
                                                .data0 = static_cast<uint8_t>(vdata)});
}

/// @brief Encode CDNA4 `ds_wrxchg_rtn_b64`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_ds_storexchg_rtn_b64(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                                 rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 254 || vdst % 2u != 0 || vaddr > 255 ||
      vdata > 254 || vdata % 2u != 0)
    return std::nullopt;
  return cdna4::build_ds(cdna4::kDsWrxchgRtnB64Ds, {.offset0 = byte_offset,
                                                    .addr = static_cast<uint8_t>(vaddr),
                                                    .data0 = static_cast<uint8_t>(vdata),
                                                    .vdst = static_cast<uint8_t>(vdst)});
}

/// @brief Encode a returning CDNA4 FLAT atomic with device scope.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_flat_atomic(uint16_t op, uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                        bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vaddr > 254 || vaddr % 2u != 0 || vsrc > 255 ||
      vdst > 255 || scope != 2)
    return std::nullopt;
  return cdna4::build_flat(op, {.sc0 = static_cast<uint8_t>(return_old_value),
                                .addr = static_cast<uint8_t>(vaddr),
                                .data = static_cast<uint8_t>(vsrc),
                                .vdst = static_cast<uint8_t>(vdst)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_flat_atomic_add_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                                uint8_t scope, rj_code_arch_t arch) {
  return build_cdna4_flat_atomic(cdna4::kFlatAtomicAddFlat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_flat_atomic_or_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                               uint8_t scope, rj_code_arch_t arch) {
  return build_cdna4_flat_atomic(cdna4::kFlatAtomicOrFlat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_flat_atomic_cmpswap_b32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                    bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 254 || vsrc % 2u != 0)
    return std::nullopt;
  return build_cdna4_flat_atomic(cdna4::kFlatAtomicCmpswapFlat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_flat_atomic_swap_b64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                 bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 254 || vsrc % 2u != 0 || vdst > 254 || vdst % 2u != 0)
    return std::nullopt;
  return build_cdna4_flat_atomic(cdna4::kFlatAtomicSwapX2Flat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_flat_atomic_add_u64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                                uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 254 || vsrc % 2u != 0 || vdst > 254 || vdst % 2u != 0)
    return std::nullopt;
  return build_cdna4_flat_atomic(cdna4::kFlatAtomicAddX2Flat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

/// @brief Encode CDNA4 `s_waitcnt` with the caller-supplied counter immediate.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_cdna4_s_waitcnt(uint16_t simm16,
                                                                             rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return build_sopp_encoding(arch, cdna4::kSWaitcntSopp, simm16);
}

/// @brief Drain both counters to which a general CDNA4 FLAT operation contributes.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_wait_flat0(rj_code_arch_t arch) {
  return build_cdna4_s_waitcnt(/*vmcnt(0), lgkmcnt(0)=*/0x0070u, arch);
}

/// @brief Drain CDNA4 LDS completion through LGKM_CNT.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_wait_lds0(rj_code_arch_t arch) {
  return build_cdna4_s_waitcnt(/*lgkmcnt(0)=*/0xc07fu, arch);
}

/// @brief Drain CDNA4 scalar-memory completion through LGKM_CNT.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_wait_scalar_load0(rj_code_arch_t arch) {
  return build_cdna4_s_wait_lds0(arch);
}

/// @brief Encode CDNA4's no-operand, workgroup-wide `s_barrier`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_cdna4_s_barrier(rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return build_sopp_encoding(arch, cdna4::kSBarrierSopp, 0);
}

/// @brief Conservatively drain FLAT/LDS traffic before a CDNA4 workgroup barrier.
///
/// `s_barrier` synchronizes waves but does not itself drain memory counters.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_s_barrier_with_memory_wait(rj_code_arch_t arch) {
  const auto wait = build_cdna4_s_wait_flat0(arch);
  const auto barrier = build_cdna4_s_barrier(arch);
  if (!wait || !barrier)
    return std::nullopt;
  return std::array<uint32_t, 2>{*wait, *barrier};
}

/// @brief Encode CDNA4 `s_trap simm16`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_cdna4_s_trap(uint16_t simm16,
                                                                          rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return build_sopp_encoding(arch, cdna4::kSTrapSopp, simm16);
}

/// @brief Encode CDNA4 `s_nop 0` as a one-cycle dependent-SALU separation.
///
/// gfx950 does not implement gfx12's `s_delay_alu` instruction.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_salu_dependency_delay(rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return build_sopp_encoding(arch, cdna4::kSNopSopp, 0);
}

/// @brief Encode CDNA4 `s_icache_inv`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_icache_inv(rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return build_sopp_encoding(arch, cdna4::kSIcacheInvSopp, 0);
}

/// @brief Encode a no-operand CDNA4 scalar data-cache operation.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_s_dcache(uint16_t op, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return cdna4::build_smem(op);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_s_dcache_inv(rj_code_arch_t arch) {
  return build_cdna4_s_dcache(cdna4::kSDcacheInvSmem, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_s_dcache_wb(rj_code_arch_t arch) {
  return build_cdna4_s_dcache(cdna4::kSDcacheWbSmem, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_s_dcache_inv_vol(rj_code_arch_t arch) {
  return build_cdna4_s_dcache(cdna4::kSDcacheInvVolSmem, arch);
}

/// @brief Encode CDNA4 `buffer_inv sc1` for coherent vector-cache retries.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_buffer_inv_sc1(rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  cdna4::MubufBuilderFields fields;
  fields.sc1 = 1;
  return cdna4::build_mubuf(cdna4::kBufferInvMubuf, fields);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_s_dcache_wb_vol(rj_code_arch_t arch) {
  return build_cdna4_s_dcache(cdna4::kSDcacheWbVolSmem, arch);
}

/// @brief Encode CDNA4 `scratch_store_dword off, vsrc, off offset:byte_offset`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_address_free_scratch_store_b32(uint16_t vsrc, uint32_t byte_offset,
                                           rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vsrc > 255 ||
      byte_offset > kMaxCdnaAddressFreeScratchDwordOffset || byte_offset % sizeof(uint32_t) != 0) {
    return std::nullopt;
  }
  return std::array<uint32_t, 2>{0xdc704000u | byte_offset,
                                 (static_cast<uint32_t>(vsrc) << 8u) | 0x007f0000u};
}

/// @brief Encode CDNA4 `scratch_load_dword vdst, off, off offset:byte_offset`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_address_free_scratch_load_b32(uint16_t vdst, uint32_t byte_offset,
                                          rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 255 ||
      byte_offset > kMaxCdnaAddressFreeScratchDwordOffset || byte_offset % sizeof(uint32_t) != 0) {
    return std::nullopt;
  }
  return std::array<uint32_t, 2>{0xdc504000u | byte_offset,
                                 (static_cast<uint32_t>(vdst) << 24u) | 0x007f0000u};
}

/// @brief Encode CDNA4 `scratch_store_dword off, vsrc, saddr`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_scratch_store_b32_saddr(uint16_t vsrc, uint16_t saddr, uint32_t byte_offset,
                                    rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vsrc > 255 || saddr > 127 ||
      byte_offset > kMaxCdnaAddressFreeScratchDwordOffset || byte_offset % sizeof(uint32_t) != 0) {
    return std::nullopt;
  }
  return std::array<uint32_t, 2>{0xdc704000u | byte_offset,
                                 (static_cast<uint32_t>(saddr) << 16u) |
                                     (static_cast<uint32_t>(vsrc) << 8u)};
}

/// @brief Encode CDNA4 `scratch_load_dword vdst, off, saddr`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna4_scratch_load_b32_saddr(uint16_t vdst, uint16_t saddr, uint32_t byte_offset,
                                   rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4 || vdst > 255 || saddr > 127 ||
      byte_offset > kMaxCdnaAddressFreeScratchDwordOffset || byte_offset % sizeof(uint32_t) != 0) {
    return std::nullopt;
  }
  return std::array<uint32_t, 2>{0xdc504000u | byte_offset,
                                 (static_cast<uint32_t>(vdst) << 24u) |
                                     (static_cast<uint32_t>(saddr) << 16u)};
}

/// @brief Encode CDNA4 `s_waitcnt vmcnt(0)` for FLAT_SCRATCH completion.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna4_s_wait_vmcnt0(rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  return 0xbf8c0f70u;
}

} // namespace rocjitsu
