// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna3_instrumentation_builder.h
/// @brief CDNA3 instruction encoders used by DBI instrumentation.

#pragma once

#include "rocjitsu/code/patch/instruction_builder.h"

namespace rocjitsu {

/// @brief Return whether @p arch selects the CDNA3 encoder backend.
[[nodiscard]] inline constexpr bool is_cdna3_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3;
}

/// @brief Encode CDNA3 `s_getreg_b32 sdst, hwreg`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_getreg_b32(uint16_t sdst, uint16_t hwreg, rj_code_arch_t arch) {
  constexpr uint16_t kSGetregB32Sopk = 17;
  if (!is_cdna3_arch(arch) || sdst > 127)
    return std::nullopt;
  return build_sopk_encoding(arch, kSGetregB32Sopk, sdst, hwreg);
}

/// @brief Encode CDNA3 `s_mov_b64`; the literal marker is rejected because
/// this single-word helper cannot append a literal payload.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_mov_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 126 || ssrc0 > 254)
    return std::nullopt;
  return build_sop1_encoding(arch, cdna3::kSMovB64Sop1, sdst, ssrc0);
}

/// @brief Encode CDNA3 `s_and_saveexec_b64` without a literal operand.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_and_saveexec_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 126 || ssrc0 > 254)
    return std::nullopt;
  return build_sop1_encoding(arch, cdna3::kSAndSaveExecB64Sop1, sdst, ssrc0);
}

/// @brief Encode CDNA3 `s_andn2_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_andn2_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  return build_sop2_encoding(arch, cdna3::kSAndn2B64Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA3 `s_and_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_and_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  return build_sop2_encoding(arch, cdna3::kSAndB64Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA3 `s_bcnt1_i32_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_bcnt1_i32_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 127 || ssrc0 > 254)
    return std::nullopt;
  return build_sop1_encoding(arch, cdna3::kSBcnt1I32B64Sop1, sdst, ssrc0);
}

/// @brief Encode CDNA3 `s_xor_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_xor_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  return build_sop2_encoding(arch, cdna3::kSXorB64Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA3 `s_sub_u32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_sub_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sop2_encoding(arch, cdna3::kSSubU32Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA3 `s_add_u32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_add_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sop2_encoding(arch, cdna3::kSAddU32Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA3 `s_cselect_b32` for SCC capture.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_cselect_b32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sop2_encoding(arch, cdna3::kSCselectB32Sop2, sdst, ssrc0, ssrc1);
}

/// @brief Encode CDNA3 `s_cmp_lg_u32` for SCC restoration.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_cmp_lg_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sopc_encoding(arch, cdna3::kSCmpLgU32Sopc, ssrc0, ssrc1);
}

/// @brief Encode CDNA3 `s_cmp_eq_u32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_cmp_eq_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sopc_encoding(arch, cdna3::kSCmpEqU32Sopc, ssrc0, ssrc1);
}

/// @brief Encode one of CDNA3's scalar conditional branches.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_cbranch(uint16_t opcode, int16_t offset_dwords, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, opcode, static_cast<uint16_t>(offset_dwords));
}

/// @brief Materialize a 32-bit literal in a CDNA3 VGPR.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_v_mov_b32_literal(uint16_t vdst, uint32_t literal, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255)
    return std::nullopt;
  return std::array<uint32_t, 2>{
      cdna3::build_vop1(cdna3::kVMovB32Vop1,
                        {.src0 = kVopLiteralSource, .vdst = static_cast<uint8_t>(vdst)})[0],
      literal};
}

/// @brief Encode CDNA3 `v_lshrrev_b32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_lshrrev_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vop2(
      cdna3::kVLshrrevB32Vop2,
      {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1), .vdst = static_cast<uint8_t>(vdst)})[0];
}

/// @brief Encode CDNA3 `v_lshlrev_b32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_lshlrev_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vop2(
      cdna3::kVLshlrevB32Vop2,
      {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1), .vdst = static_cast<uint8_t>(vdst)})[0];
}

/// @brief Encode CDNA3 `v_and_b32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_and_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vop2(
      cdna3::kVAndB32Vop2,
      {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1), .vdst = static_cast<uint8_t>(vdst)})[0];
}

/// @brief Encode CDNA3 `v_min_u32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_min_u32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vop2(
      cdna3::kVMinU32Vop2,
      {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1), .vdst = static_cast<uint8_t>(vdst)})[0];
}

/// @brief Encode CDNA3 `v_xor_b32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_xor_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vop2(
      cdna3::kVXorB32Vop2,
      {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1), .vdst = static_cast<uint8_t>(vdst)})[0];
}

/// @brief Encode a CDNA3 VOP2 instruction with an appended literal word.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_vop2_literal(uint16_t opcode, uint16_t vdst, uint32_t literal, uint16_t vsrc1,
                         rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || vsrc1 > 255)
    return std::nullopt;
  return std::array<uint32_t, 2>{cdna3::build_vop2(opcode, {.src0 = kVopLiteralSource,
                                                            .vsrc1 = static_cast<uint8_t>(vsrc1),
                                                            .vdst = static_cast<uint8_t>(vdst)})[0],
                                 literal};
}

/// @brief Encode CDNA3 `v_add_u32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_add_u32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vop2(
      cdna3::kVAddU32Vop2,
      {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1), .vdst = static_cast<uint8_t>(vdst)})[0];
}

/// @brief Encode CDNA3 multiply by a literal materialized in an explicit VGPR.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna3_v_mul_lo_u32_literal(uint16_t vdst, uint16_t literal_vgpr, uint32_t literal,
                                 uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || literal_vgpr > 255 || vsrc1 > 255 ||
      literal_vgpr == vsrc1)
    return std::nullopt;
  const auto materialize = build_cdna3_v_mov_b32_literal(literal_vgpr, literal, arch);
  if (!materialize)
    return std::nullopt;
  const auto multiply =
      cdna3::build_vop3(cdna3::kVMulLoU32Vop3, {.vdst = static_cast<uint8_t>(vdst),
                                                .src0 = vector_source_vgpr(literal_vgpr),
                                                .src1 = vector_source_vgpr(vsrc1)});
  return std::vector<uint32_t>{(*materialize)[0], (*materialize)[1], multiply[0], multiply[1]};
}

/// @brief Encode CDNA3 `v_mbcnt_lo_u32_b32`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_v_mbcnt_lo_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return cdna3::build_vop3(cdna3::kVMbcntLoU32B32Vop3,
                           {.vdst = static_cast<uint8_t>(vdst), .src0 = src0, .src1 = src1});
}

/// @brief Encode CDNA3 `v_mbcnt_hi_u32_b32`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_v_mbcnt_hi_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return cdna3::build_vop3(cdna3::kVMbcntHiU32B32Vop3,
                           {.vdst = static_cast<uint8_t>(vdst), .src0 = src0, .src1 = src1});
}

/// @brief Encode CDNA3 `v_cmp_eq_u32` with the physical VCC destination.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_cmp_eq_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vopc(cdna3::kVCmpEqU32Vopc,
                           {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

/// @brief Encode CDNA3 `v_cmp_ne_u32` with the physical VCC destination.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_cmp_ne_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vopc(cdna3::kVCmpNeU32Vopc,
                           {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

/// @brief Encode CDNA3 `v_cmp_ne_u16` with the physical VCC destination.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_cmp_ne_u16_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vopc(cdna3::kVCmpNeU16Vopc,
                           {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

/// @brief Encode CDNA3 `v_cmp_gt_u32` with the physical VCC destination.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_cmp_gt_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return cdna3::build_vopc(cdna3::kVCmpGtU32Vopc,
                           {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

/// @brief Encode CDNA3 `v_readfirstlane_b32`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_v_readfirstlane_b32(uint16_t sdst, uint16_t vsrc, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 127 || vsrc > 255)
    return std::nullopt;
  return cdna3::build_vop1(cdna3::kVReadfirstlaneB32Vop1, {.src0 = vector_source_vgpr(vsrc),
                                                           .vdst = static_cast<uint8_t>(sdst)})[0];
}

/// @brief Encode CDNA3 `v_writelane_b32 vdst, ssrc, lane`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_v_writelane_b32(uint16_t vdst, uint16_t ssrc, uint16_t lane, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || ssrc > 105 || lane > 63)
    return std::nullopt;
  return cdna3::build_vop3(cdna3::kVWritelaneB32Vop3, {.vdst = static_cast<uint8_t>(vdst),
                                                       .src0 = ssrc,
                                                       .src1 = scalar_positive_inline_u32(lane),
                                                       .src2 = 0u});
}

/// @brief Encode CDNA3 `v_readlane_b32 sdst, vsrc, lane`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_v_readlane_b32(uint16_t sdst, uint16_t vsrc, uint16_t lane, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 105 || vsrc > 255 || lane > 63)
    return std::nullopt;
  return cdna3::build_vop3(cdna3::kVReadlaneB32Vop3, {.vdst = static_cast<uint8_t>(sdst),
                                                      .src0 = vector_source_vgpr(vsrc),
                                                      .src1 = scalar_positive_inline_u32(lane),
                                                      .src2 = 0u});
}

/// @brief Add one VGPR offset to an in-place 64-bit CDNA3 address pair.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna3_v_add_u64_vgpr_offset(uint16_t address_vgpr, uint16_t offset_vgpr,
                                  rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || address_vgpr >= 255 || offset_vgpr > 255)
    return std::nullopt;
  return std::vector<uint32_t>{
      cdna3::build_vop2(cdna3::kVAddCoU32Vop2, {.src0 = vector_source_vgpr(offset_vgpr),
                                                .vsrc1 = static_cast<uint8_t>(address_vgpr),
                                                .vdst = static_cast<uint8_t>(address_vgpr)})[0],
      cdna3::build_vop2(cdna3::kVAddcCoU32Vop2,
                        {.src0 = scalar_positive_inline_u32(0),
                         .vsrc1 = static_cast<uint8_t>(address_vgpr + 1u),
                         .vdst = static_cast<uint8_t>(address_vgpr + 1u)})[0]};
}

/// @brief Add an unsigned 64-bit literal to an in-place CDNA3 address pair.
/// @details VCC is clobbered and must be saved by the caller.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna3_v_add_u64_literal(uint16_t address_vgpr, uint64_t literal, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || address_vgpr >= 255)
    return std::nullopt;

  std::vector<uint32_t> words;
  words.reserve(4);
  const uint32_t low = static_cast<uint32_t>(literal);
  const uint16_t low_src =
      low <= 64u ? scalar_positive_inline_u32(static_cast<uint16_t>(low)) : kVopLiteralSource;
  words.push_back(
      cdna3::build_vop2(cdna3::kVAddCoU32Vop2, {.src0 = low_src,
                                                .vsrc1 = static_cast<uint8_t>(address_vgpr),
                                                .vdst = static_cast<uint8_t>(address_vgpr)})[0]);
  if (low_src == kVopLiteralSource)
    words.push_back(low);

  const uint32_t high = static_cast<uint32_t>(literal >> 32u);
  const uint16_t high_src =
      high <= 64u ? scalar_positive_inline_u32(static_cast<uint16_t>(high)) : kVopLiteralSource;
  words.push_back(cdna3::build_vop2(cdna3::kVAddcCoU32Vop2,
                                    {.src0 = high_src,
                                     .vsrc1 = static_cast<uint8_t>(address_vgpr + 1u),
                                     .vdst = static_cast<uint8_t>(address_vgpr + 1u)})[0]);
  if (high_src == kVopLiteralSource)
    words.push_back(high);
  return words;
}

/// @brief Add one sign-extended 32-bit VGPR offset to a 64-bit CDNA3 address pair.
/// @details @p sign_vgpr is a scratch VGPR distinct from the address pair and offset.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna3_v_add_u64_signed_vgpr_offset(uint16_t address_vgpr, uint16_t offset_vgpr,
                                         uint16_t sign_vgpr, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || address_vgpr >= 255u || offset_vgpr > 255u || sign_vgpr > 255u ||
      sign_vgpr == offset_vgpr || sign_vgpr == address_vgpr || sign_vgpr == address_vgpr + 1u)
    return std::nullopt;
  return std::vector<uint32_t>{
      cdna3::build_vop2(cdna3::kVAshrrevI32Vop2, {.src0 = scalar_positive_inline_u32(31u),
                                                  .vsrc1 = static_cast<uint8_t>(offset_vgpr),
                                                  .vdst = static_cast<uint8_t>(sign_vgpr)})[0],
      cdna3::build_vop2(cdna3::kVAddCoU32Vop2, {.src0 = vector_source_vgpr(offset_vgpr),
                                                .vsrc1 = static_cast<uint8_t>(address_vgpr),
                                                .vdst = static_cast<uint8_t>(address_vgpr)})[0],
      cdna3::build_vop2(cdna3::kVAddcCoU32Vop2,
                        {.src0 = vector_source_vgpr(sign_vgpr),
                         .vsrc1 = static_cast<uint8_t>(address_vgpr + 1u),
                         .vdst = static_cast<uint8_t>(address_vgpr + 1u)})[0]};
}

/// @brief Add a signed 24-bit displacement to an in-place CDNA3 address pair.
[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_cdna3_v_add_u64_signed_i24(uint16_t address_vgpr, int32_t displacement, rj_code_arch_t arch) {
  constexpr int32_t kSigned24Min = -(1 << 23);
  constexpr int32_t kSigned24Max = (1 << 23) - 1;
  if (!is_cdna3_arch(arch) || address_vgpr >= 255 || displacement < kSigned24Min ||
      displacement > kSigned24Max)
    return std::nullopt;
  uint16_t low_src = kVopLiteralSource;
  if (displacement >= 0 && displacement <= 64)
    low_src = scalar_positive_inline_u32(static_cast<uint16_t>(displacement));
  else if (displacement >= -16 && displacement < 0)
    low_src = static_cast<uint16_t>(192 - displacement);
  const uint16_t high_src = displacement < 0 ? 193u : scalar_positive_inline_u32(0);
  std::vector<uint32_t> words{
      cdna3::build_vop2(cdna3::kVAddCoU32Vop2, {.src0 = low_src,
                                                .vsrc1 = static_cast<uint8_t>(address_vgpr),
                                                .vdst = static_cast<uint8_t>(address_vgpr)})[0]};
  if (low_src == kVopLiteralSource)
    words.push_back(static_cast<uint32_t>(displacement));
  words.push_back(cdna3::build_vop2(cdna3::kVAddcCoU32Vop2,
                                    {.src0 = high_src,
                                     .vsrc1 = static_cast<uint8_t>(address_vgpr + 1u),
                                     .vdst = static_cast<uint8_t>(address_vgpr + 1u)})[0]);
  return words;
}

/// @brief Drain CDNA3 vector-memory operations through VM_CNT.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_wait_vmcnt0(rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, cdna3::kSWaitcntSopp, 0x0f70u);
}

/// @brief Drain CDNA3 LDS and scalar-memory operations through LGKM_CNT.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_wait_lgkmcnt0(rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, cdna3::kSWaitcntSopp, 0xc07fu);
}

/// @brief Drain CDNA3 vector-memory and LGKM operations together.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_cdna3_s_wait_vmcnt_lgkmcnt0(rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, cdna3::kSWaitcntSopp, 0x0070u);
}

/// @brief Encode CDNA3's no-operand, workgroup-wide `s_barrier`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_cdna3_s_barrier(rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, cdna3::kSBarrierSopp, 0u);
}

/// @brief Encode CDNA3 `buffer_inv sc1` for coherent cache retries.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_buffer_inv_sc1(rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch))
    return std::nullopt;
  cdna3::MubufBuilderFields fields;
  fields.sc1 = 1;
  return cdna3::build_mubuf(cdna3::kBufferInvMubuf, fields);
}

/// @brief Encode CDNA3 `s_load_dword sdst, s[sbase:sbase+1], byte_offset`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_s_load_dword(uint16_t sdst, uint16_t sbase, uint32_t byte_offset, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || sdst > 101 || sbase > 100 || sbase % 2u != 0 ||
      byte_offset % sizeof(uint32_t) != 0 || byte_offset > 0xfffffu)
    return std::nullopt;
  return cdna3::build_smem(cdna3::kSLoadDwordSmem, {.sbase = static_cast<uint8_t>(sbase / 2u),
                                                    .sdata = static_cast<uint8_t>(sdst),
                                                    .imm = 1,
                                                    .offset = byte_offset});
}

/// @brief Encode CDNA3 `flat_store_dword v[vaddr:vaddr+1], vsrc`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_flat_store_b32(uint16_t vaddr, uint16_t vsrc, uint16_t byte_offset,
                           rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vaddr > 254 || vsrc > 255 || byte_offset > 0xfffu)
    return std::nullopt;
  return cdna3::build_flat(cdna3::kFlatStoreDwordFlat, {.offset = byte_offset,
                                                        .addr = static_cast<uint8_t>(vaddr),
                                                        .data = static_cast<uint8_t>(vsrc)});
}

/// @brief Encode CDNA3 `flat_load_dword vdst, v[vaddr:vaddr+1]`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_flat_load_b32(uint16_t vaddr, uint16_t vdst, uint16_t byte_offset,
                          rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vaddr > 254 || vdst > 255 || byte_offset > 0xfffu)
    return std::nullopt;
  return cdna3::build_flat(cdna3::kFlatLoadDwordFlat, {.offset = byte_offset,
                                                       .addr = static_cast<uint8_t>(vaddr),
                                                       .vdst = static_cast<uint8_t>(vdst)});
}

/// @brief Encode CDNA3 `ds_write_b32`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_ds_store_b32(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return cdna3::build_ds(cdna3::kDsWriteB32Ds, {.offset0 = byte_offset,
                                                .addr = static_cast<uint8_t>(vaddr),
                                                .data0 = static_cast<uint8_t>(vdata)});
}

/// @brief Encode CDNA3 `ds_write_b64` from a consecutive VGPR pair.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_ds_store_b64(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vaddr > 255 || vdata > 254)
    return std::nullopt;
  return cdna3::build_ds(cdna3::kDsWriteB64Ds, {.offset0 = byte_offset,
                                                .addr = static_cast<uint8_t>(vaddr),
                                                .data0 = static_cast<uint8_t>(vdata)});
}

/// @brief Encode CDNA3 `ds_write_b128` from four consecutive VGPRs.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_ds_store_b128(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                          rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vaddr > 255 || vdata > 252)
    return std::nullopt;
  return cdna3::build_ds(cdna3::kDsWriteB128Ds, {.offset0 = byte_offset,
                                                 .addr = static_cast<uint8_t>(vaddr),
                                                 .data0 = static_cast<uint8_t>(vdata)});
}

/// @brief Encode CDNA3 `ds_wrxchg_rtn_b64`; both value tuples must be even.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_ds_storexchg_rtn_b64(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                                 rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 254 || vdst % 2u != 0 || vaddr > 255 || vdata > 254 ||
      vdata % 2u != 0)
    return std::nullopt;
  return cdna3::build_ds(cdna3::kDsWrxchgRtnB64Ds, {.offset0 = byte_offset,
                                                    .addr = static_cast<uint8_t>(vaddr),
                                                    .data0 = static_cast<uint8_t>(vdata),
                                                    .vdst = static_cast<uint8_t>(vdst)});
}

/// @brief Encode CDNA3 `ds_wrxchg_rtn_b32`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_ds_storexchg_rtn_b32(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                                 rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return cdna3::build_ds(cdna3::kDsWrxchgRtnB32Ds, {.offset0 = byte_offset,
                                                    .addr = static_cast<uint8_t>(vaddr),
                                                    .data0 = static_cast<uint8_t>(vdata),
                                                    .vdst = static_cast<uint8_t>(vdst)});
}

/// @brief Encode CDNA3 `ds_or_rtn_b32`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_ds_or_rtn_b32(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                          rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return cdna3::build_ds(cdna3::kDsOrRtnB32Ds, {.offset0 = byte_offset,
                                                .addr = static_cast<uint8_t>(vaddr),
                                                .data0 = static_cast<uint8_t>(vdata),
                                                .vdst = static_cast<uint8_t>(vdst)});
}

/// @brief Encode CDNA3 `ds_read_b32`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_ds_load_b32(uint16_t vdst, uint16_t vaddr, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || vaddr > 255)
    return std::nullopt;
  return cdna3::build_ds(cdna3::kDsReadB32Ds, {.offset0 = byte_offset,
                                               .addr = static_cast<uint8_t>(vaddr),
                                               .vdst = static_cast<uint8_t>(vdst)});
}

/// @brief Encode a CDNA3 FLAT atomic. SC0 requests the pre-operation value in
/// @p vdst.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_flat_atomic(uint16_t op, uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                        bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vaddr > 254 || vaddr % 2u != 0 || vsrc > 255 || vdst > 255 ||
      scope != 2)
    return std::nullopt;
  return cdna3::build_flat(op, {.sc0 = static_cast<uint8_t>(return_old_value),
                                .addr = static_cast<uint8_t>(vaddr),
                                .data = static_cast<uint8_t>(vsrc),
                                .vdst = static_cast<uint8_t>(vdst)});
}

/// @brief Encode returning or non-returning CDNA3 32-bit FLAT atomic add.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_flat_atomic_add_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                                uint8_t scope, rj_code_arch_t arch) {
  return build_cdna3_flat_atomic(cdna3::kFlatAtomicAddFlat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

/// @brief Encode returning or non-returning CDNA3 32-bit FLAT atomic OR.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_flat_atomic_or_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                               uint8_t scope, rj_code_arch_t arch) {
  return build_cdna3_flat_atomic(cdna3::kFlatAtomicOrFlat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

/// @brief Encode returning CDNA3 32-bit FLAT compare-and-swap.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_flat_atomic_cmpswap_b32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                    bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 254 || vsrc % 2u != 0)
    return std::nullopt;
  return build_cdna3_flat_atomic(cdna3::kFlatAtomicCmpswapFlat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

/// @brief Encode returning CDNA3 64-bit FLAT compare-and-swap.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_flat_atomic_cmpswap_b64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                    bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 252 || vsrc % 2u != 0 || vdst > 254 || vdst % 2u != 0)
    return std::nullopt;
  return build_cdna3_flat_atomic(cdna3::kFlatAtomicCmpswapX2Flat, vaddr, vsrc, vdst,
                                 return_old_value, scope, arch);
}

/// @brief Encode returning CDNA3 64-bit FLAT swap with even source/destination tuples.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_flat_atomic_swap_b64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                 bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 254 || vsrc % 2u != 0 || vdst > 254 || vdst % 2u != 0)
    return std::nullopt;
  return build_cdna3_flat_atomic(cdna3::kFlatAtomicSwapX2Flat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

/// @brief Encode returning CDNA3 64-bit FLAT add with even source/destination tuples.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_flat_atomic_add_u64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                                uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 254 || vsrc % 2u != 0 || vdst > 254 || vdst % 2u != 0)
    return std::nullopt;
  return build_cdna3_flat_atomic(cdna3::kFlatAtomicAddX2Flat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

/// @brief Encode address-free CDNA scratch store with a dword-aligned 12-bit offset.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_address_free_scratch_store_b32(uint16_t vsrc, uint32_t byte_offset,
                                           rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vsrc > 255 || byte_offset > 0xffcu ||
      byte_offset % sizeof(uint32_t) != 0)
    return std::nullopt;
  return cdna3::build_flat(cdna3::kFlatStoreDwordFlat,
                           {.offset = static_cast<uint16_t>(byte_offset),
                            .seg = 1u,
                            .data = static_cast<uint8_t>(vsrc),
                            .saddr = 0x7fu});
}

/// @brief Encode address-free CDNA scratch load with a dword-aligned 12-bit offset.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_address_free_scratch_load_b32(uint16_t vdst, uint32_t byte_offset,
                                          rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || byte_offset > 0xffcu ||
      byte_offset % sizeof(uint32_t) != 0)
    return std::nullopt;
  return cdna3::build_flat(cdna3::kFlatLoadDwordFlat, {.offset = static_cast<uint16_t>(byte_offset),
                                                       .seg = 1u,
                                                       .saddr = 0x7fu,
                                                       .vdst = static_cast<uint8_t>(vdst)});
}

/// @brief Encode CDNA3 scratch store using an explicit scalar address register.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_scratch_store_b32_saddr(uint16_t vsrc, uint16_t saddr, uint32_t byte_offset,
                                    rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vsrc > 255 || saddr > 127 || byte_offset > 0xffcu ||
      byte_offset % sizeof(uint32_t) != 0)
    return std::nullopt;
  return cdna3::build_flat(cdna3::kFlatStoreDwordFlat,
                           {.offset = static_cast<uint16_t>(byte_offset),
                            .seg = 1u,
                            .data = static_cast<uint8_t>(vsrc),
                            .saddr = static_cast<uint8_t>(saddr)});
}

/// @brief Encode CDNA3 scratch load using an explicit scalar address register.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_cdna3_scratch_load_b32_saddr(uint16_t vdst, uint16_t saddr, uint32_t byte_offset,
                                   rj_code_arch_t arch) {
  if (!is_cdna3_arch(arch) || vdst > 255 || saddr > 127 || byte_offset > 0xffcu ||
      byte_offset % sizeof(uint32_t) != 0)
    return std::nullopt;
  return cdna3::build_flat(cdna3::kFlatLoadDwordFlat, {.offset = static_cast<uint16_t>(byte_offset),
                                                       .seg = 1u,
                                                       .saddr = static_cast<uint8_t>(saddr),
                                                       .vdst = static_cast<uint8_t>(vdst)});
}

} // namespace rocjitsu
