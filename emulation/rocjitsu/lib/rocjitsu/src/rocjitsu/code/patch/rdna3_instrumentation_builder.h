// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rdna3_instrumentation_builder.h
/// @brief RDNA3 instruction encoders used by DBI instrumentation.

#pragma once

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/operand_types.h"

namespace rocjitsu {

[[nodiscard]] inline constexpr bool is_rdna3_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3;
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_getreg_b32(uint16_t sdst, uint16_t hwreg, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 127)
    return std::nullopt;
  return build_sopk_encoding(arch, /*s_getreg_b32=*/17, sdst, hwreg);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_mov_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 126 || ssrc0 > 254)
    return std::nullopt;
  return build_sop1_encoding(arch, rdna3::kSMovB64Sop1, sdst, ssrc0);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_and_saveexec_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 126 || ssrc0 > 254)
    return std::nullopt;
  return build_sop1_encoding(arch, rdna3::kSAndSaveExecB64Sop1, sdst, ssrc0);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_andn2_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  return build_sop2_encoding(arch, rdna3::kSAndNot1B64Sop2, sdst, ssrc0, ssrc1);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_and_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  return build_sop2_encoding(arch, rdna3::kSAndB64Sop2, sdst, ssrc0, ssrc1);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_bcnt1_i32_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 127 || ssrc0 > 254)
    return std::nullopt;
  return build_sop1_encoding(arch, rdna3::kSBcnt1I32B64Sop1, sdst, ssrc0);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_xor_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  return build_sop2_encoding(arch, rdna3::kSXorB64Sop2, sdst, ssrc0, ssrc1);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_sub_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sop2_encoding(arch, rdna3::kSSubU32Sop2, sdst, ssrc0, ssrc1);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_add_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sop2_encoding(arch, rdna3::kSAddU32Sop2, sdst, ssrc0, ssrc1);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_cselect_b32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sop2_encoding(arch, rdna3::kSCselectB32Sop2, sdst, ssrc0, ssrc1);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_cmp_lg_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sopc_encoding(arch, rdna3::kSCmpLgU32Sopc, ssrc0, ssrc1);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_cmp_eq_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  return build_sopc_encoding(arch, rdna3::kSCmpEqU32Sopc, ssrc0, ssrc1);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_cbranch(uint16_t opcode, int16_t offset_dwords, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, opcode, static_cast<uint16_t>(offset_dwords));
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_s_load_dword(uint16_t sdst, uint16_t sbase, uint32_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 105 || sbase > 104 || sbase % 2u != 0 ||
      byte_offset > 0x1fffffu)
    return std::nullopt;
  return rdna3::build_smem(rdna3::kSLoadB32Smem, {.sbase = static_cast<uint8_t>(sbase / 2u),
                                                  .sdata = static_cast<uint8_t>(sdst),
                                                  .offset = byte_offset});
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_mov_b32(uint16_t vdst, uint16_t src0, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || src0 > 511)
    return std::nullopt;
  return rdna3::build_vop1(rdna3::kVMovB32Vop1,
                           {.src0 = src0, .vdst = static_cast<uint8_t>(vdst)})[0];
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_v_mov_b32_literal(uint16_t vdst, uint32_t literal, rj_code_arch_t arch) {
  const auto move = build_rdna3_v_mov_b32(vdst, kVopLiteralSource, arch);
  if (!move)
    return std::nullopt;
  return std::array<uint32_t, 2>{*move, literal};
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_vop2(uint16_t opcode, uint16_t vdst, uint16_t src0, uint16_t vsrc1,
                 rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return rdna3::build_vop2(
      opcode,
      {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1), .vdst = static_cast<uint8_t>(vdst)})[0];
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_vop2_literal(uint16_t opcode, uint16_t vdst, uint32_t literal, uint16_t vsrc1,
                         rj_code_arch_t arch) {
  const auto word = build_rdna3_vop2(opcode, vdst, kVopLiteralSource, vsrc1, arch);
  if (!word)
    return std::nullopt;
  return std::array<uint32_t, 2>{*word, literal};
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_lshrrev_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vop2(rdna3::kVLshrrevB32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_lshlrev_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vop2(rdna3::kVLshlrevB32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_and_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vop2(rdna3::kVAndB32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_min_u32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vop2(rdna3::kVMinU32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_xor_b32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vop2(rdna3::kVXorB32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_add_u32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vop2(rdna3::kVAddNcU32Vop2, vdst, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_vopc(uint16_t opcode, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return rdna3::build_vopc(opcode, {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_cmp_eq_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vopc(rdna3::kVCmpEqU32Vopc, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_cmp_ne_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vopc(rdna3::kVCmpNeU32Vopc, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_cmp_ne_u16_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vopc(rdna3::kVCmpNeU16Vopc, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_cmp_gt_u32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  return build_rdna3_vopc(rdna3::kVCmpGtU32Vopc, src0, vsrc1, arch);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_v_readfirstlane_b32(uint16_t sdst, uint16_t vsrc, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 105 || vsrc > 255)
    return std::nullopt;
  return rdna3::build_vop1(rdna3::kVReadfirstlaneB32Vop1, {.src0 = vector_source_vgpr(vsrc),
                                                           .vdst = static_cast<uint8_t>(sdst)})[0];
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_v_writelane_b32(uint16_t vdst, uint16_t ssrc, uint16_t lane, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || ssrc > 105 || lane > 63)
    return std::nullopt;
  return rdna3::build_vop3(
      rdna3::kVWritelaneB32Vop3,
      {.vdst = static_cast<uint8_t>(vdst), .src0 = ssrc, .src1 = scalar_positive_inline_u32(lane)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_v_readlane_b32(uint16_t sdst, uint16_t vsrc, uint16_t lane, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || sdst > 105 || vsrc > 255 || lane > 63)
    return std::nullopt;
  return rdna3::build_vop3(rdna3::kVReadlaneB32Vop3, {.vdst = static_cast<uint8_t>(sdst),
                                                      .src0 = vector_source_vgpr(vsrc),
                                                      .src1 = scalar_positive_inline_u32(lane)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_v_mbcnt_lo_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return rdna3::build_vop3(rdna3::kVMbcntLoU32B32Vop3,
                           {.vdst = static_cast<uint8_t>(vdst), .src0 = src0, .src1 = src1});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_v_mbcnt_hi_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return rdna3::build_vop3(rdna3::kVMbcntHiU32B32Vop3,
                           {.vdst = static_cast<uint8_t>(vdst), .src0 = src0, .src1 = src1});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_rdna3_v_mul_lo_u32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1,
                                 rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || vsrc1 > 255)
    return std::nullopt;
  const auto multiply =
      rdna3::build_vop3(rdna3::kVMulLoU32Vop3, {.vdst = static_cast<uint8_t>(vdst),
                                                .src0 = kVopLiteralSource,
                                                .src1 = vector_source_vgpr(vsrc1)});
  return std::array<uint32_t, 3>{multiply[0], multiply[1], literal};
}

[[nodiscard]] inline constexpr std::array<uint32_t, 2>
build_rdna3_v_add_co_u32(uint16_t vdst, uint16_t src0, uint16_t src1) {
  return rdna3::build_vop3_sdst_enc(rdna3::kVAddCoU32Vop3SdstEnc,
                                    {.vdst = static_cast<uint8_t>(vdst),
                                     .sdst = static_cast<uint8_t>(rdna3::OPR_SDST_VCC_LO),
                                     .src0 = src0,
                                     .src1 = src1});
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_rdna3_v_add_u64_vgpr_offset(uint16_t address_vgpr, uint16_t offset_vgpr,
                                  rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || address_vgpr >= 255 || offset_vgpr > 255)
    return std::nullopt;
  const auto low = build_rdna3_v_add_co_u32(address_vgpr, vector_source_vgpr(offset_vgpr),
                                            vector_source_vgpr(address_vgpr));
  const auto high = build_rdna3_vop2(
      rdna3::kVAddCoCiU32Vop2, static_cast<uint16_t>(address_vgpr + 1u),
      scalar_positive_inline_u32(0), static_cast<uint16_t>(address_vgpr + 1u), arch);
  if (!high)
    return std::nullopt;
  return std::vector<uint32_t>{low[0], low[1], *high};
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_rdna3_v_add_u64_literal(uint16_t address_vgpr, uint64_t literal, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || address_vgpr >= 255)
    return std::nullopt;
  const uint32_t low_literal = static_cast<uint32_t>(literal);
  const uint16_t low_src = low_literal <= 64u
                               ? scalar_positive_inline_u32(static_cast<uint16_t>(low_literal))
                               : kVopLiteralSource;
  const auto low =
      build_rdna3_v_add_co_u32(address_vgpr, low_src, vector_source_vgpr(address_vgpr));
  const uint32_t high_literal = static_cast<uint32_t>(literal >> 32u);
  const uint16_t high_src = high_literal <= 64u
                                ? scalar_positive_inline_u32(static_cast<uint16_t>(high_literal))
                                : kVopLiteralSource;
  const auto high =
      build_rdna3_vop2(rdna3::kVAddCoCiU32Vop2, static_cast<uint16_t>(address_vgpr + 1u), high_src,
                       static_cast<uint16_t>(address_vgpr + 1u), arch);
  if (!high)
    return std::nullopt;
  std::vector<uint32_t> words{low[0], low[1]};
  if (low_src == kVopLiteralSource)
    words.push_back(low_literal);
  words.push_back(*high);
  if (high_src == kVopLiteralSource)
    words.push_back(high_literal);
  return words;
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_rdna3_v_add_u64_signed_vgpr_offset(uint16_t address_vgpr, uint16_t offset_vgpr,
                                         uint16_t sign_vgpr, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || address_vgpr >= 255 || offset_vgpr > 255 || sign_vgpr > 255 ||
      sign_vgpr == offset_vgpr || sign_vgpr == address_vgpr || sign_vgpr == address_vgpr + 1u)
    return std::nullopt;
  const auto sign = build_rdna3_vop2(rdna3::kVAshrrevI32Vop2, sign_vgpr,
                                     scalar_positive_inline_u32(31), offset_vgpr, arch);
  const auto low = build_rdna3_v_add_co_u32(address_vgpr, vector_source_vgpr(offset_vgpr),
                                            vector_source_vgpr(address_vgpr));
  const auto high = build_rdna3_vop2(
      rdna3::kVAddCoCiU32Vop2, static_cast<uint16_t>(address_vgpr + 1u),
      vector_source_vgpr(sign_vgpr), static_cast<uint16_t>(address_vgpr + 1u), arch);
  if (!sign || !high)
    return std::nullopt;
  return std::vector<uint32_t>{*sign, low[0], low[1], *high};
}

[[nodiscard]] inline std::optional<std::vector<uint32_t>>
build_rdna3_v_add_u64_signed_i24(uint16_t address_vgpr, int32_t displacement, rj_code_arch_t arch) {
  constexpr int32_t kSigned24Min = -(1 << 23);
  constexpr int32_t kSigned24Max = (1 << 23) - 1;
  if (!is_rdna3_arch(arch) || address_vgpr >= 255 || displacement < kSigned24Min ||
      displacement > kSigned24Max)
    return std::nullopt;
  uint16_t low_src = kVopLiteralSource;
  if (displacement >= 0 && displacement <= 64)
    low_src = scalar_positive_inline_u32(static_cast<uint16_t>(displacement));
  else if (displacement >= -16 && displacement < 0)
    low_src = static_cast<uint16_t>(192 - displacement);
  const uint16_t high_src = displacement < 0 ? 193u : scalar_positive_inline_u32(0);
  const auto low =
      build_rdna3_v_add_co_u32(address_vgpr, low_src, vector_source_vgpr(address_vgpr));
  const auto high =
      build_rdna3_vop2(rdna3::kVAddCoCiU32Vop2, static_cast<uint16_t>(address_vgpr + 1u), high_src,
                       static_cast<uint16_t>(address_vgpr + 1u), arch);
  if (!high)
    return std::nullopt;
  std::vector<uint32_t> words{low[0], low[1]};
  if (low_src == kVopLiteralSource)
    words.push_back(static_cast<uint32_t>(displacement));
  words.push_back(*high);
  return words;
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_wait_vmcnt0(rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, rdna3::kSWaitcntSopp, 0x03f7u);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_wait_lgkmcnt0(rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, rdna3::kSWaitcntSopp, 0xfc07u);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna3_s_wait_vmcnt_lgkmcnt0(rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, rdna3::kSWaitcntSopp, 0x0007u);
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_rdna3_s_barrier(rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, rdna3::kSBarrierSopp, 0u);
}

[[nodiscard]] inline constexpr std::optional<uint32_t> build_rdna3_s_trap(uint16_t simm16,
                                                                          rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch))
    return std::nullopt;
  return build_sopp_encoding(arch, rdna3::kSTrapSopp, simm16);
}

inline constexpr uint8_t kRdna3FlatNoSaddr = static_cast<uint8_t>(rdna3::OPR_SREG_NULL);

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_flat_store_b32(uint16_t vaddr, uint16_t vsrc, uint16_t byte_offset,
                           rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vaddr > 254 || vsrc > 255 || byte_offset > 0x1fffu)
    return std::nullopt;
  return rdna3::build_flat(rdna3::kFlatStoreB32Flat, {.offset = byte_offset,
                                                      .addr = static_cast<uint8_t>(vaddr),
                                                      .data = static_cast<uint8_t>(vsrc),
                                                      .saddr = kRdna3FlatNoSaddr});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_flat_load_b32(uint16_t vaddr, uint16_t vdst, uint16_t byte_offset,
                          rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vaddr > 254 || vdst > 255 || byte_offset > 0x1fffu)
    return std::nullopt;
  return rdna3::build_flat(rdna3::kFlatLoadB32Flat, {.offset = byte_offset,
                                                     .addr = static_cast<uint8_t>(vaddr),
                                                     .saddr = kRdna3FlatNoSaddr,
                                                     .vdst = static_cast<uint8_t>(vdst)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_ds_store_b32(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return rdna3::build_ds(rdna3::kDsStoreB32Ds, {.offset0 = byte_offset,
                                                .addr = static_cast<uint8_t>(vaddr),
                                                .data0 = static_cast<uint8_t>(vdata)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_ds_store_b64(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vaddr > 255 || vdata > 254)
    return std::nullopt;
  return rdna3::build_ds(rdna3::kDsStoreB64Ds, {.offset0 = byte_offset,
                                                .addr = static_cast<uint8_t>(vaddr),
                                                .data0 = static_cast<uint8_t>(vdata)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_ds_store_b128(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                          rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vaddr > 255 || vdata > 252)
    return std::nullopt;
  return rdna3::build_ds(rdna3::kDsStoreB128Ds, {.offset0 = byte_offset,
                                                 .addr = static_cast<uint8_t>(vaddr),
                                                 .data0 = static_cast<uint8_t>(vdata)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_ds_storexchg_rtn_b64(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                                 rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 254 || vaddr > 255 || vdata > 254)
    return std::nullopt;
  return rdna3::build_ds(rdna3::kDsStorexchgRtnB64Ds, {.offset0 = byte_offset,
                                                       .addr = static_cast<uint8_t>(vaddr),
                                                       .data0 = static_cast<uint8_t>(vdata),
                                                       .vdst = static_cast<uint8_t>(vdst)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_ds_storexchg_rtn_b32(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                                 rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return rdna3::build_ds(rdna3::kDsStorexchgRtnB32Ds, {.offset0 = byte_offset,
                                                       .addr = static_cast<uint8_t>(vaddr),
                                                       .data0 = static_cast<uint8_t>(vdata),
                                                       .vdst = static_cast<uint8_t>(vdst)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_ds_or_rtn_b32(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                          rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return rdna3::build_ds(rdna3::kDsOrRtnB32Ds, {.offset0 = byte_offset,
                                                .addr = static_cast<uint8_t>(vaddr),
                                                .data0 = static_cast<uint8_t>(vdata),
                                                .vdst = static_cast<uint8_t>(vdst)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_ds_load_b32(uint16_t vdst, uint16_t vaddr, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || vaddr > 255)
    return std::nullopt;
  return rdna3::build_ds(rdna3::kDsLoadB32Ds, {.offset0 = byte_offset,
                                               .addr = static_cast<uint8_t>(vaddr),
                                               .vdst = static_cast<uint8_t>(vdst)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_flat_atomic(uint16_t opcode, uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                        bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vaddr > 254 || vsrc > 255 || vdst > 255 || scope != 2)
    return std::nullopt;
  return rdna3::build_flat(opcode, {.glc = static_cast<uint8_t>(return_old_value),
                                    .addr = static_cast<uint8_t>(vaddr),
                                    .data = static_cast<uint8_t>(vsrc),
                                    .saddr = kRdna3FlatNoSaddr,
                                    .vdst = static_cast<uint8_t>(vdst)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_flat_atomic_add_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                                uint8_t scope, rj_code_arch_t arch) {
  return build_rdna3_flat_atomic(rdna3::kFlatAtomicAddU32Flat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_flat_atomic_or_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                               uint8_t scope, rj_code_arch_t arch) {
  return build_rdna3_flat_atomic(rdna3::kFlatAtomicOrB32Flat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_flat_atomic_cmpswap_b32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                    bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 254)
    return std::nullopt;
  return build_rdna3_flat_atomic(rdna3::kFlatAtomicCmpswapB32Flat, vaddr, vsrc, vdst,
                                 return_old_value, scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_flat_atomic_cmpswap_b64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                    bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 252 || vdst > 254)
    return std::nullopt;
  return build_rdna3_flat_atomic(rdna3::kFlatAtomicCmpswapB64Flat, vaddr, vsrc, vdst,
                                 return_old_value, scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_flat_atomic_swap_b64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                 bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 254 || vdst > 254)
    return std::nullopt;
  return build_rdna3_flat_atomic(rdna3::kFlatAtomicSwapB64Flat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_flat_atomic_add_u64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                                uint8_t scope, rj_code_arch_t arch) {
  if (!return_old_value || vsrc > 254 || vdst > 254)
    return std::nullopt;
  return build_rdna3_flat_atomic(rdna3::kFlatAtomicAddU64Flat, vaddr, vsrc, vdst, return_old_value,
                                 scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_scratch_store_b32(uint16_t vsrc, uint16_t saddr, uint32_t byte_offset,
                              rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vsrc > 255 || saddr > 124 || byte_offset > 0x1ffcu ||
      byte_offset % sizeof(uint32_t) != 0)
    return std::nullopt;
  return rdna3::build_flat(rdna3::kFlatStoreB32Flat, {.offset = static_cast<uint16_t>(byte_offset),
                                                      .seg = 1u,
                                                      .data = static_cast<uint8_t>(vsrc),
                                                      .saddr = static_cast<uint8_t>(saddr)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_scratch_load_b32(uint16_t vdst, uint16_t saddr, uint32_t byte_offset,
                             rj_code_arch_t arch) {
  if (!is_rdna3_arch(arch) || vdst > 255 || saddr > 124 || byte_offset > 0x1ffcu ||
      byte_offset % sizeof(uint32_t) != 0)
    return std::nullopt;
  return rdna3::build_flat(rdna3::kFlatLoadB32Flat, {.offset = static_cast<uint16_t>(byte_offset),
                                                     .seg = 1u,
                                                     .saddr = static_cast<uint8_t>(saddr),
                                                     .vdst = static_cast<uint8_t>(vdst)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_address_free_scratch_store_b32(uint16_t vsrc, uint32_t byte_offset,
                                           rj_code_arch_t arch) {
  return build_rdna3_scratch_store_b32(vsrc, kRdna3FlatNoSaddr, byte_offset, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_address_free_scratch_load_b32(uint16_t vdst, uint32_t byte_offset,
                                          rj_code_arch_t arch) {
  return build_rdna3_scratch_load_b32(vdst, kRdna3FlatNoSaddr, byte_offset, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_scratch_store_b32_saddr(uint16_t vsrc, uint16_t saddr, uint32_t byte_offset,
                                    rj_code_arch_t arch) {
  return build_rdna3_scratch_store_b32(vsrc, saddr, byte_offset, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_rdna3_scratch_load_b32_saddr(uint16_t vdst, uint16_t saddr, uint32_t byte_offset,
                                   rj_code_arch_t arch) {
  return build_rdna3_scratch_load_b32(vdst, saddr, byte_offset, arch);
}

} // namespace rocjitsu
