// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rdna4_instrumentation_builder.h
/// @brief RDNA4-family instruction encoders used by DBI instrumentation.

#pragma once

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/operand_types.h"

namespace rocjitsu {

// The generated operand metadata is the authority for each target's packed
// VFLAT/VGLOBAL no-SADDR selector.
inline constexpr uint32_t kRdna4FlatNoSaddrEncoding = static_cast<uint32_t>(rdna4::OPR_SREG_NULL);
inline constexpr uint32_t kFlatNoSaddrGfx1250Encoding =
    static_cast<uint32_t>(gfx1250::OPR_SREG_NULL);

[[nodiscard]] inline constexpr uint32_t flat_no_saddr_encoding(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_GFX1250 ? kFlatNoSaddrGfx1250Encoding
                                            : kRdna4FlatNoSaddrEncoding;
}

/// @brief Encode an s_sleep instruction for the given target ISA.
///
/// @param delay  ISA-defined sleep delay immediate.
/// @param arch   Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t
build_s_sleep(uint16_t delay, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  return pack_sopp(sopp_op_sleep(arch), delay);
}

/// @brief Encode an s_sleep_var instruction for the given target ISA.
///
/// @param ssrc0  Scalar source operand encoding that supplies the delay.
/// @param arch   Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t
build_s_sleep_var(uint16_t ssrc0, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  return pack_sop1(sop1_op_sleep_var(arch), 0, ssrc0);
}

/// @brief Build the SOPK hwreg immediate used by s_getreg/s_setreg instructions.
[[nodiscard]] inline constexpr std::optional<uint16_t>
build_hwreg_imm(uint16_t reg_id, uint16_t offset, uint16_t size_bits) {
  if (reg_id > 63 || offset > 31 || size_bits == 0 || size_bits > 32)
    return std::nullopt;
  return static_cast<uint16_t>(reg_id | (offset << 6u) | ((size_bits - 1u) << 11u));
}

/// @brief Encode RDNA4-class `s_getreg_b32 sdst, hwreg`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_getreg_b32(uint16_t sdst, uint16_t hwreg, rj_code_arch_t arch) {
  if ((arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_GFX1250) || sdst > 127)
    return std::nullopt;
  return pack_sopk(/*op=*/17, sdst, hwreg);
}

/// @brief Encode v_mov_b32_e32 for the given target ISA.
///
/// @param vdst  Destination VGPR.
/// @param src0  VOP1 source operand encoding.
/// @param arch  Target ISA architecture.
/// @returns The encoded 32-bit instruction word.
[[nodiscard]] inline constexpr uint32_t build_v_mov_b32_e32(uint16_t vdst, uint16_t src0,
                                                            [[maybe_unused]] rj_code_arch_t arch) {
  return (0x3Fu << 25) | (static_cast<uint32_t>(vdst) << 17) | (1u << 9) | (src0 & 0x1FFu);
}

/// @brief Encode VOP2 `v_lshrrev_b32 vdst, src0, vsrc1`.
///
/// @details The operation is `vdst = vsrc1 >> src0`. @p src0 is a VOP source
/// operand encoding, so inline constants such as `scalar_positive_inline_u32(2)`
/// are allowed. @p vsrc1 is a VGPR number.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_lshrrev_b32_e32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_lshrrev_b32(arch);
  if (!op)
    return std::nullopt;
  return pack_vop2(*op, vdst, src0, vsrc1);
}

/// @brief Encode VOP2 `v_lshlrev_b32 vdst, src0, vsrc1`.
///
/// @details The operation is `vdst = vsrc1 << src0`. @p src0 is a VOP source
/// operand encoding, so inline constants such as `scalar_positive_inline_u32(16)`
/// are allowed. @p vsrc1 is a VGPR number.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_lshlrev_b32_e32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_lshlrev_b32(arch);
  if (!op)
    return std::nullopt;
  return pack_vop2(*op, vdst, src0, vsrc1);
}

/// @brief Encode VOP2 `v_add_nc_u32 vdst, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_add_nc_u32_e32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_add_nc_u32(arch);
  if (!op)
    return std::nullopt;
  return pack_vop2(*op, vdst, src0, vsrc1);
}

/// @brief Encode VOP2 `v_add_nc_u32 vdst, literal, vsrc1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_add_nc_u32_e32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1,
                               rj_code_arch_t arch) {
  if (vdst > 255 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_add_nc_u32(arch);
  if (!op)
    return std::nullopt;
  return std::array<uint32_t, 2>{pack_vop2(*op, vdst, kVopLiteralSource, vsrc1), literal};
}

/// @brief Encode RDNA4 `ds_store_b32 vaddr, vdata offset:byte_offset`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_ds_store_b32(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD8340000u | byte_offset,
                                 static_cast<uint32_t>(vaddr) |
                                     (static_cast<uint32_t>(vdata) << 8u)};
}

/// @brief Encode RDNA4/gfx1250 `ds_store_b64 vaddr, v[vdata:vdata+1]`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_ds_store_b64(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vaddr > 255 || vdata > 254)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD9340000u | byte_offset,
                                 static_cast<uint32_t>(vaddr) |
                                     (static_cast<uint32_t>(vdata) << 8u)};
}

/// @brief Encode RDNA4/gfx1250 `ds_store_b128 vaddr, v[vdata:vdata+3]`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_ds_store_b128(uint16_t vaddr, uint16_t vdata, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vaddr > 255 || vdata > 252)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xDB7C0000u | byte_offset,
                                 static_cast<uint32_t>(vaddr) |
                                     (static_cast<uint32_t>(vdata) << 8u)};
}

/// @brief Encode RDNA4 `ds_storexchg_rtn_b64 vdst, vaddr, vdata`.
///
/// @details The data and destination operands each name the first register of
/// a consecutive VGPR pair. The returned pair contains the prior 64-bit LDS
/// value after the caller waits for DSCNT.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_ds_storexchg_rtn_b64(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                           rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 254 || vaddr > 255 || vdata > 254)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD9B40000u | byte_offset,
                                 static_cast<uint32_t>(vaddr) |
                                     (static_cast<uint32_t>(vdata) << 8u) |
                                     (static_cast<uint32_t>(vdst) << 24u)};
}

/// @brief Encode RDNA4/gfx1250 `ds_storexchg_rtn_b32 vdst, vaddr, vdata`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_ds_storexchg_rtn_b32(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                           rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 255 || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD8B40000u | byte_offset,
                                 static_cast<uint32_t>(vaddr) |
                                     (static_cast<uint32_t>(vdata) << 8u) |
                                     (static_cast<uint32_t>(vdst) << 24u)};
}

/// @brief Encode RDNA4/gfx1250 `ds_or_rtn_b32 vdst, vaddr, vdata`.
///
/// @details The destination receives the prior 32-bit LDS word after the
/// caller waits for DSCNT. This is used to claim and observe independently
/// packed lazy-initialization state bits without disturbing adjacent cells.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_ds_or_rtn_b32(uint16_t vdst, uint16_t vaddr, uint16_t vdata, uint8_t byte_offset,
                    rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 255 || vaddr > 255 || vdata > 255)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD8A80000u | byte_offset,
                                 static_cast<uint32_t>(vaddr) |
                                     (static_cast<uint32_t>(vdata) << 8u) |
                                     (static_cast<uint32_t>(vdst) << 24u)};
}

/// @brief Encode RDNA4/gfx1250 `ds_load_b32 vdst, vaddr`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_ds_load_b32(uint16_t vdst, uint16_t vaddr, uint8_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 255 || vaddr > 255)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD8D80000u | byte_offset,
                                 static_cast<uint32_t>(vaddr) |
                                     (static_cast<uint32_t>(vdst) << 24u)};
}

/// @brief Encode VOP2 `v_min_u32 vdst, literal, vsrc1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_min_u32_e32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_min_u32(arch);
  if (!op)
    return std::nullopt;
  return std::array<uint32_t, 2>{pack_vop2(*op, vdst, kVopLiteralSource, vsrc1), literal};
}

/// @brief Encode VOP2 `v_and_b32 vdst, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_and_b32_e32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_and_b32(arch);
  if (!op)
    return std::nullopt;
  return pack_vop2(*op, vdst, src0, vsrc1);
}

/// @brief Encode VOP2 `v_and_b32 vdst, literal, vsrc1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_and_b32_e32_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_and_b32(arch);
  if (!op)
    return std::nullopt;
  return std::array<uint32_t, 2>{pack_vop2(*op, vdst, kVopLiteralSource, vsrc1), literal};
}

/// @brief Encode RDNA4 VOP3 `v_mul_lo_u32 vdst, literal, vsrc1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_v_mul_lo_u32_vop3_literal(uint16_t vdst, uint32_t literal, uint16_t vsrc1,
                                rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 255 || vsrc1 > 255)
    return std::nullopt;
  return std::array<uint32_t, 3>{0xD72C0000u | vdst,
                                 0x02000000u | kVopLiteralSource |
                                     (static_cast<uint32_t>(vector_source_vgpr(vsrc1)) << 9u),
                                 literal};
}

/// @brief Encode VOP2 `v_xor_b32 vdst, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_xor_b32_e32(uint16_t vdst, uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (vdst > 255 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  const std::optional<uint32_t> op = vop2_op_xor_b32(arch);
  if (!op)
    return std::nullopt;
  return pack_vop2(*op, vdst, src0, vsrc1);
}

/// @brief Encode an RDNA4 64-bit VGPR add of a zero-extended VGPR offset.
/// @details Produces `v_add_co_u32`/`v_add_co_ci_u32` and the required gfx12
/// ALU dependency wait. VCC is clobbered and must be saved by the caller.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 5>>
build_v_add_u64_vgpr_offset(uint16_t address_vgpr, uint16_t offset_vgpr, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || address_vgpr >= 255 || offset_vgpr > 255)
    return std::nullopt;
  constexpr uint16_t kVccLo = 106;
  const auto pack_vop3 = [](uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2,
                            uint16_t sdst) {
    const uint32_t w0 =
        (vdst & 0xffu) | ((sdst & 0x7fu) << 8u) | ((op & 0x3ffu) << 16u) | (0x35u << 26u);
    const uint32_t w1 = (src0 & 0x1ffu) | ((src1 & 0x1ffu) << 9u) | ((src2 & 0x1ffu) << 18u);
    return std::array<uint32_t, 2>{w0, w1};
  };
  const auto low =
      arch == ROCJITSU_CODE_ARCH_GFX1250
          ? gfx1250::build_vop3_sdst_enc(gfx1250::kVAddCoU32Vop3SdstEnc,
                                         {.vdst = static_cast<uint8_t>(address_vgpr),
                                          .sdst = kVccLo,
                                          .src0 = vector_source_vgpr(offset_vgpr),
                                          .src1 = vector_source_vgpr(address_vgpr),
                                          .src2 = scalar_positive_inline_u32(0)})
          : pack_vop3(rdna4::kVAddCoU32Vop3SdstEnc, static_cast<uint8_t>(address_vgpr),
                      vector_source_vgpr(offset_vgpr), vector_source_vgpr(address_vgpr), 0, kVccLo);
  const auto high =
      arch == ROCJITSU_CODE_ARCH_GFX1250
          ? gfx1250::build_vop3_sdst_enc(
                gfx1250::kVAddCoCiU32Vop3SdstEnc,
                {.vdst = static_cast<uint8_t>(address_vgpr + 1u),
                 .sdst = kVccLo,
                 .src0 = scalar_positive_inline_u32(0),
                 .src1 = vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)),
                 .src2 = kVccLo})
          : pack_vop3(rdna4::kVAddCoCiU32Vop3SdstEnc, static_cast<uint8_t>(address_vgpr + 1u),
                      scalar_positive_inline_u32(0),
                      vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)), kVccLo, kVccLo);
  return std::array<uint32_t, 5>{low[0], low[1], pack_sopp(rdna4::kSWaitAlu, 0xfffdu), high[0],
                                 high[1]};
}

/// @brief Encode an RDNA4 64-bit VGPR add of a signed 24-bit displacement.
/// @details Produces `v_add_co_u32`/`v_add_co_ci_u32`, including the literal
/// low dword, sign-extended high dword, and required gfx12 ALU dependency
/// wait. VCC is clobbered and must be saved by the caller.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 6>>
build_v_add_u64_signed_i24(uint16_t address_vgpr, int32_t displacement, rj_code_arch_t arch) {
  constexpr int32_t kSigned24Min = -(1 << 23);
  constexpr int32_t kSigned24Max = (1 << 23) - 1;
  if (!is_rdna4_family_arch(arch) || address_vgpr >= 255 || displacement < kSigned24Min ||
      displacement > kSigned24Max)
    return std::nullopt;
  constexpr uint16_t kVccLo = 106;
  constexpr uint16_t kScalarInlineNegativeOne = 193;
  const auto pack_vop3 = [](uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2,
                            uint16_t sdst) {
    const uint32_t w0 =
        (vdst & 0xffu) | ((sdst & 0x7fu) << 8u) | ((op & 0x3ffu) << 16u) | (0x35u << 26u);
    const uint32_t w1 = (src0 & 0x1ffu) | ((src1 & 0x1ffu) << 9u) | ((src2 & 0x1ffu) << 18u);
    return std::array<uint32_t, 2>{w0, w1};
  };
  const uint16_t high_displacement =
      displacement < 0 ? kScalarInlineNegativeOne : scalar_positive_inline_u32(0);
  uint16_t low_displacement = kVopLiteralSource;
  uint32_t low_extension = static_cast<uint32_t>(displacement);
  if (arch == ROCJITSU_CODE_ARCH_GFX1250 && displacement >= 0 && displacement <= 64) {
    low_displacement = scalar_positive_inline_u32(static_cast<uint16_t>(displacement));
    low_extension = build_s_nop(0, arch);
  } else if (arch == ROCJITSU_CODE_ARCH_GFX1250 && displacement >= -16 && displacement < 0) {
    low_displacement = static_cast<uint16_t>(192 - displacement);
    low_extension = build_s_nop(0, arch);
  }
  const auto low = arch == ROCJITSU_CODE_ARCH_GFX1250
                       ? gfx1250::build_vop3_sdst_enc(gfx1250::kVAddCoU32Vop3SdstEnc,
                                                      {.vdst = static_cast<uint8_t>(address_vgpr),
                                                       .sdst = kVccLo,
                                                       .src0 = low_displacement,
                                                       .src1 = vector_source_vgpr(address_vgpr),
                                                       .src2 = scalar_positive_inline_u32(0)})
                       : pack_vop3(rdna4::kVAddCoU32Vop3SdstEnc, static_cast<uint8_t>(address_vgpr),
                                   kVopLiteralSource, vector_source_vgpr(address_vgpr), 0, kVccLo);
  const auto high =
      arch == ROCJITSU_CODE_ARCH_GFX1250
          ? gfx1250::build_vop3_sdst_enc(
                gfx1250::kVAddCoCiU32Vop3SdstEnc,
                {.vdst = static_cast<uint8_t>(address_vgpr + 1u),
                 .sdst = kVccLo,
                 .src0 = high_displacement,
                 .src1 = vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)),
                 .src2 = kVccLo})
          : pack_vop3(rdna4::kVAddCoCiU32Vop3SdstEnc, static_cast<uint8_t>(address_vgpr + 1u),
                      high_displacement,
                      vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)), kVccLo, kVccLo);
  return std::array<uint32_t, 6>{
      low[0], low[1], low_extension, pack_sopp(rdna4::kSWaitAlu, 0xfffdu), high[0], high[1]};
}

/// @brief Encode RDNA4 `v_readfirstlane_b32 sdst, vsrc`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_readfirstlane_b32(uint16_t sdst, uint16_t vsrc, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 127 || vsrc > 255)
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return gfx1250::build_vop1(
        gfx1250::kVReadfirstlaneB32Vop1,
        {.src0 = vector_source_vgpr(vsrc), .vdst = static_cast<uint8_t>(sdst)})[0];
  return (0x3Fu << 25) | (static_cast<uint32_t>(sdst) << 17) | (2u << 9) | vector_source_vgpr(vsrc);
}

/// @brief Encode RDNA4 `v_mbcnt_lo_u32_b32 vdst, src0, src1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_mbcnt_lo_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD71F0000u | static_cast<uint32_t>(vdst),
                                 (static_cast<uint32_t>(src1) << 9u) | src0};
}

/// @brief Encode RDNA4 `v_mbcnt_hi_u32_b32 vdst, src0, src1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_mbcnt_hi_u32_b32(uint16_t vdst, uint16_t src0, uint16_t src1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 255 || src0 > 511 || src1 > 511)
    return std::nullopt;
  return std::array<uint32_t, 2>{0xD7200000u | static_cast<uint32_t>(vdst),
                                 (static_cast<uint32_t>(src1) << 9u) | src0};
}

/// @brief Encode RDNA4 `v_cmp_eq_u32_e32 vcc_lo, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_eq_u32_e32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return 0x7C940000u | (static_cast<uint32_t>(vsrc1) << 9u) | src0;
}

/// @brief Encode RDNA4-family `v_cmp_ne_u32_e32 vcc_lo, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_ne_u32_e32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return gfx1250::build_vopc(gfx1250::kVCmpNeU32Vopc,
                               {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
  return rdna4::build_vopc(rdna4::kVCmpNeU32Vopc,
                           {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

/// @brief Encode RDNA4-family `v_cmp_ne_u16_e32 vcc_lo, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_ne_u16_e32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250)
    return gfx1250::build_vopc(gfx1250::kVCmpNeU16Vopc,
                               {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
  return rdna4::build_vopc(rdna4::kVCmpNeU16Vopc,
                           {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

/// @brief Encode RDNA4 `v_cmp_gt_u32_e32 vcc_lo, src0, vsrc1`.
///
/// Useful as `src0=capacity, vsrc1=slot`, which tests `slot < capacity` while
/// keeping the immediate-like operand in the source position this encoding
/// accepts.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_v_cmp_gt_u32_e32_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return 0x7C980000u | (static_cast<uint32_t>(vsrc1) << 9u) | src0;
}

/// @brief Encode RDNA4/gfx1250 `v_cmp_gt_u32_e32 vcc_lo, literal, vsrc1`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_v_cmp_gt_u32_e32_vcc_literal(uint32_t literal, uint16_t vsrc1, rj_code_arch_t arch) {
  const auto compare = build_v_cmp_gt_u32_e32_vcc(kVopLiteralSource, vsrc1, arch);
  if (!compare)
    return std::nullopt;
  return std::array<uint32_t, 2>{*compare, literal};
}

/// @brief Encode RDNA4 `v_mov_b32` in VOP3 form with a literal source.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_v_mov_b32_e64_literal(uint16_t vdst, uint32_t literal, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 255)
    return std::nullopt;
  return std::array<uint32_t, 3>{0xD5810000u | static_cast<uint32_t>(vdst), 0x000000FFu, literal};
}

/// @brief Encode RDNA4 `flat_store_b32 v[vaddr:vaddr+1], vsrc`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_store_b32_vaddr_vsrc(uint16_t vaddr, uint16_t vsrc, rj_code_arch_t arch,
                                uint32_t byte_offset = 0) {
  if (!is_rdna4_family_arch(arch) || vaddr > 255 || vsrc > 255 || byte_offset > 0xffffffu)
    return std::nullopt;
  return std::array<uint32_t, 3>{0xEC068000u | flat_no_saddr_encoding(arch),
                                 static_cast<uint32_t>(vsrc) << 23u,
                                 static_cast<uint32_t>(vaddr) | (byte_offset << 8u)};
}

/// @brief Encode RDNA4 `flat_load_b32 vdst, v[vaddr:vaddr+1]`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_load_b32_vaddr_vdst(uint16_t vaddr, uint16_t vdst, rj_code_arch_t arch,
                               uint32_t byte_offset = 0) {
  if (!is_rdna4_family_arch(arch) || vaddr > 255 || vdst > 255 || byte_offset > 0xffffffu)
    return std::nullopt;
  return std::array<uint32_t, 3>{0xEC050000u | flat_no_saddr_encoding(arch),
                                 static_cast<uint32_t>(vdst),
                                 static_cast<uint32_t>(vaddr) | (byte_offset << 8u)};
}

/// @brief Encode gfx12 `scratch_store_b32 off, vsrc, off offset:byte_offset`.
///
/// @details The address-free form uses only the implicit per-lane scratch base
/// and the positive signed-24-bit immediate. DBI spill slots are dword
/// aligned, so unaligned offsets are rejected even though the ISA field itself
/// is byte-addressed.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_address_free_scratch_store_b32(uint16_t vsrc, uint32_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vsrc > 255 ||
      byte_offset > kMaxAddressFreeScratchDwordOffset || byte_offset % sizeof(uint32_t) != 0) {
    return std::nullopt;
  }
  constexpr uint32_t kRdna4ScratchNoSaddr = 0x7c;
  return std::array<uint32_t, 3>{0xed068000u | kRdna4ScratchNoSaddr,
                                 static_cast<uint32_t>(vsrc) << 23u, byte_offset << 8u};
}

/// @brief Encode gfx12 `scratch_load_b32 vdst, off, off offset:byte_offset`.
/// @copydetails build_address_free_scratch_store_b32
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_address_free_scratch_load_b32(uint16_t vdst, uint32_t byte_offset, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 255 ||
      byte_offset > kMaxAddressFreeScratchDwordOffset || byte_offset % sizeof(uint32_t) != 0) {
    return std::nullopt;
  }
  constexpr uint32_t kRdna4ScratchNoSaddr = 0x7c;
  return std::array<uint32_t, 3>{0xed050000u | kRdna4ScratchNoSaddr, static_cast<uint32_t>(vdst),
                                 byte_offset << 8u};
}

/// @brief Encode gfx12 `scratch_store_b32 off, vsrc, saddr offset:byte_offset`.
///
/// @details The explicit scalar offset form is used for a dynamic-stack frame
/// whose base is held in @p saddr. Unlike the address-free fixed-segment form,
/// offsets are relative to the frame selected by the caller.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_scratch_store_b32_saddr(uint16_t vsrc, uint16_t saddr, uint32_t byte_offset,
                              rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vsrc > 255 || saddr > 127 ||
      byte_offset > kMaxAddressFreeScratchDwordOffset || byte_offset % sizeof(uint32_t) != 0) {
    return std::nullopt;
  }
  return std::array<uint32_t, 3>{0xed068000u | saddr, static_cast<uint32_t>(vsrc) << 23u,
                                 byte_offset << 8u};
}

/// @brief Encode gfx12 `scratch_load_b32 vdst, off, saddr offset:byte_offset`.
/// @copydetails build_scratch_store_b32_saddr
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_scratch_load_b32_saddr(uint16_t vdst, uint16_t saddr, uint32_t byte_offset,
                             rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vdst > 255 || saddr > 127 ||
      byte_offset > kMaxAddressFreeScratchDwordOffset || byte_offset % sizeof(uint32_t) != 0) {
    return std::nullopt;
  }
  return std::array<uint32_t, 3>{0xed050000u | saddr, static_cast<uint32_t>(vdst),
                                 byte_offset << 8u};
}

/// @brief Encode gfx12 `s_wait_storecnt 0`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_wait_storecnt0(rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return pack_sopp(rdna4::kSWaitStorecntSopp, 0);
}

/// @brief Encode gfx12 `s_wait_storecnt_dscnt 0`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_storecnt_dscnt0(rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return pack_sopp(rdna4::kSWaitStorecntDscntSopp, 0);
}

/// @brief Encode gfx12 `s_wait_loadcnt_dscnt 0`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_loadcnt_dscnt0(rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return pack_sopp(rdna4::kSWaitLoadcntDscntSopp, 0);
}

/// @brief Encode gfx12 `s_wait_loadcnt 0`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_wait_loadcnt0(rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return pack_sopp(rdna4::kSWaitLoadcntSopp, 0);
}

/// @brief Encode gfx12 `s_wait_alu depctr_sa_sdst(0)`.
///
/// This is the conservative scalar dependency drain emitted by LLVM before an
/// indirect branch consumes an SGPR pair that was just constructed by SALU
/// instructions. A one-cycle `s_delay_alu` is not sufficient for a chained
/// low/high PC calculation.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_wait_alu_sa_sdst0(rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return pack_sopp(rdna4::kSWaitAlu, 0xff9eu);
}

/// @brief Encode RDNA4 `flat_atomic_add_u32 vdst, v[vaddr:vaddr+1], vsrc`.
///
/// @details Uses the no-SADDR flat form. `return_old_value=true` encodes the
/// GFX12 atomic-return TH value so the old memory value is written to @p vdst.
/// @p scope is the two-bit RDNA4 SCOPE field; device scope is 2.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_atomic_add_u32_vaddr_vsrc_vdst(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                          bool return_old_value, uint8_t scope,
                                          rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vaddr > 255 || vsrc > 255 || vdst > 255 || scope > 3)
    return std::nullopt;
  constexpr uint32_t kRdna4AtomicReturnTh = 1;
  const uint32_t th = return_old_value ? kRdna4AtomicReturnTh : 0u;
  return std::array<uint32_t, 3>{0xEC0D4000u | flat_no_saddr_encoding(arch),
                                 static_cast<uint32_t>(vdst) |
                                     (static_cast<uint32_t>(scope) << 18u) | (th << 20u) |
                                     (static_cast<uint32_t>(vsrc) << 23u),
                                 static_cast<uint32_t>(vaddr)};
}

/// @brief Encode RDNA4 `flat_atomic_or_b32 vdst, v[vaddr:vaddr+1], vsrc`.
///
/// @details Uses the no-SADDR flat form. `return_old_value=true` encodes the
/// GFX12 atomic-return TH value so the old memory value is written to @p vdst.
/// @p scope is the two-bit RDNA4 SCOPE field; device scope is 2.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_atomic_or_u32_vaddr_vsrc_vdst(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                         bool return_old_value, uint8_t scope,
                                         rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vaddr > 255 || vsrc > 255 || vdst > 255 || scope > 3)
    return std::nullopt;
  constexpr uint32_t kRdna4AtomicReturnTh = 1;
  const uint32_t th = return_old_value ? kRdna4AtomicReturnTh : 0u;
  return std::array<uint32_t, 3>{0xEC0F4000u | flat_no_saddr_encoding(arch),
                                 static_cast<uint32_t>(vdst) |
                                     (static_cast<uint32_t>(scope) << 18u) | (th << 20u) |
                                     (static_cast<uint32_t>(vsrc) << 23u),
                                 static_cast<uint32_t>(vaddr)};
}

/// @brief Encode RDNA4 `flat_atomic_cmpswap_b32 vdst, v[vaddr:vaddr+1],
///        v[vsrc:vsrc+1]` with `vsrc=new_value` and `vsrc+1=compare_value`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                              bool return_old_value, uint8_t scope,
                                              rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vaddr > 254 || vsrc > 254 || vdst > 255 || scope > 3)
    return std::nullopt;
  constexpr uint32_t kRdna4AtomicReturnTh = 1;
  const uint32_t th = return_old_value ? kRdna4AtomicReturnTh : 0u;
  return std::array<uint32_t, 3>{0xEC0D0000u | flat_no_saddr_encoding(arch),
                                 static_cast<uint32_t>(vdst) |
                                     (static_cast<uint32_t>(scope) << 18u) | (th << 20u) |
                                     (static_cast<uint32_t>(vsrc) << 23u),
                                 static_cast<uint32_t>(vaddr)};
}

/// @brief Encode RDNA4 `flat_atomic_swap_b64 v[vdst:vdst+1], v[vaddr:vaddr+1],
///        v[vsrc:vsrc+1]`.
///
/// @details Uses the no-SADDR flat form. `return_old_value=true` encodes the
/// GFX12 atomic-return TH value so the old 64-bit memory value is written to
/// @p vdst/@p vdst+1. @p scope is the two-bit RDNA4 SCOPE field; device scope
/// is 2.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_atomic_swap_b64_vaddr_vsrc_vdst(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                           bool return_old_value, uint8_t scope,
                                           rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vaddr > 254 || vsrc > 254 || vdst > 254 || scope > 3)
    return std::nullopt;
  constexpr uint32_t kRdna4AtomicReturnTh = 1;
  const uint32_t th = return_old_value ? kRdna4AtomicReturnTh : 0u;
  return std::array<uint32_t, 3>{0xEC104000u | flat_no_saddr_encoding(arch),
                                 static_cast<uint32_t>(vdst) |
                                     (static_cast<uint32_t>(scope) << 18u) | (th << 20u) |
                                     (static_cast<uint32_t>(vsrc) << 23u),
                                 static_cast<uint32_t>(vaddr)};
}

/// @brief Encode RDNA4 `flat_atomic_add_u64 v[vdst:vdst+1],
///        v[vaddr:vaddr+1], v[vsrc:vsrc+1]`.
///
/// @details A zero source with `return_old_value=true` is an atomic 64-bit
/// snapshot that leaves memory unchanged.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_flat_atomic_add_u64_vaddr_vsrc_vdst(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                          bool return_old_value, uint8_t scope,
                                          rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || vaddr > 254 || vsrc > 254 || vdst > 254 || scope > 3)
    return std::nullopt;
  constexpr uint32_t kRdna4AtomicReturnTh = 1;
  const uint32_t th = return_old_value ? kRdna4AtomicReturnTh : 0u;
  return std::array<uint32_t, 3>{0xEC10C000u | flat_no_saddr_encoding(arch),
                                 static_cast<uint32_t>(vdst) |
                                     (static_cast<uint32_t>(scope) << 18u) | (th << 20u) |
                                     (static_cast<uint32_t>(vsrc) << 23u),
                                 static_cast<uint32_t>(vaddr)};
}

/// @brief Encode RDNA4 `s_sub_u32 sdst, ssrc0, ssrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_sub_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  constexpr uint32_t kRdna4Sop2SubU32 = 1;
  return pack_sop2(kRdna4Sop2SubU32, sdst, ssrc0, ssrc1);
}

/// @brief Encode RDNA4 `s_add_u32 sdst, ssrc0, ssrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna4_s_add_u32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  constexpr uint32_t kRdna4Sop2AddU32 = 2;
  return pack_sop2(kRdna4Sop2AddU32, sdst, ssrc0, ssrc1);
}

/// @brief Encode RDNA4 `s_mov_b64 s[sdst:sdst+1], s[ssrc0:ssrc0+1]`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_mov_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 126 || ssrc0 > 255)
    return std::nullopt;
  return pack_sop1(1, sdst, ssrc0);
}

/// @brief Encode RDNA4 `s_and_saveexec_b64 s[sdst:sdst+1], s[ssrc0:ssrc0+1]`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_and_saveexec_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 126 || ssrc0 > 255)
    return std::nullopt;
  return pack_sop1(0x21, sdst, ssrc0);
}

/// @brief Encode RDNA4 `s_and_not1_b64 s[sdst:sdst+1],
/// s[ssrc0:ssrc0+1], s[ssrc1:ssrc1+1]`.
///
/// This is the gfx12 spelling of the older `s_andn2_b64` operation. It is
/// useful when an instrumentation loop must remove an already-processed lane
/// partition from a saved EXEC mask without disturbing EXEC itself.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_and_not1_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  constexpr uint32_t kRdna4Sop2AndNot1B64 = 0x23;
  return pack_sop2(kRdna4Sop2AndNot1B64, sdst, ssrc0, ssrc1);
}

/// @brief Encode RDNA4/gfx1250 `s_and_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_and_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  constexpr uint32_t kRdna4Sop2AndB64 = 0x17;
  return pack_sop2(kRdna4Sop2AndB64, sdst, ssrc0, ssrc1);
}

/// @brief Encode RDNA4/gfx1250 `s_bcnt1_i32_b64`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_bcnt1_i32_b64(uint16_t sdst, uint16_t ssrc0, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 127 || ssrc0 > 254)
    return std::nullopt;
  constexpr uint32_t kRdna4Sop1Bcnt1I32B64 = 25u;
  return pack_sop1(kRdna4Sop1Bcnt1I32B64, sdst, ssrc0);
}

/// @brief Encode RDNA4 `s_xor_b64 s[sdst:sdst+1],
/// s[ssrc0:ssrc0+1], s[ssrc1:ssrc1+1]`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_xor_b64(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 126 || ssrc0 > 254 || ssrc1 > 254)
    return std::nullopt;
  constexpr uint32_t kRdna4Sop2XorB64 = 0x1b;
  return pack_sop2(kRdna4Sop2XorB64, sdst, ssrc0, ssrc1);
}

/// @brief Encode RDNA4 `s_cselect_b32 sdst, ssrc0, ssrc1`.
///
/// This is useful for materializing SCC without changing it: choose inline 1
/// when SCC is set and inline 0 otherwise.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna4_s_cselect_b32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || sdst > 127 || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  constexpr uint32_t kRdna4Sop2CselectB32 = 0x30;
  return pack_sop2(kRdna4Sop2CselectB32, sdst, ssrc0, ssrc1);
}

/// @brief Encode RDNA4 `s_cmp_lg_u32 ssrc0, ssrc1`.
///
/// Comparing a previously materialized SCC value with inline zero restores
/// SCC to that saved Boolean value.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_rdna4_s_cmp_lg_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  constexpr uint32_t kRdna4SopcCmpLgU32 = 7;
  return pack_sopc(kRdna4SopcCmpLgU32, ssrc0, ssrc1);
}

/// @brief Encode RDNA4 `s_cmp_eq_u32 ssrc0, ssrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_cmp_eq_u32(uint16_t ssrc0, uint16_t ssrc1, rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch) || ssrc0 > 255 || ssrc1 > 255)
    return std::nullopt;
  constexpr uint32_t kRdna4SopcCmpEqU32 = 6;
  return pack_sopc(kRdna4SopcCmpEqU32, ssrc0, ssrc1);
}

/// @brief Encode RDNA4 `s_cbranch_scc0`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_scc0(int16_t offset_dwords,
                                                                            rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  constexpr uint32_t kRdna4SoppCbranchScc0 = 33;
  return pack_sopp(kRdna4SoppCbranchScc0, static_cast<uint16_t>(offset_dwords));
}

/// @brief Encode RDNA4 `s_cbranch_scc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_scc1(int16_t offset_dwords,
                                                                            rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  constexpr uint32_t kRdna4SoppCbranchScc1 = 34;
  return pack_sopp(kRdna4SoppCbranchScc1, static_cast<uint16_t>(offset_dwords));
}

/// @brief Encode RDNA4 `s_cbranch_vccz`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_vccz(int16_t offset_dwords,
                                                                            rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  constexpr uint32_t kRdna4SoppCbranchVccz = 35;
  return pack_sopp(kRdna4SoppCbranchVccz, static_cast<uint16_t>(offset_dwords));
}

/// @brief Encode RDNA4 `s_cbranch_vccnz`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_vccnz(int16_t offset_dwords,
                                                                             rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  constexpr uint32_t kRdna4SoppCbranchVccnz = 36;
  return pack_sopp(kRdna4SoppCbranchVccnz, static_cast<uint16_t>(offset_dwords));
}

/// @brief Encode RDNA4 `s_cbranch_execz`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_execz(int16_t offset_dwords,
                                                                             rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  constexpr uint32_t kRdna4SoppCbranchExecz = 0x25;
  return pack_sopp(kRdna4SoppCbranchExecz, static_cast<uint16_t>(offset_dwords));
}

/// @brief Encode the RDNA4 workgroup-wide barrier signal/wait pair.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_barrier_signal_all(rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return pack_sop1(/*s_barrier_signal=*/0x4e, /*sdst=*/0, /*barrier_id=-1=*/0xc1);
}

[[nodiscard]] inline constexpr std::optional<uint32_t>
build_s_barrier_wait_all(rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  return pack_sopp(/*s_barrier_wait=*/0x14, /*barrier_id=-1=*/0xffff);
}

/// @brief Encode RDNA4 `s_cbranch_execnz`.
[[nodiscard]] inline constexpr std::optional<uint32_t> build_s_cbranch_execnz(int16_t offset_dwords,
                                                                              rj_code_arch_t arch) {
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  constexpr uint32_t kRdna4SoppCbranchExecnz = 0x26;
  return pack_sopp(kRdna4SoppCbranchExecnz, static_cast<uint16_t>(offset_dwords));
}

} // namespace rocjitsu
