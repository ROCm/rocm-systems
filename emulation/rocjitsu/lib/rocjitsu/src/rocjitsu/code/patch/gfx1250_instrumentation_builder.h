// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_instrumentation_builder.h
/// @brief gfx1250 instruction encoders used by DBI instrumentation.

#pragma once

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand_types.h"

namespace rocjitsu {

// Keep the packed no-SADDR field tied to the generated ISA metadata. This is
// both the semantic null selector and the encoding accepted by LLVM/RocJITsu.
inline constexpr uint8_t kGfx1250FlatNoSaddrEncoding = static_cast<uint8_t>(gfx1250::OPR_SREG_NULL);

/// @brief Encode gfx1250 `s_set_vgpr_msb simm16`.
///
/// Instrumentation uses this to select the low 256-register window while it
/// executes its own vector instructions, then restores the guest's operand
/// bank selection before executing displaced code.  The low byte selects the
/// new layout and the high byte records the previous layout.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_gfx1250_s_set_vgpr_msb(uint16_t packed_transition, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250)
    return std::nullopt;
  return 0xBF860000u | packed_transition;
}

/// @brief Encode a gfx1250 VGPR-bank transition.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_gfx1250_s_set_vgpr_msb_transition(uint8_t previous_mode, uint8_t new_mode,
                                        rj_code_arch_t arch) {
  return build_gfx1250_s_set_vgpr_msb(
      static_cast<uint16_t>((static_cast<uint16_t>(previous_mode) << 8u) | new_mode), arch);
}

/// @brief Encode gfx1250 `s_call_i64 s[sdst:sdst+1], simm16`.
///
/// The immediate is a signed dword offset from the instruction following the
/// call. The instruction writes that following PC to the destination pair.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_gfx1250_s_call_i64(uint16_t sdst, int16_t simm16, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || sdst > 104 || sdst % 2u != 0u)
    return std::nullopt;
  return pack_sopk(gfx1250::kSCallI64Sopk, sdst, static_cast<uint16_t>(simm16));
}

/// @brief Encode gfx1250 `v_cmp_ne_u16_e32 vcc_lo, src0, vsrc1`.
[[nodiscard]] inline constexpr std::optional<uint32_t>
build_gfx1250_v_cmp_ne_u16_vcc(uint16_t src0, uint16_t vsrc1, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || src0 > 511 || vsrc1 > 255)
    return std::nullopt;
  return gfx1250::build_vopc(gfx1250::kVCmpNeU16Vopc,
                             {.src0 = src0, .vsrc1 = static_cast<uint8_t>(vsrc1)})[0];
}

/// @brief Encode gfx1250 `v_writelane_b32 vdst, ssrc, lane`.
///
/// Unlike ordinary vector writes, this fixed-lane transfer executes
/// independently of EXEC and can therefore preserve scalar state across an
/// instrumentation body entered with no active lanes.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_gfx1250_v_writelane_b32(uint16_t vdst, uint16_t ssrc, uint16_t lane, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || vdst > 255 || ssrc > 105 || lane > 63)
    return std::nullopt;
  return gfx1250::build_vop3(gfx1250::kVWritelaneB32Vop3, {.vdst = static_cast<uint8_t>(vdst),
                                                           .src0 = ssrc,
                                                           .src1 = scalar_positive_inline_u32(lane),
                                                           .src2 = scalar_positive_inline_u32(0)});
}

/// @brief Encode gfx1250 `v_readlane_b32 sdst, vsrc, lane`.
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 2>>
build_gfx1250_v_readlane_b32(uint16_t sdst, uint16_t vsrc, uint16_t lane, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || sdst > 105 || vsrc > 255 || lane > 63)
    return std::nullopt;
  return gfx1250::build_vop3(gfx1250::kVReadlaneB32Vop3, {.vdst = static_cast<uint8_t>(sdst),
                                                          .src0 = vector_source_vgpr(vsrc),
                                                          .src1 = scalar_positive_inline_u32(lane),
                                                          .src2 = scalar_positive_inline_u32(0)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_store_b32(uint16_t vaddr, uint16_t vsrc, uint32_t byte_offset,
                             rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || vaddr > 255 || vsrc > 255 || byte_offset > 0xffffffu)
    return std::nullopt;
  return gfx1250::build_vflat(gfx1250::kFlatStoreB32Vflat, {.saddr = kGfx1250FlatNoSaddrEncoding,
                                                            .vsrc = static_cast<uint8_t>(vsrc),
                                                            .vaddr = static_cast<uint8_t>(vaddr),
                                                            .ioffset = byte_offset});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_load_b32(uint16_t vaddr, uint16_t vdst, uint32_t byte_offset,
                            rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || vaddr > 255 || vdst > 255 || byte_offset > 0xffffffu)
    return std::nullopt;
  return gfx1250::build_vflat(gfx1250::kFlatLoadB32Vflat, {.saddr = kGfx1250FlatNoSaddrEncoding,
                                                           .vdst = static_cast<uint8_t>(vdst),
                                                           .vaddr = static_cast<uint8_t>(vaddr),
                                                           .ioffset = byte_offset});
}

template <uint16_t Opcode>
[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_atomic(uint16_t vaddr, uint16_t vsrc, uint16_t vdst, bool return_old_value,
                          uint8_t scope, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || vaddr > 255 || vsrc > 255 || vdst > 255 || scope > 3)
    return std::nullopt;
  return gfx1250::build_vflat(Opcode, {.saddr = kGfx1250FlatNoSaddrEncoding,
                                       .vdst = static_cast<uint8_t>(vdst),
                                       .scope = scope,
                                       .th = static_cast<uint8_t>(return_old_value ? 1u : 0u),
                                       .vsrc = static_cast<uint8_t>(vsrc),
                                       .vaddr = static_cast<uint8_t>(vaddr)});
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_atomic_add_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                  bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  return build_gfx1250_flat_atomic<gfx1250::kFlatAtomicAddU32Vflat>(vaddr, vsrc, vdst,
                                                                    return_old_value, scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_atomic_or_u32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                 bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  return build_gfx1250_flat_atomic<gfx1250::kFlatAtomicOrB32Vflat>(vaddr, vsrc, vdst,
                                                                   return_old_value, scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_atomic_cmpswap_b32(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                      bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (vaddr > 254 || vsrc > 254)
    return std::nullopt;
  return build_gfx1250_flat_atomic<gfx1250::kFlatAtomicCmpswapB32Vflat>(
      vaddr, vsrc, vdst, return_old_value, scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_atomic_cmpswap_b64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                      bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (vaddr > 254 || vsrc > 252 || vdst > 254)
    return std::nullopt;
  return build_gfx1250_flat_atomic<gfx1250::kFlatAtomicCmpswapB64Vflat>(
      vaddr, vsrc, vdst, return_old_value, scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_atomic_swap_b64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                   bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (vaddr > 254 || vsrc > 254 || vdst > 254)
    return std::nullopt;
  return build_gfx1250_flat_atomic<gfx1250::kFlatAtomicSwapB64Vflat>(vaddr, vsrc, vdst,
                                                                     return_old_value, scope, arch);
}

[[nodiscard]] inline constexpr std::optional<std::array<uint32_t, 3>>
build_gfx1250_flat_atomic_add_u64(uint16_t vaddr, uint16_t vsrc, uint16_t vdst,
                                  bool return_old_value, uint8_t scope, rj_code_arch_t arch) {
  if (vaddr > 254 || vsrc > 254 || vdst > 254)
    return std::nullopt;
  return build_gfx1250_flat_atomic<gfx1250::kFlatAtomicAddU64Vflat>(vaddr, vsrc, vdst,
                                                                    return_old_value, scope, arch);
}

} // namespace rocjitsu
