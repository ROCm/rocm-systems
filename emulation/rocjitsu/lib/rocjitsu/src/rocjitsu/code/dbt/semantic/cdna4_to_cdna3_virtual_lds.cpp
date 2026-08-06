// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/semantic/cdna4_to_cdna3_virtual_lds.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/semantic/cdna3_lds.h"
#include "rocjitsu/code/dbt/semantic/cdna3_scratch.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

inline constexpr uint16_t kCdna3ScalarNull = 0x7F;
inline constexpr uint16_t kCdnaWaitcntAll0 = 0x0000;
inline constexpr uint32_t kFlatGlobalPositiveImm13Max = 4095;
inline constexpr uint32_t kCdnaOrdinarySgprLimit = 102;
inline constexpr uint32_t kCdnaSpecialSgprTailReserve = 8;
inline constexpr uint32_t kCdnaSmemImmediateByteOffsetMax = 0x1FFFFF;

/// @brief Common fields for CDNA3 FLAT instructions targeting global memory.
struct FlatGlobalOperands {
  uint16_t signed_offset13 = 0;
  bool sc0 = false;
  bool sc1 = false;
  bool nt = false;
  uint8_t addr = 0;
  uint8_t saddr = 0;
  bool acc = false;
};

[[nodiscard]] std::pair<uint32_t, uint32_t>
build_cdna3_smem_load(uint16_t op, uint8_t dst, uint8_t sbase, uint32_t byte_offset) {
  const auto words = cdna3::build_smem(
      op,
      {.sbase = static_cast<uint8_t>(sbase / 2), .sdata = dst, .imm = 1, .offset = byte_offset});
  return {words[0], words[1]};
}

[[nodiscard]] std::pair<uint32_t, uint32_t>
build_cdna3_vop2_literal(uint16_t op, uint8_t vdst, uint8_t vsrc1, uint32_t literal) {
  return {cdna3::build_vop2(op, {.src0 = 0xFF, .vsrc1 = vsrc1, .vdst = vdst})[0], literal};
}

[[nodiscard]] std::pair<uint32_t, uint32_t>
build_cdna3_flat_global(const FlatGlobalOperands &src, uint16_t op, uint8_t data, bool is_load) {
  cdna3::FlatBuilderFields fields{.offset = src.signed_offset13,
                                  .seg = 2,
                                  .sc0 = static_cast<uint8_t>(src.sc0),
                                  .nt = static_cast<uint8_t>(src.nt),
                                  .sc1 = static_cast<uint8_t>(src.sc1),
                                  .addr = src.addr,
                                  .saddr = src.saddr,
                                  .acc = static_cast<uint8_t>(src.acc)};
  if (is_load)
    fields.vdst = data;
  else
    fields.data = data;
  auto words = cdna3::build_flat(op, fields);
  words[0] |= static_cast<uint32_t>(src.signed_offset13 & 0x1000u);
  return {words[0], words[1]};
}

struct VirtualLdsVgprRange {
  uint16_t base = 0;
  uint16_t count = 0;
};

struct VirtualLdsAddressTemp {
  uint8_t reg = 0;
  bool spilled = false;
  uint32_t spill_offset = 0;
};

struct VirtualLdsTempRange {
  std::array<VirtualLdsAddressTemp, 4> temps{};
  uint8_t count = 0;

  [[nodiscard]] uint8_t base() const { return temps[0].reg; }
};

/// @brief Return true when two VGPR ranges overlap.
[[nodiscard]] bool virtual_lds_vgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_count,
                                                   uint16_t rhs_base, uint16_t rhs_count) {
  return lhs_base < rhs_base + rhs_count && rhs_base < lhs_base + lhs_count;
}

/// @brief Return true when @p reg is inside any reserved virtual-LDS temp range.
[[nodiscard]] bool virtual_lds_vgpr_is_forbidden(uint16_t reg,
                                                 std::span<const VirtualLdsVgprRange> forbidden) {
  return std::ranges::any_of(forbidden, [reg](const VirtualLdsVgprRange &range) {
    return virtual_lds_vgpr_ranges_overlap(reg, 1, range.base, range.count);
  });
}

/// @brief Return true when any register in [@p reg, @p reg + @p count) is reserved.
[[nodiscard]] bool
virtual_lds_vgpr_range_is_forbidden(uint16_t reg, uint16_t count,
                                    std::span<const VirtualLdsVgprRange> forbidden) {
  return std::ranges::any_of(forbidden, [reg, count](const VirtualLdsVgprRange &range) {
    return virtual_lds_vgpr_ranges_overlap(reg, count, range.base, range.count);
  });
}

/// @brief Pick one VGPR temporary for a virtual-LDS address calculation.
///
/// @details The returned temp is either newly descriptor-backed or marked for
/// per-use private-scratch spilling when the descriptor cannot grow. The caller
/// provides @p forbidden for registers already live inside the same replacement
/// sequence, such as source address/data/destination tuples.
[[nodiscard]] std::optional<VirtualLdsAddressTemp>
choose_virtual_lds_address_temp(TranslationContext &context,
                                std::span<const VirtualLdsVgprRange> forbidden) {
  // Prefer a descriptor-backed temporary before falling back to flat scratch.
  // Virtual-LDS lowering often runs in hand-written assembly kernels that did
  // not request flat-scratch initialization SGPRs because the source kernel did
  // not use scratch. Growing the ordinary VGPR count is cheaper and avoids
  // introducing a new flat-scratch ABI dependency. The chosen register is
  // reusable across independent DS replacements; `forbidden` prevents reusing it
  // twice inside one replacement sequence.
  const uint32_t first_extra_vgpr = context.num_vgprs;
  // Descriptor recomputation can move the AccVGPR base upward when semantic
  // lowering requests more ordinary VGPRs. Do not cap scratch growth at the
  // source ACCUM_OFFSET here; doing so makes tiny kernels whose descriptor
  // decodes as v0..v3 + a0..a3 fail before the recompute has a chance to move
  // the accumulator window.
  const uint32_t extra_limit = 256;
  for (uint32_t candidate = first_extra_vgpr; candidate < extra_limit; ++candidate) {
    if (virtual_lds_vgpr_is_forbidden(static_cast<uint16_t>(candidate), forbidden))
      continue;
    context.require_vgprs(candidate + 1);
    return VirtualLdsAddressTemp{.reg = static_cast<uint8_t>(candidate)};
  }

  // If the descriptor cannot grow without colliding with the AccVGPR window or
  // the architectural VGPR limit, preserve an existing ordinary VGPR around the
  // replacement sequence with per-lane scratch.
  const uint32_t initial_vgprs = std::min<uint32_t>(context.num_vgprs, 256);
  for (uint32_t reg = initial_vgprs; reg > 0; --reg) {
    const uint16_t candidate = static_cast<uint16_t>(reg - 1);
    if (virtual_lds_vgpr_is_forbidden(candidate, forbidden))
      continue;
    return VirtualLdsAddressTemp{
        .reg = static_cast<uint8_t>(candidate),
        .spilled = true,
    };
  }

  return std::nullopt;
}

/// @brief Pick a contiguous VGPR run for wide virtual-LDS data/address operands.
///
/// @details CDNA3 FLAT/GLOBAL encodings require even-aligned tuples for some
/// wide operands. The helper first tries to grow the descriptor, then falls
/// back to a spill-backed borrowed run if all extra VGPRs are unavailable.
[[nodiscard]] std::optional<VirtualLdsTempRange>
choose_virtual_lds_temp_range(TranslationContext &context,
                              std::span<const VirtualLdsVgprRange> forbidden, uint16_t count,
                              uint16_t alignment = 1) {
  if (count == 0 || count > 4)
    return std::nullopt;
  if (alignment == 0)
    alignment = 1;

  // Wide FLAT/GLOBAL operations need their data/destination operands in a
  // contiguous VGPR run. Prefer growing the descriptor because those extra
  // VGPRs require no flat-scratch save/restore around the replacement.
  const uint32_t first_extra_vgpr = context.num_vgprs;
  const uint32_t extra_limit = 256;
  for (uint32_t candidate = first_extra_vgpr; candidate + count <= extra_limit; ++candidate) {
    if ((candidate % alignment) != 0)
      continue;
    if (virtual_lds_vgpr_range_is_forbidden(static_cast<uint16_t>(candidate), count, forbidden))
      continue;
    context.require_vgprs(candidate + count);
    VirtualLdsTempRange range{};
    range.count = static_cast<uint8_t>(count);
    for (uint16_t i = 0; i < count; ++i)
      range.temps[i].reg = static_cast<uint8_t>(candidate + i);
    return range;
  }

  // Descriptor-full kernels can still borrow an existing contiguous run by
  // saving it in private scratch. This is slower, but keeps virtual LDS real
  // instead of skipping kernels whose data operands alias the temporary address
  // pair.
  const uint32_t initial_vgprs = std::min<uint32_t>(context.num_vgprs, 256);
  if (initial_vgprs >= count) {
    for (uint32_t reg = initial_vgprs - count + 1; reg > 0; --reg) {
      const uint16_t candidate = static_cast<uint16_t>(reg - 1);
      if ((candidate % alignment) != 0)
        continue;
      if (virtual_lds_vgpr_range_is_forbidden(candidate, count, forbidden))
        continue;
      VirtualLdsTempRange range{};
      range.count = static_cast<uint8_t>(count);
      for (uint16_t i = 0; i < count; ++i) {
        range.temps[i].reg = static_cast<uint8_t>(candidate + i);
        range.temps[i].spilled = true;
      }
      return range;
    }
  }

  return std::nullopt;
}

/// @brief Assign private-scratch slots to spill-backed virtual-LDS VGPR temps.
///
/// @details Returns the first byte after the assigned temp slots so callers can
/// reserve adjacent extra words for non-temp state, such as saved SGPR-pair
/// values or a saved high address VGPR.
///
/// @returns The first byte past the assigned slots, or std::nullopt if the spill
/// reservation overflows the 32-bit private segment (the caller must fail the
/// lowering rather than emit save/restore at a wrapped offset).
std::optional<uint32_t>
assign_virtual_lds_spill_offsets(TranslationContext &context,
                                 const std::vector<VirtualLdsAddressTemp *> &temps,
                                 uint32_t extra_dwords = 0) {
  uint32_t spilled_count = 0;
  for (const VirtualLdsAddressTemp *temp : temps) {
    if (temp != nullptr && temp->spilled)
      ++spilled_count;
  }
  const uint32_t total_dwords = spilled_count + extra_dwords;
  if (total_dwords == 0)
    return 0u;

  const auto base_offset = context.reserve_semantic_spill_dwords(total_dwords);
  if (!base_offset)
    return std::nullopt;
  uint32_t index = 0;
  for (VirtualLdsAddressTemp *temp : temps) {
    if (temp == nullptr || !temp->spilled)
      continue;
    temp->spill_offset = *base_offset + index * sizeof(uint32_t);
    ++index;
  }
  return *base_offset + spilled_count * sizeof(uint32_t);
}

/// @brief Emit a CDNA3 flat-scratch store used to preserve a borrowed VGPR temp.
void emit_cdna3_scratch_store_b32(std::vector<uint32_t> &words, uint8_t data, uint32_t byte_offset);
/// @brief Emit a CDNA3 flat-scratch load used to restore a borrowed VGPR temp.
void emit_cdna3_scratch_load_b32(std::vector<uint32_t> &words, uint8_t dst, uint32_t byte_offset);
/// @brief Build a CDNA3 v_mov_b32 word for VGPR/scalar copies.
[[nodiscard]] uint32_t build_cdna3_v_mov_b32(uint8_t vdst, uint16_t src0);
/// @brief Build a CDNA3 v_readfirstlane_b32 word for moving one lane into an SGPR.
[[nodiscard]] uint32_t build_cdna3_v_readfirstlane_b32(uint8_t sdst, uint8_t vsrc);
/// @brief Build CDNA3 v_add_co_u32 words and route carry-out to @p sdst.
[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_v_add_co_u32(uint8_t sdst, uint8_t vdst,
                                                                     uint16_t src0, uint16_t src1);
/// @brief Build CDNA3 v_addc_co_u32 words with carry-in from @p carry_sgpr.
[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_v_addc_co_u32(uint8_t sdst, uint8_t vdst,
                                                                      uint16_t src0, uint16_t src1,
                                                                      uint8_t carry_sgpr);
/// @brief Build a CDNA3 s_cbranch_execz guard.
[[nodiscard]] uint32_t build_cdna3_s_cbranch_execz(uint16_t simm16);
/// @brief Build a CDNA3 s_add_u32 word.
[[nodiscard]] uint32_t build_cdna3_s_add_u32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1);
/// @brief Build a CDNA3 s_addc_u32 word.
[[nodiscard]] uint32_t build_cdna3_s_addc_u32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1);
/// @brief Build a CDNA3 s_mul_i32 word.
[[nodiscard]] uint32_t build_cdna3_s_mul_i32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1);

/// @brief Choose two VGPR temps for preserving a borrowed virtual-LDS SGPR pair.
[[nodiscard]] std::optional<std::array<VirtualLdsAddressTemp, 2>>
choose_virtual_lds_base_spill_temps(TranslationContext &context,
                                    std::vector<VirtualLdsVgprRange> &forbidden) {
  std::array<VirtualLdsAddressTemp, 2> temps{};
  for (VirtualLdsAddressTemp &temp : temps) {
    auto chosen = choose_virtual_lds_address_temp(context, forbidden);
    if (!chosen)
      return std::nullopt;
    temp = *chosen;
    forbidden.push_back({.base = temp.reg, .count = 1});
  }
  return temps;
}

/// @brief Emit stores for every spill-backed temp in @p temps.
void emit_virtual_lds_temp_spill_stores(std::vector<uint32_t> &words,
                                        const std::vector<VirtualLdsAddressTemp *> &temps) {
  bool emitted = false;
  for (VirtualLdsAddressTemp *temp : temps) {
    if (temp == nullptr || !temp->spilled)
      continue;
    emit_cdna3_scratch_store_b32(words, temp->reg, temp->spill_offset);
    emitted = true;
  }
  if (emitted)
    Cdna3ScratchEmitter::append_wait(words);
}

/// @brief Emit loads for every spill-backed temp in @p temps.
void emit_virtual_lds_temp_spill_loads(std::vector<uint32_t> &words,
                                       const std::vector<VirtualLdsAddressTemp *> &temps) {
  bool emitted = false;
  for (VirtualLdsAddressTemp *temp : temps) {
    if (temp == nullptr || !temp->spilled)
      continue;
    emit_cdna3_scratch_load_b32(words, temp->reg, temp->spill_offset);
    emitted = true;
  }
  if (emitted)
    Cdna3ScratchEmitter::append_wait(words);
}

/// @brief Save a borrowed SGPR pair and load the virtual-LDS backing base.
///
/// @details Descriptor-full kernels borrow the selected SGPR pair only around
/// each lowered DS operation. This setup moves the old scalar values through
/// VGPR temps, then reloads the runtime backing pointer into the borrowed pair.
void emit_virtual_lds_base_spill_setup(std::vector<uint32_t> &words,
                                       const TranslationContext &context,
                                       const std::array<VirtualLdsAddressTemp, 2> &temps,
                                       uint32_t saved_sgpr_offset) {
  const auto base = static_cast<uint8_t>(context.virtual_lds_base_sgpr);
  words.push_back(build_cdna3_v_mov_b32(temps[0].reg, base));
  words.push_back(build_cdna3_v_mov_b32(temps[1].reg, static_cast<uint8_t>(base + 1)));
  if (context.virtual_lds_base_pointer_spilled) {
    // The descriptor-selected pointer SGPR pair is guest-owned after entry. For
    // descriptor-full kernels, the entry prologue saved the backing pointer in
    // persistent private scratch; each per-use borrow reloads it through VGPRs
    // so the original scalar pair can be restored after the virtual LDS access.
    emit_cdna3_scratch_store_b32(words, temps[0].reg, saved_sgpr_offset);
    emit_cdna3_scratch_store_b32(words, temps[1].reg, saved_sgpr_offset + sizeof(uint32_t));
    emit_cdna3_scratch_load_b32(words, temps[0].reg, context.virtual_lds_base_pointer_spill_offset);
    emit_cdna3_scratch_load_b32(words, temps[1].reg,
                                context.virtual_lds_base_pointer_spill_offset + sizeof(uint32_t));
    Cdna3ScratchEmitter::append_wait(words);
    words.push_back(build_cdna3_v_readfirstlane_b32(base, temps[0].reg));
    words.push_back(build_cdna3_v_readfirstlane_b32(static_cast<uint8_t>(base + 1), temps[1].reg));
    return;
  }
  auto [load0, load1] =
      build_cdna3_smem_load(cdna3::kSLoadDwordx2Smem, base,
                            static_cast<uint8_t>(context.virtual_lds_kernarg_segment_ptr_sgpr),
                            context.virtual_lds_kernarg_pointer_offset);
  words.push_back(load0);
  words.push_back(load1);
  words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
}

/// @brief Restore a borrowed virtual-LDS SGPR pair after a lowered DS operation.
void emit_virtual_lds_base_spill_restore(std::vector<uint32_t> &words,
                                         const TranslationContext &context,
                                         const std::array<VirtualLdsAddressTemp, 2> &temps,
                                         uint32_t saved_sgpr_offset) {
  const auto base = static_cast<uint8_t>(context.virtual_lds_base_sgpr);
  if (context.virtual_lds_base_pointer_spilled) {
    emit_cdna3_scratch_load_b32(words, temps[0].reg, saved_sgpr_offset);
    emit_cdna3_scratch_load_b32(words, temps[1].reg, saved_sgpr_offset + sizeof(uint32_t));
    Cdna3ScratchEmitter::append_wait(words);
  }
  words.push_back(build_cdna3_v_readfirstlane_b32(base, temps[0].reg));
  words.push_back(build_cdna3_v_readfirstlane_b32(static_cast<uint8_t>(base + 1), temps[1].reg));
}

/// @brief Guard a spill-per-use virtual-LDS sequence when EXEC is empty.
///
/// @details Borrowed SGPR setup uses vector instructions to move scalar state
/// through VGPR temps. If EXEC is zero, those vector moves would not update the
/// temps, so descriptor-full kernels must skip the whole replacement sequence.
[[nodiscard]] bool guard_virtual_lds_execz(std::vector<uint32_t> &words,
                                           const TranslationContext &context) {
  if (!context.virtual_lds_base_sgpr_spill_per_use)
    return true;
  if (words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max()))
    return false;
  words.insert(words.begin(), build_cdna3_s_cbranch_execz(static_cast<uint16_t>(words.size())));
  return true;
}

/// @brief Emit a CDNA3 flat-scratch b32 store.
void emit_cdna3_scratch_store_b32(std::vector<uint32_t> &words, uint8_t data,
                                  uint32_t byte_offset) {
  Cdna3ScratchEmitter::append_store_dword(words, data, byte_offset);
}

/// @brief Emit a CDNA3 flat-scratch b32 load.
void emit_cdna3_scratch_load_b32(std::vector<uint32_t> &words, uint8_t dst, uint32_t byte_offset) {
  Cdna3ScratchEmitter::append_load_dword(words, dst, byte_offset);
}

/// @brief Add a 32-bit literal to a VGPR and append the two-word CDNA3 VOP2 encoding.
void emit_cdna3_v_add_u32_literal(std::vector<uint32_t> &words, uint8_t vdst, uint8_t vsrc1,
                                  uint32_t literal) {
  auto [w0, w1] = build_cdna3_vop2_literal(cdna3::kVAddU32Vop2, vdst, vsrc1, literal);
  words.push_back(w0);
  words.push_back(w1);
}

/// @brief Encode @p vgpr in the scalar-source operand namespace used by VOP encodings.
[[nodiscard]] constexpr uint16_t vgpr_src(uint8_t vgpr) {
  return static_cast<uint16_t>(256u + vgpr);
}

/// @brief Build a two-word CDNA3 VOP3P instruction used by AccVGPR copy helpers.
[[nodiscard]] std::pair<uint32_t, uint32_t>
build_cdna3_vop3p(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1 = 0, uint16_t src2 = 0) {
  const auto words = cdna3::build_vop3p(
      op,
      {.vdst = vdst, .op_sel_hi_2 = 1, .src0 = src0, .src1 = src1, .src2 = src2, .op_sel_hi = 3});
  return {words[0], words[1]};
}

/// @brief Build a two-word CDNA3 VOP3 instruction with an SGPR carry/status destination.
[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_vop3_sdst(uint16_t op, uint8_t sdst,
                                                                  uint8_t vdst, uint16_t src0,
                                                                  uint16_t src1,
                                                                  uint16_t src2 = 0) {
  const auto words = cdna3::build_vop3_sdst_enc(
      op, {.vdst = vdst, .sdst = sdst, .src0 = src0, .src1 = src1, .src2 = src2});
  return {words[0], words[1]};
}

[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_v_add_co_u32(uint8_t sdst, uint8_t vdst,
                                                                     uint16_t src0, uint16_t src1) {
  return build_cdna3_vop3_sdst(cdna3::kVAddCoU32Vop3SdstEnc, sdst, vdst, src0, src1);
}

[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_v_addc_co_u32(uint8_t sdst, uint8_t vdst,
                                                                      uint16_t src0, uint16_t src1,
                                                                      uint8_t carry_sgpr) {
  return build_cdna3_vop3_sdst(cdna3::kVAddcCoU32Vop3SdstEnc, sdst, vdst, src0, src1, carry_sgpr);
}

/// @brief Emit a 64-bit VGPR address add for spill-per-use virtual LDS.
void emit_virtual_lds_full_vgpr_address(std::vector<uint32_t> &words,
                                        const TranslationContext &context, uint8_t addr_low,
                                        uint8_t addr_high, uint8_t base_low, uint8_t base_high) {
  // Descriptor-full kernels cannot keep a dedicated virtual-LDS base SGPR
  // live. In spill-per-use mode the borrowed SGPR pair is already saved and
  // restored around this sequence, so use it only as the temporary carry mask
  // for a correct 64-bit VGPR address calculation. The memory instruction
  // itself is encoded with SADDR=null and therefore does not depend on a
  // borrowed scalar base during VMEM issue.
  const auto carry_sgpr = static_cast<uint8_t>(context.virtual_lds_base_sgpr);
  auto [add_lo0, add_lo1] =
      build_cdna3_v_add_co_u32(carry_sgpr, addr_low, vgpr_src(addr_low), vgpr_src(base_low));
  words.push_back(add_lo0);
  words.push_back(add_lo1);
  auto [add_hi0, add_hi1] = build_cdna3_v_addc_co_u32(
      carry_sgpr, addr_high, scalar_positive_inline_u32(0), vgpr_src(base_high), carry_sgpr);
  words.push_back(add_hi0);
  words.push_back(add_hi1);
}

/// @brief Build one CDNA3 VOP1 instruction word.
[[nodiscard]] uint32_t build_cdna3_vop1(uint16_t op, uint8_t vdst, uint16_t src0) {
  return cdna3::build_vop1(op, {.src0 = src0, .vdst = vdst})[0];
}

[[nodiscard]] uint32_t build_cdna3_v_mov_b32(uint8_t vdst, uint16_t src0) {
  return build_cdna3_vop1(cdna3::kVMovB32Vop1, vdst, src0);
}

/// @brief Emit CDNA3 AccVGPR-to-VGPR copy used when staging virtual-LDS data.
void emit_cdna3_accvgpr_read_b32(std::vector<uint32_t> &words, uint8_t dst_vgpr, uint8_t src_acc) {
  auto [w0, w1] = build_cdna3_vop3p(cdna3::kVAccvgprReadVop3p, dst_vgpr, vgpr_src(src_acc));
  words.push_back(w0);
  words.push_back(w1);
}

/// @brief Emit CDNA3 VGPR-to-AccVGPR copy used when restoring staged load results.
void emit_cdna3_accvgpr_write_b32(std::vector<uint32_t> &words, uint8_t dst_acc, uint8_t src_vgpr) {
  auto [w0, w1] = build_cdna3_vop3p(cdna3::kVAccvgprWriteVop3p, dst_acc, vgpr_src(src_vgpr));
  words.push_back(w0);
  words.push_back(w1);
}

[[nodiscard]] uint32_t build_cdna3_v_readfirstlane_b32(uint8_t sdst, uint8_t vsrc) {
  // VOP1 SRC0 uses the scalar-source operand namespace. Plain values 0..127
  // select SGPR/special scalar operands such as EXEC_LO/EXEC_HI; VGPR operands
  // are encoded as 256 + vN. `v_readfirstlane_b32` must read the per-lane VGPR
  // temp, not one of those scalar operands, because virtual-LDS spill restore
  // uses it to rebuild the borrowed backing-pointer SGPR pair.
  return build_cdna3_vop1(cdna3::kVReadfirstlaneB32Vop1, sdst, static_cast<uint16_t>(256u + vsrc));
}

/// @brief Copy a contiguous VGPR run into a temporary run.
void emit_virtual_lds_copy_to_temp_range(std::vector<uint32_t> &words,
                                         const VirtualLdsTempRange &range, uint8_t src_base) {
  for (uint8_t i = 0; i < range.count; ++i) {
    words.push_back(
        build_cdna3_v_mov_b32(range.temps[i].reg, vgpr_src(static_cast<uint8_t>(src_base + i))));
  }
}

/// @brief Copy a contiguous AccVGPR run into a VGPR temporary run.
void emit_virtual_lds_copy_acc_to_temp_range(std::vector<uint32_t> &words,
                                             const VirtualLdsTempRange &range, uint8_t src_base) {
  for (uint8_t i = 0; i < range.count; ++i)
    emit_cdna3_accvgpr_read_b32(words, range.temps[i].reg, static_cast<uint8_t>(src_base + i));
}

/// @brief Copy a temporary run back into ordinary VGPR destinations.
void emit_virtual_lds_copy_from_temp_range(std::vector<uint32_t> &words,
                                           const VirtualLdsTempRange &range, uint8_t dst_base) {
  for (uint8_t i = 0; i < range.count; ++i) {
    words.push_back(
        build_cdna3_v_mov_b32(static_cast<uint8_t>(dst_base + i), vgpr_src(range.temps[i].reg)));
  }
}

/// @brief Copy a temporary run back into AccVGPR destinations.
void emit_virtual_lds_copy_temp_to_acc_range(std::vector<uint32_t> &words,
                                             const VirtualLdsTempRange &range, uint8_t dst_base) {
  for (uint8_t i = 0; i < range.count; ++i)
    emit_cdna3_accvgpr_write_b32(words, static_cast<uint8_t>(dst_base + i), range.temps[i].reg);
}

[[nodiscard]] uint32_t build_cdna3_s_cbranch_execz(uint16_t simm16) {
  return cdna3::build_sopp(cdna3::kSCbranchExeczSopp, {.simm16 = simm16})[0];
}

[[nodiscard]] uint32_t build_cdna3_s_add_u32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return cdna3::build_sop2(cdna3::kSAddU32Sop2, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                                 .ssrc1 = static_cast<uint8_t>(ssrc1),
                                                 .sdst = sdst})[0];
}

[[nodiscard]] uint32_t build_cdna3_s_addc_u32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return cdna3::build_sop2(cdna3::kSAddcU32Sop2, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                                  .sdst = sdst})[0];
}

[[nodiscard]] uint32_t build_cdna3_s_mul_i32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return cdna3::build_sop2(cdna3::kSMulI32Sop2, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                                 .ssrc1 = static_cast<uint8_t>(ssrc1),
                                                 .sdst = sdst})[0];
}

/// @brief Reserve the SGPR state needed by virtual-LDS address lowering.
///
/// @details The reservation is based on decoded ordinary SGPR operands plus ABI
/// SGPRs, not just the descriptor allocation count. That avoids placing DBT's
/// backing-pointer pair in CDNA special SGPR territory while still allowing a
/// spill-per-use fallback for descriptor-full kernels.
[[nodiscard]] std::optional<VirtualLdsBaseSgprReservation>
reserve_cdna3_virtual_lds_base_sgpr_pair(TranslationContext &context, KernelBlockScope blocks,
                                         const KdTranslation &translation, rj_code_arch_t arch) {
  if (!arch_is_cdna(arch))
    return std::nullopt;

  auto note_sgpr_ref = [](uint32_t &count, RegisterRef ref) {
    if (ref.cls != RegClass::SGPR)
      return;
    // Implicit operands include architectural special registers such as EXEC
    // and VCC. Those live in the descriptor tail and must not make virtual-LDS
    // scratch selection think every ordinary SGPR is already occupied.
    if (ref.index >= kCdnaOrdinarySgprLimit)
      return;
    count = std::max<uint32_t>(count, static_cast<uint32_t>(ref.index) + ref.width);
  };

  uint32_t ordinary_floor = 0;
  for (BasicBlock *block : blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      for (int i = 0; i < inst.num_src_operands(); ++i) {
        if (const Operand *operand = inst.src_operand(i)) {
          if (auto ref = operand->to_register_ref())
            note_sgpr_ref(ordinary_floor, *ref);
        }
      }
      for (int i = 0; i < inst.num_dst_operands(); ++i) {
        if (const Operand *operand = inst.dst_operand(i)) {
          if (auto ref = operand->to_register_ref())
            note_sgpr_ref(ordinary_floor, *ref);
        }
      }

      // Do not fold implicit uses/defs into the ordinary SGPR floor. They can
      // describe architectural state such as EXEC/VCC/SCC rather than guest
      // ordinary scalar registers, and counting them here forces small kernels
      // into descriptor-full virtual-LDS spill mode. Explicit operands plus the
      // descriptor ABI SGPR fields below are the values that matter for choosing
      // a non-conflicting backing-pointer pair.
    }
  }

  ordinary_floor = std::max<uint32_t>(ordinary_floor, translation.target_user_sgpr_count);
  auto include_sgpr = [&](int16_t sgpr, uint32_t width) {
    if (sgpr >= 0)
      ordinary_floor = std::max<uint32_t>(ordinary_floor, static_cast<uint32_t>(sgpr) + width);
  };
  include_sgpr(translation.has_kernarg_segment_ptr ? translation.kernarg_segment_ptr_sgpr : -1, 2);
  include_sgpr(translation.has_dispatch_ptr ? translation.dispatch_ptr_sgpr : -1, 2);
  include_sgpr(translation.workgroup_id_sgpr_x, 1);
  include_sgpr(translation.workgroup_id_sgpr_y, 1);
  include_sgpr(translation.workgroup_id_sgpr_z, 1);

  // Virtual LDS flat/global operations need an ordinary 64-bit SGPR base, and
  // the entry prologue needs one scalar scratch register to compute the
  // per-workgroup byte offset. COMPUTE_PGM_RSRC1's SGPR count is an allocation
  // total, not the highest ordinary `sN` named by the kernel: VCC, flat scratch,
  // XNACK, and granularity padding can live in the tail. Starting new scratch
  // at the descriptor total can therefore put DBT temporaries in special SGPR
  // territory. Derive the ordinary floor from decoded operands and reserve a
  // conservative CDNA special-SGPR tail when asking the descriptor to grow.
  const uint32_t current = std::max(ordinary_floor, context.required_sgpr_count);
  const uint32_t base = (current + 1u) & ~1u;
  if (base + 4 <= kCdnaOrdinarySgprLimit) {
    context.require_sgprs(base + 4 + kCdnaSpecialSgprTailReserve);
    return VirtualLdsBaseSgprReservation{.base = static_cast<uint16_t>(base),
                                         .prologue_temp = static_cast<uint16_t>(base + 2)};
  }

  const uint32_t allocated_ordinary = std::min<uint32_t>(ordinary_floor, kCdnaOrdinarySgprLimit);
  if (allocated_ordinary < 2)
    return std::nullopt;
  (void)blocks;

  const uint32_t borrowed_temp = (allocated_ordinary - 2u) & ~1u;
  if (base + 2 <= kCdnaOrdinarySgprLimit) {
    context.require_sgprs(base + 2 + kCdnaSpecialSgprTailReserve);
    return VirtualLdsBaseSgprReservation{.base = static_cast<uint16_t>(base),
                                         .prologue_temp = static_cast<uint16_t>(borrowed_temp)};
  }

  if (allocated_ordinary < 4)
    return std::nullopt;

  // The descriptor is already full, so do not permanently reuse an existing
  // pair based on static analysis. Borrow the last descriptor-backed ordinary
  // pair only inside each lowered LDS memory sequence, saving and restoring the
  // original scalar value through VGPR temps. A neighboring high SGPR handles
  // entry-only offset math before guest scalar values become meaningful.
  const uint32_t spill_base = (allocated_ordinary - 2u) & ~1u;
  const uint32_t temp_base = (spill_base >= 2) ? ((spill_base - 2u) & ~1u) : 0;
  if (temp_base == spill_base)
    return std::nullopt;
  // `ordinary_floor` comes from decoded source operands, not necessarily from
  // the descriptor's encoded SGPR allocation. Some code objects name high
  // scalar registers while advertising a small granulated SGPR count. Borrowed
  // virtual-LDS SADDR pairs must still be inside the target wave allocation or
  // vector memory observes an out-of-range scalar source.
  context.require_sgprs(allocated_ordinary);
  return VirtualLdsBaseSgprReservation{.base = static_cast<uint16_t>(spill_base),
                                       .prologue_temp = static_cast<uint16_t>(temp_base),
                                       .spill_per_use = true};
}

/// @brief Append the target entry prologue that initializes a virtual-LDS backing pointer.
///
/// @details The prologue consumes runtime state written by the HSA hook, folds
/// the current workgroup's backing-buffer stride into the base pointer, and
/// optionally spills that pointer for descriptor-full kernels that borrow the
/// base SGPR pair around each lowered DS instruction.
[[nodiscard]] bool append_cdna3_virtual_lds_entry_prologue(KdTranslation &translation) {
  if (!translation.needs_lds_overflow_buf)
    return true;
  const uint16_t pointer_base_sgpr = translation.kernarg_segment_ptr_sgpr;
  if (!translation.has_kernarg_segment_ptr)
    return false;
  if ((translation.virtual_lds_lowering.base_sgpr % 2) != 0 || (pointer_base_sgpr % 2) != 0)
    return false;
  if ((translation.virtual_lds_lowering.prologue_temp_sgpr % 2) != 0)
    return false;
  if (static_cast<uint32_t>(translation.virtual_lds_lowering.base_sgpr) + 1 >=
          kCdnaOrdinarySgprLimit ||
      static_cast<uint32_t>(translation.virtual_lds_lowering.prologue_temp_sgpr) + 1 >=
          kCdnaOrdinarySgprLimit ||
      pointer_base_sgpr > 126)
    return false;
  if (translation.lds_overflow_kernarg_pointer_offset > kCdnaSmemImmediateByteOffsetMax)
    return false;
  if (translation.lds_overflow_kernarg_pointer_offset >
      kCdnaSmemImmediateByteOffsetMax - kVirtualLdsRuntimeStateBytes)
    return false;
  if (translation.kernarg_wrapper_original_pointer_offset > kCdnaSmemImmediateByteOffsetMax)
    return false;
  if (translation.user_sgpr_repair_count != 0 &&
      static_cast<uint32_t>(translation.user_sgpr_repair_start) +
              translation.user_sgpr_repair_count + 1u >
          126u) {
    return false;
  }

  auto valid_workgroup_sgpr = [](int16_t sgpr) { return sgpr < 0 || sgpr <= 126; };
  if (!valid_workgroup_sgpr(translation.lds_overflow_workgroup_id_sgpr_x) ||
      !valid_workgroup_sgpr(translation.lds_overflow_workgroup_id_sgpr_y) ||
      !valid_workgroup_sgpr(translation.lds_overflow_workgroup_id_sgpr_z)) {
    return false;
  }

  const auto base = static_cast<uint8_t>(translation.virtual_lds_lowering.base_sgpr);
  const auto temp = static_cast<uint8_t>(translation.virtual_lds_lowering.prologue_temp_sgpr);
  const auto product = static_cast<uint8_t>(temp + 1);
  uint8_t state_sbase = static_cast<uint8_t>(pointer_base_sgpr);
  uint32_t state_offset = translation.lds_overflow_kernarg_pointer_offset;

  auto append_smem_load_dword = [&](uint8_t dst, uint8_t sbase, uint32_t offset) {
    auto [load0, load1] = build_cdna3_smem_load(cdna3::kSLoadDwordSmem, dst, sbase, offset);
    translation.prologue_words.push_back(load0);
    translation.prologue_words.push_back(load1);
    Cdna3ScratchEmitter::append_wait(translation.prologue_words);
  };
  auto append_smem_load_dwordx2 = [&](uint8_t dst, uint8_t sbase, uint32_t offset) {
    auto [load0, load1] = build_cdna3_smem_load(cdna3::kSLoadDwordx2Smem, dst, sbase, offset);
    translation.prologue_words.push_back(load0);
    translation.prologue_words.push_back(load1);
    translation.prologue_words.push_back(
        cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
  };
  auto append_stride_term = [&](int16_t workgroup_id_sgpr, uint32_t stride_offset) {
    if (workgroup_id_sgpr < 0)
      return;
    append_smem_load_dword(product, state_sbase, state_offset + stride_offset);
    translation.prologue_words.push_back(
        build_cdna3_s_mul_i32(product, static_cast<uint16_t>(workgroup_id_sgpr), product));
    translation.prologue_words.push_back(build_cdna3_s_add_u32(temp, temp, product));
  };

  auto append_restore_guest_user_sgprs = [&]() {
    if (translation.source_has_kernarg_segment_ptr) {
      // The wrapper pointer is DBT-owned. Restore the source kernarg pointer
      // after all DBT state loads so LLVM-emitted guest code observes the same
      // SGPR value it would have received without translation.
      append_smem_load_dwordx2(temp, state_sbase,
                               translation.kernarg_wrapper_original_pointer_offset);
      translation.prologue_words.push_back(
          build_s_mov_b32(pointer_base_sgpr, temp, ROCJITSU_CODE_ARCH_CDNA3));
      translation.prologue_words.push_back(
          build_s_mov_b32(static_cast<uint16_t>(pointer_base_sgpr + 1),
                          static_cast<uint16_t>(temp + 1), ROCJITSU_CODE_ARCH_CDNA3));
    }

    // If DBT inserted a target-only kernarg segment pointer, CP shifted every
    // later initialized SGPR up by two. Move source-visible values back down
    // before entering the translated guest body.
    for (uint16_t i = 0; i < translation.user_sgpr_repair_count; ++i) {
      const uint16_t dst = static_cast<uint16_t>(translation.user_sgpr_repair_start + i);
      const uint16_t src = static_cast<uint16_t>(dst + 2);
      translation.prologue_words.push_back(build_s_mov_b32(dst, src, ROCJITSU_CODE_ARCH_CDNA3));
    }
  };

  // The hook allocates one backing slice per workgroup. Convert the
  // descriptor-selected workgroup ids into a byte offset using dispatch-time
  // stride fields:
  //   offset = wg_x * stride_x + wg_y * stride_y + wg_z * stride_z.
  translation.prologue_words.push_back(
      build_s_mov_b32(temp, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA3));
  append_stride_term(translation.lds_overflow_workgroup_id_sgpr_x, kVirtualLdsStateStrideXOffset);
  append_stride_term(translation.lds_overflow_workgroup_id_sgpr_y, kVirtualLdsStateStrideYOffset);
  append_stride_term(translation.lds_overflow_workgroup_id_sgpr_z, kVirtualLdsStateStrideZOffset);

  append_smem_load_dwordx2(base, state_sbase, state_offset + kVirtualLdsStateBackingBaseOffset);
  translation.prologue_words.push_back(build_cdna3_s_add_u32(base, base, temp));
  translation.prologue_words.push_back(build_cdna3_s_addc_u32(static_cast<uint8_t>(base + 1),
                                                              static_cast<uint8_t>(base + 1),
                                                              scalar_positive_inline_u32(0)));
  if (translation.virtual_lds_lowering.base_sgpr_spill_per_use) {
    if (!translation.virtual_lds_lowering.base_pointer_spilled)
      return false;
    if (translation.target_vgpr_count < 2 || translation.target_vgpr_count > 256)
      return false;

    // Spill-per-use mode borrows a guest SGPR pair for each lowered LDS
    // operation. The kernarg/dispatch pointer SGPR pair is not preserved by the
    // guest body, so consume it at the descriptor entry and save the backing
    // pointer in private scratch before any original instruction can clobber
    // the pointer base. Descriptor recomputation after body lowering may raise
    // target_vgpr_count; keep the originally chosen temps stable so the
    // recomputed descriptor reproduces the already-emitted prologue exactly.
    if (!translation.virtual_lds_lowering.entry_temp_vgprs_valid) {
      translation.virtual_lds_lowering.entry_temp_vgpr_lo =
          static_cast<uint8_t>(translation.target_vgpr_count - 2);
      translation.virtual_lds_lowering.entry_temp_vgpr_hi =
          static_cast<uint8_t>(translation.target_vgpr_count - 1);
      translation.virtual_lds_lowering.entry_temp_vgprs_valid = true;
    }
    if (translation.virtual_lds_lowering.entry_temp_vgpr_hi >= translation.target_vgpr_count)
      return false;
    const uint8_t temp_lo = translation.virtual_lds_lowering.entry_temp_vgpr_lo;
    const uint8_t temp_hi = translation.virtual_lds_lowering.entry_temp_vgpr_hi;
    translation.prologue_words.push_back(build_cdna3_v_mov_b32(
        temp_lo, static_cast<uint8_t>(translation.virtual_lds_lowering.base_sgpr)));
    translation.prologue_words.push_back(build_cdna3_v_mov_b32(
        temp_hi, static_cast<uint8_t>(translation.virtual_lds_lowering.base_sgpr + 1)));
    emit_cdna3_scratch_store_b32(translation.prologue_words, temp_lo,
                                 translation.virtual_lds_lowering.base_pointer_spill_offset);
    emit_cdna3_scratch_store_b32(translation.prologue_words, temp_hi,
                                 translation.virtual_lds_lowering.base_pointer_spill_offset +
                                     sizeof(uint32_t));
    translation.prologue_words.push_back(
        cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
    append_restore_guest_user_sgprs();
    return true;
  }

  append_restore_guest_user_sgprs();
  return true;
}

/// @brief CDNA4 DS opcode metadata needed to emit an equivalent CDNA3 GLOBAL op.
struct VirtualLdsDsOp {
  bool is_load = false;
  uint8_t flat_op = 0;
  uint8_t vgpr_count = 1;
  uint32_t two_addr_stride_bytes = 0;
  uint8_t read2_dst_delta = 0;
};

/// @brief Build common CDNA3 FLAT/GLOBAL operands for virtual-LDS memory traffic.
[[nodiscard]] FlatGlobalOperands
make_virtual_lds_flat_global_operands(uint16_t signed_offset13, uint8_t addr, uint8_t saddr) {
  FlatGlobalOperands operands{};
  operands.signed_offset13 = signed_offset13;
  operands.addr = addr;
  operands.saddr = saddr;
  // Virtual LDS uses global memory as a workgroup-local backing store. On
  // GFX940-class FLAT/GLOBAL instructions, SC[1:0] = 1 encodes group scope,
  // matching the native LDS producer/consumer sharing domain for ordinary
  // same-workgroup dispatches.
  operands.sc0 = true;
  operands.sc1 = false;
  return operands;
}

/// @brief Map a CDNA4 DS memory opcode to its virtual-LDS lowering metadata.
///
/// @details Only LDS storage operations are listed here. Cross-lane DS
/// operations such as bpermute are intentionally omitted so they continue to
/// use the DS unit directly.
[[nodiscard]] std::optional<VirtualLdsDsOp> virtual_lds_ds_op(uint16_t opcode) {
  switch (opcode) {
  case cdna4::kDsWriteB32Ds:
    return VirtualLdsDsOp{.is_load = false, .flat_op = cdna3::kFlatStoreDwordFlat};
  case cdna4::kDsWrite2B32Ds:
    return VirtualLdsDsOp{.is_load = false,
                          .flat_op = cdna3::kFlatStoreDwordFlat,
                          .vgpr_count = 1,
                          .two_addr_stride_bytes = 4};
  case cdna4::kDsWrite2st64B32Ds:
    return VirtualLdsDsOp{.is_load = false,
                          .flat_op = cdna3::kFlatStoreDwordFlat,
                          .vgpr_count = 1,
                          .two_addr_stride_bytes = 256};
  case cdna4::kDsWriteB8Ds:
    return VirtualLdsDsOp{.is_load = false, .flat_op = cdna3::kFlatStoreByteFlat};
  case cdna4::kDsWriteB16Ds:
    return VirtualLdsDsOp{.is_load = false, .flat_op = cdna3::kFlatStoreShortFlat};
  case cdna4::kDsReadB32Ds:
    return VirtualLdsDsOp{.is_load = true, .flat_op = cdna3::kFlatLoadDwordFlat};
  case cdna4::kDsRead2B32Ds:
    return VirtualLdsDsOp{.is_load = true,
                          .flat_op = cdna3::kFlatLoadDwordFlat,
                          .vgpr_count = 1,
                          .two_addr_stride_bytes = 4,
                          .read2_dst_delta = 1};
  case cdna4::kDsRead2st64B32Ds:
    return VirtualLdsDsOp{.is_load = true,
                          .flat_op = cdna3::kFlatLoadDwordFlat,
                          .vgpr_count = 1,
                          .two_addr_stride_bytes = 256,
                          .read2_dst_delta = 1};
  case cdna4::kDsReadI8Ds:
    return VirtualLdsDsOp{.is_load = true, .flat_op = cdna3::kFlatLoadSbyteFlat};
  case cdna4::kDsReadU8Ds:
    return VirtualLdsDsOp{.is_load = true, .flat_op = cdna3::kFlatLoadUbyteFlat};
  case cdna4::kDsReadU16Ds:
    return VirtualLdsDsOp{.is_load = true, .flat_op = cdna3::kFlatLoadUshortFlat};
  case cdna4::kDsWriteB64Ds:
    return VirtualLdsDsOp{
        .is_load = false, .flat_op = cdna3::kFlatStoreDwordx2Flat, .vgpr_count = 2};
  case cdna4::kDsWrite2B64Ds:
    return VirtualLdsDsOp{.is_load = false,
                          .flat_op = cdna3::kFlatStoreDwordx2Flat,
                          .vgpr_count = 2,
                          .two_addr_stride_bytes = 8};
  case cdna4::kDsWrite2st64B64Ds:
    return VirtualLdsDsOp{.is_load = false,
                          .flat_op = cdna3::kFlatStoreDwordx2Flat,
                          .vgpr_count = 2,
                          .two_addr_stride_bytes = 512};
  case cdna4::kDsWriteB8D16HiDs:
    return VirtualLdsDsOp{.is_load = false, .flat_op = cdna3::kFlatStoreByteD16HiFlat};
  case cdna4::kDsWriteB16D16HiDs:
    return VirtualLdsDsOp{.is_load = false, .flat_op = cdna3::kFlatStoreShortD16HiFlat};
  case cdna4::kDsReadU16D16Ds:
    return VirtualLdsDsOp{.is_load = true, .flat_op = cdna3::kFlatLoadShortD16Flat};
  case cdna4::kDsReadU16D16HiDs:
    return VirtualLdsDsOp{.is_load = true, .flat_op = cdna3::kFlatLoadShortD16HiFlat};
  case cdna4::kDsReadB64Ds:
    return VirtualLdsDsOp{.is_load = true, .flat_op = cdna3::kFlatLoadDwordx2Flat, .vgpr_count = 2};
  case cdna4::kDsRead2B64Ds:
    return VirtualLdsDsOp{.is_load = true,
                          .flat_op = cdna3::kFlatLoadDwordx2Flat,
                          .vgpr_count = 2,
                          .two_addr_stride_bytes = 8,
                          .read2_dst_delta = 2};
  case cdna4::kDsRead2st64B64Ds:
    return VirtualLdsDsOp{.is_load = true,
                          .flat_op = cdna3::kFlatLoadDwordx2Flat,
                          .vgpr_count = 2,
                          // ST64 offsets are scaled by 64 elements, not by a
                          // fixed byte count. B64 therefore uses 8 * 64 bytes,
                          // matching the write form and AMD's DS pseudocode.
                          .two_addr_stride_bytes = 512,
                          .read2_dst_delta = 2};
  case cdna4::kDsWriteB96Ds:
    return VirtualLdsDsOp{
        .is_load = false, .flat_op = cdna3::kFlatStoreDwordx3Flat, .vgpr_count = 3};
  case cdna4::kDsWriteB128Ds:
    return VirtualLdsDsOp{
        .is_load = false, .flat_op = cdna3::kFlatStoreDwordx4Flat, .vgpr_count = 4};
  case cdna4::kDsReadB96Ds:
    return VirtualLdsDsOp{.is_load = true, .flat_op = cdna3::kFlatLoadDwordx3Flat, .vgpr_count = 3};
  case cdna4::kDsReadB128Ds:
    return VirtualLdsDsOp{.is_load = true, .flat_op = cdna3::kFlatLoadDwordx4Flat, .vgpr_count = 4};
  default:
    return std::nullopt;
  }
}

/// @brief Return true when @p inst uses the CDNA4 DS encoding family.
[[nodiscard]] bool is_cdna4_ds_encoding(const Instruction &inst) {
  return inst.encoding_id() >= cdna4::encoding::kDs &&
         inst.encoding_id() <= cdna4::encoding::kDsHi7;
}

/// @brief Return true when a CDNA4 MUBUF instruction targets LDS.
[[nodiscard]] bool is_cdna4_mubuf_lds_instruction(const Instruction &inst) {
  if (inst.encoding_id() != cdna4::encoding::kMubuf)
    return false;

  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || static_cast<size_t>(inst.size()) < sizeof(cdna4::MubufMachineInst))
    return false;

  cdna4::MubufMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  return src.lds != 0;
}

/// @brief Return true when @p inst needs a virtual-LDS body variant.
///
/// @details The scan is intentionally broader than the lowering table: any
/// mnemonic with an LDS suffix or MUBUF LDS bit means the kernel cannot be made
/// a zero-hardware-LDS sidecar until the access is either lowered or rejected
/// with a clear diagnostic.
[[nodiscard]] bool source_instruction_uses_virtualizable_lds_impl(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic.find("_lds") != std::string_view::npos)
    return true;
  if (is_cdna4_mubuf_lds_instruction(inst))
    return true;
  if (mnemonic == "ds_read_b64_tr_b16")
    return true;
  return is_cdna4_ds_encoding(inst) && virtual_lds_ds_op(inst.opcode()).has_value();
}

/// @brief Lower one CDNA4 DS memory instruction to CDNA3 GLOBAL memory traffic.
///
/// @details The HSA hook provides a per-workgroup backing buffer and the entry
/// prologue initializes its base pointer. This instruction-level lowering
/// preserves native DS address/data aliasing behavior by staging overlapping
/// operands through VGPR temps and private scratch when required.
[[nodiscard]] ExpandResult
lower_cdna4_to_cdna3_virtual_lds_ds_instruction(const Instruction &inst,
                                                TranslationContext &context) {
  if (!context.virtualize_lds || !is_cdna4_ds_encoding(inst))
    return ExpandResult::not_handled();

  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || static_cast<size_t>(inst.size()) < sizeof(cdna4::DsMachineInst)) {
    return ExpandResult::failed(std::string(inst.mnemonic()) + ": missing DS source encoding",
                                {"Decode the source DS instruction before virtual LDS lowering."});
  }

  cdna4::DsMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto op = virtual_lds_ds_op(static_cast<uint16_t>(src.op));
  if (!op) {
    if ((inst.flags() & MEMORY_OP) == 0)
      return ExpandResult::not_handled();
    return ExpandResult::failed(
        std::string(inst.mnemonic()) + ": virtual LDS lowering does not support this DS opcode",
        {"Add a virtual-LDS lowering for this DS memory operation before translating this "
         "kernel with hardware LDS set to zero."});
  }
  if (src.gds != 0) {
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                    ": virtual LDS lowering does not support GDS",
                                {"Keep virtual LDS lowering limited to ordinary LDS accesses."});
  }
  const bool uses_acc = src.acc != 0;
  if (context.virtual_lds_base_sgpr > 126 || (context.virtual_lds_base_sgpr % 2) != 0) {
    return ExpandResult::failed(
        std::string(inst.mnemonic()) + ": virtual LDS backing-buffer SGPR pair is not encodable",
        {"Reserve an even SGPR pair that CDNA3 flat/global instructions can encode as saddr."});
  }
  if (src.addr == std::numeric_limits<uint8_t>::max()) {
    return ExpandResult::failed(
        std::string(inst.mnemonic()) + ": virtual LDS source address has no high VGPR",
        {"Add a spill-based lowering for DS address v255 before using it as a CDNA3 GLOBAL "
         "64-bit VADDR pair."});
  }

  std::vector<uint32_t> words;
  const uint8_t addr = static_cast<uint8_t>(src.addr);
  const uint8_t addr_high = static_cast<uint8_t>(addr + 1);
  const bool use_full_vgpr_address = context.virtual_lds_base_sgpr_spill_per_use;
  auto require_source_vgprs = [&](uint32_t base, uint32_t count) {
    if (count != 0)
      context.require_vgprs(base + count);
  };
  // Native DS uses a 32-bit LDS byte offset in `addr`. Virtual-LDS lowering
  // materializes that offset as the low half of a 64-bit GLOBAL address tuple.
  // Even source address registers can use `addr:addr+1` directly; odd address
  // registers are copied into a private even tuple below.  Hand-written
  // assembly can also name high DS data/destination registers without the
  // descriptor reflecting them. Record those requirements before descriptor
  // recomputation so the host launch allocates every VGPR the translated body
  // actually reads or writes.
  require_source_vgprs(addr, 2);

  if (op->two_addr_stride_bytes != 0 && op->is_load) {
    const uint32_t byte_offset0 = static_cast<uint32_t>(src.offset0) * op->two_addr_stride_bytes;
    const uint32_t byte_offset1 = static_cast<uint32_t>(src.offset1) * op->two_addr_stride_bytes;
    const uint16_t total_dst_vgprs = static_cast<uint16_t>(op->read2_dst_delta + op->vgpr_count);
    if (!uses_acc)
      require_source_vgprs(src.vdst, total_dst_vgprs);
    struct Read2Access {
      uint8_t vdst = 0;
      uint32_t byte_offset = 0;
      bool clobbers_addr_low = false;
      bool clobbers_addr_high = false;
    };
    std::array<Read2Access, 2> accesses = {
        Read2Access{
            .vdst = static_cast<uint8_t>(src.vdst),
            .byte_offset = byte_offset0,
            .clobbers_addr_low = !uses_acc && virtual_lds_vgpr_ranges_overlap(
                                                  static_cast<uint16_t>(src.vdst), op->vgpr_count,
                                                  static_cast<uint16_t>(addr), 1),
            .clobbers_addr_high = !uses_acc && virtual_lds_vgpr_ranges_overlap(
                                                   static_cast<uint16_t>(src.vdst), op->vgpr_count,
                                                   static_cast<uint16_t>(addr_high), 1)},
        Read2Access{.vdst = static_cast<uint8_t>(src.vdst + op->read2_dst_delta),
                    .byte_offset = byte_offset1,
                    .clobbers_addr_low =
                        !uses_acc && virtual_lds_vgpr_ranges_overlap(
                                         static_cast<uint16_t>(src.vdst + op->read2_dst_delta),
                                         op->vgpr_count, static_cast<uint16_t>(addr), 1),
                    .clobbers_addr_high =
                        !uses_acc && virtual_lds_vgpr_ranges_overlap(
                                         static_cast<uint16_t>(src.vdst + op->read2_dst_delta),
                                         op->vgpr_count, static_cast<uint16_t>(addr_high), 1)}};
    auto clobbers_addr_pair = [](const Read2Access &access) {
      return access.clobbers_addr_low || access.clobbers_addr_high;
    };
    const uint32_t address_clobber_count =
        (clobbers_addr_pair(accesses[0]) ? 1u : 0u) + (clobbers_addr_pair(accesses[1]) ? 1u : 0u);
    const bool needs_aligned_read2_results =
        op->vgpr_count >= 2 && ((accesses[0].vdst % 2) != 0 || (accesses[1].vdst % 2) != 0);
    const bool stage_read2_results = address_clobber_count > 1 || needs_aligned_read2_results;
    if (!stage_read2_results && clobbers_addr_pair(accesses[0]))
      std::swap(accesses[0], accesses[1]);

    const bool needs_materialized_offset =
        byte_offset0 > kFlatGlobalPositiveImm13Max || byte_offset1 > kFlatGlobalPositiveImm13Max;
    const bool needs_preserved_addr = needs_materialized_offset ||
                                      (!stage_read2_results && address_clobber_count != 0) ||
                                      use_full_vgpr_address;
    const bool result_clobbers_addr_low =
        stage_read2_results && !uses_acc
            ? virtual_lds_vgpr_ranges_overlap(static_cast<uint16_t>(src.vdst), total_dst_vgprs,
                                              static_cast<uint16_t>(addr), 1)
            : accesses[1].clobbers_addr_low;
    const bool result_clobbers_addr_high =
        stage_read2_results && !uses_acc
            ? virtual_lds_vgpr_ranges_overlap(static_cast<uint16_t>(src.vdst), total_dst_vgprs,
                                              static_cast<uint16_t>(addr_high), 1)
            : accesses[1].clobbers_addr_high;
    const bool restore_addr_low =
        (needs_materialized_offset || use_full_vgpr_address) && !result_clobbers_addr_low;
    const bool restore_addr_high = !result_clobbers_addr_high;

    std::vector<VirtualLdsVgprRange> forbidden = {
        {.base = static_cast<uint16_t>(addr), .count = 2},
    };
    if (!uses_acc)
      forbidden.push_back({.base = static_cast<uint16_t>(src.vdst), .count = total_dst_vgprs});
    std::optional<VirtualLdsTempRange> address_pair_temp;
    if ((addr % 2) != 0) {
      // CDNA3 GLOBAL/FLAT instructions encode the 64-bit VGPR address operand
      // with an even-base register class. CDNA4 DS only names a 32-bit LDS
      // offset VGPR, so odd DS address registers must be copied into a private
      // even pair before using them as GLOBAL VADDR.
      address_pair_temp = choose_virtual_lds_temp_range(context, forbidden, 2, 2);
      if (!address_pair_temp) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS read2 lowering cannot allocate an even GLOBAL address pair",
            {"Add a more general spill path for kernels whose DS operands cover every ordinary "
             "VGPR."});
      }
      forbidden.push_back({.base = address_pair_temp->base(), .count = address_pair_temp->count});
    }
    const uint8_t global_addr = address_pair_temp ? address_pair_temp->base() : addr;
    const uint8_t global_addr_high =
        address_pair_temp ? static_cast<uint8_t>(address_pair_temp->base() + 1) : addr_high;
    std::optional<VirtualLdsAddressTemp> base_temp;
    if (needs_preserved_addr) {
      base_temp = choose_virtual_lds_address_temp(context, forbidden);
      if (!base_temp) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS read2 lowering cannot preserve the source address VGPR",
            {"Add a more general spill path for kernels whose DS operands cover every ordinary "
             "VGPR."});
      }
      forbidden.push_back({.base = base_temp->reg, .count = 1});
    }

    std::optional<VirtualLdsTempRange> staged_read2_temp;
    if (stage_read2_results) {
      // Either both read2 destinations overlap the GLOBAL VADDR pair, or at
      // least one wide destination tuple has an odd base that CDNA3 GLOBAL
      // cannot encode. Load into a private contiguous run first, then copy the
      // completed results into the architectural destination range after the
      // address pair is no longer needed.
      staged_read2_temp = choose_virtual_lds_temp_range(context, forbidden, total_dst_vgprs,
                                                        op->vgpr_count >= 2 ? 2 : 1);
      if (!staged_read2_temp) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS read2 lowering cannot stage overlapping results",
            {"Add a more general spill path for kernels whose DS operands cover every ordinary "
             "VGPR."});
      }
      forbidden.push_back({.base = staged_read2_temp->base(), .count = staged_read2_temp->count});
    }

    std::optional<std::array<VirtualLdsAddressTemp, 2>> base_spill_temps;
    if (context.virtual_lds_base_sgpr_spill_per_use) {
      std::vector<VirtualLdsVgprRange> base_spill_forbidden = {
          {.base = static_cast<uint16_t>(addr), .count = 1},
      };
      if (!uses_acc)
        base_spill_forbidden.push_back(
            {.base = static_cast<uint16_t>(src.vdst), .count = total_dst_vgprs});
      if (use_full_vgpr_address || !restore_addr_high)
        base_spill_forbidden.push_back({.base = static_cast<uint16_t>(addr_high), .count = 1});
      if (base_temp)
        base_spill_forbidden.push_back({.base = base_temp->reg, .count = 1});
      if (address_pair_temp)
        base_spill_forbidden.push_back(
            {.base = address_pair_temp->base(), .count = address_pair_temp->count});
      if (staged_read2_temp)
        base_spill_forbidden.push_back(
            {.base = staged_read2_temp->base(), .count = staged_read2_temp->count});
      base_spill_temps = choose_virtual_lds_base_spill_temps(context, base_spill_forbidden);
      if (!base_spill_temps) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS SGPR spill lowering cannot find VGPR save temps",
            {"Add a more general spill path for kernels whose DS operands cover every ordinary "
             "VGPR."});
      }
    }

    const uint32_t base_sgpr_spill_dwords =
        base_spill_temps && context.virtual_lds_base_pointer_spilled ? 2u : 0u;
    const uint32_t extra_spill_dwords = base_sgpr_spill_dwords + (restore_addr_high ? 1u : 0u);
    std::vector<VirtualLdsAddressTemp *> spill_temps;
    if (address_pair_temp) {
      for (uint8_t i = 0; i < address_pair_temp->count; ++i)
        spill_temps.push_back(&address_pair_temp->temps[i]);
    }
    if (base_temp)
      spill_temps.push_back(&*base_temp);
    if (staged_read2_temp) {
      for (uint8_t i = 0; i < staged_read2_temp->count; ++i)
        spill_temps.push_back(&staged_read2_temp->temps[i]);
    }
    if (base_spill_temps) {
      spill_temps.push_back(&(*base_spill_temps)[0]);
      spill_temps.push_back(&(*base_spill_temps)[1]);
    }
    const auto extra_spill_base_offset_opt =
        assign_virtual_lds_spill_offsets(context, spill_temps, extra_spill_dwords);
    if (!extra_spill_base_offset_opt)
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
          ": virtual-LDS spill offset overflows the 32-bit private segment");
    const uint32_t extra_spill_base_offset = *extra_spill_base_offset_opt;
    const uint32_t base_sgpr_save_offset = extra_spill_base_offset;
    uint32_t addr_high_spill_offset = 0;
    if (restore_addr_high) {
      addr_high_spill_offset = extra_spill_base_offset + base_sgpr_spill_dwords * sizeof(uint32_t);
      emit_cdna3_scratch_store_b32(words, addr_high, addr_high_spill_offset);
      words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
    }
    emit_virtual_lds_temp_spill_stores(words, spill_temps);
    if (base_spill_temps)
      emit_virtual_lds_base_spill_setup(words, context, *base_spill_temps, base_sgpr_save_offset);

    // Native ds_read2 issues two LDS reads from the same original address VGPR.
    // If either destination aliases the source address pair, perform that load
    // last and keep a private low-half copy for preparing the GLOBAL VADDR.
    const uint8_t preserved_base = base_temp ? base_temp->reg : addr;
    if (base_temp)
      emit_cdna3_v_add_u32_literal(words, preserved_base, addr, 0);
    if (address_pair_temp)
      emit_cdna3_v_add_u32_literal(words, global_addr, addr, 0);

    auto emit_read2_load = [&](const Read2Access &access) {
      uint16_t flat_offset = static_cast<uint16_t>(access.byte_offset);
      if (access.byte_offset > kFlatGlobalPositiveImm13Max) {
        flat_offset = 0;
        emit_cdna3_v_add_u32_literal(words, global_addr, preserved_base, access.byte_offset);
      } else if (base_temp) {
        emit_cdna3_v_add_u32_literal(words, global_addr, preserved_base, 0);
      }
      if (use_full_vgpr_address) {
        assert(base_spill_temps && "spill-per-use virtual LDS should have base VGPR temps");
        emit_virtual_lds_full_vgpr_address(words, context, global_addr, global_addr_high,
                                           (*base_spill_temps)[0].reg, (*base_spill_temps)[1].reg);
      } else {
        words.push_back(build_cdna3_v_mov_b32(global_addr_high, scalar_positive_inline_u32(0)));
      }

      const auto operands = make_virtual_lds_flat_global_operands(
          flat_offset, global_addr,
          use_full_vgpr_address ? kCdna3ScalarNull
                                : static_cast<uint8_t>(context.virtual_lds_base_sgpr));
      auto flat_operands = operands;
      flat_operands.acc = uses_acc && !staged_read2_temp;
      const uint8_t vdst =
          staged_read2_temp
              ? static_cast<uint8_t>(staged_read2_temp->base() + (access.vdst - src.vdst))
              : access.vdst;
      auto [w0, w1] = build_cdna3_flat_global(flat_operands, op->flat_op, vdst, true);
      words.push_back(w0);
      words.push_back(w1);
    };

    emit_read2_load(accesses[0]);
    emit_read2_load(accesses[1]);
    // Native DS participates in lgkmcnt. Complete the replacement VMEM before
    // exposing control to instructions that were scheduled around LDS waits.
    words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
    if (restore_addr_low)
      emit_cdna3_v_add_u32_literal(words, addr, preserved_base, 0);
    if (base_spill_temps)
      emit_virtual_lds_base_spill_restore(words, context, *base_spill_temps, base_sgpr_save_offset);
    if (restore_addr_high) {
      emit_cdna3_scratch_load_b32(words, addr_high, addr_high_spill_offset);
      words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
    }
    if (staged_read2_temp) {
      if (uses_acc)
        emit_virtual_lds_copy_temp_to_acc_range(words, *staged_read2_temp,
                                                static_cast<uint8_t>(src.vdst));
      else
        emit_virtual_lds_copy_from_temp_range(words, *staged_read2_temp,
                                              static_cast<uint8_t>(src.vdst));
    }

    std::vector<VirtualLdsAddressTemp *> restore_temps;
    if (base_spill_temps) {
      restore_temps.push_back(&(*base_spill_temps)[1]);
      restore_temps.push_back(&(*base_spill_temps)[0]);
    }
    if (staged_read2_temp) {
      for (uint8_t i = staged_read2_temp->count; i > 0; --i)
        restore_temps.push_back(&staged_read2_temp->temps[i - 1]);
    }
    if (base_temp)
      restore_temps.push_back(&*base_temp);
    if (address_pair_temp) {
      for (uint8_t i = address_pair_temp->count; i > 0; --i)
        restore_temps.push_back(&address_pair_temp->temps[i - 1]);
    }
    emit_virtual_lds_temp_spill_loads(words, restore_temps);
    if (!guard_virtual_lds_execz(words, context)) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) + ": virtual LDS SGPR spill guard branch is out of range",
          {"Reduce the per-instruction virtual-LDS spill sequence or add a long guard branch."});
    }
    return ExpandResult::success(std::move(words));
  }

  if (op->two_addr_stride_bytes != 0) {
    const uint32_t byte_offset0 = static_cast<uint32_t>(src.offset0) * op->two_addr_stride_bytes;
    const uint32_t byte_offset1 = static_cast<uint32_t>(src.offset1) * op->two_addr_stride_bytes;
    const bool needs_materialized_offset =
        byte_offset0 > kFlatGlobalPositiveImm13Max || byte_offset1 > kFlatGlobalPositiveImm13Max;
    if (!uses_acc) {
      require_source_vgprs(src.data0, op->vgpr_count);
      require_source_vgprs(src.data1, op->vgpr_count);
    }

    std::vector<VirtualLdsVgprRange> forbidden = {
        {.base = static_cast<uint16_t>(addr), .count = 2},
    };
    if (!uses_acc) {
      forbidden.push_back({.base = static_cast<uint16_t>(src.data0), .count = op->vgpr_count});
      forbidden.push_back({.base = static_cast<uint16_t>(src.data1), .count = op->vgpr_count});
    }
    std::optional<VirtualLdsTempRange> address_pair_temp;
    if ((addr % 2) != 0) {
      address_pair_temp = choose_virtual_lds_temp_range(context, forbidden, 2, 2);
      if (!address_pair_temp) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS write2 lowering cannot allocate an even GLOBAL address pair",
            {"Add a more general spill path for kernels whose DS operands cover every ordinary "
             "VGPR."});
      }
      forbidden.push_back({.base = address_pair_temp->base(), .count = address_pair_temp->count});
    }
    const uint8_t global_addr = address_pair_temp ? address_pair_temp->base() : addr;
    const uint8_t global_addr_high =
        address_pair_temp ? static_cast<uint8_t>(address_pair_temp->base() + 1) : addr_high;
    std::optional<VirtualLdsAddressTemp> base_temp;
    if (needs_materialized_offset || use_full_vgpr_address) {
      base_temp = choose_virtual_lds_address_temp(context, forbidden);
      if (!base_temp) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS write2 lowering cannot preserve the source address VGPR",
            {"Add a more general spill path for kernels whose DS operands cover every ordinary "
             "VGPR."});
      }
      forbidden.push_back({.base = base_temp->reg, .count = 1});
    }

    auto data_needs_temp = [&](uint16_t data) {
      const bool needs_aligned_data = op->vgpr_count >= 2 && (data % 2) != 0;
      const bool overlaps_high =
          !uses_acc && virtual_lds_vgpr_ranges_overlap(data, op->vgpr_count,
                                                       static_cast<uint16_t>(global_addr_high), 1);
      const bool overlaps_low_when_materialized =
          !uses_acc && (needs_materialized_offset || use_full_vgpr_address) &&
          virtual_lds_vgpr_ranges_overlap(data, op->vgpr_count, static_cast<uint16_t>(global_addr),
                                          1);
      return needs_aligned_data || overlaps_high || overlaps_low_when_materialized;
    };
    std::optional<VirtualLdsTempRange> data0_temp;
    std::optional<VirtualLdsTempRange> data1_temp;
    auto choose_data_temp = [&](uint16_t data, std::optional<VirtualLdsTempRange> &temp) -> bool {
      if (!data_needs_temp(data))
        return true;
      temp = choose_virtual_lds_temp_range(context, forbidden, op->vgpr_count,
                                           op->vgpr_count >= 2 ? 2 : 1);
      if (!temp)
        return false;
      forbidden.push_back({.base = temp->base(), .count = temp->count});
      return true;
    };
    if (!choose_data_temp(static_cast<uint16_t>(src.data0), data0_temp) ||
        !choose_data_temp(static_cast<uint16_t>(src.data1), data1_temp)) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
              ": virtual LDS write2 store data overlaps the address VGPR pair",
          {"Add contiguous data temporary support for wide stores whose data operands overlap the "
           "GLOBAL address pair."});
    }

    std::optional<std::array<VirtualLdsAddressTemp, 2>> base_spill_temps;
    if (context.virtual_lds_base_sgpr_spill_per_use) {
      std::vector<VirtualLdsVgprRange> base_spill_forbidden = {
          {.base = static_cast<uint16_t>(addr), .count = 1},
      };
      if (use_full_vgpr_address)
        base_spill_forbidden.push_back({.base = static_cast<uint16_t>(addr_high), .count = 1});
      if (base_temp)
        base_spill_forbidden.push_back({.base = base_temp->reg, .count = 1});
      if (address_pair_temp)
        base_spill_forbidden.push_back(
            {.base = address_pair_temp->base(), .count = address_pair_temp->count});
      if (data0_temp)
        base_spill_forbidden.push_back({.base = data0_temp->base(), .count = data0_temp->count});
      else if (!uses_acc)
        base_spill_forbidden.push_back(
            {.base = static_cast<uint16_t>(src.data0), .count = op->vgpr_count});
      if (data1_temp)
        base_spill_forbidden.push_back({.base = data1_temp->base(), .count = data1_temp->count});
      else if (!uses_acc)
        base_spill_forbidden.push_back(
            {.base = static_cast<uint16_t>(src.data1), .count = op->vgpr_count});
      base_spill_temps = choose_virtual_lds_base_spill_temps(context, base_spill_forbidden);
      if (!base_spill_temps) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS SGPR spill lowering cannot find VGPR save temps",
            {"Add a more general spill path for kernels whose DS operands cover every ordinary "
             "VGPR."});
      }
    }

    const uint32_t base_sgpr_spill_dwords =
        base_spill_temps && context.virtual_lds_base_pointer_spilled ? 2u : 0u;
    std::vector<VirtualLdsAddressTemp *> spill_temps;
    if (address_pair_temp) {
      for (uint8_t i = 0; i < address_pair_temp->count; ++i)
        spill_temps.push_back(&address_pair_temp->temps[i]);
    }
    if (base_temp)
      spill_temps.push_back(&*base_temp);
    if (data0_temp) {
      for (uint8_t i = 0; i < data0_temp->count; ++i)
        spill_temps.push_back(&data0_temp->temps[i]);
    }
    if (data1_temp) {
      for (uint8_t i = 0; i < data1_temp->count; ++i)
        spill_temps.push_back(&data1_temp->temps[i]);
    }
    if (base_spill_temps) {
      spill_temps.push_back(&(*base_spill_temps)[0]);
      spill_temps.push_back(&(*base_spill_temps)[1]);
    }
    const auto extra_spill_base_offset_opt =
        assign_virtual_lds_spill_offsets(context, spill_temps, base_sgpr_spill_dwords + 1u);
    if (!extra_spill_base_offset_opt)
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
          ": virtual-LDS spill offset overflows the 32-bit private segment");
    const uint32_t extra_spill_base_offset = *extra_spill_base_offset_opt;
    const uint32_t base_sgpr_save_offset = extra_spill_base_offset;
    const uint32_t addr_high_spill_offset =
        extra_spill_base_offset + base_sgpr_spill_dwords * sizeof(uint32_t);
    emit_cdna3_scratch_store_b32(words, addr_high, addr_high_spill_offset);
    words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
    emit_virtual_lds_temp_spill_stores(words, spill_temps);
    if (base_spill_temps)
      emit_virtual_lds_base_spill_setup(words, context, *base_spill_temps, base_sgpr_save_offset);

    if (data0_temp) {
      if (uses_acc)
        emit_virtual_lds_copy_acc_to_temp_range(words, *data0_temp,
                                                static_cast<uint8_t>(src.data0));
      else
        emit_virtual_lds_copy_to_temp_range(words, *data0_temp, static_cast<uint8_t>(src.data0));
    }
    if (data1_temp) {
      if (uses_acc)
        emit_virtual_lds_copy_acc_to_temp_range(words, *data1_temp,
                                                static_cast<uint8_t>(src.data1));
      else
        emit_virtual_lds_copy_to_temp_range(words, *data1_temp, static_cast<uint8_t>(src.data1));
    }
    const uint8_t preserved_base = base_temp ? base_temp->reg : addr;
    if (base_temp)
      emit_cdna3_v_add_u32_literal(words, preserved_base, addr, 0);
    if (address_pair_temp)
      emit_cdna3_v_add_u32_literal(words, global_addr, addr, 0);

    auto emit_write2_store = [&](uint8_t data, uint32_t byte_offset, bool data_is_acc) {
      uint16_t flat_offset = static_cast<uint16_t>(byte_offset);
      if (byte_offset > kFlatGlobalPositiveImm13Max) {
        flat_offset = 0;
        emit_cdna3_v_add_u32_literal(words, global_addr, preserved_base, byte_offset);
      } else if (base_temp) {
        emit_cdna3_v_add_u32_literal(words, global_addr, preserved_base, 0);
      }
      if (use_full_vgpr_address) {
        assert(base_spill_temps && "spill-per-use virtual LDS should have base VGPR temps");
        emit_virtual_lds_full_vgpr_address(words, context, global_addr, global_addr_high,
                                           (*base_spill_temps)[0].reg, (*base_spill_temps)[1].reg);
      } else {
        words.push_back(build_cdna3_v_mov_b32(global_addr_high, scalar_positive_inline_u32(0)));
      }

      const auto operands = make_virtual_lds_flat_global_operands(
          flat_offset, global_addr,
          use_full_vgpr_address ? kCdna3ScalarNull
                                : static_cast<uint8_t>(context.virtual_lds_base_sgpr));
      auto flat_operands = operands;
      flat_operands.acc = data_is_acc;
      auto [w0, w1] = build_cdna3_flat_global(flat_operands, op->flat_op, data, false);
      words.push_back(w0);
      words.push_back(w1);
    };

    emit_write2_store(data0_temp ? data0_temp->base() : static_cast<uint8_t>(src.data0),
                      byte_offset0, uses_acc && !data0_temp);
    emit_write2_store(data1_temp ? data1_temp->base() : static_cast<uint8_t>(src.data1),
                      byte_offset1, uses_acc && !data1_temp);
    words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
    if (base_temp)
      emit_cdna3_v_add_u32_literal(words, addr, preserved_base, 0);
    if (base_spill_temps)
      emit_virtual_lds_base_spill_restore(words, context, *base_spill_temps, base_sgpr_save_offset);
    emit_cdna3_scratch_load_b32(words, addr_high, addr_high_spill_offset);
    words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
    std::vector<VirtualLdsAddressTemp *> restore_temps;
    if (base_spill_temps) {
      restore_temps.push_back(&(*base_spill_temps)[1]);
      restore_temps.push_back(&(*base_spill_temps)[0]);
    }
    if (data1_temp) {
      for (uint8_t i = data1_temp->count; i > 0; --i)
        restore_temps.push_back(&data1_temp->temps[i - 1]);
    }
    if (data0_temp) {
      for (uint8_t i = data0_temp->count; i > 0; --i)
        restore_temps.push_back(&data0_temp->temps[i - 1]);
    }
    if (base_temp)
      restore_temps.push_back(&*base_temp);
    if (address_pair_temp) {
      for (uint8_t i = address_pair_temp->count; i > 0; --i)
        restore_temps.push_back(&address_pair_temp->temps[i - 1]);
    }
    emit_virtual_lds_temp_spill_loads(words, restore_temps);
    if (!guard_virtual_lds_execz(words, context)) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) + ": virtual LDS SGPR spill guard branch is out of range",
          {"Reduce the per-instruction virtual-LDS spill sequence or add a long guard branch."});
    }
    return ExpandResult::success(std::move(words));
  }

  const uint32_t ds_offset = (static_cast<uint32_t>(src.offset1) << 8) | src.offset0;
  uint16_t flat_offset = static_cast<uint16_t>(ds_offset);
  const bool needs_materialized_offset = ds_offset > kFlatGlobalPositiveImm13Max;
  if (!uses_acc)
    require_source_vgprs(op->is_load ? src.vdst : src.data0, op->vgpr_count);
  const bool load_clobbers_addr_low =
      !uses_acc && op->is_load &&
      virtual_lds_vgpr_ranges_overlap(static_cast<uint16_t>(src.vdst), op->vgpr_count,
                                      static_cast<uint16_t>(addr), 1);
  const bool load_clobbers_addr_high =
      !uses_acc && op->is_load &&
      virtual_lds_vgpr_ranges_overlap(static_cast<uint16_t>(src.vdst), op->vgpr_count,
                                      static_cast<uint16_t>(addr_high), 1);
  std::optional<VirtualLdsAddressTemp> base_temp;
  std::optional<VirtualLdsTempRange> store_data_temp;
  std::optional<VirtualLdsTempRange> load_result_temp;
  std::vector<VirtualLdsVgprRange> forbidden = {
      {.base = static_cast<uint16_t>(addr), .count = 2},
  };
  if (!uses_acc) {
    forbidden.push_back(
        {.base = op->is_load ? static_cast<uint16_t>(src.vdst) : static_cast<uint16_t>(src.data0),
         .count = op->vgpr_count});
  }
  std::optional<VirtualLdsTempRange> address_pair_temp;
  if ((addr % 2) != 0) {
    address_pair_temp = choose_virtual_lds_temp_range(context, forbidden, 2, 2);
    if (!address_pair_temp) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
              ": virtual LDS lowering cannot allocate an even GLOBAL address pair",
          {"Add a more general spill path for kernels whose DS operands cover every ordinary "
           "VGPR."});
    }
    forbidden.push_back({.base = address_pair_temp->base(), .count = address_pair_temp->count});
  }
  const uint8_t global_addr = address_pair_temp ? address_pair_temp->base() : addr;
  const uint8_t global_addr_high =
      address_pair_temp ? static_cast<uint8_t>(address_pair_temp->base() + 1) : addr_high;
  if (needs_materialized_offset ||
      (use_full_vgpr_address && (!op->is_load || !load_clobbers_addr_low))) {
    base_temp = choose_virtual_lds_address_temp(context, forbidden);
    if (!base_temp) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
              ": virtual LDS lowering cannot preserve the source address VGPR",
          {"Add a more general spill path for kernels whose DS operands cover every ordinary "
           "VGPR."});
    }
    forbidden.push_back({.base = base_temp->reg, .count = 1});
  }

  const bool data_needs_temp =
      !op->is_load &&
      ((op->vgpr_count >= 2 && (src.data0 % 2) != 0) ||
       (!uses_acc &&
        virtual_lds_vgpr_ranges_overlap(static_cast<uint16_t>(src.data0), op->vgpr_count,
                                        static_cast<uint16_t>(global_addr_high), 1)) ||
       (!uses_acc && (needs_materialized_offset || use_full_vgpr_address) &&
        virtual_lds_vgpr_ranges_overlap(static_cast<uint16_t>(src.data0), op->vgpr_count,
                                        static_cast<uint16_t>(global_addr), 1)));
  if (data_needs_temp) {
    store_data_temp = choose_virtual_lds_temp_range(context, forbidden, op->vgpr_count,
                                                    op->vgpr_count >= 2 ? 2 : 1);
    if (!store_data_temp) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
              ": virtual LDS store lowering cannot preserve overlapping store data",
          {"Add a more general spill path for kernels whose DS operands cover every ordinary "
           "VGPR."});
    }
    forbidden.push_back({.base = store_data_temp->base(), .count = store_data_temp->count});
  }
  if (op->is_load && op->vgpr_count >= 2 && (src.vdst % 2) != 0) {
    load_result_temp = choose_virtual_lds_temp_range(context, forbidden, op->vgpr_count, 2);
    if (!load_result_temp) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
              ": virtual LDS load lowering cannot allocate an aligned result tuple",
          {"Add a more general spill path for kernels whose DS operands cover every ordinary "
           "VGPR."});
    }
    forbidden.push_back({.base = load_result_temp->base(), .count = load_result_temp->count});
  }

  const bool restore_addr_high = !load_clobbers_addr_high;
  std::optional<std::array<VirtualLdsAddressTemp, 2>> base_spill_temps;
  if (context.virtual_lds_base_sgpr_spill_per_use) {
    std::vector<VirtualLdsVgprRange> base_spill_forbidden = {
        {.base = static_cast<uint16_t>(addr), .count = 1},
    };
    if (use_full_vgpr_address || !restore_addr_high)
      base_spill_forbidden.push_back({.base = static_cast<uint16_t>(addr_high), .count = 1});
    if (base_temp)
      base_spill_forbidden.push_back({.base = base_temp->reg, .count = 1});
    if (address_pair_temp)
      base_spill_forbidden.push_back(
          {.base = address_pair_temp->base(), .count = address_pair_temp->count});
    if (op->is_load) {
      if (!uses_acc)
        base_spill_forbidden.push_back(
            {.base = static_cast<uint16_t>(src.vdst), .count = op->vgpr_count});
      if (load_result_temp)
        base_spill_forbidden.push_back(
            {.base = load_result_temp->base(), .count = load_result_temp->count});
    } else if (store_data_temp) {
      base_spill_forbidden.push_back(
          {.base = store_data_temp->base(), .count = store_data_temp->count});
    } else if (!uses_acc) {
      base_spill_forbidden.push_back(
          {.base = static_cast<uint16_t>(src.data0), .count = op->vgpr_count});
    }
    base_spill_temps = choose_virtual_lds_base_spill_temps(context, base_spill_forbidden);
    if (!base_spill_temps) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
              ": virtual LDS SGPR spill lowering cannot find VGPR save temps",
          {"Add a more general spill path for kernels whose DS operands cover every ordinary "
           "VGPR."});
    }
  }

  const uint32_t base_sgpr_spill_dwords =
      base_spill_temps && context.virtual_lds_base_pointer_spilled ? 2u : 0u;
  const uint32_t extra_spill_dwords = base_sgpr_spill_dwords + (restore_addr_high ? 1u : 0u);
  std::vector<VirtualLdsAddressTemp *> spill_temps;
  if (address_pair_temp) {
    for (uint8_t i = 0; i < address_pair_temp->count; ++i)
      spill_temps.push_back(&address_pair_temp->temps[i]);
  }
  if (base_temp)
    spill_temps.push_back(&*base_temp);
  if (store_data_temp) {
    for (uint8_t i = 0; i < store_data_temp->count; ++i)
      spill_temps.push_back(&store_data_temp->temps[i]);
  }
  if (load_result_temp) {
    for (uint8_t i = 0; i < load_result_temp->count; ++i)
      spill_temps.push_back(&load_result_temp->temps[i]);
  }
  if (base_spill_temps) {
    spill_temps.push_back(&(*base_spill_temps)[0]);
    spill_temps.push_back(&(*base_spill_temps)[1]);
  }
  const auto extra_spill_base_offset_opt =
      assign_virtual_lds_spill_offsets(context, spill_temps, extra_spill_dwords);
  if (!extra_spill_base_offset_opt)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                ": virtual-LDS spill offset overflows the 32-bit private segment");
  const uint32_t extra_spill_base_offset = *extra_spill_base_offset_opt;
  const uint32_t base_sgpr_save_offset = extra_spill_base_offset;
  // Save the source high half before any spill-per-use SGPR setup can borrow it
  // as a temporary. It is restored only after the borrowed SGPR pair is restored.
  uint32_t addr_high_spill_offset = 0;
  if (restore_addr_high) {
    addr_high_spill_offset = extra_spill_base_offset + base_sgpr_spill_dwords * sizeof(uint32_t);
    emit_cdna3_scratch_store_b32(words, addr_high, addr_high_spill_offset);
    words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
  }
  emit_virtual_lds_temp_spill_stores(words, spill_temps);

  if (store_data_temp) {
    if (uses_acc)
      emit_virtual_lds_copy_acc_to_temp_range(words, *store_data_temp,
                                              static_cast<uint8_t>(src.data0));
    else
      emit_virtual_lds_copy_to_temp_range(words, *store_data_temp, static_cast<uint8_t>(src.data0));
  }
  const uint8_t preserved_base = base_temp ? base_temp->reg : addr;
  if (base_temp)
    emit_cdna3_v_add_u32_literal(words, preserved_base, addr, 0);
  if (address_pair_temp)
    emit_cdna3_v_add_u32_literal(words, global_addr, addr, 0);
  if (needs_materialized_offset) {
    emit_cdna3_v_add_u32_literal(words, global_addr, preserved_base, ds_offset);
    flat_offset = 0;
  }

  // Use an encodable GLOBAL VADDR pair. The shared CDNA3 access emitter owns the
  // high-half initialization and the descriptor-full borrowed-SGPR protocol.
  const bool restore_addr_low =
      (needs_materialized_offset || use_full_vgpr_address) && !load_clobbers_addr_low;
  const uint8_t data =
      op->is_load ? (load_result_temp ? load_result_temp->base() : static_cast<uint8_t>(src.vdst))
                  : (store_data_temp ? store_data_temp->base() : static_cast<uint8_t>(src.data0));
  std::optional<Cdna3VirtualLdsBorrowScratch> borrow_scratch;
  if (base_spill_temps) {
    borrow_scratch = Cdna3VirtualLdsBorrowScratch{
        .pointer_vgpr_lo = (*base_spill_temps)[0].reg,
        .pointer_vgpr_hi = (*base_spill_temps)[1].reg,
        .saved_sgpr_private_offset = base_sgpr_save_offset,
    };
  }
  if (!append_cdna3_virtual_lds_access(
          words, context,
          Cdna3VirtualLdsAccess{.is_load = op->is_load,
                                .op = op->flat_op,
                                .data_vgpr = data,
                                .address_vgpr = global_addr,
                                .byte_offset = flat_offset,
                                .acc = uses_acc && !store_data_temp && !load_result_temp},
          borrow_scratch)) {
    return ExpandResult::failed(
        std::string(inst.mnemonic()) + ": virtual LDS access is not encodable",
        {"Provide an even GLOBAL address pair and non-overlapping spill-per-use scratch."});
  }
  if (restore_addr_low)
    emit_cdna3_v_add_u32_literal(words, addr, preserved_base, 0);
  if (restore_addr_high) {
    emit_cdna3_scratch_load_b32(words, addr_high, addr_high_spill_offset);
    words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
  }
  if (load_result_temp) {
    if (uses_acc)
      emit_virtual_lds_copy_temp_to_acc_range(words, *load_result_temp,
                                              static_cast<uint8_t>(src.vdst));
    else
      emit_virtual_lds_copy_from_temp_range(words, *load_result_temp,
                                            static_cast<uint8_t>(src.vdst));
  }
  std::vector<VirtualLdsAddressTemp *> restore_temps;
  if (base_spill_temps) {
    restore_temps.push_back(&(*base_spill_temps)[1]);
    restore_temps.push_back(&(*base_spill_temps)[0]);
  }
  if (store_data_temp) {
    for (uint8_t i = store_data_temp->count; i > 0; --i)
      restore_temps.push_back(&store_data_temp->temps[i - 1]);
  }
  if (load_result_temp) {
    for (uint8_t i = load_result_temp->count; i > 0; --i)
      restore_temps.push_back(&load_result_temp->temps[i - 1]);
  }
  if (base_temp)
    restore_temps.push_back(&*base_temp);
  if (address_pair_temp) {
    for (uint8_t i = address_pair_temp->count; i > 0; --i)
      restore_temps.push_back(&address_pair_temp->temps[i - 1]);
  }
  emit_virtual_lds_temp_spill_loads(words, restore_temps);
  if (!guard_virtual_lds_execz(words, context)) {
    return ExpandResult::failed(
        std::string(inst.mnemonic()) + ": virtual LDS SGPR spill guard branch is out of range",
        {"Reduce the per-instruction virtual-LDS spill sequence or add a long guard branch."});
  }
  return ExpandResult::success(std::move(words));
}

} // namespace

bool cdna4_to_cdna3_source_instruction_uses_virtualizable_lds(const Instruction &inst) {
  return source_instruction_uses_virtualizable_lds_impl(inst);
}

std::optional<VirtualLdsBaseSgprReservation> reserve_cdna4_to_cdna3_virtual_lds_base_sgpr_pair(
    TranslationContext &context, KernelBlockScope blocks, const KdTranslation &translation) {
  return reserve_cdna3_virtual_lds_base_sgpr_pair(context, blocks, translation,
                                                  ROCJITSU_CODE_ARCH_CDNA3);
}

bool append_cdna4_to_cdna3_virtual_lds_entry_prologue(KdTranslation &translation) {
  return append_cdna3_virtual_lds_entry_prologue(translation);
}

ExpandResult lower_cdna4_to_cdna3_virtual_lds_instruction(const Instruction &inst,
                                                          TranslationContext &context) {
  return lower_cdna4_to_cdna3_virtual_lds_ds_instruction(inst, context);
}

} // namespace rocjitsu
