// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_cdna3.cpp
/// @brief CDNA4-to-CDNA3 handwritten semantic expansion rules.

#include "rocjitsu/code/dbt/semantic/rules.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/vop3.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

// CDNA3 instruction builders used by the narrow semantic rules below. These
// helpers intentionally only cover the encodings the lowering emits; keeping
// them local avoids adding a broad assembler abstraction for a handful of
// code-cave sequences.

[[nodiscard]] std::pair<uint32_t, uint32_t>
build_cdna3_vop3(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1 = 0, uint16_t src2 = 0) {
  cdna3::Vop3MachineInst dst{};
  dst.encoding = 0x34;
  dst.op = op;
  dst.vdst = vdst;
  dst.src0 = src0 & 0x1FF;
  dst.src1 = src1 & 0x1FF;
  dst.src2 = src2 & 0x1FF;

  uint32_t words[2]{};
  std::memcpy(words, &dst, sizeof(dst));
  return {words[0], words[1]};
}

/// @brief Build a CDNA3 VOP3P-MFMA instruction word pair.
/// @details The wide-K lowering materializes A/B operands in ordinary VGPRs, so
/// the emitted narrow MFMA clears the source ACC selector while preserving the
/// destination AccVGPR selector from the CDNA4 instruction.
[[nodiscard]] std::pair<uint32_t, uint32_t>
build_cdna3_vop3p_mfma(uint8_t op, const cdna4::Vop3pMfmaMachineInst &src, uint8_t vdst,
                       uint8_t acc_cd, uint16_t src0, uint16_t src1, uint16_t src2) {
  cdna3::Vop3pMfmaMachineInst dst{};
  dst.encoding = 0x1A7;
  dst.op = op;
  dst.vdst = vdst;
  dst.cbsz = src.cbsz;
  dst.abid = src.abid;
  dst.acc_cd = acc_cd;
  dst.src0 = src0 & 0x1FF;
  dst.src1 = src1 & 0x1FF;
  dst.src2 = src2 & 0x1FF;
  dst.acc = 0;
  dst.blgp = src.blgp;

  uint32_t words[2]{};
  std::memcpy(words, &dst, sizeof(dst));
  return {words[0], words[1]};
}

[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_ds(uint8_t op, uint8_t vdst, uint8_t addr,
                                                           uint8_t data0 = 0, uint8_t data1 = 0,
                                                           uint8_t offset0 = 0,
                                                           uint8_t offset1 = 0) {
  cdna3::DsMachineInst dst{};
  dst.encoding = 0x36;
  dst.op = op;
  dst.offset0 = offset0;
  dst.offset1 = offset1;
  dst.addr = addr;
  dst.data0 = data0;
  dst.data1 = data1;
  dst.vdst = vdst;

  uint32_t words[2]{};
  std::memcpy(words, &dst, sizeof(dst));
  return {words[0], words[1]};
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_s_mov_b32_lit(uint8_t sdst,
                                                                          uint32_t literal) {
  return {pack_sop1(0, sdst, 0xFF), literal};
}

constexpr uint8_t kExecLo = 126;
constexpr uint16_t kInlineConst0 = 128;
constexpr uint16_t kInlineConstNeg1 = 193;

// BinaryTranslator currently reserves 128 VGPRs for all semantic lowerings.
// Scratch allocation must remain inside that descriptor headroom until the DBT
// pipeline grows per-lowering resource reporting.
constexpr uint16_t kSemanticScratchVgprLimit = 128;

// CDNA3 VOP3 opcodes used by the bitop3, wide-K, and DS transpose expansions.
constexpr uint16_t kCdna3OpMovB32 = 321;
constexpr uint16_t kCdna3OpLshrrevB32 = 272;
constexpr uint16_t kCdna3OpLshlrevB32 = 274;
constexpr uint16_t kCdna3OpAndB32 = 275;
constexpr uint16_t kCdna3OpOrB32 = 276;
constexpr uint16_t kCdna3OpXorB32 = 277;
constexpr uint16_t kCdna3OpAddU32 = 308;
constexpr uint16_t kCdna3OpPermB32 = 493;
constexpr uint16_t kCdna3OpMbcntLoU32B32 = 652;
constexpr uint16_t kCdna3OpMbcntHiU32B32 = 653;

// CDNA3 VOP3P-MFMA opcodes used by CDNA4 wide-K MFMA expansion.
constexpr uint8_t kCdna3OpMfmaF32_32x32x8F16 = 76;
constexpr uint8_t kCdna3OpMfmaF32_16x16x16F16 = 77;

// CDNA3 DS opcodes used to synthesize CDNA4 transposed LDS reads.
constexpr uint8_t kCdna3DsOpBpermuteB32 = 63;
constexpr uint8_t kCdna3DsOpReadB64 = 118;

constexpr uint8_t kCdna3SoppOpWaitcnt = 12;
constexpr uint16_t kCdnaWaitcntLgkmcnt0 = 0xC07F;

void emit_cdna3_vop3(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint16_t src0,
                     uint16_t src1 = 0, uint16_t src2 = 0) {
  auto [w0, w1] = build_cdna3_vop3(op, vdst, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_ds(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst, uint8_t addr,
                   uint8_t data0 = 0, uint8_t data1 = 0, uint8_t offset0 = 0, uint8_t offset1 = 0) {
  auto [w0, w1] = build_cdna3_ds(op, vdst, addr, data0, data1, offset0, offset1);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_lgkm_wait(std::vector<uint32_t> &words) {
  // GFX9/CDNA s_waitcnt encodes "lgkmcnt(0)" as lgkm=0 while leaving VM/EXP at
  // their no-wait maxima. The DS read/bpermute sequences below require the
  // loaded/permuted data before issuing dependent VALU instructions.
  words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntLgkmcnt0));
}

[[nodiscard]] constexpr uint16_t vgpr_src(uint8_t reg) { return static_cast<uint16_t>(256 + reg); }

void emit_s_mov_b32_lit(std::vector<uint32_t> &words, uint8_t sdst, uint32_t literal) {
  auto [w0, w1] = build_s_mov_b32_lit(sdst, literal);
  words.push_back(w0);
  words.push_back(w1);
}

[[maybe_unused]] void emit_cdna3_exec_mask(std::vector<uint32_t> &words, uint64_t mask) {
  emit_s_mov_b32_lit(words, kExecLo, static_cast<uint32_t>(mask));
  emit_s_mov_b32_lit(words, kExecLo + 1, static_cast<uint32_t>(mask >> 32));
}

void emit_cdna3_mfma(std::vector<uint32_t> &words, uint8_t op,
                     const cdna4::Vop3pMfmaMachineInst &src, uint16_t src0, uint16_t src1,
                     uint16_t src2) {
  auto [w0, w1] = build_cdna3_vop3p_mfma(op, src, static_cast<uint8_t>(src.vdst),
                                         static_cast<uint8_t>(src.acc_cd), src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_mfma_to_vgpr(std::vector<uint32_t> &words, uint8_t op,
                             const cdna4::Vop3pMfmaMachineInst &src, uint8_t vdst, uint16_t src0,
                             uint16_t src1, uint16_t src2) {
  auto [w0, w1] = build_cdna3_vop3p_mfma(op, src, vdst, 0, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
}

/// @brief Find a dead contiguous VGPR run that stays below a semantic scratch limit.
/// @details This wraps LivenessAnalysis::find_free_run() for semantic lowerings
/// that must allocate temporary VGPRs inside the descriptor headroom reserved by
/// the DBT pipeline. The returned base is at or after @p search_start and the
/// half-open run [base, base + @p count) is guaranteed not to cross @p limit.
/// @returns The first matching VGPR base, or std::nullopt if no suitable run is
/// available before @p limit.
[[nodiscard]] std::optional<uint16_t>
find_free_vgpr_run_below(const LivenessAnalysis &liveness, const Instruction &inst, uint16_t count,
                         uint16_t search_start, uint16_t limit) {
  auto candidate = liveness.find_free_run(&inst, count, search_start);
  if (!candidate || *candidate + count > limit)
    return std::nullopt;
  return candidate;
}

// -----------------------------------------------------------------------------
// V_BITOP3 expansions.
// -----------------------------------------------------------------------------

/// @brief Convert the 3-input truth table into algebraic-normal-form coefficients.
/// @details Truth-table bit index is {S0[i], S1[i], S2[i]}: bit 2 is S0, bit 1
/// is S1, and bit 0 is S2. ANF lets CDNA3 synthesize the LUT from AND/XOR.
[[nodiscard]] std::array<uint8_t, 8> bitop3_anf_coefficients(uint8_t truth_table) {
  std::array<uint8_t, 8> coeff{};
  for (uint8_t mask = 0; mask < coeff.size(); ++mask)
    coeff[mask] = static_cast<uint8_t>((truth_table >> mask) & 0x1);

  for (uint8_t variable_mask : {uint8_t{4}, uint8_t{2}, uint8_t{1}}) {
    for (uint8_t mask = 0; mask < coeff.size(); ++mask) {
      if ((mask & variable_mask) != 0)
        coeff[mask] ^= coeff[mask ^ variable_mask];
    }
  }
  return coeff;
}

[[nodiscard]] bool vdst_aliases_any_vgpr_source(uint8_t vdst, const std::array<uint16_t, 3> &src) {
  const uint16_t encoded_vdst = static_cast<uint16_t>(256 + vdst);
  return src[0] == encoded_vdst || src[1] == encoded_vdst || src[2] == encoded_vdst;
}

[[nodiscard]] bool bitop3_needs_product_term(const std::array<uint8_t, 8> &coeff) {
  for (uint8_t mask = 1; mask < coeff.size(); ++mask) {
    if (coeff[mask] != 0 && std::popcount(mask) >= 2)
      return true;
  }
  return false;
}

template <typename Bitop3Inst>
std::vector<uint32_t> lower_cdna4_bitop3_to_cdna3(const Bitop3Inst &inst,
                                                  const LivenessAnalysis &liveness, bool is_b16) {
  // V_BITOP3 is a three-input bitwise LUT. CDNA4 encodes the eight LUT bits in
  // VOP3 modifier fields; CDNA3 has no equivalent instruction, so the lowering
  // emits the LUT as algebraic normal form over GF(2):
  //
  //   ttbl_index = (S0_bit << 2) | (S1_bit << 1) | S2_bit
  //   coeff[] = mobius_transform(ttbl[])
  //   result = coeff[0]
  //          ^ coeff[1] & S2
  //          ^ coeff[2] & S1
  //          ^ coeff[3] & S1 & S2
  //          ^ coeff[4] & S0
  //          ^ coeff[5] & S0 & S2
  //          ^ coeff[6] & S0 & S1
  //          ^ coeff[7] & S0 & S1 & S2
  //
  // Multiplication in that expression is bitwise AND, addition is XOR, and a
  // constant one term is materialized as -1 so every lane bit sees true. The B16
  // form computes the same 32-bit LUT, then clears the high half with a
  // left/right shift pair.
  const uint8_t vdst = static_cast<uint8_t>(inst.vdst.encoding_value());
  const std::array<uint16_t, 3> src = {static_cast<uint16_t>(inst.src0.encoding_value()),
                                       static_cast<uint16_t>(inst.src1.encoding_value()),
                                       static_cast<uint16_t>(inst.src2.encoding_value())};

  if (is_b16 && inst.inst_.op_sel != 0)
    // NYI: OP_SEL selects B16 source/destination halves. Source-half selection
    // can be lowered by shifting selected high halves down before the LUT, but
    // OP_SEL[3] is read-modify-write: it writes the high half of vdst while
    // preserving the old low half. The generated operand metadata currently
    // treats vdst only as a destination, so liveness may allocate vdst as
    // scratch and clobber the implicit source value. Until that implicit vdst
    // read is modeled, only lower the canonical OP_SEL=0 form instead of
    // silently producing wrong code.
    return {};
  // V_BITOP3 overloads VOP3 modifier fields as TTBL bits instead of ordinary
  // arithmetic modifiers: {OMOD[1:0], ABS[2:0], NEG[2:0]}.
  const uint8_t truth_table = static_cast<uint8_t>(
      ((inst.inst_.omod & 0x3) << 6) | ((inst.inst_.abs & 0x7) << 3) | (inst.inst_.neg & 0x7));
  const auto coeff = bitop3_anf_coefficients(truth_table);

  const bool needs_acc_temp = vdst_aliases_any_vgpr_source(vdst, src);
  const bool needs_term_temp = bitop3_needs_product_term(coeff);

  // Scratch policy:
  //   - No product terms and no vdst/source alias: use vdst as the accumulator.
  //   - vdst/source alias only: use one scratch accumulator, then copy to vdst.
  //   - Any product term: use two scratch VGPRs, one accumulator and one AND
  //     term. This keeps the generated sequence simple and prevents the AND temp
  //     from aliasing the accumulator. Liveness may choose vdst as scratch when
  //     vdst is dead before the original instruction; that is fine because the
  //     final result still lands in vdst.
  const uint16_t scratch_count =
      needs_term_temp ? 2 : static_cast<uint16_t>(needs_acc_temp ? 1 : 0);

  uint8_t acc = vdst;
  uint8_t term = 0;
  if (scratch_count != 0) {
    auto scratch =
        find_free_vgpr_run_below(liveness, inst, scratch_count, 0, kSemanticScratchVgprLimit);
    if (!scratch)
      return {};

    acc = static_cast<uint8_t>(*scratch);
    if (needs_term_temp)
      term = static_cast<uint8_t>(*scratch + 1);
  }

  std::vector<uint32_t> words;

  auto src_for_variable = [&](uint8_t variable_mask) -> uint16_t {
    switch (variable_mask) {
    case 4:
      return src[0];
    case 2:
      return src[1];
    default:
      return src[2];
    }
  };

  auto emit_mov = [&](uint8_t dst, uint16_t src0) {
    emit_cdna3_vop3(words, kCdna3OpMovB32, dst, src0);
  };
  auto emit_and = [&](uint8_t dst, uint16_t src0, uint16_t src1) {
    emit_cdna3_vop3(words, kCdna3OpAndB32, dst, src0, src1);
  };
  auto emit_xor = [&](uint8_t dst, uint16_t src0, uint16_t src1) {
    emit_cdna3_vop3(words, kCdna3OpXorB32, dst, src0, src1);
  };

  bool acc_initialized = false;
  if (coeff[0] != 0) {
    emit_mov(acc, kInlineConstNeg1);
    acc_initialized = true;
  }

  for (uint8_t mask = 1; mask < coeff.size(); ++mask) {
    if (coeff[mask] == 0)
      continue;

    std::array<uint16_t, 3> variables{};
    uint8_t variable_count = 0;
    for (uint8_t variable_mask : {uint8_t{4}, uint8_t{2}, uint8_t{1}}) {
      if ((mask & variable_mask) != 0)
        variables[variable_count++] = src_for_variable(variable_mask);
    }

    uint16_t term_src = variables[0];
    if (variable_count >= 2) {
      emit_and(term, variables[0], variables[1]);
      if (variable_count == 3)
        emit_and(term, vgpr_src(term), variables[2]);
      term_src = vgpr_src(term);
    }

    if (!acc_initialized) {
      emit_mov(acc, term_src);
      acc_initialized = true;
    } else {
      emit_xor(acc, vgpr_src(acc), term_src);
    }
  }

  if (!acc_initialized)
    emit_mov(acc, kInlineConst0);

  if (is_b16) {
    // The B16 form writes a zero-extended low half. Shift left then logical
    // shift right to clear bits 31:16 without needing a separate 0xffff mask,
    // which CDNA3 cannot encode as an inline VALU operand.
    const uint16_t shift16 = scalar_positive_inline_u32(16);
    emit_cdna3_vop3(words, kCdna3OpLshlrevB32, acc, shift16, vgpr_src(acc));
    emit_cdna3_vop3(words, kCdna3OpLshrrevB32, acc, shift16, vgpr_src(acc));
  }

  if (acc != vdst)
    emit_mov(vdst, vgpr_src(acc));

  return words;
}

// -----------------------------------------------------------------------------
// Wide-K MFMA expansions.
// -----------------------------------------------------------------------------

enum class WideKMfmaShape {
  F32_16x16x32_F16,
  F32_32x32x16_F16,
};

struct WideKMfmaLowering {
  WideKMfmaShape shape;
  uint8_t narrow_op;
  uint8_t dst_regs;
  uint8_t wide_src_regs;
  uint8_t narrow_src_regs;
};

[[nodiscard]] constexpr WideKMfmaLowering lowering_for_shape(WideKMfmaShape shape) {
  switch (shape) {
  case WideKMfmaShape::F32_16x16x32_F16:
    return {shape, kCdna3OpMfmaF32_16x16x16F16, 4, 4, 2};
  case WideKMfmaShape::F32_32x32x16_F16:
    return {shape, kCdna3OpMfmaF32_32x32x8F16, 16, 4, 2};
  }
  return {shape, 0, 0, 0, 0};
}

[[nodiscard]] bool ranges_overlap(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                  uint16_t rhs_count) {
  return lhs_base < rhs_base + rhs_count && rhs_base < lhs_base + lhs_count;
}

[[nodiscard]] bool wide_mfma_needs_partial_accum_scratch(const cdna4::Vop3pMfmaMachineInst &mfma,
                                                         const WideKMfmaLowering &lowering) {
  // acc_cd=1 writes the AccVGPR bank. Because this lowering currently rejects
  // ACC-selected A/B sources, the original A/B operands are ordinary VGPRs and
  // cannot be clobbered by an AccVGPR partial accumulator.
  if (mfma.acc_cd != 0)
    return false;

  const uint16_t dst_base = static_cast<uint16_t>(mfma.vdst);
  const uint16_t src0_base = static_cast<uint16_t>(mfma.src0 - 256);
  const uint16_t src1_base = static_cast<uint16_t>(mfma.src1 - 256);
  return ranges_overlap(dst_base, lowering.dst_regs, src0_base, lowering.wide_src_regs) ||
         ranges_overlap(dst_base, lowering.dst_regs, src1_base, lowering.wide_src_regs);
}

std::vector<uint32_t> lower_wide_k_mfma_f16_cdna4_to_cdna3(const Instruction &inst,
                                                           const LivenessAnalysis &liveness,
                                                           WideKMfmaShape shape) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop3pMfmaMachineInst))
    return {};

  cdna4::Vop3pMfmaMachineInst mfma{};
  std::memcpy(&mfma, raw, sizeof(mfma));
  const WideKMfmaLowering lowering = lowering_for_shape(shape);
  if (lowering.narrow_op == 0)
    return {};

  // CDNA4's wide-K F16 forms double the K dimension by doubling each contiguous
  // A/B VGPR source window. CDNA3 has the same output layout for the narrower-K
  // forms, so the lowering emits two narrow MFMAs over the low and high halves
  // of the source windows:
  //
  //   partial = mfma_narrow(A[0:1], B[0:1], C)
  //   D       = mfma_narrow(A[2:3], B[2:3], partial)
  //
  // When D is an AccVGPR destination, `partial` is the final destination and the
  // second instruction reads it back through src2. CDNA3 resolves src2 encodings
  // 256-511 to the AccVGPR bank when acc_cd=1. When D is an ordinary VGPR
  // destination that overlaps either full A/B source window, the first MFMA must
  // instead write a dead VGPR run; otherwise it could clobber source registers
  // that the second MFMA has not read yet.
  //
  // NYI: non-default cbsz/abid/blgp/acc modifiers need validation against the
  // two-instruction expansion before this can preserve them safely.
  if (mfma.cbsz != 0 || mfma.abid != 0 || mfma.blgp != 0 || mfma.acc != 0)
    return {};
  // SRC0/SRC1 are OPR_SRC_VGPR_OR_ACCVGPR operands. The ISA defines the CDNA4
  // wide forms as 128-bit source windows and the CDNA3 narrow forms as 64-bit
  // source windows; the operand value is the base of that contiguous window, and
  // 64-bit-or-wider VGPR/AccVGPR operands are even-aligned by the ISA. Since this
  // rule rejects ACC-selected A/B sources above and assumes the original CDNA4
  // instruction is well-formed, the split can use src and src + narrow_src_regs
  // directly without a packing step.
  // The original accumulator is only consumed by the first narrow MFMA; the
  // second consumes the partial accumulator produced by the first. Forward src2
  // unchanged and rely on the original CDNA4 instruction being well-formed.
  // VDST has the same operand size in the CDNA4 wide form and the emitted CDNA3
  // narrow form. Forward the original destination base and acc_cd; destination
  // window validity is part of the source instruction's ISA contract.

  const bool needs_scratch = wide_mfma_needs_partial_accum_scratch(mfma, lowering);
  uint8_t partial_vdst = static_cast<uint8_t>(mfma.vdst);
  uint16_t partial_src2 = static_cast<uint16_t>(256 + mfma.vdst);
  if (needs_scratch) {
    std::optional<uint16_t> scratch =
        find_free_vgpr_run_below(liveness, inst, lowering.dst_regs, 0, kSemanticScratchVgprLimit);
    // NYI: if no dead VGPR run exists, the general solution is to spill a live
    // VGPR range and use it for the partial accumulator. That waits on spill
    // manager integration, so reject for now rather than clobbering live inputs.
    if (!scratch)
      return {};
    partial_vdst = static_cast<uint8_t>(*scratch);
    partial_src2 = static_cast<uint16_t>(256 + partial_vdst);
  }

  std::vector<uint32_t> words;
  if (needs_scratch) {
    emit_cdna3_mfma_to_vgpr(words, lowering.narrow_op, mfma, partial_vdst,
                            static_cast<uint16_t>(mfma.src0), static_cast<uint16_t>(mfma.src1),
                            static_cast<uint16_t>(mfma.src2));
  } else {
    emit_cdna3_mfma(words, lowering.narrow_op, mfma, static_cast<uint16_t>(mfma.src0),
                    static_cast<uint16_t>(mfma.src1), static_cast<uint16_t>(mfma.src2));
  }
  emit_cdna3_mfma(words, lowering.narrow_op, mfma,
                  static_cast<uint16_t>(mfma.src0 + lowering.narrow_src_regs),
                  static_cast<uint16_t>(mfma.src1 + lowering.narrow_src_regs), partial_src2);
  return words;
}

// -----------------------------------------------------------------------------
// DS transpose expansions.
// -----------------------------------------------------------------------------

void emit_cdna3_b16_transpose_halfword(std::vector<uint32_t> &words, uint8_t halfword_dst,
                                       uint8_t gather_tmp, uint8_t lane_byte_addr, uint8_t raw_lo,
                                       uint8_t raw_hi, uint8_t halfword_selector) {
  emit_cdna3_ds(words, kCdna3DsOpBpermuteB32, halfword_dst, lane_byte_addr, raw_lo);
  emit_cdna3_ds(words, kCdna3DsOpBpermuteB32, gather_tmp, lane_byte_addr, raw_hi);
  emit_cdna3_lgkm_wait(words);
  emit_cdna3_vop3(words, kCdna3OpPermB32, halfword_dst, vgpr_src(gather_tmp),
                  vgpr_src(halfword_dst), vgpr_src(halfword_selector));
}

void emit_cdna3_pack_low_b16_pair(std::vector<uint32_t> &words, uint8_t dst, uint8_t halfword_lo,
                                  uint8_t halfword_hi, uint8_t shifted_hi_tmp, uint8_t mask_tmp) {
  // Pack the low 16 bits of two VGPR values into a raw 32-bit payload. This is
  // not an FP16 conversion; `v_pack_b32_f16` can canonicalize/change FP
  // payloads, so build the packed destination with integer operations:
  //
  //   dst = (halfword_hi[15:0] << 16) | halfword_lo[15:0]
  //
  // CDNA3 cannot inline 0xffff as a VALU source, so synthesize the mask from
  // -1 >> 16. The helper may clobber halfword_lo, shifted_hi_tmp, and mask_tmp.
  emit_cdna3_vop3(words, kCdna3OpMovB32, mask_tmp, kInlineConstNeg1);
  emit_cdna3_vop3(words, kCdna3OpLshrrevB32, mask_tmp, scalar_positive_inline_u32(16),
                  vgpr_src(mask_tmp));
  emit_cdna3_vop3(words, kCdna3OpAndB32, halfword_lo, vgpr_src(mask_tmp), vgpr_src(halfword_lo));
  emit_cdna3_vop3(words, kCdna3OpAndB32, shifted_hi_tmp, vgpr_src(mask_tmp), vgpr_src(halfword_hi));
  emit_cdna3_vop3(words, kCdna3OpLshlrevB32, shifted_hi_tmp, scalar_positive_inline_u32(16),
                  vgpr_src(shifted_hi_tmp));
  emit_cdna3_vop3(words, kCdna3OpOrB32, dst, vgpr_src(halfword_lo), vgpr_src(shifted_hi_tmp));
}

std::vector<uint32_t> lower_ds_read_b64_tr_b16_cdna4_to_cdna3(const Instruction &inst,
                                                              const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::DsMachineInst))
    return {};

  cdna4::DsMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.gds != 0)
    // CDNA4 DS encodings can select GDS, but CDNA3 reserves GDS=1 for this
    // instruction. Do not translate that variant into an illegal CDNA3 encoding.
    return {};
  if (src.acc != 0)
    // DS ACC redirects VDST into the AccVGPR file. This lowering rebuilds the
    // result with ordinary VALU writes, so AccVGPR destinations need a separate
    // implementation before they can be translated safely.
    return {};

  const uint8_t vdst = static_cast<uint8_t>(src.vdst);
  const uint8_t addr = static_cast<uint8_t>(src.addr);
  // VDST is a 64-bit destination, so it names a contiguous two-register pair.
  // Pair validity is part of the source instruction's ISA contract.

  constexpr uint16_t kScratchCount = 8;
  uint16_t scratch_start =
      std::max<uint16_t>(static_cast<uint16_t>(vdst + 2), static_cast<uint16_t>(addr + 1));
  if ((scratch_start & 1) != 0)
    ++scratch_start;

  std::optional<uint16_t> scratch;
  // The pack helper wants several even/odd register relationships to stay
  // simple, so search only for an even-aligned run. If liveness first reports
  // an odd free run, advance past it and keep looking instead of accepting a
  // scratch layout that would make the emitted DS transpose sequence harder to
  // reason about.
  for (uint16_t search = scratch_start; search + kScratchCount <= kSemanticScratchVgprLimit;) {
    auto candidate =
        find_free_vgpr_run_below(liveness, inst, kScratchCount, search, kSemanticScratchVgprLimit);
    if (!candidate)
      break;
    if ((*candidate & 1) == 0) {
      scratch = candidate;
      break;
    }
    search = static_cast<uint16_t>(*candidate + 1);
    if ((search & 1) != 0)
      ++search;
  }
  if (!scratch)
    return {};

  const uint8_t raw_lo = static_cast<uint8_t>(*scratch + 0);
  const uint8_t raw_hi = static_cast<uint8_t>(*scratch + 1);
  const uint8_t lane_base = static_cast<uint8_t>(*scratch + 2);
  const uint8_t halfword_selector = static_cast<uint8_t>(*scratch + 3);
  const uint8_t tmp = static_cast<uint8_t>(*scratch + 4);
  const uint8_t halfword_lo = static_cast<uint8_t>(*scratch + 5);
  const uint8_t halfword_hi = static_cast<uint8_t>(*scratch + 6);
  const uint8_t gather_tmp = static_cast<uint8_t>(*scratch + 7);

  std::vector<uint32_t> words;

  // CDNA4 ds_read_b64_tr_b16 loads four transposed halfwords per lane from the
  // LDS read footprint. CDNA3 only has the non-transposed ds_read_b64, so the
  // expansion reconstructs the transposed result through the DS crossbar:
  //
  //   raw_lo:raw_hi = ds_read_b64(addr, offset0, offset1)
  //   lane          = mbcnt(exec)
  //   selector      = ((lane & 3) * 2) | (((lane & 3) * 2 + 1) << 8)
  //   lane_base     = ((lane & 0x30) << 2) | (lane & 0x0c)
  //   h0            = halfword_at(lane_base +  0, selector, raw_lo, raw_hi)
  //   h1            = halfword_at(lane_base + 16, selector, raw_lo, raw_hi)
  //   h2            = halfword_at(lane_base + 32, selector, raw_lo, raw_hi)
  //   h3            = halfword_at(lane_base + 48, selector, raw_lo, raw_hi)
  //   vdst          = pack_u16_pair(h0, h1)
  //   vdst+1        = pack_u16_pair(h2, h3)
  //
  // halfword_at() is emitted as two ds_bpermute_b32 operations followed by
  // v_perm_b32 so the selector can choose the required halfword from either
  // 32-bit half of the original b64 read. pack_u16_pair() is deliberately
  // integer mask/shift/or instead of v_pack_b32_f16 because this DS op moves
  // raw 16-bit payloads, not FP16 values.
  emit_cdna3_ds(words, kCdna3DsOpReadB64, raw_lo, addr, 0, 0, src.offset0, src.offset1);
  emit_cdna3_lgkm_wait(words);

  emit_cdna3_vop3(words, kCdna3OpMbcntLoU32B32, tmp, kInlineConstNeg1, kInlineConst0);
  emit_cdna3_vop3(words, kCdna3OpMbcntHiU32B32, tmp, kInlineConstNeg1, vgpr_src(tmp));

  // Build the ds_bpermute byte addresses that recover each halfword in the
  // transposed 4x16-lane pattern, then pack pairs of halfwords back into the
  // two 32-bit destination registers produced by ds_read_b64_tr_b16.
  emit_cdna3_vop3(words, kCdna3OpAndB32, halfword_selector, scalar_positive_inline_u32(3),
                  vgpr_src(tmp));
  emit_cdna3_vop3(words, kCdna3OpLshlrevB32, halfword_selector, scalar_positive_inline_u32(1),
                  vgpr_src(halfword_selector));

  emit_cdna3_vop3(words, kCdna3OpAndB32, lane_base, scalar_positive_inline_u32(0x30),
                  vgpr_src(tmp));
  emit_cdna3_vop3(words, kCdna3OpLshlrevB32, lane_base, scalar_positive_inline_u32(2),
                  vgpr_src(lane_base));
  emit_cdna3_vop3(words, kCdna3OpAndB32, tmp, scalar_positive_inline_u32(0x0c), vgpr_src(tmp));
  emit_cdna3_vop3(words, kCdna3OpOrB32, lane_base, vgpr_src(lane_base), vgpr_src(tmp));

  emit_cdna3_vop3(words, kCdna3OpAddU32, tmp, scalar_positive_inline_u32(1),
                  vgpr_src(halfword_selector));
  emit_cdna3_vop3(words, kCdna3OpLshlrevB32, tmp, scalar_positive_inline_u32(8), vgpr_src(tmp));
  emit_cdna3_vop3(words, kCdna3OpOrB32, halfword_selector, vgpr_src(halfword_selector),
                  vgpr_src(tmp));

  emit_cdna3_b16_transpose_halfword(words, halfword_lo, gather_tmp, lane_base, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_vop3(words, kCdna3OpAddU32, tmp, scalar_positive_inline_u32(16), vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, halfword_hi, gather_tmp, tmp, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_pack_low_b16_pair(words, vdst, halfword_lo, halfword_hi, tmp, gather_tmp);

  emit_cdna3_vop3(words, kCdna3OpAddU32, tmp, scalar_positive_inline_u32(32), vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, halfword_lo, gather_tmp, tmp, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_vop3(words, kCdna3OpAddU32, tmp, scalar_positive_inline_u32(48), vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, halfword_hi, gather_tmp, tmp, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_pack_low_b16_pair(words, static_cast<uint8_t>(vdst + 1), halfword_lo, halfword_hi, tmp,
                               gather_tmp);

  return words;
}

std::vector<uint32_t> expand_v_bitop3_b16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                         uint64_t, const LivenessAnalysis &liveness,
                                                         const LaneLayout *, const LaneLayout *) {
  // The rule table only routes V_BITOP3_B16 here, so use the generated
  // instruction type directly instead of re-decoding ordinary operands.
  return lower_cdna4_bitop3_to_cdna3(static_cast<const cdna4::VBitop3B16Vop3 &>(inst), liveness,
                                     true);
}

std::vector<uint32_t> expand_v_bitop3_b32_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                         uint64_t, const LivenessAnalysis &liveness,
                                                         const LaneLayout *, const LaneLayout *) {
  // The rule table only routes V_BITOP3_B32 here, so use the generated
  // instruction type directly instead of re-decoding ordinary operands.
  return lower_cdna4_bitop3_to_cdna3(static_cast<const cdna4::VBitop3B32Vop3 &>(inst), liveness,
                                     false);
}

std::vector<uint32_t> expand_ds_read_b64_tr_b16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                               uint64_t,
                                                               const LivenessAnalysis &liveness,
                                                               const LaneLayout *,
                                                               const LaneLayout *) {
  return lower_ds_read_b64_tr_b16_cdna4_to_cdna3(inst, liveness);
}

std::vector<uint32_t> expand_mfma_f32_16x16x32_f16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                                  uint64_t,
                                                                  const LivenessAnalysis &liveness,
                                                                  const LaneLayout *,
                                                                  const LaneLayout *) {
  return lower_wide_k_mfma_f16_cdna4_to_cdna3(inst, liveness, WideKMfmaShape::F32_16x16x32_F16);
}

std::vector<uint32_t> expand_mfma_f32_32x32x16_f16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                                  uint64_t,
                                                                  const LivenessAnalysis &liveness,
                                                                  const LaneLayout *,
                                                                  const LaneLayout *) {
  return lower_wide_k_mfma_f16_cdna4_to_cdna3(inst, liveness, WideKMfmaShape::F32_32x32x16_F16);
}

constexpr uint16_t kEncVop3 = 0x1A4;
constexpr uint16_t kEncVop3pMfma = 0x1A7;
constexpr uint16_t kEncDsReadB64TrB16 = 0x1B3;

constexpr uint16_t kCdna4Op_v_mfma_f32_16x16x32_f16 = 84;
constexpr uint16_t kCdna4Op_v_mfma_f32_32x32x16_f16 = 85;
constexpr uint16_t kCdna4Op_v_bitop3_b16 = 563;
constexpr uint16_t kCdna4Op_v_bitop3_b32 = 564;
constexpr uint16_t kCdna4Op_ds_read_b64_tr_b16 = 227;

// Table MUST be sorted by (src_encoding_id, src_opcode) for binary search.
const TranslationRule kExpandRules_cdna4_to_cdna3[] = {
    {kEncVop3, kCdna4Op_v_bitop3_b16, RuleAction::Expand, 0, 0, nullptr,
     expand_v_bitop3_b16_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop3, kCdna4Op_v_bitop3_b32, RuleAction::Expand, 0, 0, nullptr,
     expand_v_bitop3_b32_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop3pMfma, kCdna4Op_v_mfma_f32_16x16x32_f16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_16x16x32_f16_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop3pMfma, kCdna4Op_v_mfma_f32_32x32x16_f16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_32x32x16_f16_cdna4_to_cdna3, nullptr, nullptr},
    {kEncDsReadB64TrB16, kCdna4Op_ds_read_b64_tr_b16, RuleAction::Expand, 0, 0, nullptr,
     expand_ds_read_b64_tr_b16_cdna4_to_cdna3, nullptr, nullptr},
};

} // namespace

std::span<const TranslationRule> semantic_expand_rules_cdna4_to_cdna3() {
  return std::span<const TranslationRule>(kExpandRules_cdna4_to_cdna3);
}

} // namespace rocjitsu
