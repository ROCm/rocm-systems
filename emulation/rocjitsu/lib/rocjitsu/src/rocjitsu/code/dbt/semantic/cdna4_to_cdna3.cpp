// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_cdna3.cpp
/// @brief CDNA4-to-CDNA3 handwritten semantic expansion rules.

#include "rocjitsu/code/dbt/semantic/rules.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/vop3.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <string>
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
build_cdna3_vop3p_mfma(uint16_t op, const cdna4::Vop3pMfmaMachineInst &src, uint8_t vdst,
                       uint8_t acc_cd, uint16_t src0, uint16_t src1, uint16_t src2) {
  cdna3::Vop3pMfmaMachineInst dst{};
  dst.encoding = cdna3::encoding::kVop3pMfma;
  dst.op = op & 0x7F;
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

[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_ds(uint16_t op, uint8_t vdst, uint8_t addr,
                                                           uint8_t data0 = 0, uint8_t data1 = 0,
                                                           uint8_t offset0 = 0,
                                                           uint8_t offset1 = 0) {
  cdna3::DsMachineInst dst{};
  dst.encoding = 0x36;
  dst.op = op & 0xFF;
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

[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_mubuf(const cdna4::MubufMachineInst &src,
                                                              uint16_t op, uint8_t vdata) {
  cdna3::MubufMachineInst dst{};
  dst.encoding = 0x38;
  dst.op = op & 0x7F;
  dst.offset = src.offset;
  dst.offen = src.offen;
  dst.idxen = src.idxen;
  dst.sc0 = src.sc0;
  dst.sc1 = src.sc1;
  dst.lds = 0;
  dst.nt = src.nt;
  dst.vaddr = src.vaddr;
  dst.vdata = vdata;
  dst.srsrc = src.srsrc;
  dst.acc = 0;
  dst.soffset = src.soffset;

  uint32_t words[2]{};
  std::memcpy(words, &dst, sizeof(dst));
  return {words[0], words[1]};
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_s_mov_b32_lit(uint8_t sdst,
                                                                          uint32_t literal) {
  return {pack_sop1(cdna3::kSMovB32, sdst, 0xFF), literal};
}

[[nodiscard]] constexpr uint32_t build_s_mov_b64(uint8_t sdst, uint16_t ssrc0) {
  return pack_sop1(cdna3::kSMovB64, sdst, ssrc0);
}

constexpr uint8_t kExecLo = 126;
constexpr uint16_t kInlineConst0 = 128;
constexpr uint16_t kInlineConstNeg1 = 193;
constexpr uint16_t kM0 = 124;

constexpr uint16_t kCdnaWaitcntLgkmcnt0 = 0xC07F;
constexpr uint16_t kCdnaWaitcntAll0 = 0x0000;

void emit_cdna3_vop3(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint16_t src0,
                     uint16_t src1 = 0, uint16_t src2 = 0) {
  auto [w0, w1] = build_cdna3_vop3(op, vdst, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_ds(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint8_t addr,
                   uint8_t data0 = 0, uint8_t data1 = 0, uint8_t offset0 = 0, uint8_t offset1 = 0) {
  auto [w0, w1] = build_cdna3_ds(op, vdst, addr, data0, data1, offset0, offset1);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_lgkm_wait(std::vector<uint32_t> &words) {
  // GFX9/CDNA s_waitcnt encodes "lgkmcnt(0)" as lgkm=0 while leaving VM/EXP at
  // their no-wait maxima. The DS read/bpermute sequences below require the
  // loaded/permuted data before issuing dependent VALU instructions.
  words.push_back(pack_sopp(cdna3::kSWaitcntSopp, kCdnaWaitcntLgkmcnt0));
}

void emit_cdna3_wait_all(std::vector<uint32_t> &words) {
  // The load-to-LDS expansion below consumes a just-issued VMEM load through a
  // following DS write.  Waiting all counters is stronger than the hardware
  // `vmcnt(0)` dependency we strictly need, but it keeps this first lowering
  // conservative and matches the monolithic CDNA waitcnt encoding.
  words.push_back(pack_sopp(cdna3::kSWaitcntSopp, kCdnaWaitcntAll0));
}

[[nodiscard]] constexpr uint16_t vgpr_src(uint8_t reg) { return static_cast<uint16_t>(256 + reg); }

void emit_cdna3_mubuf(std::vector<uint32_t> &words, const cdna4::MubufMachineInst &src, uint16_t op,
                      uint8_t vdata) {
  auto [w0, w1] = build_cdna3_mubuf(src, op, vdata);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_s_mov_b32_lit(std::vector<uint32_t> &words, uint8_t sdst, uint32_t literal) {
  auto [w0, w1] = build_s_mov_b32_lit(sdst, literal);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_s_mov_b64(std::vector<uint32_t> &words, uint8_t sdst, uint16_t ssrc0) {
  words.push_back(build_s_mov_b64(sdst, ssrc0));
}

void emit_cdna3_exec_mask(std::vector<uint32_t> &words, uint64_t mask) {
  // TODO: Optimize the common all-lanes case by emitting one s_mov_b64 -1
  // instead of two literal s_mov_b32 instructions.
  emit_s_mov_b32_lit(words, kExecLo, static_cast<uint32_t>(mask));
  emit_s_mov_b32_lit(words, kExecLo + 1, static_cast<uint32_t>(mask >> 32));
}

void emit_cdna3_mfma(std::vector<uint32_t> &words, uint16_t op,
                     const cdna4::Vop3pMfmaMachineInst &src, uint16_t src0, uint16_t src1,
                     uint16_t src2) {
  auto [w0, w1] = build_cdna3_vop3p_mfma(op, src, static_cast<uint8_t>(src.vdst),
                                         static_cast<uint8_t>(src.acc_cd), src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_mfma_to_vgpr(std::vector<uint32_t> &words, uint16_t op,
                             const cdna4::Vop3pMfmaMachineInst &src, uint8_t vdst, uint16_t src0,
                             uint16_t src1, uint16_t src2) {
  auto [w0, w1] = build_cdna3_vop3p_mfma(op, src, vdst, 0, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
}

[[nodiscard]] ExpandResult failed_existing_expand_rule(const Instruction &inst,
                                                       const std::string &problem,
                                                       std::vector<std::string> work = {}) {
  if (work.empty()) {
    work = {"Check this rule's operand restrictions and scratch allocation.",
            "Implement the unsupported form or add a narrower legalization entry."};
  }
  return ExpandResult::failed(std::string(inst.mnemonic()) + ": " + problem, std::move(work));
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
ExpandResult lower_cdna4_bitop3_to_cdna3(const Bitop3Inst &inst, const LivenessAnalysis &liveness,
                                         TranslationContext &context, bool is_b16) {
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
    return failed_existing_expand_rule(
        inst, "B16 form has non-zero op_sel",
        {
            "Implement OP_SEL handling for canonical vdst read-modify-write lowering.",
        });
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
    auto scratch = liveness.find_free_run(&inst, scratch_count);
    if (!scratch)
      return failed_existing_expand_rule(
          inst, "No free VGPR run for temporary bitop temporaries",
          {"Check scratch-pressure assumptions and test with higher debug_min_free_vgpr.",
           "Add spill-backed lowering for high-pressure callers."});

    acc = static_cast<uint8_t>(*scratch);
    if (needs_term_temp)
      term = static_cast<uint8_t>(*scratch + 1);
    // Scratch may be above the original ordinary VGPR allocation. Feed that
    // exact window back to descriptor translation so the patched kernel can
    // legally address the generated temporaries and, for CDNA targets with
    // AccVGPRs, so ACCUM_OFFSET can move above the ordinary scratch window.
    context.require_vgprs(static_cast<uint32_t>(*scratch) + scratch_count);
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
    emit_cdna3_vop3(words, cdna3::kVMovB32Vop3, dst, src0);
  };
  auto emit_and = [&](uint8_t dst, uint16_t src0, uint16_t src1) {
    emit_cdna3_vop3(words, cdna3::kVAndB32Vop3, dst, src0, src1);
  };
  auto emit_xor = [&](uint8_t dst, uint16_t src0, uint16_t src1) {
    emit_cdna3_vop3(words, cdna3::kVXorB32Vop3, dst, src0, src1);
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
    emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, acc, shift16, vgpr_src(acc));
    emit_cdna3_vop3(words, cdna3::kVLshrrevB32Vop3, acc, shift16, vgpr_src(acc));
  }

  if (acc != vdst)
    emit_mov(vdst, vgpr_src(acc));

  return ExpandResult::success(std::move(words));
}

// -----------------------------------------------------------------------------
// V_PERMLANE*_SWAP expansions.
// -----------------------------------------------------------------------------

ExpandResult lower_permlane32_swap_b32_cdna4_to_cdna3(const Instruction &inst,
                                                      const LivenessAnalysis &liveness,
                                                      TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop1MachineInst))
    return failed_existing_expand_rule(
        inst, "No decodable VOP1 instruction encoding",
        {"Decode the source VOP1 instruction before applying the swap lowering."});

  cdna4::Vop1MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.src0 < 256 || src.src0 > 511)
    // Only plain VGPR sources have been validated for this first lowering.
    // Scalar sources, literals, DPP, and SDWA encodings need separate operand
    // handling before they can be lowered safely.
    return failed_existing_expand_rule(
        inst, "swap source is not plain VGPR form",
        {"Add lowerings for scalar/LITERAL/DPP/SDWA source operands before enabling this form.",
         "Keep this rule limited to VOP1/e32 VGPR swap until operand decoding is complete."});

  const uint8_t vdst = static_cast<uint8_t>(src.vdst);
  const uint8_t vsrc = static_cast<uint8_t>(src.src0 - 256);

  // The generated CDNA4 metadata models this swap instruction as two outputs,
  // but semantically both operands are also read before either is overwritten.
  // Start VGPR scratch after both architectural operands so liveness cannot
  // hand us vdst/vsrc merely because their old values are not represented as
  // ordinary source uses.
  const uint16_t scratch_start =
      static_cast<uint16_t>(std::max<uint16_t>(vdst, vsrc) + uint16_t{1});
  auto scratch = liveness.find_free_run(&inst, 3, scratch_start);
  if (!scratch)
    return failed_existing_expand_rule(
        inst, "No free VGPR scratch window for swap temporaries",
        {"Provide one SGPR pair for EXEC save/restore and three temporary VGPRs for lane address "
         "and pre-write swap values."});

  const uint8_t lane = static_cast<uint8_t>(*scratch);
  const uint8_t partner_addr = static_cast<uint8_t>(*scratch + 1);
  const uint8_t from_dst_high = static_cast<uint8_t>(*scratch + 2);
  const uint8_t from_src_low = lane;
  // Use a descriptor-backed SGPR pair for EXEC save/restore. The CDNA4 metadata
  // does not model all implicit SGPR buffer-resource lifetimes, so a
  // liveness-selected guest pair can corrupt later memory operations.
  const uint8_t saved_exec = static_cast<uint8_t>((context.num_sgprs + 1u) & ~1u);

  context.require_sgprs(static_cast<uint32_t>(saved_exec) + 2);
  context.require_vgprs(static_cast<uint32_t>(*scratch) + 3);

  std::vector<uint32_t> words;

  // CDNA4 v_permlane32_swap_b32 ignores EXEC and swaps rows 2/3 of VDST with
  // rows 0/1 of SRC0:
  //
  //   SRC0 lanes  0..31 = old VDST lanes 32..63
  //   VDST lanes 32..63 = old SRC0 lanes  0..31
  //
  // CDNA3 has no row-swap instruction.  Run the data-gather portion with EXEC
  // forced to all lanes so mbcnt() produces physical lane IDs and ds_bpermute
  // can read both source half-waves.  Only after both old values are captured do
  // we narrow EXEC to the low and high halves for the two destructive writes.
  emit_s_mov_b64(words, saved_exec, kExecLo);
  emit_cdna3_exec_mask(words, UINT64_MAX);

  emit_cdna3_vop3(words, cdna3::kVMbcntLoU32B32Vop3, lane, kInlineConstNeg1, kInlineConst0);
  emit_cdna3_vop3(words, cdna3::kVMbcntHiU32B32Vop3, lane, kInlineConstNeg1, vgpr_src(lane));
  emit_cdna3_vop3(words, cdna3::kVXorB32Vop3, partner_addr, scalar_positive_inline_u32(32),
                  vgpr_src(lane));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, partner_addr, scalar_positive_inline_u32(2),
                  vgpr_src(partner_addr));

  emit_cdna3_ds(words, cdna3::kDsBpermuteB32Ds, from_dst_high, partner_addr, vdst);
  emit_cdna3_ds(words, cdna3::kDsBpermuteB32Ds, from_src_low, partner_addr, vsrc);
  emit_cdna3_lgkm_wait(words);

  emit_cdna3_exec_mask(words, 0x00000000FFFFFFFFull);
  emit_cdna3_vop3(words, cdna3::kVMovB32Vop3, vsrc, vgpr_src(from_dst_high));

  emit_cdna3_exec_mask(words, 0xFFFFFFFF00000000ull);
  emit_cdna3_vop3(words, cdna3::kVMovB32Vop3, vdst, vgpr_src(from_src_low));

  emit_s_mov_b64(words, kExecLo, saved_exec);
  return ExpandResult::success(std::move(words));
}

ExpandResult lower_cvt_pk_f16_f32_cdna4_to_cdna3(const Instruction &inst,
                                                 const LivenessAnalysis &liveness,
                                                 TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop3MachineInst))
    return failed_existing_expand_rule(inst, "No decodable VOP3 instruction encoding",
                                       {"Decode the source VOP3 instruction before lowering."});

  cdna4::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || src.op_sel != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0)
    // Source modifiers would have to be replayed on the two scalar conversions
    // before packing, so reject them until this form is implemented and tested.
    return failed_existing_expand_rule(
        inst, "Unsupported modifiers on packed F16 conversion (abs/op_sel/clamp/omod/neg)",
        {"Implement modifier handling when source modifiers are present."});
  if (src.src0 < 256 || src.src0 > 511 || src.src1 < 256 || src.src1 > 511)
    // Keep this first lowering to ordinary VGPR sources. The CDNA3 scalar
    // v_cvt_f16_f32 instruction can accept other operand classes, but the
    // scratch/alias reasoning below has only been validated for VGPR operands.
    return failed_existing_expand_rule(
        inst, "Packed F16 conversion operands are not both VGPR registers",
        {"Add operand-class-specific lowering for scalar, literal, DPP, or SDWA forms."});

  const uint8_t vdst = static_cast<uint8_t>(src.vdst);
  const uint8_t src0_vgpr = static_cast<uint8_t>(src.src0 - 256);
  const uint8_t src1_vgpr = static_cast<uint8_t>(src.src1 - 256);
  const uint16_t scratch_start =
      static_cast<uint16_t>(std::max<uint16_t>({vdst, src0_vgpr, src1_vgpr}) + uint16_t{1});
  auto scratch = liveness.find_free_run(&inst, 2, scratch_start);
  if (!scratch)
    return failed_existing_expand_rule(
        inst, "No free VGPR scratch window for packed conversion temporaries",
        {"Provide two temporary VGPRs so destination/source aliases are safe."});

  const uint8_t lo = static_cast<uint8_t>(*scratch);
  const uint8_t hi = static_cast<uint8_t>(*scratch + 1);
  context.require_vgprs(static_cast<uint32_t>(*scratch) + 2);

  std::vector<uint32_t> words;
  // CDNA4 v_cvt_pk_f16_f32 packs two f32-to-f16 conversions into one VGPR:
  //
  //   VDST[15:0]  = f32_to_f16(SRC0)
  //   VDST[31:16] = f32_to_f16(SRC1)
  //
  // CDNA3 lacks this exact packed instruction but has the scalar
  // v_cvt_f16_f32 with the same conversion helper used by the generated VM
  // model. Convert both sources before writing VDST so forms such as
  // `v_cvt_pk_f16_f32 v31, v31, v32` do not clobber a still-needed source.
  emit_cdna3_vop3(words, cdna3::kVCvtF16F32Vop3, lo, static_cast<uint16_t>(src.src0));
  emit_cdna3_vop3(words, cdna3::kVCvtF16F32Vop3, hi, static_cast<uint16_t>(src.src1));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, hi, scalar_positive_inline_u32(16), vgpr_src(hi));
  emit_cdna3_vop3(words, cdna3::kVOrB32Vop3, vdst, vgpr_src(lo), vgpr_src(hi));
  return ExpandResult::success(std::move(words));
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
    return {shape, cdna3::kVMfmaF3216x16x16F16Vop3pMfma, 4, 4, 2};
  case WideKMfmaShape::F32_32x32x16_F16:
    return {shape, cdna3::kVMfmaF3232x32x8F16Vop3pMfma, 16, 4, 2};
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

ExpandResult lower_wide_k_mfma_f16_cdna4_to_cdna3(const Instruction &inst,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context,
                                                  WideKMfmaShape shape) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop3pMfmaMachineInst))
    return failed_existing_expand_rule(
        inst, "Could not decode wide-K MFMA instruction encoding",
        {"Decode the source instruction before lowering this MFMA form."});

  cdna4::Vop3pMfmaMachineInst mfma{};
  std::memcpy(&mfma, raw, sizeof(mfma));
  const WideKMfmaLowering lowering = lowering_for_shape(shape);
  if (lowering.narrow_op == 0)
    return failed_existing_expand_rule(
        inst, "Unsupported wide-K MFMA shape",
        {"Implement a lowering path for this wide-K MFMA shape before enabling this path."});

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
    return failed_existing_expand_rule(
        inst, "MFMA wide-K lowering only supports default cbsz/abid/blgp/acc modifiers",
        {"Validate and implement non-default MFMA modifiers before enabling this form."});
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
    std::optional<uint16_t> scratch = liveness.find_free_run(&inst, lowering.dst_regs);
    // NYI: if no dead VGPR run exists, the general solution is to spill a live
    // VGPR range and use it for the partial accumulator. That waits on spill
    // manager integration, so reject for now rather than clobbering live inputs.
    if (!scratch)
      return failed_existing_expand_rule(inst,
                                         "No free VGPR window for partial accumulator scratch",
                                         {"Add spill-backed lowering for partial accumulator "
                                          "when live inputs consume scratch registers."});
    partial_vdst = static_cast<uint8_t>(*scratch);
    partial_src2 = static_cast<uint16_t>(256 + partial_vdst);
    // The partial accumulator is an ordinary VGPR tuple emitted only by this
    // lowering. If liveness selects registers outside the descriptor's original
    // ordinary VGPR count, descriptor feedback must grow the allocation before
    // the translated code object is launched.
    context.require_vgprs(static_cast<uint32_t>(*scratch) + lowering.dst_regs);
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
  return ExpandResult::success(std::move(words));
}

// -----------------------------------------------------------------------------
// DS transpose expansions.
// -----------------------------------------------------------------------------

void emit_cdna3_b16_transpose_halfword(std::vector<uint32_t> &words, uint8_t halfword_dst,
                                       uint8_t gather_tmp, uint8_t lane_byte_addr, uint8_t raw_lo,
                                       uint8_t raw_hi, uint8_t halfword_selector) {
  emit_cdna3_ds(words, cdna3::kDsBpermuteB32Ds, halfword_dst, lane_byte_addr, raw_lo);
  emit_cdna3_ds(words, cdna3::kDsBpermuteB32Ds, gather_tmp, lane_byte_addr, raw_hi);
  emit_cdna3_lgkm_wait(words);
  emit_cdna3_vop3(words, cdna3::kVPermB32Vop3, halfword_dst, vgpr_src(gather_tmp),
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
  emit_cdna3_vop3(words, cdna3::kVMovB32Vop3, mask_tmp, kInlineConstNeg1);
  emit_cdna3_vop3(words, cdna3::kVLshrrevB32Vop3, mask_tmp, scalar_positive_inline_u32(16),
                  vgpr_src(mask_tmp));
  emit_cdna3_vop3(words, cdna3::kVAndB32Vop3, halfword_lo, vgpr_src(mask_tmp),
                  vgpr_src(halfword_lo));
  emit_cdna3_vop3(words, cdna3::kVAndB32Vop3, shifted_hi_tmp, vgpr_src(mask_tmp),
                  vgpr_src(halfword_hi));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, shifted_hi_tmp, scalar_positive_inline_u32(16),
                  vgpr_src(shifted_hi_tmp));
  emit_cdna3_vop3(words, cdna3::kVOrB32Vop3, dst, vgpr_src(halfword_lo), vgpr_src(shifted_hi_tmp));
}

ExpandResult lower_ds_read_b64_tr_b16_cdna4_to_cdna3(const Instruction &inst,
                                                     const LivenessAnalysis &liveness,
                                                     TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::DsMachineInst))
    return failed_existing_expand_rule(inst, "No decodable DS instruction encoding",
                                       {"Decode the source DS instruction before lowering."});

  cdna4::DsMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.gds != 0)
    // CDNA4 DS encodings can select GDS, but CDNA3 reserves GDS=1 for this
    // instruction. Do not translate that variant into an illegal CDNA3 encoding.
    return failed_existing_expand_rule(
        inst, "DS transpose lower reads from GDS, which is illegal on CDNA3",
        {"Ensure the source DS read uses ordinary LDS addressing before enabling this rule."});
  if (src.acc != 0)
    // DS ACC redirects VDST into the AccVGPR file. This lowering rebuilds the
    // result with ordinary VALU writes, so AccVGPR destinations need a separate
    // implementation before they can be translated safely.
    return failed_existing_expand_rule(
        inst, "DS transpose lowering does not support destination ACCVGPR results",
        {"Define an AccVGPR destination path for transposed DS read lowering."});

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
  for (uint16_t search = scratch_start;;) {
    auto candidate = liveness.find_free_run(&inst, kScratchCount, search);
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
    return failed_existing_expand_rule(
        inst, "No even-aligned VGPR window found for transpose scratch",
        {"Provide enough aligned scratch registers or add a spill-backed "
         "transpose lowering."});

  const uint8_t raw_lo = static_cast<uint8_t>(*scratch + 0);
  const uint8_t raw_hi = static_cast<uint8_t>(*scratch + 1);
  const uint8_t lane_base = static_cast<uint8_t>(*scratch + 2);
  const uint8_t halfword_selector = static_cast<uint8_t>(*scratch + 3);
  const uint8_t tmp = static_cast<uint8_t>(*scratch + 4);
  const uint8_t halfword_lo = static_cast<uint8_t>(*scratch + 5);
  const uint8_t halfword_hi = static_cast<uint8_t>(*scratch + 6);
  const uint8_t gather_tmp = static_cast<uint8_t>(*scratch + 7);
  // Liveness may choose a scratch window beyond the guest kernel's original
  // VGPR allocation. The kernel descriptor must be grown to cover those
  // generated temporaries before the translated CDNA3 code can legally use them.
  context.require_vgprs(static_cast<uint32_t>(*scratch) + kScratchCount);

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
  emit_cdna3_ds(words, cdna3::kDsReadB64Ds, raw_lo, addr, 0, 0, src.offset0, src.offset1);
  emit_cdna3_lgkm_wait(words);

  emit_cdna3_vop3(words, cdna3::kVMbcntLoU32B32Vop3, tmp, kInlineConstNeg1, kInlineConst0);
  emit_cdna3_vop3(words, cdna3::kVMbcntHiU32B32Vop3, tmp, kInlineConstNeg1, vgpr_src(tmp));

  // Build the ds_bpermute byte addresses that recover each halfword in the
  // transposed 4x16-lane pattern, then pack pairs of halfwords back into the
  // two 32-bit destination registers produced by ds_read_b64_tr_b16.
  emit_cdna3_vop3(words, cdna3::kVAndB32Vop3, halfword_selector, scalar_positive_inline_u32(3),
                  vgpr_src(tmp));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, halfword_selector, scalar_positive_inline_u32(1),
                  vgpr_src(halfword_selector));

  emit_cdna3_vop3(words, cdna3::kVAndB32Vop3, lane_base, scalar_positive_inline_u32(0x30),
                  vgpr_src(tmp));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, lane_base, scalar_positive_inline_u32(2),
                  vgpr_src(lane_base));
  emit_cdna3_vop3(words, cdna3::kVAndB32Vop3, tmp, scalar_positive_inline_u32(0x0c), vgpr_src(tmp));
  emit_cdna3_vop3(words, cdna3::kVOrB32Vop3, lane_base, vgpr_src(lane_base), vgpr_src(tmp));

  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, tmp, scalar_positive_inline_u32(1),
                  vgpr_src(halfword_selector));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, tmp, scalar_positive_inline_u32(8),
                  vgpr_src(tmp));
  emit_cdna3_vop3(words, cdna3::kVOrB32Vop3, halfword_selector, vgpr_src(halfword_selector),
                  vgpr_src(tmp));

  // TODO: Gather both destination halfword pairs in one pass to reduce the
  // number of dependent ds_bpermute/VALU operations in this correctness-first
  // lowering.
  emit_cdna3_b16_transpose_halfword(words, halfword_lo, gather_tmp, lane_base, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, tmp, scalar_positive_inline_u32(16),
                  vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, halfword_hi, gather_tmp, tmp, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_pack_low_b16_pair(words, vdst, halfword_lo, halfword_hi, tmp, gather_tmp);

  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, tmp, scalar_positive_inline_u32(32),
                  vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, halfword_lo, gather_tmp, tmp, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, tmp, scalar_positive_inline_u32(48),
                  vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, halfword_hi, gather_tmp, tmp, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_pack_low_b16_pair(words, static_cast<uint8_t>(vdst + 1), halfword_lo, halfword_hi, tmp,
                               gather_tmp);

  return ExpandResult::success(std::move(words));
}

[[nodiscard]] ExpandResult expand_mubuf_load_to_lds_cdna4_to_cdna3(const Instruction &inst,
                                                                   const LivenessAnalysis &liveness,
                                                                   TranslationContext &context,
                                                                   uint8_t num_dwords) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::MubufMachineInst))
    return failed_existing_expand_rule(inst, "Could not decode source MUBUF fields",
                                       {"Decode the source MUBUF fields before lowering."});

  cdna4::MubufMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.lds == 0)
    return ExpandResult::not_handled();
  if (src.offen != 0 && src.idxen != 0) {
    return failed_existing_expand_rule(
        inst, "MUBUF LDS destination lowering does not support combined offen+idxen",
        {"Add an assembler/ISA test before enabling the combined offen+idxen LDS form.",
         "Verify how the MUBUF address tuple maps when VDATA is reserved by LDS DMA."});
  }
  if (src.acc != 0) {
    return failed_existing_expand_rule(
        inst, "MUBUF LDS destination lowering does not support ACC source modifier",
        {"Add tests for MUBUF LDS forms with ACC modifiers before enabling them.",
         "Keep the scratch VGPR load destination in ordinary VGPRs, then define how the "
         "source ACC bit should interact with an LDS architectural destination."});
  }
  if (src.vdata != 0) {
    return failed_existing_expand_rule(
        inst, "MUBUF LDS destination lowering only supports raw VDATA field = 0",
        {"Validate nonzero raw VDATA bits for MUBUF LDS-destination forms before enabling "
         "this lowering.",
         "Treat the LDS write as the architectural destination; do not assume raw VDATA is "
         "a real VGPR result when lds=1."});
  }
  if (num_dwords == 2) {
    return failed_existing_expand_rule(
        inst, "CDNA4 LDS-destination MUBUF docs do not list buffer_load_dwordx2",
        {"CDNA4 LDS-destination MUBUF documentation does not list buffer_load_dwordx2.",
         "Add an ISA-backed test before enabling a synthetic dwordx2 LDS lowering."});
  }

  const uint8_t data_count = num_dwords;
  // LDS-destination MUBUF loads use the vector address operands for the global
  // memory request and write the response to LDS at:
  //
  //   DWORD:  LDS[m0 + lane_id * 4 ...]
  //   DWORDX3/DWORDX4: LDS[m0 + lane_id * 16 ...]
  //
  // The CDNA3 fallback materializes the response in scratch VGPRs, waits for
  // it, computes the same per-lane LDS byte address, and stores the contiguous
  // payload with the matching DS write width.  dwordx3 writes the low 12 bytes
  // of the 16-byte lane slot and leaves the fourth dword untouched; dwordx4
  // writes the full slot.  `vdata` is deliberately scratch: in the LDS form the
  // architectural destination is LDS, not the source instruction's vdata
  // register tuple.
  //
  // Start scratch after the ordinary VGPRs, not after the existing AccVGPR
  // window. If the chosen scratch overlaps the source accumulator window,
  // require_vgprs() records the new ordinary VGPR end and descriptor
  // recomputation moves ACCUM_OFFSET above it while preserving num_agprs.
  const uint8_t vaddr_count = (src.offen && src.idxen) ? 2 : ((src.offen || src.idxen) ? 1 : 0);
  const uint16_t scratch_count = static_cast<uint16_t>(data_count + 1);
  const uint16_t scratch_start =
      static_cast<uint16_t>(std::max<uint32_t>(src.vaddr + vaddr_count, context.num_vgprs));
  auto scratch = liveness.find_free_run(&inst, scratch_count, scratch_start);
  if (!scratch) {
    return failed_existing_expand_rule(
        inst, "No scratch VGPR window found for temporary global-load destination and LDS address",
        {"Provide scratch VGPRs for the temporary global-load destination and LDS address.",
         "Add a spill-backed lowering for high-pressure inputs if no dead VGPR run exists."});
  }

  const uint8_t data = static_cast<uint8_t>(*scratch);
  const uint8_t lds_addr = static_cast<uint8_t>(*scratch + data_count);
  // Use a new descriptor-backed SGPR pair for EXEC save/restore instead of a
  // liveness-selected guest pair.  The LDS form hides important SGPR resource
  // uses inside buffer descriptors, and clobbering one of those descriptors for
  // a temporary save can redirect later global loads.
  const uint8_t saved_exec = static_cast<uint8_t>((context.num_sgprs + 1u) & ~1u);

  uint16_t load_op = 0;
  uint16_t ds_op = cdna3::kDsWriteB32Ds;
  uint16_t lane_stride_shift = 2;
  switch (num_dwords) {
  case 1:
    load_op = cdna3::kBufferLoadDwordMubuf;
    ds_op = cdna3::kDsWriteB32Ds;
    lane_stride_shift = 2;
    break;
  case 3:
    load_op = cdna3::kBufferLoadDwordx3Mubuf;
    ds_op = cdna3::kDsWriteB96Ds;
    lane_stride_shift = 4;
    break;
  case 4:
    load_op = cdna3::kBufferLoadDwordx4Mubuf;
    ds_op = cdna3::kDsWriteB128Ds;
    lane_stride_shift = 4;
    break;
  default:
    return failed_existing_expand_rule(inst,
                                       "Unsupported MUBUF LDS load width in CDNA4->CDNA3 lowering",
                                       {"Unsupported MUBUF LDS load width."});
  }
  context.require_sgprs(static_cast<uint32_t>(saved_exec) + 2);
  context.require_vgprs(static_cast<uint32_t>(*scratch) + scratch_count);

  std::vector<uint32_t> words;
  // MUBUF-to-LDS uses physical TID-in-wave for the LDS lane slot, not the
  // active-lane prefix.  Compute that address with EXEC forced to all lanes,
  // then restore the original EXEC before issuing the global load and DS write
  // so inactive lanes keep the same side-effect behavior as the source
  // instruction.
  emit_s_mov_b64(words, saved_exec, kExecLo);
  emit_cdna3_exec_mask(words, UINT64_MAX);
  emit_cdna3_vop3(words, cdna3::kVMbcntLoU32B32Vop3, lds_addr, kInlineConstNeg1, kInlineConst0);
  emit_cdna3_vop3(words, cdna3::kVMbcntHiU32B32Vop3, lds_addr, kInlineConstNeg1,
                  vgpr_src(lds_addr));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, lds_addr,
                  scalar_positive_inline_u32(lane_stride_shift), vgpr_src(lds_addr));
  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, lds_addr, kM0, vgpr_src(lds_addr));
  emit_s_mov_b64(words, kExecLo, saved_exec);

  emit_cdna3_mubuf(words, src, load_op, data);
  emit_cdna3_wait_all(words);
  emit_cdna3_ds(words, ds_op, 0, lds_addr, data);
  // Native buffer_load_* ... lds exposes a VMEM-completed LDS side effect. The
  // fallback sequence creates that side effect with an explicit DS write, so
  // wait for the DS operation here before branching back to code that was
  // scheduled around the original LDS-DMA instruction.
  emit_cdna3_lgkm_wait(words);

  return ExpandResult::success(std::move(words));
}

ExpandResult expand_v_bitop3_b16_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                TranslationContext &context, const LaneLayout *,
                                                const LaneLayout *) {
  // The rule table only routes V_BITOP3_B16 here, so use the generated
  // instruction type directly instead of re-decoding ordinary operands.
  return lower_cdna4_bitop3_to_cdna3(static_cast<const cdna4::VBitop3B16Vop3 &>(inst), liveness,
                                     context, true);
}

ExpandResult expand_v_bitop3_b32_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                const LivenessAnalysis &liveness,
                                                TranslationContext &context, const LaneLayout *,
                                                const LaneLayout *) {
  // The rule table only routes V_BITOP3_B32 here, so use the generated
  // instruction type directly instead of re-decoding ordinary operands.
  return lower_cdna4_bitop3_to_cdna3(static_cast<const cdna4::VBitop3B32Vop3 &>(inst), liveness,
                                     context, false);
}

ExpandResult expand_permlane32_swap_b32_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                       const LivenessAnalysis &liveness,
                                                       TranslationContext &context,
                                                       const LaneLayout *, const LaneLayout *) {
  return lower_permlane32_swap_b32_cdna4_to_cdna3(inst, liveness, context);
}

ExpandResult expand_cvt_pk_f16_f32_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context, const LaneLayout *,
                                                  const LaneLayout *) {
  return lower_cvt_pk_f16_f32_cdna4_to_cdna3(inst, liveness, context);
}

ExpandResult expand_ds_read_b64_tr_b16_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                      const LivenessAnalysis &liveness,
                                                      TranslationContext &context,
                                                      const LaneLayout *, const LaneLayout *) {
  return lower_ds_read_b64_tr_b16_cdna4_to_cdna3(inst, liveness, context);
}

ExpandResult expand_mfma_f32_16x16x32_f16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                         uint64_t, const LivenessAnalysis &liveness,
                                                         TranslationContext &context,
                                                         const LaneLayout *, const LaneLayout *) {
  return lower_wide_k_mfma_f16_cdna4_to_cdna3(inst, liveness, context,
                                              WideKMfmaShape::F32_16x16x32_F16);
}

ExpandResult expand_mfma_f32_32x32x16_f16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                         uint64_t, const LivenessAnalysis &liveness,
                                                         TranslationContext &context,
                                                         const LaneLayout *, const LaneLayout *) {
  return lower_wide_k_mfma_f16_cdna4_to_cdna3(inst, liveness, context,
                                              WideKMfmaShape::F32_32x32x16_F16);
}

ExpandResult expand_buffer_load_dwordx3_lds_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                           uint64_t,
                                                           const LivenessAnalysis &liveness,
                                                           TranslationContext &context,
                                                           const LaneLayout *, const LaneLayout *) {
  return expand_mubuf_load_to_lds_cdna4_to_cdna3(inst, liveness, context, 3);
}

ExpandResult expand_buffer_load_dwordx4_lds_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                           uint64_t,
                                                           const LivenessAnalysis &liveness,
                                                           TranslationContext &context,
                                                           const LaneLayout *, const LaneLayout *) {
  return expand_mubuf_load_to_lds_cdna4_to_cdna3(inst, liveness, context, 4);
}

// Table MUST be sorted by (src_encoding_id, src_opcode) for binary search.
const TranslationRule kExpandRules_cdna4_to_cdna3[] = {
    {cdna4::encoding::kVop1, cdna4::kVPermlane32SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi1, cdna4::kVPermlane32SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi2, cdna4::kVPermlane32SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi3, cdna4::kVPermlane32SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16, RuleAction::Expand, 0, 0, nullptr,
     expand_v_bitop3_b16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B32, RuleAction::Expand, 0, 0, nullptr,
     expand_v_bitop3_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3OpHi4, cdna4::kVCvtPkF16F32, RuleAction::Expand, 0, 0, nullptr,
     expand_cvt_pk_f16_f32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3pMfma, cdna4::kVMfmaF3216x16x32F16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_16x16x32_f16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3pMfma, cdna4::kVMfmaF3232x32x16F16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_32x32x16_f16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kDsHi3, cdna4::kDsReadB64TrB16, RuleAction::Expand, 0, 0, nullptr,
     expand_ds_read_b64_tr_b16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx3, RuleAction::Expand, 0, 0, nullptr,
     expand_buffer_load_dwordx3_lds_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx4, RuleAction::Expand, 0, 0, nullptr,
     expand_buffer_load_dwordx4_lds_cdna4_to_cdna3, nullptr, nullptr},
};

} // namespace

std::span<const TranslationRule> semantic_expand_rules_cdna4_to_cdna3() {
  return std::span<const TranslationRule>(kExpandRules_cdna4_to_cdna3);
}

} // namespace rocjitsu
