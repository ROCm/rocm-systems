// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_cdna3.cpp
/// @brief CDNA4-to-CDNA3 handwritten semantic expansion rules.

#include "rocjitsu/code/dbt/semantic/rules.h"

#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/dbt/semantic/cdna3_lds.h"
#include "rocjitsu/code/dbt/semantic/cdna3_scratch.h"
#include "rocjitsu/code/dbt/semantic_scratch.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

// These small emission adapters describe the operands needed by each lowering;
// the generated builders own the CDNA3 binary layouts and fixed encoding bits.
// Keeping that split means semantic rules no longer serialize MachineInst
// bitfields or repeat format constants, while remaining explicit about every
// source value and scratch register they emit.

[[nodiscard]] std::array<uint32_t, 2> build_cdna3_vop3(uint16_t op, uint8_t vdst, uint16_t src0,
                                                       uint16_t src1 = 0, uint16_t src2 = 0) {
  cdna3::Vop3BuilderFields fields{};
  fields.vdst = vdst;
  fields.src0 = src0;
  fields.src1 = src1;
  fields.src2 = src2;
  return cdna3::build_vop3(op, fields);
}

[[nodiscard]] std::pair<uint32_t, uint32_t>
build_cdna3_vop3p(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1 = 0, uint16_t src2 = 0) {
  const auto words = cdna3::build_vop3p(
      op,
      {.vdst = vdst, .op_sel_hi_2 = 1, .src0 = src0, .src1 = src1, .src2 = src2, .op_sel_hi = 3});
  return {words[0], words[1]};
}

/// @brief Build a CDNA3 VOP3P-MFMA instruction word pair.
/// @details Preserve both the source and destination accumulator selectors.
/// Some wide-K rules keep an AccVGPR source while only narrowing K, so clearing
/// the source selector would reinterpret that operand as an ordinary VGPR.
[[nodiscard]] std::array<uint32_t, 2>
build_cdna3_vop3p_mfma(uint16_t op, const cdna4::Vop3pMfmaMachineInst &src, uint8_t vdst,
                       uint8_t acc_cd, uint16_t src0, uint16_t src1, uint16_t src2) {
  cdna3::Vop3pMfmaBuilderFields fields{};
  fields.vdst = vdst;
  fields.cbsz = src.cbsz;
  fields.abid = src.abid;
  fields.acc_cd = acc_cd;
  fields.src0 = src0;
  fields.src1 = src1;
  fields.src2 = src2;
  fields.acc = src.acc;
  fields.blgp = src.blgp;
  return cdna3::build_vop3p_mfma(op, fields);
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_s_mov_b32_lit(uint8_t sdst,
                                                                          uint32_t literal) {
  const auto inst = cdna3::build_sop1(cdna3::kSMovB32, {.ssrc0 = 0xFF, .sdst = sdst});
  return {inst[0], literal};
}

[[nodiscard]] constexpr uint32_t build_s_mov_b64(uint8_t sdst, uint16_t ssrc0) {
  return cdna3::build_sop1(cdna3::kSMovB64,
                           {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = sdst})[0];
}

constexpr uint8_t kExecLo = cdna4::OPR_SRC_EXEC_LO;
constexpr uint16_t kInlineConst0 = cdna4::OPR_SRC_POS_INT_MIN;
constexpr uint16_t kInlineConstNeg1 = cdna4::OPR_SRC_NEG_INT_MIN;
constexpr uint16_t kInlineConstFloatOne = cdna4::OPR_SRC_FLOAT_ONE;
constexpr uint16_t kM0 = cdna4::OPR_SRC_M0;
constexpr uint16_t kSrcDpp8Lo = amdgpu::SRC_DPP8_LO;
constexpr uint16_t kSrcDpp8Hi = amdgpu::SRC_DPP8_HI;
constexpr uint16_t kSrcSdwa = amdgpu::SRC_SDWA;
constexpr uint16_t kSrcDpp = amdgpu::SRC_DPP;
// The two-dword source marker is an encoding discriminator rather than an
// OPR_SRC predefined value in the XML. Keep that one exceptional value local.
constexpr uint16_t kSrcLiteral64 = 254;
constexpr uint16_t kSrcLiteral32 = cdna4::OPR_SRC_SRC_LITERAL;
constexpr uint8_t kSdwaWord0 = 4;
constexpr uint8_t kSdwaWord1 = 5;
constexpr uint8_t kSdwaDword = 6;

constexpr uint16_t kCdnaWaitcntLgkmcnt0 = 0xC07F;
constexpr uint16_t kCdnaWaitcntAll0 = 0x0000;
constexpr uint32_t kCdna3MaxOrdinarySgprs = 102;
constexpr uint32_t kCdna3SpecialSgprTailReserve = 8;
constexpr uint32_t kFlatGlobalPositiveImm13Max = 4095;

struct VgprForbiddenRange {
  uint16_t base = 0;
  uint16_t count = 0;
};

void emit_cdna3_vop3(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint16_t src0,
                     uint16_t src1 = 0, uint16_t src2 = 0) {
  auto [w0, w1] = build_cdna3_vop3(op, vdst, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_vop2_literal(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint8_t vsrc1,
                             uint32_t literal) {
  words.push_back(cdna3::build_vop2(op, {.src0 = 0xFF, .vsrc1 = vsrc1, .vdst = vdst})[0]);
  words.push_back(literal);
}

void emit_cdna3_ds(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint8_t addr,
                   uint8_t data0 = 0, uint8_t data1 = 0, uint8_t offset0 = 0, uint8_t offset1 = 0) {
  const auto encoded = cdna3::build_ds(op, {.offset0 = offset0,
                                            .offset1 = offset1,
                                            .addr = addr,
                                            .data0 = data0,
                                            .data1 = data1,
                                            .vdst = vdst});
  words.insert(words.end(), encoded.begin(), encoded.end());
}

void emit_cdna3_lgkm_wait(std::vector<uint32_t> &words) {
  // GFX9/CDNA s_waitcnt encodes "lgkmcnt(0)" as lgkm=0 while leaving VM/EXP at
  // their no-wait maxima. The DS read/bpermute sequences below require the
  // loaded/permuted data before issuing dependent VALU instructions.
  words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntLgkmcnt0})[0]);
}

void emit_cdna3_wait_all(std::vector<uint32_t> &words) {
  // The load-to-LDS expansion below consumes a just-issued VMEM load through a
  // following DS write.  Waiting all counters is stronger than the hardware
  // `vmcnt(0)` dependency we strictly need, but it keeps this first lowering
  // conservative and matches the monolithic CDNA waitcnt encoding.
  words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
}

[[nodiscard]] constexpr uint16_t vgpr_src(uint8_t reg) { return static_cast<uint16_t>(256 + reg); }

void emit_cdna3_accvgpr_write_b32(std::vector<uint32_t> &words, uint8_t acc_dst, uint8_t src_vgpr) {
  auto [w0, w1] = build_cdna3_vop3p(cdna3::kVAccvgprWriteVop3p, acc_dst, vgpr_src(src_vgpr));
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_f32_to_bf16_rne(std::vector<uint32_t> &words, uint8_t dst_half, uint8_t t0,
                                uint8_t t1, uint8_t t2, uint16_t src) {
  // Convert one FP32 payload to BF16 with round-to-nearest-even, matching the
  // authoritative guest reference util::f32_to_bf16_rne (data_types.h):
  //
  //   if ((f & 0x7f800000) != 0x7f800000)          // finite / subnormal
  //     f += 0x7fff + ((f >> 16) & 1);             //   RNE bias
  //   else if (f & 0xffff)                         // NaN whose payload is in
  //     f |= 0x10000;                              //   the truncated low bits
  //   return f >> 16;                              // keep it a NaN, not Inf
  //
  // CDNA3 has no CDNA4 packed BF16 conversion, so this is lowered by hand. The
  // earlier lowering applied the RNE bias unconditionally, which turned a NaN
  // like 0x7F800001 into +Inf (a real value change, not ~1ulp drift). This
  // sequence is fully branchless and clobbers no VCC/SGPR state: it computes the
  // rounded value and the NaN-preserving value in parallel, then selects between
  // them with an arithmetic mask derived from whether the exponent is all-ones.
  constexpr uint32_t kExpMask = 0x7f800000u;
  const uint8_t f = dst_half; // working copy of the source bits
  emit_cdna3_vop3(words, cdna3::kVMovB32Vop3, f, src);

  // rounded (t0) = f + 0x7fff + ((f >> 16) & 1)
  emit_cdna3_vop3(words, cdna3::kVLshrrevB32Vop3, t0, scalar_positive_inline_u32(16), vgpr_src(f));
  emit_cdna3_vop3(words, cdna3::kVAndB32Vop3, t0, scalar_positive_inline_u32(1), vgpr_src(t0));
  emit_cdna3_vop2_literal(words, cdna3::kVAddU32Vop2, t0, t0, 0x7fffu);
  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, t0, vgpr_src(f), vgpr_src(t0));

  // special (t1) = f | (((f & 0xffff) != 0) << 16)  -- preserve NaN-ness
  emit_cdna3_vop3(words, cdna3::kVBfeU32Vop3, t1, vgpr_src(f), scalar_positive_inline_u32(0),
                  scalar_positive_inline_u32(16));
  emit_cdna3_vop3(words, cdna3::kVMinU32Vop3, t1, scalar_positive_inline_u32(1), vgpr_src(t1));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, t1, scalar_positive_inline_u32(16), vgpr_src(t1));
  emit_cdna3_vop3(words, cdna3::kVOrB32Vop3, t1, vgpr_src(f), vgpr_src(t1));

  // mask (t2) = (exp field all ones) ? 0xFFFFFFFF : 0, computed arithmetically:
  //   d = (f & 0x7f800000) ^ 0x7f800000   -> 0 iff exponent is all-ones
  //   mask = 0 - ((d - 1) >> 31)          -> all-ones iff d == 0
  emit_cdna3_vop2_literal(words, cdna3::kVAndB32Vop2, t2, f, kExpMask);
  emit_cdna3_vop2_literal(words, cdna3::kVXorB32Vop2, t2, t2, kExpMask);
  emit_cdna3_vop3(words, cdna3::kVSubU32Vop3, t2, vgpr_src(t2), scalar_positive_inline_u32(1));
  emit_cdna3_vop3(words, cdna3::kVLshrrevB32Vop3, t2, scalar_positive_inline_u32(31), vgpr_src(t2));
  emit_cdna3_vop3(words, cdna3::kVSubU32Vop3, t2, scalar_positive_inline_u32(0), vgpr_src(t2));

  // dst = bfi(mask, special, rounded) = (mask & special) | (~mask & rounded)
  emit_cdna3_vop3(words, cdna3::kVBfiB32Vop3, dst_half, vgpr_src(t2), vgpr_src(t1), vgpr_src(t0));
  emit_cdna3_vop3(words, cdna3::kVLshrrevB32Vop3, dst_half, scalar_positive_inline_u32(16),
                  vgpr_src(dst_half));
}

/// @brief Materialize preservation chosen by the architecture-neutral allocator.
/// @details The CDNA3 emitter validates the lease before encoding it. Leases in
/// this file come exclusively from the emitter's published allocation policy,
/// so a failure here indicates an internal allocation/emission contract bug.
void emit_cdna3_scratch_save(std::vector<uint32_t> &words, const SemanticScratchLease &lease) {
  [[maybe_unused]] const bool emitted = Cdna3ScratchEmitter::append_save(words, lease);
  assert(emitted && "CDNA3 scratch allocator returned an unencodable save lease");
}

void emit_cdna3_scratch_restore(std::vector<uint32_t> &words, const SemanticScratchLease &lease) {
  [[maybe_unused]] const bool emitted = Cdna3ScratchEmitter::append_restore(words, lease);
  assert(emitted && "CDNA3 scratch allocator returned an unencodable restore lease");
}

[[nodiscard]] std::optional<uint8_t> vgpr_operand_index(uint16_t encoded_src) {
  if (encoded_src < 256 || encoded_src > 511)
    return std::nullopt;
  return static_cast<uint8_t>(encoded_src - 256);
}

[[nodiscard]] bool is_vop1_modifier_or_literal_source(uint16_t encoded_src) {
  // The e32 VOP1 source namespace reserves these selectors for DPP, SDWA, and
  // literal suffix encodings. Those forms change how the source payload is read
  // or require extra instruction words, so semantic rules that only rewrite the
  // base instruction must reject them until they are decoded and tested.
  return encoded_src == kSrcDpp8Lo || encoded_src == kSrcDpp8Hi || encoded_src == kSrcSdwa ||
         encoded_src == kSrcDpp || encoded_src == kSrcLiteral64 || encoded_src == kSrcLiteral32;
}

[[nodiscard]] bool is_vop3p_modifier_or_literal_source(uint16_t encoded_src) {
  // VOP3P uses the same source-selector reservations for DPP/SDWA/literal
  // suffixes as the other VALU encodings. This rule only rewrites the two base
  // dwords, so reject forms that need an extra payload or alternate source
  // interpretation until they are decoded and tested explicitly.
  return encoded_src == kSrcDpp8Lo || encoded_src == kSrcDpp8Hi || encoded_src == kSrcSdwa ||
         encoded_src == kSrcDpp || encoded_src == kSrcLiteral64 || encoded_src == kSrcLiteral32;
}

[[nodiscard]] std::optional<SemanticScratchLease>
choose_one_vgpr_temp_or_spill(const Instruction &inst, const LivenessAnalysis &liveness,
                              TranslationContext &context, uint8_t forbidden,
                              std::optional<uint8_t> preferred_victim) {
  SemanticScratchRequest request;
  request.count = 1;
  request.preferred_victim_base = preferred_victim;
  request.forbidden.expand({RegClass::VGPR, forbidden, 1});
  SemanticScratchAllocator allocator(inst, liveness, context,
                                     Cdna3ScratchEmitter::allocation_policy());
  return allocator.acquire_vgprs(request).lease;
}

[[nodiscard]] std::optional<SemanticScratchLease>
choose_vgpr_window_or_spill(const Instruction &inst, const LivenessAnalysis &liveness,
                            TranslationContext &context, uint16_t count, uint16_t alignment,
                            std::initializer_list<VgprForbiddenRange> forbidden_ranges) {
  SemanticScratchRequest request;
  request.count = count;
  request.alignment = alignment;
  for (const VgprForbiddenRange &range : forbidden_ranges) {
    if (range.count != 0)
      request.forbidden.expand({RegClass::VGPR, range.base, static_cast<uint8_t>(range.count)});
  }
  SemanticScratchAllocator allocator(inst, liveness, context,
                                     Cdna3ScratchEmitter::allocation_policy());
  return allocator.acquire_vgprs(request).lease;
}

/// @brief Reserve non-overlapping private slots for one borrowed virtual-LDS SGPR pair.
///
/// @details A spill-backed semantic scratch window remains live until the end of
/// the replacement sequence. The access emitter must therefore save the borrowed
/// scalar pair after that window instead of reusing the per-instruction spill
/// base and overwriting victim VGPRs that have not yet been restored.
[[nodiscard]] std::optional<Cdna3VirtualLdsBorrowScratch>
make_cdna3_virtual_lds_borrow_scratch(TranslationContext &context,
                                      const SemanticScratchLease &window, uint8_t pointer_vgpr_lo,
                                      uint8_t pointer_vgpr_hi) {
  if (!context.virtual_lds_base_sgpr_spill_per_use)
    return std::nullopt;

  // Reconstruct the per-lowering frame and consume the window's preservation
  // range before assigning the scalar-pair slots. SemanticSpillFrame owns the
  // non-overlap arithmetic; the lowering no longer compares raw reservation
  // bases or manually extends a combined allocation.
  SemanticSpillFrame frame(context);
  if (window.spilled &&
      !frame.allocate_dwords(window.count, sizeof(uint32_t), Cdna3ScratchEmitter::kMaxDwordOffset))
    return std::nullopt;
  const auto saved_sgpr =
      frame.allocate_dwords(2, sizeof(uint32_t), Cdna3ScratchEmitter::kMaxDwordOffset);
  if (!saved_sgpr)
    return std::nullopt;

  return Cdna3VirtualLdsBorrowScratch{.pointer_vgpr_lo = pointer_vgpr_lo,
                                      .pointer_vgpr_hi = pointer_vgpr_hi,
                                      .saved_sgpr_private_offset = saved_sgpr->byte_offset};
}

void emit_cdna3_mubuf(std::vector<uint32_t> &words, const cdna4::MubufMachineInst &src, uint16_t op,
                      uint8_t vdata) {
  const auto encoded = cdna3::build_mubuf(op, {.offset = static_cast<uint16_t>(src.offset),
                                               .offen = static_cast<uint8_t>(src.offen),
                                               .idxen = static_cast<uint8_t>(src.idxen),
                                               .sc0 = static_cast<uint8_t>(src.sc0),
                                               .sc1 = static_cast<uint8_t>(src.sc1),
                                               .nt = static_cast<uint8_t>(src.nt),
                                               .vaddr = static_cast<uint8_t>(src.vaddr),
                                               .vdata = vdata,
                                               .srsrc = static_cast<uint8_t>(src.srsrc),
                                               .soffset = static_cast<uint8_t>(src.soffset)});
  words.insert(words.end(), encoded.begin(), encoded.end());
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

[[nodiscard]] std::optional<uint8_t>
choose_cdna3_exec_save_sgpr(const Instruction &inst, const LivenessAnalysis &liveness,
                            const TranslationContext &context) {
  const auto overlaps_virtual_lds_base = [&](uint32_t candidate) {
    if (!context.virtualize_lds)
      return false;
    return candidate < context.virtual_lds_base_sgpr + 2u &&
           context.virtual_lds_base_sgpr < candidate + 2u;
  };

  // Prefer growing the descriptor with a fresh pair just above the original
  // SGPR allocation. Virtual-LDS mode also reserves a DBT-owned SGPR pair for
  // the backing pointer; clobbering that pair with an EXEC save turns the next
  // virtual-LDS global access into a wild pointer.
  if (context.num_sgprs < kCdna3MaxOrdinarySgprs) {
    const uint32_t preferred = (context.num_sgprs + 1u) & ~1u;
    for (uint32_t candidate = preferred; candidate + 1u < kCdna3MaxOrdinarySgprs; candidate += 2u) {
      if (!overlaps_virtual_lds_base(candidate))
        return static_cast<uint8_t>(candidate);
    }
  }

  // When the descriptor cannot grow, fall back only to a pair proven dead at
  // this replacement point. The old downward search borrowed high SGPR pairs
  // without dataflow proof; if the guest read that pair later, restoring EXEC
  // still left the guest scalar contents corrupted.
  uint16_t search_start = 0;
  while (auto candidate = liveness.find_free_sgpr_pair(&inst, search_start)) {
    if (!overlaps_virtual_lds_base(*candidate))
      return static_cast<uint8_t>(*candidate);
    search_start = static_cast<uint16_t>(*candidate + 2);
  }
  return std::nullopt;
}

void require_cdna3_exec_save_sgpr(TranslationContext &context, uint8_t saved_exec) {
  // COMPUTE_PGM_RSRC1's SGPR count is the rounded wave allocation, not the
  // highest ordinary scalar register available to DBT. VCC, flat scratch, XNACK,
  // and granularity padding can live in the descriptor tail. When a semantic
  // lowering introduces an ordinary SGPR pair for EXEC save/restore, grow the
  // descriptor far enough that the architectural special-register tail moves
  // above that generated pair.
  context.require_sgprs(static_cast<uint32_t>(saved_exec) + 2u + kCdna3SpecialSgprTailReserve);
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

[[nodiscard]] std::array<VgprForbiddenRange, 4>
bitop3_spill_forbidden_ranges(uint8_t vdst, const std::array<uint16_t, 3> &src) {
  // Spill-backed scratch may borrow a live VGPR window and restore it after the
  // synthesized LUT sequence. That is not safe for source operands, because the
  // sequence still needs their original values while the scratch window is
  // clobbered. It is also not safe for VDST, because the final restore would
  // overwrite the architectural result. Dead-register scratch still uses the
  // faster liveness-only path below.
  std::array<VgprForbiddenRange, 4> forbidden{};
  forbidden[0] = {static_cast<uint16_t>(vdst), 1};
  size_t next = 1;
  for (uint16_t encoded : src) {
    if (const auto vgpr = vgpr_operand_index(encoded)) {
      bool seen = *vgpr == vdst;
      for (size_t i = 1; i < next; ++i)
        seen = seen || forbidden[i].base == *vgpr;
      if (!seen && next < forbidden.size())
        forbidden[next++] = {static_cast<uint16_t>(*vgpr), 1};
    }
  }
  return forbidden;
}

[[nodiscard]] bool bitop3_needs_product_term(const std::array<uint8_t, 8> &coeff) {
  for (uint8_t mask = 1; mask < coeff.size(); ++mask) {
    if (coeff[mask] != 0 && std::popcount(mask) >= 2)
      return true;
  }
  return false;
}

void emit_cdna3_b16_zero_extend(std::vector<uint32_t> &words, uint8_t reg) {
  // V_BITOP3_B16 computes a 16-bit LUT result and writes that value through the
  // generated 16-bit destination operand. CDNA3's B32 integer ops leave their
  // full 32-bit result in the destination, so every B16 expansion path must
  // explicitly clear bits 31:16 before returning.
  const uint16_t shift16 = scalar_positive_inline_u32(16);
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, reg, shift16, vgpr_src(reg));
  emit_cdna3_vop3(words, cdna3::kVLshrrevB32Vop3, reg, shift16, vgpr_src(reg));
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

  if (truth_table == 0xec || truth_table == 0xf8 || truth_table == 0xfe) {
    std::vector<uint32_t> words;
    if (truth_table == 0xec) {
      // S1 | (S0 & S2)
      emit_cdna3_vop3(words, cdna3::kVAndOrB32Vop3, vdst, src[0], src[2], src[1]);
    } else if (truth_table == 0xf8) {
      // S0 | (S1 & S2)
      emit_cdna3_vop3(words, cdna3::kVAndOrB32Vop3, vdst, src[1], src[2], src[0]);
    } else {
      // S0 | S1 | S2
      emit_cdna3_vop3(words, cdna3::kVOr3B32Vop3, vdst, src[0], src[1], src[2]);
    }
    if (is_b16)
      emit_cdna3_b16_zero_extend(words, vdst);
    return ExpandResult::success(std::move(words));
  }

  if (!is_b16 && truth_table == 0xc8) {
    uint8_t term = vdst;
    std::optional<SemanticScratchLease> scratch_term;
    if (vdst_aliases_any_vgpr_source(vdst, src)) {
      const auto forbidden = bitop3_spill_forbidden_ranges(vdst, src);
      scratch_term = choose_vgpr_window_or_spill(
          inst, liveness, context, 1, 1, {forbidden[0], forbidden[1], forbidden[2], forbidden[3]});
      if (!scratch_term)
        return failed_existing_expand_rule(
            inst, "No VGPR or spill-backed scratch for compact bitop temporary",
            {"Provide one temporary VGPR or a spillable non-source, non-destination VGPR."});
      term = static_cast<uint8_t>(scratch_term->base);
    }

    std::vector<uint32_t> words;
    if (scratch_term)
      emit_cdna3_scratch_save(words, *scratch_term);

    auto emit_and = [&](uint8_t dst, uint16_t src0, uint16_t src1) {
      emit_cdna3_vop3(words, cdna3::kVAndB32Vop3, dst, src0, src1);
    };
    auto emit_or = [&](uint8_t dst, uint16_t src0, uint16_t src1) {
      emit_cdna3_vop3(words, cdna3::kVOrB32Vop3, dst, src0, src1);
    };

    // S1 & (S0 | S2)
    emit_or(term, src[0], src[2]);
    emit_and(vdst, src[1], vgpr_src(term));

    if (scratch_term)
      emit_cdna3_scratch_restore(words, *scratch_term);
    return ExpandResult::success(std::move(words));
  }

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
  std::optional<SemanticScratchLease> scratch_lease;
  if (scratch_count != 0) {
    const auto forbidden = bitop3_spill_forbidden_ranges(vdst, src);
    scratch_lease =
        choose_vgpr_window_or_spill(inst, liveness, context, scratch_count, 1,
                                    {forbidden[0], forbidden[1], forbidden[2], forbidden[3]});
    if (!scratch_lease)
      return failed_existing_expand_rule(
          inst, "No VGPR or spill-backed scratch window for temporary bitop temporaries",
          {"Provide a spillable scratch window that does not overlap VDST or any VGPR source."});

    acc = static_cast<uint8_t>(scratch_lease->base);
    if (needs_term_temp)
      term = static_cast<uint8_t>(scratch_lease->base + 1);
  }

  std::vector<uint32_t> words;
  if (scratch_lease)
    emit_cdna3_scratch_save(words, *scratch_lease);

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
    emit_cdna3_b16_zero_extend(words, acc);
  }

  if (acc != vdst)
    emit_mov(vdst, vgpr_src(acc));

  if (scratch_lease)
    emit_cdna3_scratch_restore(words, *scratch_lease);

  return ExpandResult::success(std::move(words));
}

// -----------------------------------------------------------------------------
// V_PERMLANE*_SWAP expansions.
// -----------------------------------------------------------------------------

ExpandResult lower_permlane_swap_b32_cdna4_to_cdna3(const Instruction &inst,
                                                    const LivenessAnalysis &liveness,
                                                    TranslationContext &context,
                                                    uint8_t half_wave_lanes) {
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

  // Swap operands are read/write values, so the normal liveness query protects
  // their pre-write contents. Explicitly forbid only the architectural outputs:
  // those registers are intentionally written near the end of the lowering and
  // cannot be restored as scratch victims.
  auto scratch = choose_vgpr_window_or_spill(inst, liveness, context, 3, 1, {{vdst, 1}, {vsrc, 1}});
  if (!scratch)
    return failed_existing_expand_rule(
        inst, "No free VGPR scratch window for swap temporaries",
        {"Provide one SGPR pair for EXEC save/restore and three temporary VGPRs for lane address "
         "and pre-write swap values.",
         "Allow the spill-backed VGPR scratch path to preserve live non-output registers."});

  const uint8_t lane = scratch->base;
  const uint8_t partner_addr = static_cast<uint8_t>(scratch->base + 1);
  const uint8_t from_dst_high = static_cast<uint8_t>(scratch->base + 2);
  const uint8_t from_src_low = lane;
  // Prefer a fresh descriptor-backed SGPR pair for EXEC save/restore. When the
  // descriptor cannot grow, choose_cdna3_exec_save_sgpr() falls back only to a
  // pair that liveness proves dead at this replacement point.
  const std::optional<uint8_t> saved_exec = choose_cdna3_exec_save_sgpr(inst, liveness, context);
  if (!saved_exec)
    return failed_existing_expand_rule(
        inst, "No free descriptor-backed SGPR pair for EXEC save/restore",
        {"Add SGPR spill-backed EXEC save/restore before lowering this instruction in "
         "descriptor-full kernels."});

  require_cdna3_exec_save_sgpr(context, *saved_exec);

  std::vector<uint32_t> words;

  // CDNA4 v_permlane*_swap_b32 swaps the selected upper lane group of VDST with
  // the selected lower lane group of SRC0:
  //
  //   SRC0 lanes 0..N-1 = old VDST lanes N..2N-1
  //   VDST lanes N..2N-1 = old SRC0 lanes 0..N-1
  //
  // For the 16-lane form this operation applies to rows 0/1 and rows 2/3; for
  // the 32-lane form it applies to the single 0/1 half-wave pair. CDNA3 has no
  // row-swap instruction. Run the data-gather portion with EXEC forced to all
  // lanes so mbcnt() produces physical lane IDs and ds_bpermute can read both
  // source half-waves. Only after both old values are captured do we narrow EXEC
  // to the exact write lane groups. Do not intersect those masks with the
  // original EXEC: generated execution semantics mark V_PERMLANE*_SWAP_B32 as
  // EXEC-ignoring, and later control flow may re-enable lanes that depend on
  // these swapped values.
  emit_s_mov_b64(words, *saved_exec, kExecLo);
  emit_cdna3_exec_mask(words, UINT64_MAX);
  if (scratch->spilled) {
    // The gather runs under all lanes so it can read both sides of the row
    // swap, so spilled scratch victims must be preserved for every lane, not
    // only the entry EXEC mask.
    emit_cdna3_scratch_save(words, *scratch);
  }

  emit_cdna3_vop3(words, cdna3::kVMbcntLoU32B32Vop3, lane, kInlineConstNeg1, kInlineConst0);
  emit_cdna3_vop3(words, cdna3::kVMbcntHiU32B32Vop3, lane, kInlineConstNeg1, vgpr_src(lane));
  emit_cdna3_vop3(words, cdna3::kVXorB32Vop3, partner_addr,
                  scalar_positive_inline_u32(half_wave_lanes), vgpr_src(lane));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, partner_addr, scalar_positive_inline_u32(2),
                  vgpr_src(partner_addr));

  emit_cdna3_ds(words, cdna3::kDsBpermuteB32Ds, from_dst_high, partner_addr, vdst);
  emit_cdna3_ds(words, cdna3::kDsBpermuteB32Ds, from_src_low, partner_addr, vsrc);
  emit_cdna3_lgkm_wait(words);

  const uint64_t group_mask = (uint64_t{1} << half_wave_lanes) - 1u;
  uint64_t low_mask = 0;
  for (uint8_t lane = 0; lane < 64; lane = static_cast<uint8_t>(lane + 2 * half_wave_lanes))
    low_mask |= group_mask << lane;
  const uint64_t high_mask = low_mask << half_wave_lanes;
  emit_cdna3_exec_mask(words, low_mask);
  emit_cdna3_vop3(words, cdna3::kVMovB32Vop3, vsrc, vgpr_src(from_dst_high));

  emit_cdna3_exec_mask(words, high_mask);
  emit_cdna3_vop3(words, cdna3::kVMovB32Vop3, vdst, vgpr_src(from_src_low));

  if (scratch->spilled) {
    // The gather ran with all lanes enabled, so every lane of each scratch
    // victim may have been clobbered. Restore the spill window under the same
    // all-lane mask before restoring the source EXEC.
    emit_cdna3_exec_mask(words, UINT64_MAX);
    emit_cdna3_scratch_restore(words, *scratch);
  }
  emit_s_mov_b64(words, kExecLo, *saved_exec);
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
  // CDNA4 v_cvt_pk_f16_f32 packs two f32-to-f16 conversions into one VGPR:
  //
  //   VDST[15:0]  = f32_to_f16(SRC0)
  //   VDST[31:16] = f32_to_f16(SRC1)
  //
  // CDNA3's packed form is RTZ, so use scalar conversions to preserve the
  // source instruction's ordinary f32->f16 rounding semantics before packing.
  const uint8_t vdst = static_cast<uint8_t>(src.vdst);
  const std::optional<uint8_t> src0_vgpr = vgpr_operand_index(static_cast<uint16_t>(src.src0));
  const std::optional<uint8_t> src1_vgpr = vgpr_operand_index(static_cast<uint16_t>(src.src1));
  const bool vdst_is_src1 = src1_vgpr && vdst == *src1_vgpr;

  const std::optional<uint8_t> preferred_temp =
      vdst_is_src1 ? src0_vgpr : (src1_vgpr ? src1_vgpr : src0_vgpr);
  auto temp = choose_one_vgpr_temp_or_spill(inst, liveness, context, vdst, preferred_temp);
  if (!temp) {
    return failed_existing_expand_rule(
        inst, "No VGPR temporary or scratch spill slot for packed F16 conversion",
        {"Provide one temporary VGPR or an encodable per-lane scratch spill slot."});
  }
  std::vector<uint32_t> words;
  emit_cdna3_scratch_save(words, *temp);

  if (vdst_is_src1) {
    // When VDST aliases SRC1, materialize the low half in the temporary first
    // so the second conversion can still read SRC1's original value.
    emit_cdna3_vop3(words, cdna3::kVCvtF16F32Vop3, static_cast<uint8_t>(temp->base),
                    static_cast<uint16_t>(src.src0));
    emit_cdna3_vop3(words, cdna3::kVCvtF16F32Vop3, vdst, static_cast<uint16_t>(src.src1));
    emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, vdst, scalar_positive_inline_u32(16),
                    vgpr_src(vdst));
    emit_cdna3_vop3(words, cdna3::kVOrB32Vop3, vdst, vgpr_src(static_cast<uint8_t>(temp->base)),
                    vgpr_src(vdst));
  } else {
    // Otherwise VDST can hold the low half directly and only the high half
    // needs a temporary register.
    emit_cdna3_vop3(words, cdna3::kVCvtF16F32Vop3, vdst, static_cast<uint16_t>(src.src0));
    emit_cdna3_vop3(words, cdna3::kVCvtF16F32Vop3, static_cast<uint8_t>(temp->base),
                    static_cast<uint16_t>(src.src1));
    emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, static_cast<uint8_t>(temp->base),
                    scalar_positive_inline_u32(16), vgpr_src(static_cast<uint8_t>(temp->base)));
    emit_cdna3_vop3(words, cdna3::kVOrB32Vop3, vdst, vgpr_src(vdst),
                    vgpr_src(static_cast<uint8_t>(temp->base)));
  }

  emit_cdna3_scratch_restore(words, *temp);
  return ExpandResult::success(std::move(words));
}

ExpandResult lower_cvt_pk_bf16_f32_cdna4_to_cdna3(const Instruction &inst,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop3MachineInst))
    return failed_existing_expand_rule(inst, "No decodable VOP3 instruction encoding",
                                       {"Decode the source VOP3 instruction before lowering."});

  cdna4::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.abs != 0 || src.op_sel != 0 || src.clamp != 0 || src.omod != 0 || src.neg != 0)
    return failed_existing_expand_rule(
        inst, "Unsupported modifiers on packed BF16 conversion (abs/op_sel/clamp/omod/neg)",
        {"Implement modifier handling when source modifiers are present."});
  if (src.src0 > 511 || src.src1 > 511)
    return failed_existing_expand_rule(
        inst, "Packed BF16 conversion operands use unsupported source encodings",
        {"Add operand-class-specific lowering for literal, DPP, or SDWA forms."});

  const uint8_t vdst = static_cast<uint8_t>(src.vdst);
  const std::optional<uint8_t> src1_vgpr = vgpr_operand_index(static_cast<uint16_t>(src.src1));
  // The RNE sequence converts SRC0 first and SRC1 second. SRC0 is consumed
  // before the low conversion clobbers any scratch register, but SRC1 must
  // survive until the second conversion. Spill-backed scratch borrows live
  // VGPRs, so never borrow the high source for this lowering.
  // The NaN-safe RNE lowering needs one result register per half plus three
  // shared scratch temporaries (the temps are reused across the two halves
  // because the low conversion completes before the high one begins).
  auto scratch = choose_vgpr_window_or_spill(
      inst, liveness, context, 5, 1,
      {{vdst, 1},
       {static_cast<uint16_t>(src1_vgpr.value_or(0)), static_cast<uint16_t>(src1_vgpr ? 1 : 0)}});
  if (!scratch)
    return failed_existing_expand_rule(
        inst, "No VGPR scratch window for packed BF16 conversion temporaries",
        {"Provide five temporary VGPRs or spill-backed temporaries so BF16 "
         "rounding, NaN preservation, and destination/source aliases are safe."});
  std::vector<uint32_t> words;
  emit_cdna3_scratch_save(words, *scratch);

  const uint8_t lo = scratch->base;
  const uint8_t hi = static_cast<uint8_t>(scratch->base + 1);
  const uint8_t t0 = static_cast<uint8_t>(scratch->base + 2);
  const uint8_t t1 = static_cast<uint8_t>(scratch->base + 3);
  const uint8_t t2 = static_cast<uint8_t>(scratch->base + 4);
  emit_cdna3_f32_to_bf16_rne(words, lo, t0, t1, t2, static_cast<uint16_t>(src.src0));
  emit_cdna3_f32_to_bf16_rne(words, hi, t0, t1, t2, static_cast<uint16_t>(src.src1));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, hi, scalar_positive_inline_u32(16), vgpr_src(hi));
  emit_cdna3_vop3(words, cdna3::kVOrB32Vop3, vdst, vgpr_src(lo), vgpr_src(hi));

  emit_cdna3_scratch_restore(words, *scratch);
  return ExpandResult::success(std::move(words));
}

ExpandResult lower_cvt_f32_bf16_vop1_cdna4_to_cdna3(const Instruction &inst, uint64_t offset,
                                                    std::span<const uint8_t> source_text) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop1MachineInst))
    return failed_existing_expand_rule(inst, "No decodable VOP1 instruction encoding",
                                       {"Decode the source VOP1 instruction before lowering."});

  cdna4::Vop1MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.src0 == kSrcSdwa) {
    if (offset > source_text.size() ||
        source_text.size() - offset < sizeof(cdna4::Vop1VopSdwaMachineInst)) {
      return failed_existing_expand_rule(
          inst, "Source text does not contain full VOP1 SDWA payload",
          {"Pass the original two-word SDWA encoding into semantic expansion."});
    }

    cdna4::Vop1VopSdwaMachineInst sdwa{};
    std::memcpy(&sdwa, source_text.data() + offset, sizeof(sdwa));
    if (sdwa.src0 != kSrcSdwa || sdwa.op != src.op || sdwa.vdst != src.vdst)
      return failed_existing_expand_rule(
          inst, "Source text VOP1 SDWA payload does not match decoded instruction",
          {"Use the source instruction offset to read the original modifier payload."});

    if (sdwa.dst_sel != kSdwaDword || sdwa.clamp != 0 || sdwa.omod != 0 ||
        (sdwa.src0_sel != kSdwaWord0 && sdwa.src0_sel != kSdwaWord1) || sdwa.src0_sext != 0 ||
        sdwa.src0_neg != 0 || sdwa.src0_abs != 0 || sdwa.s0 != 0) {
      return failed_existing_expand_rule(inst, "Unsupported BF16-to-FP32 VOP1 SDWA form",
                                         {"This lowering currently supports only VGPR "
                                          "WORD_0/WORD_1 source selection into a full DWORD "
                                          "destination, with no SDWA source modifiers."});
    }

    std::vector<uint32_t> words;
    const uint8_t vdst = static_cast<uint8_t>(sdwa.vdst);
    const uint16_t vsrc = vgpr_src(static_cast<uint8_t>(sdwa.vsrc0));
    if (sdwa.src0_sel == kSdwaWord0) {
      // WORD_0 selects SRC0[15:0] with zero extension. Shifting the full source
      // register left by 16 is equivalent: the unselected high half is shifted
      // out, and the selected low half lands in FP32[31:16].
      emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, vdst, scalar_positive_inline_u32(16), vsrc);
    } else {
      // WORD_1 selects SRC0[31:16]. First move that half into bits [15:0], then
      // use the same BF16-to-FP32 left shift. Reusing VDST is safe even when it
      // aliases VSRC because the original source is consumed by the first VALU.
      emit_cdna3_vop3(words, cdna3::kVLshrrevB32Vop3, vdst, scalar_positive_inline_u32(16), vsrc);
      emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, vdst, scalar_positive_inline_u32(16),
                      vgpr_src(vdst));
    }
    return ExpandResult::success(std::move(words));
  }

  if (is_vop1_modifier_or_literal_source(static_cast<uint16_t>(src.src0))) {
    return failed_existing_expand_rule(
        inst, "Unsupported BF16-to-FP32 VOP1 source modifier or literal",
        {"Add DPP, SDWA, or literal-aware source handling before enabling this form."});
  }

  std::vector<uint32_t> words;
  // BF16 has the same sign and exponent layout as FP32, with the mantissa
  // truncated to the high 7 fraction bits. Converting BF16 bits in SRC0[15:0]
  // to FP32 is therefore just placing those 16 bits in FP32[31:16] and zeroing
  // FP32[15:0]. A left shift preserves NaNs/Infs/signs without needing any
  // floating-point hardware support on the target.
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, static_cast<uint8_t>(src.vdst),
                  scalar_positive_inline_u32(16), static_cast<uint16_t>(src.src0));
  return ExpandResult::success(std::move(words));
}

// -----------------------------------------------------------------------------
// V_DOT2_F32_BF16 expansion.
// -----------------------------------------------------------------------------

void emit_cdna3_bf16_half_as_f32(std::vector<uint32_t> &words, uint8_t dst, uint16_t src,
                                 bool high_half, bool negate) {
  // BF16 and FP32 use the same sign/exponent field widths. Widening a BF16
  // payload to FP32 is therefore an exact bit placement into FP32[31:16], with
  // FP32[15:0] cleared. Low packed halves need one left shift; high packed
  // halves first move down to bits 15:0 and then use the same placement.
  if (high_half) {
    emit_cdna3_vop3(words, cdna3::kVLshrrevB32Vop3, dst, scalar_positive_inline_u32(16), src);
    emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, dst, scalar_positive_inline_u32(16),
                    vgpr_src(dst));
  } else {
    emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, dst, scalar_positive_inline_u32(16), src);
  }
  if (negate) {
    // V_DOT2 neg modifiers apply to the selected packed multiplicands, not the
    // explicit S2 addend. After BF16 widening, negation is just an FP32 sign-bit
    // flip; use a VOP2 literal because 0x80000000 is not an inline constant.
    emit_cdna3_vop2_literal(words, cdna3::kVXorB32Vop2, dst, dst, 0x80000000u);
  }
}

ExpandResult lower_dot2_f32_bf16_cdna4_to_cdna3(const Instruction &inst,
                                                const LivenessAnalysis &liveness,
                                                TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop3pMachineInst))
    return failed_existing_expand_rule(inst, "No decodable VOP3P instruction encoding",
                                       {"Decode the source VOP3P instruction before lowering."});

  cdna4::Vop3pMachineInst dot{};
  std::memcpy(&dot, raw, sizeof(dot));
  if (is_vop3p_modifier_or_literal_source(static_cast<uint16_t>(dot.src0)) ||
      is_vop3p_modifier_or_literal_source(static_cast<uint16_t>(dot.src1)) ||
      is_vop3p_modifier_or_literal_source(static_cast<uint16_t>(dot.src2))) {
    return failed_existing_expand_rule(
        inst, "BF16 dot2 operands use unsupported source encodings",
        {"Add operand-class-specific lowering for literal, DPP, or SDWA forms."});
  }

  const auto src0_vgpr = vgpr_operand_index(static_cast<uint16_t>(dot.src0));
  const auto src1_vgpr = vgpr_operand_index(static_cast<uint16_t>(dot.src1));
  if (!src0_vgpr || !src1_vgpr) {
    return failed_existing_expand_rule(inst,
                                       "BF16 dot2 packed multiplicands must be plain VGPR operands",
                                       {"Add scalar-source handling after validating CDNA4 operand "
                                        "constraints for this dot form."});
  }

  std::vector<VgprForbiddenRange> forbidden = {{static_cast<uint16_t>(dot.vdst), 1},
                                               {static_cast<uint16_t>(*src0_vgpr), 1},
                                               {static_cast<uint16_t>(*src1_vgpr), 1}};
  if (const auto src2_vgpr = vgpr_operand_index(static_cast<uint16_t>(dot.src2))) {
    forbidden.push_back({static_cast<uint16_t>(*src2_vgpr), 1});
  }

  constexpr uint16_t kScratchCount = 4;
  auto scratch =
      choose_vgpr_window_or_spill(inst, liveness, context, kScratchCount, 1,
                                  {forbidden[0], forbidden[1], forbidden[2],
                                   forbidden.size() > 3 ? forbidden[3] : VgprForbiddenRange{}});
  if (!scratch) {
    return failed_existing_expand_rule(
        inst, "No VGPR scratch window for BF16 dot2 temporaries",
        {"Provide four temporary VGPRs or spill-backed temporaries for the widened BF16 "
         "multiplicands."});
  }
  const uint8_t a_lo = scratch->base;
  const uint8_t b_lo = static_cast<uint8_t>(scratch->base + 1);
  const uint8_t a_hi = static_cast<uint8_t>(scratch->base + 2);
  const uint8_t b_hi = static_cast<uint8_t>(scratch->base + 3);

  std::vector<uint32_t> words;
  emit_cdna3_scratch_save(words, *scratch);

  // CDNA4 V_DOT2_F32_BF16 is documented in this order:
  //
  //   tmp  = bf16_to_f32(S0[15:0])  * bf16_to_f32(S1[15:0])
  //   tmp += bf16_to_f32(S0[31:16]) * bf16_to_f32(S1[31:16])
  //   tmp += S2.f32
  //   D.f32 = tmp
  //
  // CDNA3 has no BF16 dot2 instruction, so widen both packed BF16 operands into
  // FP32 payloads and replay the documented product/product/addend order with
  // scalar FP32 operations. OPSEL chooses the low-product halves for SRC0/SRC1.
  // The high-product selectors are split in the VOP3P encoding table: bit 14
  // (`op_sel_hi_2` in the generated struct) selects SRC0, while OPSEL_HI bit 1
  // selects SRC1. OPSEL bit 2 and OPSEL_HI bit 0 are the corresponding SRC2
  // selectors, but this instruction's S2 operand is a plain FP32 addend, not a
  // packed 16-bit source, so those bits are ignored. NEG/NEG_HI flip only the
  // selected multiplicand signs; the ISA note says NEG/ABS do not affect S2.
  // The lowering materializes all four multiplicands before writing VDST so
  // VDST may safely alias any source, including the addend source used by the
  // final add. CLMP is a final-result modifier, so it runs only after S2 is
  // added.
  emit_cdna3_bf16_half_as_f32(words, a_lo, static_cast<uint16_t>(dot.src0),
                              (dot.op_sel & 0x1u) != 0, (dot.neg & 0x1u) != 0);
  emit_cdna3_bf16_half_as_f32(words, b_lo, static_cast<uint16_t>(dot.src1),
                              (dot.op_sel & 0x2u) != 0, (dot.neg & 0x2u) != 0);
  emit_cdna3_bf16_half_as_f32(words, a_hi, static_cast<uint16_t>(dot.src0), dot.op_sel_hi_2 != 0,
                              (dot.neg_hi & 0x1u) != 0);
  emit_cdna3_bf16_half_as_f32(words, b_hi, static_cast<uint16_t>(dot.src1),
                              (dot.op_sel_hi & 0x2u) != 0, (dot.neg_hi & 0x2u) != 0);
  emit_cdna3_vop3(words, cdna3::kVMulF32Vop3, a_lo, vgpr_src(a_lo), vgpr_src(b_lo));
  emit_cdna3_vop3(words, cdna3::kVMulF32Vop3, a_hi, vgpr_src(a_hi), vgpr_src(b_hi));
  emit_cdna3_vop3(words, cdna3::kVAddF32Vop3, a_lo, vgpr_src(a_lo), vgpr_src(a_hi));
  emit_cdna3_vop3(words, cdna3::kVAddF32Vop3, static_cast<uint8_t>(dot.vdst), vgpr_src(a_lo),
                  static_cast<uint16_t>(dot.src2));
  if (dot.clamp) {
    emit_cdna3_vop3(words, cdna3::kVMed3F32Vop3, static_cast<uint8_t>(dot.vdst),
                    vgpr_src(static_cast<uint8_t>(dot.vdst)), kInlineConst0, kInlineConstFloatOne);
  }

  emit_cdna3_scratch_restore(words, *scratch);

  return ExpandResult::success(std::move(words));
}

// -----------------------------------------------------------------------------
// Wide-K MFMA expansions.
// -----------------------------------------------------------------------------

enum class WideKMfmaShape {
  F32_16x16x32_F16,
  F32_32x32x16_F16,
  F32_16x16x32_BF16,
  F32_32x32x16_BF16,
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
  case WideKMfmaShape::F32_16x16x32_BF16:
    return {shape, cdna3::kVMfmaF3216x16x16Bf16Vop3pMfma, 4, 4, 2};
  case WideKMfmaShape::F32_32x32x16_BF16:
    return {shape, cdna3::kVMfmaF3232x32x8Bf16Vop3pMfma, 16, 4, 2};
  }
  return {shape, 0, 0, 0, 0};
}

[[nodiscard]] bool ranges_overlap(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                  uint16_t rhs_count) {
  return lhs_base < rhs_base + rhs_count && rhs_base < lhs_base + lhs_count;
}

[[nodiscard]] bool wide_mfma_needs_partial_accum_scratch(const cdna4::Vop3pMfmaMachineInst &mfma,
                                                         const WideKMfmaLowering &lowering) {
  // acc_cd=1 writes the partial accumulator into the AccVGPR bank, so ordinary
  // A/B VGPR windows cannot be clobbered. The two-bit `acc` source selector is
  // independent: bit 0 moves source A into AccVGPRs, bit 1 moves source B into
  // AccVGPRs. Only ordinary source windows can alias an ordinary partial
  // accumulator destination.
  if (mfma.acc_cd != 0)
    return false;

  const uint16_t dst_base = static_cast<uint16_t>(mfma.vdst);
  const uint16_t src0_base = static_cast<uint16_t>(mfma.src0 - 256);
  const uint16_t src1_base = static_cast<uint16_t>(mfma.src1 - 256);
  const bool src0_is_acc = (mfma.acc & 0x1u) != 0;
  const bool src1_is_acc = (mfma.acc & 0x2u) != 0;
  return (!src0_is_acc &&
          ranges_overlap(dst_base, lowering.dst_regs, src0_base, lowering.wide_src_regs)) ||
         (!src1_is_acc &&
          ranges_overlap(dst_base, lowering.dst_regs, src1_base, lowering.wide_src_regs));
}

ExpandResult lower_wide_k_mfma_cdna4_to_cdna3(const Instruction &inst,
                                              const LivenessAnalysis &liveness,
                                              TranslationContext &context, WideKMfmaShape shape) {
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

  // CDNA4's wide-K F16/BF16 forms double the K dimension by doubling each
  // contiguous A/B VGPR source window. CDNA3 has the same output layout for the
  // narrower-K forms, so the lowering emits two narrow MFMAs over the high and
  // low halves of the source windows:
  //
  //   partial = mfma_narrow(A[2:3], B[2:3], C)
  //   D       = mfma_narrow(A[0:1], B[0:1], partial)
  //
  // The ISA manuals define the CDNA4 instruction as one wide multiply-add, not
  // as an ordered pair of narrow MFMAs.  Because this emulation must expose a
  // rounded F32 partial accumulator between the two CDNA3 instructions, the
  // split order is observable at BF16 last-bit precision. Experimentally,
  // low-then-high produced a one-BF16-ULP drift in one output element. Treat
  // this as a fixed layout decision, not a tuning knob: do not flip back to
  // low-half-first unless new ISA evidence or a more direct hardware result
  // contradicts the captured high-first behavior.
  //
  // When D is an AccVGPR destination, `partial` is the final destination and the
  // second instruction reads it back through src2. CDNA3 resolves src2 encodings
  // 256-511 to the AccVGPR bank when acc_cd=1. Forward the A/B AccVGPR source
  // selector (`acc`) to both narrow MFMAs: the split only changes each source
  // window's base, not which register file supplies the window.
  //
  // When D is an ordinary VGPR destination that overlaps either ordinary full
  // A/B source window, the first MFMA must instead write a dead VGPR run;
  // otherwise it could clobber source registers that the second MFMA has not
  // read yet.
  //
  // NYI: non-default cbsz/abid/blgp modifiers need validation against the
  // two-instruction expansion before this can preserve them safely.
  if (mfma.cbsz != 0 || mfma.abid != 0 || mfma.blgp != 0)
    return failed_existing_expand_rule(
        inst, "MFMA wide-K lowering only supports default cbsz/abid/blgp modifiers",
        {"Validate and implement non-default MFMA modifiers before enabling this form."});
  // SRC0/SRC1 are OPR_SRC_VGPR_OR_ACCVGPR operands. The ISA defines the CDNA4
  // wide forms as 128-bit source windows and the CDNA3 narrow forms as 64-bit
  // source windows; the operand value is the base of that contiguous window, and
  // 64-bit-or-wider VGPR/AccVGPR operands are even-aligned by the ISA. The split
  // can use src and src + narrow_src_regs directly without a packing step.
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
    // The emitted CDNA3 MFMA writes the temporary partial accumulator as an
    // ordinary VGPR tuple. LLVM's gfx942 disassembler models these narrow MFMA
    // tuple operands with an even-base register class (VReg_*_Align2); choosing
    // a merely-dead odd tuple such as v[93:96] creates an unencodable host MFMA.
    constexpr uint16_t kMfmaTupleBaseAlignment = 2;
    SemanticScratchAllocator scratch_allocator(inst, liveness, context,
                                               Cdna3ScratchEmitter::allocation_policy());
    SemanticScratchRequest scratch_request;
    scratch_request.count = lowering.dst_regs;
    scratch_request.alignment = kMfmaTupleBaseAlignment;
    scratch_request.allow_spill = false;
    const SemanticScratchResult scratch = scratch_allocator.acquire_vgprs(scratch_request);
    // NYI: if no dead VGPR run exists, the general solution is to spill a live
    // VGPR range and use it for the partial accumulator. That waits on spill
    // manager integration, so reject for now rather than clobbering live inputs.
    if (!scratch)
      return failed_existing_expand_rule(inst,
                                         "No free VGPR window for partial accumulator scratch",
                                         {"Add spill-backed lowering for partial accumulator "
                                          "when live inputs consume scratch registers."});
    partial_vdst = static_cast<uint8_t>(scratch.lease->base);
    partial_src2 = static_cast<uint16_t>(256 + partial_vdst);
  }

  std::vector<uint32_t> words;
  if (needs_scratch) {
    emit_cdna3_mfma_to_vgpr(words, lowering.narrow_op, mfma, partial_vdst,
                            static_cast<uint16_t>(mfma.src0 + lowering.narrow_src_regs),
                            static_cast<uint16_t>(mfma.src1 + lowering.narrow_src_regs),
                            static_cast<uint16_t>(mfma.src2));
  } else {
    emit_cdna3_mfma(words, lowering.narrow_op, mfma,
                    static_cast<uint16_t>(mfma.src0 + lowering.narrow_src_regs),
                    static_cast<uint16_t>(mfma.src1 + lowering.narrow_src_regs),
                    static_cast<uint16_t>(mfma.src2));
  }
  emit_cdna3_mfma(words, lowering.narrow_op, mfma, static_cast<uint16_t>(mfma.src0),
                  static_cast<uint16_t>(mfma.src1), partial_src2);
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
  const bool dst_is_acc = src.acc != 0;
  const uint32_t ds_offset = (static_cast<uint32_t>(src.offset1) << 8) | src.offset0;
  const uint8_t vdst = static_cast<uint8_t>(src.vdst);
  const uint8_t addr = static_cast<uint8_t>(src.addr);
  // VDST is a 64-bit destination, so it names a contiguous two-register pair.
  // Pair validity is part of the source instruction's ISA contract.

  constexpr uint16_t kScratchCount = 10;
  // The pack helper wants several even/odd register relationships to stay
  // simple, so require an even-aligned window. Do not use the LDS address
  // operand as a destination in the opening DS read. For ordinary-VGPR
  // destinations, also keep the output pair out of the scratch window because
  // a spill restore would otherwise overwrite the translated result. ACC
  // destinations name the AccVGPR file, so the same numeric ordinary VGPRs can
  // still be used when ordinary VGPR liveness permits it.
  auto scratch = dst_is_acc ? choose_vgpr_window_or_spill(inst, liveness, context, kScratchCount, 2,
                                                          {{static_cast<uint16_t>(addr), 1}})
                            : choose_vgpr_window_or_spill(inst, liveness, context, kScratchCount, 2,
                                                          {{static_cast<uint16_t>(vdst), 2},
                                                           {static_cast<uint16_t>(addr), 1}});
  if (!scratch)
    return failed_existing_expand_rule(
        inst, "No even-aligned VGPR window found for transpose scratch",
        {"Provide enough aligned scratch registers or add a spill-backed "
         "transpose lowering."});

  const uint8_t raw_lo = static_cast<uint8_t>(scratch->base + 0);
  const uint8_t raw_hi = static_cast<uint8_t>(scratch->base + 1);
  const uint8_t lane_base = static_cast<uint8_t>(scratch->base + 2);
  const uint8_t halfword_selector = static_cast<uint8_t>(scratch->base + 3);
  const uint8_t tmp = static_cast<uint8_t>(scratch->base + 4);
  const uint8_t h0 = static_cast<uint8_t>(scratch->base + 5);
  const uint8_t h1 = static_cast<uint8_t>(scratch->base + 6);
  const uint8_t h2 = static_cast<uint8_t>(scratch->base + 7);
  const uint8_t h3 = static_cast<uint8_t>(scratch->base + 8);
  const uint8_t gather_tmp = static_cast<uint8_t>(scratch->base + 9);
  const auto borrow_scratch = context.virtual_lds_base_sgpr_spill_per_use
                                  ? make_cdna3_virtual_lds_borrow_scratch(context, *scratch, h2, h3)
                                  : std::optional<Cdna3VirtualLdsBorrowScratch>{};
  if (context.virtual_lds_base_sgpr_spill_per_use && !borrow_scratch) {
    return failed_existing_expand_rule(
        inst, "virtual LDS transpose lowering cannot reserve borrowed-SGPR spill state",
        {"Provide two non-overlapping scratch VGPRs and private spill slots for the virtual-LDS "
         "access emitter."});
  }
  // SemanticScratchAllocator has already fed any descriptor growth required by
  // this window back to the kernel translation context.
  const std::optional<uint8_t> saved_exec = choose_cdna3_exec_save_sgpr(inst, liveness, context);
  if (!saved_exec)
    return failed_existing_expand_rule(
        inst, "No free descriptor-backed SGPR pair for EXEC save/restore",
        {"Add SGPR spill-backed EXEC save/restore before lowering this instruction in "
         "descriptor-full kernels."});
  require_cdna3_exec_save_sgpr(context, *saved_exec);

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
  emit_s_mov_b64(words, *saved_exec, kExecLo);
  emit_cdna3_exec_mask(words, UINT64_MAX);
  if (scratch->spilled) {
    // ds_read_b64_tr_b16 is only valid with all lanes active, and the lowering
    // already materializes that EXEC state. Save the live victim window under
    // the same all-lane contract before reusing it for transpose temporaries.
    emit_cdna3_scratch_save(words, *scratch);
  }

  if (context.virtualize_lds) {
    // CDNA4 DS names a 32-bit LDS byte offset VGPR, but CDNA3 GLOBAL/FLAT
    // encodes VADDR as an even-based 64-bit VGPR pair. The generic virtual-LDS
    // DS lowering stages odd DS address registers before emitting GLOBAL. This
    // handwritten transpose rule has to do the same because it bypasses that
    // helper. Stage through `tmp:tmp+1`, which is even because the transpose
    // scratch window is even-aligned, and which is not live until after the
    // virtual load completes. The shared access emitter also owns the
    // descriptor-full backing-pointer setup and restore around this operation.
    const uint8_t virtual_addr = tmp;
    uint32_t virtual_offset = ds_offset;
    emit_cdna3_vop2_literal(words, cdna3::kVAddU32Vop2, virtual_addr, addr,
                            ds_offset > kFlatGlobalPositiveImm13Max ? ds_offset : 0);
    if (ds_offset > kFlatGlobalPositiveImm13Max)
      virtual_offset = 0;
    if (!append_cdna3_virtual_lds_access(
            words, context,
            Cdna3VirtualLdsAccess{.is_load = true,
                                  .op = static_cast<uint8_t>(cdna3::kFlatLoadDwordFlat + 1),
                                  .data_vgpr = raw_lo,
                                  .address_vgpr = virtual_addr,
                                  .byte_offset = static_cast<uint16_t>(virtual_offset)},
            borrow_scratch)) {
      return failed_existing_expand_rule(
          inst, "virtual LDS transpose access is not encodable",
          {"Provide an even GLOBAL address pair and non-overlapping spill-per-use scratch."});
    }
  } else {
    emit_cdna3_ds(words, cdna3::kDsReadB64Ds, raw_lo, addr, 0, 0, src.offset0, src.offset1);
    emit_cdna3_lgkm_wait(words);
  }

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
  emit_cdna3_b16_transpose_halfword(words, h0, gather_tmp, lane_base, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, tmp, scalar_positive_inline_u32(16),
                  vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, h1, gather_tmp, tmp, raw_lo, raw_hi, halfword_selector);
  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, tmp, scalar_positive_inline_u32(32),
                  vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, h2, gather_tmp, tmp, raw_lo, raw_hi, halfword_selector);
  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, tmp, scalar_positive_inline_u32(48),
                  vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, h3, gather_tmp, tmp, raw_lo, raw_hi, halfword_selector);

  emit_s_mov_b64(words, kExecLo, *saved_exec);
  if (dst_is_acc) {
    // CDNA3 VALU instructions cannot write AccVGPR destinations directly. Pack
    // into ordinary scratch after the raw LDS payload is no longer needed, then
    // copy the completed 32-bit words into the architectural AccVGPR pair.
    emit_cdna3_pack_low_b16_pair(words, raw_lo, h0, h1, tmp, gather_tmp);
    emit_cdna3_pack_low_b16_pair(words, raw_hi, h2, h3, tmp, gather_tmp);
    emit_cdna3_accvgpr_write_b32(words, vdst, raw_lo);
    emit_cdna3_accvgpr_write_b32(words, static_cast<uint8_t>(vdst + 1), raw_hi);
  } else {
    emit_cdna3_pack_low_b16_pair(words, vdst, h0, h1, tmp, gather_tmp);
    emit_cdna3_pack_low_b16_pair(words, static_cast<uint8_t>(vdst + 1), h2, h3, tmp, gather_tmp);
  }
  if (scratch->spilled) {
    // The transpose gather uses all lanes to reconstruct the full source
    // footprint. Restore spill victims for all lanes after the architectural
    // destination has been written under the original EXEC mask.
    emit_cdna3_exec_mask(words, UINT64_MAX);
    emit_cdna3_scratch_restore(words, *scratch);
    emit_s_mov_b64(words, kExecLo, *saved_exec);
  }

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
  // Let liveness choose any dead ordinary VGPR run. The original MUBUF source
  // operands are live-before this instruction, so a vaddr that is dead after
  // the source LDS-DMA instruction is still protected until the emitted MUBUF
  // consumes it. Starting at the descriptor count would ignore that liveness
  // fact and unnecessarily force high-pressure kernels into descriptor growth
  // or spill paths.
  const uint8_t vaddr_count = (src.offen && src.idxen) ? 2 : ((src.offen || src.idxen) ? 1 : 0);
  const bool virtual_lds = context.virtualize_lds;
  const uint16_t virtual_address_offset =
      static_cast<uint16_t>((static_cast<uint16_t>(data_count) + 1u) & ~1u);
  const uint16_t borrow_temp_count =
      virtual_lds && context.virtual_lds_base_sgpr_spill_per_use ? 2u : 0u;
  const uint16_t scratch_count =
      virtual_lds ? static_cast<uint16_t>(virtual_address_offset + 2u + borrow_temp_count)
                  : static_cast<uint16_t>(data_count + 1u);
  // CDNA3 encodes 96-bit and 128-bit MUBUF/DS VGPR tuples with an aligned
  // register class.
  // If the materialized data tuple starts on an odd VGPR, llvm-objdump reports
  // the generated buffer_load_dwordx{3,4}/ds_write_b{96,128} operands as
  // invalid and real Qwen Tensile kernels produce corrupted MLP outputs.  The
  // extra +1 scratch register holds the scalar LDS byte address; only the
  // payload tuple needs this alignment.
  const uint16_t scratch_alignment = virtual_lds || data_count >= 3 ? 2 : 1;
  auto scratch =
      choose_vgpr_window_or_spill(inst, liveness, context, scratch_count, scratch_alignment,
                                  {{static_cast<uint16_t>(src.vaddr), vaddr_count}});
  if (!scratch) {
    return failed_existing_expand_rule(
        inst, "No scratch VGPR window found for temporary global-load destination and LDS address",
        {"Provide scratch VGPRs for the temporary global-load destination and LDS address.",
         "Add a spill-backed lowering for high-pressure inputs if no dead VGPR run exists."});
  }

  const uint8_t data = scratch->base;
  const uint8_t lds_addr =
      static_cast<uint8_t>(scratch->base + (virtual_lds ? virtual_address_offset : data_count));
  std::optional<Cdna3VirtualLdsBorrowScratch> borrow_scratch;
  if (context.virtual_lds_base_sgpr_spill_per_use) {
    const uint8_t pointer_temp_lo = static_cast<uint8_t>(lds_addr + 2u);
    const uint8_t pointer_temp_hi = static_cast<uint8_t>(lds_addr + 3u);
    borrow_scratch =
        make_cdna3_virtual_lds_borrow_scratch(context, *scratch, pointer_temp_lo, pointer_temp_hi);
    if (!borrow_scratch) {
      return failed_existing_expand_rule(
          inst, "virtual LDS MUBUF lowering cannot reserve borrowed-SGPR spill state",
          {"Provide two non-overlapping scratch VGPRs and private spill slots for the virtual-LDS "
           "access emitter."});
    }
  }
  // Prefer a new descriptor-backed SGPR pair for EXEC save/restore. If the
  // descriptor is already full, the helper falls back only to a liveness-proven
  // dead pair instead of blindly borrowing a high guest SGPR.
  const std::optional<uint8_t> saved_exec = choose_cdna3_exec_save_sgpr(inst, liveness, context);
  if (!saved_exec)
    return failed_existing_expand_rule(
        inst, "No free descriptor-backed SGPR pair for EXEC save/restore",
        {"Add SGPR spill-backed EXEC save/restore before lowering this instruction in "
         "descriptor-full kernels."});

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
  require_cdna3_exec_save_sgpr(context, *saved_exec);

  std::vector<uint32_t> words;
  // MUBUF-to-LDS uses physical TID-in-wave for the LDS lane slot, not the
  // active-lane prefix.  Compute that address with EXEC forced to all lanes,
  // then restore the original EXEC before issuing the global load and DS write
  // so inactive lanes keep the same side-effect behavior as the source
  // instruction.
  emit_s_mov_b64(words, *saved_exec, kExecLo);
  emit_cdna3_exec_mask(words, UINT64_MAX);
  if (scratch->spilled) {
    // The all-lane address calculation below clobbers `lds_addr` for lanes
    // outside the original EXEC mask. Save every lane of the spilled window so
    // restoring it later preserves dormant lanes as well as active ones.
    emit_cdna3_scratch_save(words, *scratch);
  }
  emit_cdna3_vop3(words, cdna3::kVMbcntLoU32B32Vop3, lds_addr, kInlineConstNeg1, kInlineConst0);
  emit_cdna3_vop3(words, cdna3::kVMbcntHiU32B32Vop3, lds_addr, kInlineConstNeg1,
                  vgpr_src(lds_addr));
  emit_cdna3_vop3(words, cdna3::kVLshlrevB32Vop3, lds_addr,
                  scalar_positive_inline_u32(lane_stride_shift), vgpr_src(lds_addr));
  emit_cdna3_vop3(words, cdna3::kVAddU32Vop3, lds_addr, kM0, vgpr_src(lds_addr));
  emit_s_mov_b64(words, kExecLo, *saved_exec);

  emit_cdna3_mubuf(words, src, load_op, data);
  emit_cdna3_wait_all(words);
  if (virtual_lds) {
    if (!append_cdna3_virtual_lds_access(
            words, context,
            Cdna3VirtualLdsAccess{
                .is_load = false,
                .op = static_cast<uint8_t>(cdna3::kFlatStoreDwordFlat + data_count - 1),
                .data_vgpr = data,
                .address_vgpr = lds_addr},
            borrow_scratch)) {
      return failed_existing_expand_rule(
          inst, "virtual LDS MUBUF access is not encodable",
          {"Provide an even GLOBAL address pair and non-overlapping spill-per-use scratch."});
    }
  } else {
    emit_cdna3_ds(words, ds_op, 0, lds_addr, data);
    // Native buffer_load_* ... lds exposes a VMEM-completed LDS side effect. The
    // fallback sequence creates that side effect with an explicit DS write, so
    // wait for the DS operation here before branching back to code that was
    // scheduled around the original LDS-DMA instruction.
    emit_cdna3_lgkm_wait(words);
  }
  if (scratch->spilled) {
    emit_cdna3_exec_mask(words, UINT64_MAX);
    emit_cdna3_scratch_restore(words, *scratch);
    emit_s_mov_b64(words, kExecLo, *saved_exec);
  }

  return ExpandResult::success(std::move(words));
}

ExpandResult expand_v_bitop3_b16_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                std::span<const uint8_t>,
                                                const LivenessAnalysis &liveness,
                                                TranslationContext &context, const LaneLayout *,
                                                const LaneLayout *) {
  // The rule table only routes V_BITOP3_B16 here, so use the generated
  // instruction type directly instead of re-decoding ordinary operands.
  return lower_cdna4_bitop3_to_cdna3(static_cast<const cdna4::VBitop3B16Vop3 &>(inst), liveness,
                                     context, true);
}

ExpandResult expand_v_bitop3_b32_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                std::span<const uint8_t>,
                                                const LivenessAnalysis &liveness,
                                                TranslationContext &context, const LaneLayout *,
                                                const LaneLayout *) {
  // The rule table only routes V_BITOP3_B32 here, so use the generated
  // instruction type directly instead of re-decoding ordinary operands.
  return lower_cdna4_bitop3_to_cdna3(static_cast<const cdna4::VBitop3B32Vop3 &>(inst), liveness,
                                     context, false);
}

ExpandResult expand_permlane32_swap_b32_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                       std::span<const uint8_t>,
                                                       const LivenessAnalysis &liveness,
                                                       TranslationContext &context,
                                                       const LaneLayout *, const LaneLayout *) {
  return lower_permlane_swap_b32_cdna4_to_cdna3(inst, liveness, context, 32);
}

ExpandResult expand_permlane16_swap_b32_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                       std::span<const uint8_t>,
                                                       const LivenessAnalysis &liveness,
                                                       TranslationContext &context,
                                                       const LaneLayout *, const LaneLayout *) {
  return lower_permlane_swap_b32_cdna4_to_cdna3(inst, liveness, context, 16);
}

ExpandResult expand_cvt_pk_f16_f32_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                  std::span<const uint8_t>,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context, const LaneLayout *,
                                                  const LaneLayout *) {
  return lower_cvt_pk_f16_f32_cdna4_to_cdna3(inst, liveness, context);
}

ExpandResult expand_cvt_pk_bf16_f32_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                   std::span<const uint8_t>,
                                                   const LivenessAnalysis &liveness,
                                                   TranslationContext &context, const LaneLayout *,
                                                   const LaneLayout *) {
  return lower_cvt_pk_bf16_f32_cdna4_to_cdna3(inst, liveness, context);
}

ExpandResult expand_cvt_f32_bf16_vop1_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                     uint64_t offset,
                                                     std::span<const uint8_t> source_text,
                                                     const LivenessAnalysis &, TranslationContext &,
                                                     const LaneLayout *, const LaneLayout *) {
  return lower_cvt_f32_bf16_vop1_cdna4_to_cdna3(inst, offset, source_text);
}

ExpandResult expand_ds_read_b64_tr_b16_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                      std::span<const uint8_t>,
                                                      const LivenessAnalysis &liveness,
                                                      TranslationContext &context,
                                                      const LaneLayout *, const LaneLayout *) {
  return lower_ds_read_b64_tr_b16_cdna4_to_cdna3(inst, liveness, context);
}

ExpandResult expand_mfma_f32_16x16x32_f16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                         uint64_t, std::span<const uint8_t>,
                                                         const LivenessAnalysis &liveness,
                                                         TranslationContext &context,
                                                         const LaneLayout *, const LaneLayout *) {
  return lower_wide_k_mfma_cdna4_to_cdna3(inst, liveness, context,
                                          WideKMfmaShape::F32_16x16x32_F16);
}

ExpandResult expand_mfma_f32_32x32x16_f16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                         uint64_t, std::span<const uint8_t>,
                                                         const LivenessAnalysis &liveness,
                                                         TranslationContext &context,
                                                         const LaneLayout *, const LaneLayout *) {
  return lower_wide_k_mfma_cdna4_to_cdna3(inst, liveness, context,
                                          WideKMfmaShape::F32_32x32x16_F16);
}

ExpandResult expand_mfma_f32_16x16x32_bf16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                          uint64_t, std::span<const uint8_t>,
                                                          const LivenessAnalysis &liveness,
                                                          TranslationContext &context,
                                                          const LaneLayout *, const LaneLayout *) {
  return lower_wide_k_mfma_cdna4_to_cdna3(inst, liveness, context,
                                          WideKMfmaShape::F32_16x16x32_BF16);
}

ExpandResult expand_mfma_f32_32x32x16_bf16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                          uint64_t, std::span<const uint8_t>,
                                                          const LivenessAnalysis &liveness,
                                                          TranslationContext &context,
                                                          const LaneLayout *, const LaneLayout *) {
  return lower_wide_k_mfma_cdna4_to_cdna3(inst, liveness, context,
                                          WideKMfmaShape::F32_32x32x16_BF16);
}

ExpandResult expand_buffer_load_dwordx3_lds_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                           uint64_t, std::span<const uint8_t>,
                                                           const LivenessAnalysis &liveness,
                                                           TranslationContext &context,
                                                           const LaneLayout *, const LaneLayout *) {
  return expand_mubuf_load_to_lds_cdna4_to_cdna3(inst, liveness, context, 3);
}

ExpandResult expand_buffer_load_dword_lds_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                         uint64_t, std::span<const uint8_t>,
                                                         const LivenessAnalysis &liveness,
                                                         TranslationContext &context,
                                                         const LaneLayout *, const LaneLayout *) {
  return expand_mubuf_load_to_lds_cdna4_to_cdna3(inst, liveness, context, 1);
}

ExpandResult expand_buffer_load_dwordx4_lds_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                           uint64_t, std::span<const uint8_t>,
                                                           const LivenessAnalysis &liveness,
                                                           TranslationContext &context,
                                                           const LaneLayout *, const LaneLayout *) {
  return expand_mubuf_load_to_lds_cdna4_to_cdna3(inst, liveness, context, 4);
}

ExpandResult expand_dot2_f32_bf16_cdna4_to_cdna3(const Instruction &inst, uint32_t, uint64_t,
                                                 std::span<const uint8_t>,
                                                 const LivenessAnalysis &liveness,
                                                 TranslationContext &context, const LaneLayout *,
                                                 const LaneLayout *) {
  return lower_dot2_f32_bf16_cdna4_to_cdna3(inst, liveness, context);
}

// Table MUST be sorted by (src_encoding_id, src_opcode) for binary search.
constexpr TranslationRule kExpandRules_cdna4_to_cdna3[] = {
    {cdna4::encoding::kVop1, cdna4::kVPermlane16SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane16_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1, cdna4::kVPermlane32SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1, cdna4::kVCvtF32Bf16Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_cvt_f32_bf16_vop1_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi1, cdna4::kVPermlane16SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane16_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi1, cdna4::kVPermlane32SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi1, cdna4::kVCvtF32Bf16Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_cvt_f32_bf16_vop1_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi2, cdna4::kVPermlane16SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane16_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi2, cdna4::kVPermlane32SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi2, cdna4::kVCvtF32Bf16Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_cvt_f32_bf16_vop1_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi3, cdna4::kVPermlane16SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane16_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi3, cdna4::kVPermlane32SwapB32Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop1Hi3, cdna4::kVCvtF32Bf16Vop1, RuleAction::Expand, 0, 0, nullptr,
     expand_cvt_f32_bf16_vop1_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B16Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_v_bitop3_b16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3OpHi4, cdna4::kVBitop3B32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_v_bitop3_b32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3OpHi4, cdna4::kVCvtPkF16F32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_cvt_pk_f16_f32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3OpHi4, cdna4::kVCvtPkBf16F32Vop3, RuleAction::Expand, 0, 0, nullptr,
     expand_cvt_pk_bf16_f32_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3p, cdna4::kVDot2F32Bf16Vop3p, RuleAction::Expand, 0, 0, nullptr,
     expand_dot2_f32_bf16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3pMfma, cdna4::kVMfmaF3216x16x32Bf16Vop3pMfma, RuleAction::Expand, 0, 0,
     nullptr, expand_mfma_f32_16x16x32_bf16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3pMfma, cdna4::kVMfmaF3232x32x16Bf16Vop3pMfma, RuleAction::Expand, 0, 0,
     nullptr, expand_mfma_f32_32x32x16_bf16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3pMfma, cdna4::kVMfmaF3216x16x32F16Vop3pMfma, RuleAction::Expand, 0, 0,
     nullptr, expand_mfma_f32_16x16x32_f16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kVop3pMfma, cdna4::kVMfmaF3232x32x16F16Vop3pMfma, RuleAction::Expand, 0, 0,
     nullptr, expand_mfma_f32_32x32x16_f16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kDsHi3, cdna4::kDsReadB64TrB16Ds, RuleAction::Expand, 0, 0, nullptr,
     expand_ds_read_b64_tr_b16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kDsHi7, cdna4::kDsReadB64TrB16Ds, RuleAction::Expand, 0, 0, nullptr,
     expand_ds_read_b64_tr_b16_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordMubuf, RuleAction::Expand, 0, 0, nullptr,
     expand_buffer_load_dword_lds_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx3Mubuf, RuleAction::Expand, 0, 0, nullptr,
     expand_buffer_load_dwordx3_lds_cdna4_to_cdna3, nullptr, nullptr},
    {cdna4::encoding::kMubuf, cdna4::kBufferLoadDwordx4Mubuf, RuleAction::Expand, 0, 0, nullptr,
     expand_buffer_load_dwordx4_lds_cdna4_to_cdna3, nullptr, nullptr},
};

static_assert(translation_rules_sorted(kExpandRules_cdna4_to_cdna3),
              "the CDNA4-to-CDNA3 rule table must stay sorted by (encoding id, opcode)");

} // namespace

std::span<const TranslationRule> semantic_expand_rules_cdna4_to_cdna3() {
  return std::span<const TranslationRule>(kExpandRules_cdna4_to_cdna3);
}

} // namespace rocjitsu
