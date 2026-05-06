// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna4_to_cdna3.cpp
/// @brief Semantic expand rules for translating CDNA4 instructions to CDNA3.

#include "rocjitsu/code/dbt/translations/translations.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <vector>

namespace rocjitsu {
namespace {

// CDNA4 -> CDNA3 expand rules. CDNA3 keeps the same CDNA/GFX9 base encoding
// model for most instructions, but CDNA4 adds opcodes that require semantic
// synthesis:
//   - V_BITOP3_B16/B32 encode a ternary bitwise LUT in normal VOP3 modifier
//     fields; CDNA3 has only ordinary binary boolean ops.
//   - V_MFMA_F32_* wide-K F16 forms double K versus the CDNA3 forms. They
//     expand to two narrow-K MFMAs after reshaping the source operand layout.
//   - DS_READ_B64_TR_B16 performs an LDS load plus a wave-level matrix
//     transpose; CDNA3 has the ordinary LDS load and ds_bpermute machinery, but
//     not this fused transposed load.

// -----------------------------------------------------------------------------
// CDNA3 instruction builder utilities.
// -----------------------------------------------------------------------------

/// @brief Build a CDNA3 VOP3 instruction word pair.
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
build_cdna3_vop3p_mfma(uint8_t op, const cdna4::Vop3pMfmaMachineInst &src, uint16_t src0,
                       uint16_t src1, uint16_t src2) {
  cdna3::Vop3pMfmaMachineInst dst{};
  dst.encoding = 0x1A7;
  dst.op = op;
  dst.vdst = src.vdst;
  dst.cbsz = src.cbsz;
  dst.abid = src.abid;
  dst.acc_cd = src.acc_cd;
  dst.src0 = src0 & 0x1FF;
  dst.src1 = src1 & 0x1FF;
  dst.src2 = src2 & 0x1FF;
  dst.acc = 0;
  dst.blgp = src.blgp;

  uint32_t words[2]{};
  std::memcpy(words, &dst, sizeof(dst));
  return {words[0], words[1]};
}

/// @brief Build a CDNA3 DS instruction word pair.
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

[[maybe_unused]] [[nodiscard]] constexpr uint32_t build_s_mov_b64(uint8_t sdst, uint16_t ssrc0) {
  return pack_sop1(1, sdst, ssrc0);
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_s_mov_b32_lit(uint8_t sdst,
                                                                          uint32_t literal) {
  return {pack_sop1(0, sdst, 0xFF), literal};
}

constexpr uint8_t kExecLo = 126;
constexpr uint8_t kM0 = 124;
constexpr uint16_t kInlineConst0 = 128;
constexpr uint16_t kInlineConstNeg1 = 193;

// CDNA3 scalar and vector opcodes used by the expansion rules below.
constexpr uint16_t kCdna3Sop2OpAndB64 = 13;
constexpr uint16_t kCdna3OpMovB32 = 321;
constexpr uint16_t kCdna3OpLshrrevB32 = 272;
constexpr uint16_t kCdna3OpLshlrevB32 = 274;
constexpr uint16_t kCdna3OpAndB32 = 275;
constexpr uint16_t kCdna3OpOrB32 = 276;
constexpr uint16_t kCdna3OpXorB32 = 277;
constexpr uint16_t kCdna3OpAddU32 = 308;
constexpr uint16_t kCdna3OpCvtF16F32 = 330;
constexpr uint16_t kCdna3OpPermB32 = 493;
constexpr uint16_t kCdna3OpMbcntLoU32B32 = 652;
constexpr uint16_t kCdna3OpMbcntHiU32B32 = 653;
constexpr uint16_t kCdna3OpPackB32F16 = 672;

// CDNA3 VOP3P-MFMA opcodes used by CDNA4 wide-K MFMA expansion.
constexpr uint8_t kCdna3OpMfmaF32_32x32x8F16 = 76;
constexpr uint8_t kCdna3OpMfmaF32_16x16x16F16 = 77;

// CDNA3 DS opcodes used to synthesize CDNA4 transposed LDS reads.
constexpr uint8_t kCdna3DsOpBpermuteB32 = 63;
constexpr uint8_t kCdna3DsOpReadB64 = 118;
constexpr uint8_t kCdna3DsOpWriteB128 = 223;

constexpr uint8_t kCdna3SoppOpWaitcnt = 12;
constexpr uint16_t kCdnaWaitcntLgkmcnt0 = 0xC07F;
constexpr uint16_t kCdnaWaitcntVmcnt0 = 0x0F70;
constexpr uint8_t kCdna3FlatOpLoadDwordx4 = 23;

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
  // their no-wait maxima. This is conservative for the DS read/bpermute
  // sequences that call this helper; TODO: see whether these waits can be
  // removed, hoisted, or narrowed once the translator has stronger DS
  // dependency modeling.
  words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntLgkmcnt0));
}

void emit_cdna3_vmem_wait(std::vector<uint32_t> &words) {
  // GFX9/CDNA s_waitcnt encodes "vmcnt(0)" with EXP/LGKM at their no-wait
  // maxima. The global_load_lds_dwordxN expansion emits more VMEM operations
  // than the guest instruction, so it drains those replacement loads locally
  // instead of perturbing the surrounding guest wait-count schedule.
  words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntVmcnt0));
}

[[nodiscard]] constexpr uint16_t vgpr_src(uint8_t reg) { return static_cast<uint16_t>(256 + reg); }

void emit_s_mov_b32_lit(std::vector<uint32_t> &words, uint8_t sdst, uint32_t literal) {
  auto [w0, w1] = build_s_mov_b32_lit(sdst, literal);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_s_mov_b64(std::vector<uint32_t> &words, uint8_t sdst, uint16_t ssrc0) {
  words.push_back(build_s_mov_b64(sdst, ssrc0));
}

void emit_s_mov_b64_lit(std::vector<uint32_t> &words, uint8_t sdst, uint64_t literal) {
  emit_s_mov_b32_lit(words, sdst, static_cast<uint32_t>(literal));
  emit_s_mov_b32_lit(words, static_cast<uint8_t>(sdst + 1), static_cast<uint32_t>(literal >> 32));
}

void emit_s_and_b64(std::vector<uint32_t> &words, uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  words.push_back(pack_sop2(kCdna3Sop2OpAndB64, sdst, ssrc0, ssrc1));
}

[[maybe_unused]] void emit_cdna3_exec_mask(std::vector<uint32_t> &words, uint64_t mask) {
  emit_s_mov_b32_lit(words, kExecLo, static_cast<uint32_t>(mask));
  emit_s_mov_b32_lit(words, kExecLo + 1, static_cast<uint32_t>(mask >> 32));
}

[[nodiscard]] bool vgpr_run_overlaps(uint16_t base, uint16_t count, uint16_t reg) {
  return reg >= base && reg < static_cast<uint16_t>(base + count);
}

std::optional<uint16_t> find_free_run_avoiding(const LivenessAnalysis &liveness,
                                               const Instruction &inst, uint16_t count,
                                               uint16_t avoid0, uint16_t avoid1,
                                               uint16_t search_start = 0) {
  uint16_t search = search_start;
  while (true) {
    auto candidate = liveness.find_free_run(&inst, count, search);
    if (!candidate)
      return std::nullopt;
    if (!vgpr_run_overlaps(*candidate, count, avoid0) &&
        !vgpr_run_overlaps(*candidate, count, avoid1))
      return candidate;
    search = static_cast<uint16_t>(*candidate + 1);
  }
}

std::optional<uint16_t> find_free_sgpr_pair_avoiding(const LivenessAnalysis &liveness,
                                                     const Instruction &inst, uint16_t avoid_pair) {
  uint16_t search = 0;
  while (true) {
    auto candidate = liveness.find_free_sgpr_pair(&inst, search);
    if (!candidate)
      return std::nullopt;
    if (*candidate + 1 < avoid_pair || *candidate > avoid_pair + 1)
      return candidate;
    search = static_cast<uint16_t>(*candidate + 2);
  }
}

void emit_cdna3_mfma(std::vector<uint32_t> &words, uint8_t op,
                     const cdna4::Vop3pMfmaMachineInst &src, uint16_t src0, uint16_t src1,
                     uint16_t src2) {
  auto [w0, w1] = build_cdna3_vop3p_mfma(op, src, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_flat_load_dwordx4(std::vector<uint32_t> &words,
                                  const cdna4::FlatGlblMachineInst &src, uint8_t vdst) {
  cdna3::FlatMachineInst dst{};
  dst.encoding = src.encoding;
  dst.op = kCdna3FlatOpLoadDwordx4;
  dst.offset = src.offset & 0x0FFF;
  dst.seg = src.seg;
  dst.sc0 = src.sc0;
  dst.nt = src.nt;
  dst.sc1 = src.sc1;
  dst.addr = src.addr;
  // Preserve the GLOBAL address mode.  saddr=0x7f means the address comes from
  // the VGPR pair in addr; clearing it would reinterpret addr as a 32-bit
  // offset from an SGPR base and can fault on real hardware.
  dst.saddr = src.saddr;
  dst.acc = src.acc;
  dst.vdst = vdst;

  uint32_t emitted[2]{};
  std::memcpy(emitted, &dst, sizeof(dst));
  words.push_back(emitted[0]);
  words.push_back(emitted[1]);
}

void emit_cdna3_ds_write_b128(std::vector<uint32_t> &words, uint8_t addr, uint8_t data) {
  cdna3::DsMachineInst dst{};
  dst.encoding = 0x36;
  dst.op = kCdna3DsOpWriteB128;
  dst.addr = addr;
  dst.data0 = data;

  uint32_t emitted[2]{};
  std::memcpy(emitted, &dst, sizeof(dst));
  words.push_back(emitted[0]);
  words.push_back(emitted[1]);
}

void emit_cdna3_lds_dma_dwordx4_addr(std::vector<uint32_t> &words, uint8_t dst,
                                     uint8_t exec_save_sgpr) {
  // GLOBAL_LOAD_LDS takes a wave-uniform LDS base in M0 and implicitly adds
  // 16 * lane_id for the four-dword vector payload.  DS_WRITE_B128 has no such
  // implicit lane offset, so materialize the byte address in a scratch VGPR.
  //
  // MBCNT counts active lanes before the current lane. Temporarily enabling
  // all lanes turns that into the physical lane id, matching GLOBAL_LOAD_LDS
  // even when the guest instruction itself is EXEC-masked.
  emit_s_mov_b64(words, exec_save_sgpr, kExecLo);
  emit_cdna3_exec_mask(words, 0xFFFFFFFFFFFFFFFFULL);
  emit_cdna3_vop3(words, kCdna3OpMbcntLoU32B32, dst, kInlineConstNeg1, kInlineConst0);
  emit_cdna3_vop3(words, kCdna3OpMbcntHiU32B32, dst, kInlineConstNeg1, vgpr_src(dst));
  emit_cdna3_vop3(words, kCdna3OpLshlrevB32, dst, scalar_positive_inline_u32(4), vgpr_src(dst));
  emit_cdna3_vop3(words, kCdna3OpAddU32, dst, kM0, vgpr_src(dst));
  emit_s_mov_b64(words, kExecLo, exec_save_sgpr);
}

// -----------------------------------------------------------------------------
// V_BITOP3 expansions.
// -----------------------------------------------------------------------------

struct Bitop3Operands {
  uint8_t vdst = 0;
  uint16_t src[3]{};
  uint8_t truth_table = 0;
};

/// @brief Extract the overloaded V_BITOP3 truth table from a CDNA4 VOP3 word pair.
/// @details The ISA defines TTBL as { OMOD[1:0], ABS[2:0], NEG[2:0] }. Normal
/// VOP3 output modifiers, abs, and neg controls are disabled for these opcodes;
/// those encoding fields are data bits, not arithmetic modifiers.
[[nodiscard]] std::optional<Bitop3Operands> decode_cdna4_bitop3(const Instruction &inst) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop3MachineInst))
    return std::nullopt;

  cdna4::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));

  Bitop3Operands operands;
  operands.vdst = static_cast<uint8_t>(src.vdst);
  operands.src[0] = static_cast<uint16_t>(src.src0);
  operands.src[1] = static_cast<uint16_t>(src.src1);
  operands.src[2] = static_cast<uint16_t>(src.src2);
  operands.truth_table =
      static_cast<uint8_t>(((src.omod & 0x3) << 6) | ((src.abs & 0x7) << 3) | (src.neg & 0x7));
  return operands;
}

/// @brief Convert the 3-input truth table into algebraic-normal-form coefficients.
/// @details The truth-table index is exactly the ISA index {S0[i], S1[i], S2[i]}:
/// bit 2 is S0, bit 1 is S1, bit 0 is S2. The resulting coefficients map
/// directly to AND/XOR over CDNA3 bitwise instructions.
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

[[nodiscard]] bool vdst_aliases_any_vgpr_source(uint8_t vdst, const uint16_t src[3]) {
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

std::vector<uint32_t> lower_cdna4_bitop3_to_cdna3(const Instruction &inst,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context, bool is_b16) {
  auto decoded = decode_cdna4_bitop3(inst);
  if (!decoded)
    return {};
  const Bitop3Operands &op = *decoded;
  const auto coeff = bitop3_anf_coefficients(op.truth_table);

  const bool needs_acc_temp = vdst_aliases_any_vgpr_source(op.vdst, op.src);
  const bool needs_term_temp = bitop3_needs_product_term(coeff);
  const uint16_t scratch_count =
      static_cast<uint16_t>((needs_acc_temp ? 1 : 0) + (needs_term_temp ? 1 : 0));

  uint8_t acc = op.vdst;
  uint8_t term = 0;
  if (scratch_count != 0) {
    auto scratch = liveness.find_free_run(&inst, scratch_count, op.vdst + 1);
    if (!scratch)
      scratch = liveness.find_free_run(&inst, scratch_count);
    if (!scratch)
      return {};

    context.require_vgprs(*scratch + scratch_count);
    uint16_t next = *scratch;
    if (needs_acc_temp)
      acc = static_cast<uint8_t>(next++);
    if (needs_term_temp)
      term = static_cast<uint8_t>(next++);
  }

  // Emit the ternary LUT as algebraic-normal-form boolean code. Each enabled
  // one-input term becomes a v_mov_b32 or v_xor_b32 against the accumulator.
  // Each enabled two-/three-input product first materializes a scratch term
  // with v_and_b32, then xors that term into the accumulator. B16 variants keep
  // only the low half by shifting left and right by 16 before the final move
  // back to vdst.
  std::vector<uint32_t> words;

  auto src_for_variable = [&](uint8_t variable_mask) -> uint16_t {
    switch (variable_mask) {
    case 4:
      return op.src[0];
    case 2:
      return op.src[1];
    default:
      return op.src[2];
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
        emit_and(term, static_cast<uint16_t>(256 + term), variables[2]);
      term_src = static_cast<uint16_t>(256 + term);
    }

    if (!acc_initialized) {
      emit_mov(acc, term_src);
      acc_initialized = true;
    } else {
      emit_xor(acc, static_cast<uint16_t>(256 + acc), term_src);
    }
  }

  if (!acc_initialized)
    emit_mov(acc, kInlineConst0);

  if (is_b16) {
    const uint16_t shift16 = scalar_positive_inline_u32(16);
    emit_cdna3_vop3(words, kCdna3OpLshlrevB32, acc, shift16, static_cast<uint16_t>(256 + acc));
    emit_cdna3_vop3(words, kCdna3OpLshrrevB32, acc, shift16, static_cast<uint16_t>(256 + acc));
  }

  if (acc != op.vdst)
    emit_mov(op.vdst, static_cast<uint16_t>(256 + acc));

  return words;
}

// -----------------------------------------------------------------------------
// Packed F32->F16 conversion expansion.
// -----------------------------------------------------------------------------

std::vector<uint32_t> lower_v_cvt_pk_f16_f32_cdna4_to_cdna3(const Instruction &inst,
                                                            const LivenessAnalysis &liveness,
                                                            TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop3MachineInst))
    return {};

  cdna4::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));

  // The CDNA3 packed conversion opcode is the explicit RTZ form. CDNA4's
  // V_CVT_PK_F16_F32 follows the normal F32->F16 rounding path, so lower it as
  // two scalar half conversions followed by a pure halfword pack.
  std::optional<uint16_t> scratch = liveness.find_free_run(&inst, 2, src.vdst + 1);
  if (!scratch)
    scratch = liveness.find_free_run(&inst, 2);
  if (!scratch)
    return {};
  context.require_vgprs(*scratch + 2);

  const uint8_t lo = static_cast<uint8_t>(*scratch);
  const uint8_t hi = static_cast<uint8_t>(*scratch + 1);

  std::vector<uint32_t> words;
  emit_cdna3_vop3(words, kCdna3OpCvtF16F32, lo, static_cast<uint16_t>(src.src0));
  emit_cdna3_vop3(words, kCdna3OpCvtF16F32, hi, static_cast<uint16_t>(src.src1));
  emit_cdna3_vop3(words, kCdna3OpPackB32F16, static_cast<uint8_t>(src.vdst), vgpr_src(lo),
                  vgpr_src(hi));
  return words;
}

// -----------------------------------------------------------------------------
// Row-grouped permlane swap expansion.
// -----------------------------------------------------------------------------

[[nodiscard]] std::optional<uint8_t> decode_vop1_src_vgpr(uint16_t src0) {
  if (src0 >= 256 && src0 <= 511)
    return static_cast<uint8_t>(src0 - 256);

  // V_PERMLANE*_SWAP_B32 declares SRC0 as OPR_SRC_VGPR. Reject known non-VGPR
  // extension sentinels, but tolerate compact VGPR numbering if the decoder
  // ever sees that form for this VGPR-only operand.
  if (src0 < 249)
    return static_cast<uint8_t>(src0);
  return std::nullopt;
}

std::vector<uint32_t> lower_v_permlane32_swap_b32_cdna4_to_cdna3(const Instruction &inst,
                                                                 const LivenessAnalysis &liveness,
                                                                 TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop1MachineInst))
    return {};

  cdna4::Vop1MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  auto src0_opt = decode_vop1_src_vgpr(static_cast<uint16_t>(src.src0));
  if (!src0_opt)
    return {};

  const uint8_t vdst = static_cast<uint8_t>(src.vdst);
  const uint8_t src0 = *src0_opt;

  auto exec_save = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save)
    return {};
  auto exec_mask = find_free_sgpr_pair_avoiding(liveness, inst, *exec_save);
  if (!exec_mask)
    return {};
  context.require_sgprs(std::max<uint16_t>(*exec_save + 2, *exec_mask + 2));

  constexpr uint16_t kScratchCount = 3;
  const uint16_t preferred_start = std::max<uint16_t>(vdst, src0) + 1;
  auto scratch = find_free_run_avoiding(liveness, inst, kScratchCount, vdst, src0, preferred_start);
  if (!scratch)
    scratch = find_free_run_avoiding(liveness, inst, kScratchCount, vdst, src0);
  if (!scratch)
    return {};
  context.require_vgprs(*scratch + kScratchCount);

  const uint8_t lane_addr = static_cast<uint8_t>(*scratch);
  const uint8_t staged_dst = static_cast<uint8_t>(*scratch + 1);
  const uint8_t staged_src = static_cast<uint8_t>(*scratch + 2);
  const uint8_t exec_save_sgpr = static_cast<uint8_t>(*exec_save);
  const uint8_t exec_mask_sgpr = static_cast<uint8_t>(*exec_mask);

  std::vector<uint32_t> words;

  // Build the partner-lane byte address with EXEC forced to all lanes. MBCNT
  // derives lane_id from EXEC, and DS_BPERMUTE only sources active lanes, so
  // the staging phase must run with a full wave even when the guest EXEC mask
  // is partial. The only clobbered inactive-lane state is scratch VGPR state.
  emit_s_mov_b64(words, exec_save_sgpr, kExecLo);
  emit_cdna3_exec_mask(words, 0xFFFFFFFFFFFFFFFFULL);
  emit_cdna3_vop3(words, kCdna3OpMbcntLoU32B32, lane_addr, kInlineConstNeg1, kInlineConst0);
  emit_cdna3_vop3(words, kCdna3OpMbcntHiU32B32, lane_addr, kInlineConstNeg1, vgpr_src(lane_addr));
  emit_cdna3_vop3(words, kCdna3OpXorB32, lane_addr, scalar_positive_inline_u32(32),
                  vgpr_src(lane_addr));
  emit_cdna3_vop3(words, kCdna3OpLshlrevB32, lane_addr, scalar_positive_inline_u32(2),
                  vgpr_src(lane_addr));

  emit_cdna3_ds(words, kCdna3DsOpBpermuteB32, staged_dst, lane_addr, vdst);
  emit_cdna3_ds(words, kCdna3DsOpBpermuteB32, staged_src, lane_addr, src0);
  emit_cdna3_lgkm_wait(words);

  // CDNA4 swaps vdst lanes 32-63 with src0 lanes 0-31. Preserve the original
  // EXEC mask for the writeback, so inactive guest lanes keep their values.
  emit_s_mov_b64_lit(words, exec_mask_sgpr, 0x00000000FFFFFFFFULL);
  emit_s_and_b64(words, kExecLo, exec_save_sgpr, exec_mask_sgpr);
  emit_cdna3_vop3(words, kCdna3OpMovB32, src0, vgpr_src(staged_dst));

  emit_s_mov_b64_lit(words, exec_mask_sgpr, 0xFFFFFFFF00000000ULL);
  emit_s_and_b64(words, kExecLo, exec_save_sgpr, exec_mask_sgpr);
  emit_cdna3_vop3(words, kCdna3OpMovB32, vdst, vgpr_src(staged_src));

  emit_s_mov_b64(words, kExecLo, exec_save_sgpr);
  return words;
}

// -----------------------------------------------------------------------------
// Global-to-LDS vector async-copy expansion.
// -----------------------------------------------------------------------------

std::vector<uint32_t> lower_global_load_lds_dwordx4_cdna4_to_cdna3(const Instruction &inst,
                                                                   const LivenessAnalysis &liveness,
                                                                   TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::FlatGlblMachineInst))
    return {};

  cdna4::FlatGlblMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));

  // CDNA3 has no decodable GLOBAL_LOAD_LDS vector form. Materialize the copy
  // as an ordinary flat/global vector load into scratch VGPRs, then store those
  // four dwords into LDS through DS_WRITE_B128. GLOBAL_LOAD_LDS uses M0 plus
  // an implicit per-lane vector-byte offset as the LDS address; DS writes need
  // that address materialized explicitly in a VGPR.
  if (src.offset > 0x0FFFu)
    return {};

  auto exec_save = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save)
    return {};
  context.require_sgprs(*exec_save + 2);

  constexpr uint16_t kScratchCount = 5;
  const uint16_t preferred_start = static_cast<uint16_t>(src.addr + 2);
  auto scratch = find_free_run_avoiding(liveness, inst, kScratchCount, src.addr, src.addr + 1,
                                        preferred_start);
  if (!scratch)
    scratch = find_free_run_avoiding(liveness, inst, kScratchCount, src.addr, src.addr + 1);
  if (!scratch)
    return {};
  context.require_vgprs(*scratch + kScratchCount);

  const uint8_t data = static_cast<uint8_t>(*scratch);
  const uint8_t lds_addr = static_cast<uint8_t>(*scratch + 4);

  std::vector<uint32_t> words;
  emit_cdna3_lds_dma_dwordx4_addr(words, lds_addr, static_cast<uint8_t>(*exec_save));
  emit_cdna3_flat_load_dwordx4(words, src, data);
  emit_cdna3_vmem_wait(words);
  emit_cdna3_ds_write_b128(words, lds_addr, data);
  emit_cdna3_lgkm_wait(words);
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

[[nodiscard]] bool is_arch_vgpr_run(uint16_t src, uint8_t regs) {
  return src >= 256 && src + regs <= 512;
}

[[nodiscard]] bool is_even_aligned_arch_vgpr_run(uint16_t src, uint8_t regs) {
  return is_arch_vgpr_run(src, regs) && ((src - 256) % 2 == 0);
}

std::vector<uint32_t> lower_wide_k_mfma_f16_cdna4_to_cdna3(const Instruction &inst,
                                                           const LivenessAnalysis &,
                                                           TranslationContext &,
                                                           WideKMfmaShape shape) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::Vop3pMfmaMachineInst))
    return {};

  cdna4::Vop3pMfmaMachineInst mfma{};
  std::memcpy(&mfma, raw, sizeof(mfma));
  const WideKMfmaLowering lowering = lowering_for_shape(shape);
  if (lowering.narrow_op == 0)
    return {};

  const uint16_t accum_src = static_cast<uint16_t>(256 + mfma.vdst);

  // Triton usually starts a chain from inline zero and then accumulates into
  // the same AccVGPR destination window. Static, unrolled matmuls can also seed
  // one destination window from another AccVGPR window. That is legal for this
  // lowering because the scratch registers live in architectural VGPRs, while
  // src2 is read from the accumulator bank by the emitted narrow MFMAs.
  if (mfma.cbsz != 0 || mfma.abid != 0 || mfma.blgp != 0 || mfma.acc != 0)
    return {};
  const uint16_t src2 = static_cast<uint16_t>(mfma.src2);
  const bool src2_is_acc_window =
      src2 >= 256 && static_cast<uint16_t>(src2 + lowering.dst_regs) <= 512;
  if (src2 != kInlineConst0 && !src2_is_acc_window)
    return {};
  if (!is_even_aligned_arch_vgpr_run(static_cast<uint16_t>(mfma.src0), lowering.wide_src_regs) ||
      !is_even_aligned_arch_vgpr_run(static_cast<uint16_t>(mfma.src1), lowering.wide_src_regs))
    return {};
  if (static_cast<uint16_t>(mfma.vdst) + lowering.dst_regs > 256)
    return {};

  // CDNA4's wide-K F16 forms double the K dimension by doubling the contiguous
  // VGPR source window: four source VGPRs instead of the CDNA3 narrow form's
  // two. The matrix operation is a sum over K, so split the source window into
  // two legal CDNA3 narrow-K MFMAs and reassociate the accumulation:
  //
  //   D = C + dot(A[0:1], B[0:1]) + dot(A[2:3], B[2:3])
  //
  // The first narrow MFMA uses the original C operand. The second narrow MFMA
  // uses the destination accumulator window as C, preserving the original
  // destination layout without any lane repacking or EXEC manipulation.
  std::vector<uint32_t> words;
  emit_cdna3_mfma(words, lowering.narrow_op, mfma, static_cast<uint16_t>(mfma.src0),
                  static_cast<uint16_t>(mfma.src1), static_cast<uint16_t>(mfma.src2));
  emit_cdna3_mfma(words, lowering.narrow_op, mfma,
                  static_cast<uint16_t>(mfma.src0 + lowering.narrow_src_regs),
                  static_cast<uint16_t>(mfma.src1 + lowering.narrow_src_regs), accum_src);
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

std::vector<uint32_t> lower_ds_read_b64_tr_b16_cdna4_to_cdna3(const Instruction &inst,
                                                              const LivenessAnalysis &liveness,
                                                              TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || static_cast<size_t>(inst.size()) < sizeof(cdna4::DsMachineInst))
    return {};

  cdna4::DsMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.gds != 0)
    return {};

  const uint8_t vdst = static_cast<uint8_t>(src.vdst);
  const uint8_t addr = static_cast<uint8_t>(src.addr);
  if (src.vdst > 254)
    return {};

  constexpr uint16_t kScratchCount = 8;
  uint16_t scratch_start =
      std::max<uint16_t>(static_cast<uint16_t>(vdst + 2), static_cast<uint16_t>(addr + 1));
  if ((scratch_start & 1) != 0)
    ++scratch_start;

  std::optional<uint16_t> scratch;
  for (uint16_t search = scratch_start; search + kScratchCount <= 256;) {
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
    return {};
  const uint32_t scratch_end = *scratch + kScratchCount;
  if (scratch_end > 256)
    return {};
  context.require_vgprs(scratch_end);

  const uint8_t raw_lo = static_cast<uint8_t>(*scratch + 0);
  const uint8_t raw_hi = static_cast<uint8_t>(*scratch + 1);
  const uint8_t lane_base = static_cast<uint8_t>(*scratch + 2);
  const uint8_t halfword_selector = static_cast<uint8_t>(*scratch + 3);
  const uint8_t tmp = static_cast<uint8_t>(*scratch + 4);
  const uint8_t halfword_lo = static_cast<uint8_t>(*scratch + 5);
  const uint8_t halfword_hi = static_cast<uint8_t>(*scratch + 6);
  const uint8_t gather_tmp = static_cast<uint8_t>(*scratch + 7);

  std::vector<uint32_t> words;

  emit_cdna3_ds(words, kCdna3DsOpReadB64, raw_lo, addr, 0, 0, src.offset0, src.offset1);
  emit_cdna3_lgkm_wait(words);

  emit_cdna3_vop3(words, kCdna3OpMbcntLoU32B32, tmp, kInlineConstNeg1, kInlineConst0);
  emit_cdna3_vop3(words, kCdna3OpMbcntHiU32B32, tmp, kInlineConstNeg1, vgpr_src(tmp));

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
  emit_cdna3_vop3(words, kCdna3OpPackB32F16, vdst, vgpr_src(halfword_lo), vgpr_src(halfword_hi));

  emit_cdna3_vop3(words, kCdna3OpAddU32, tmp, scalar_positive_inline_u32(32), vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, halfword_lo, gather_tmp, tmp, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_vop3(words, kCdna3OpAddU32, tmp, scalar_positive_inline_u32(48), vgpr_src(lane_base));
  emit_cdna3_b16_transpose_halfword(words, halfword_hi, gather_tmp, tmp, raw_lo, raw_hi,
                                    halfword_selector);
  emit_cdna3_vop3(words, kCdna3OpPackB32F16, static_cast<uint8_t>(vdst + 1), vgpr_src(halfword_lo),
                  vgpr_src(halfword_hi));

  return words;
}

std::vector<uint32_t> expand_v_bitop3_b16_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                         uint64_t, const LivenessAnalysis &liveness,
                                                         TranslationContext &context,
                                                         const LaneLayout *, const LaneLayout *) {
  return lower_cdna4_bitop3_to_cdna3(inst, liveness, context, true);
}

std::vector<uint32_t> expand_v_bitop3_b32_cdna4_to_cdna3(const Instruction &inst, uint32_t,
                                                         uint64_t, const LivenessAnalysis &liveness,
                                                         TranslationContext &context,
                                                         const LaneLayout *, const LaneLayout *) {
  return lower_cdna4_bitop3_to_cdna3(inst, liveness, context, false);
}

std::vector<uint32_t> expand_ds_read_b64_tr_b16_cdna4_to_cdna3(
    const Instruction &inst, uint32_t, uint64_t, const LivenessAnalysis &liveness,
    TranslationContext &context, const LaneLayout *, const LaneLayout *) {
  return lower_ds_read_b64_tr_b16_cdna4_to_cdna3(inst, liveness, context);
}

std::vector<uint32_t> expand_v_cvt_pk_f16_f32_cdna4_to_cdna3(
    const Instruction &inst, uint32_t, uint64_t, const LivenessAnalysis &liveness,
    TranslationContext &context, const LaneLayout *, const LaneLayout *) {
  return lower_v_cvt_pk_f16_f32_cdna4_to_cdna3(inst, liveness, context);
}

std::vector<uint32_t> expand_v_permlane32_swap_b32_cdna4_to_cdna3(
    const Instruction &inst, uint32_t, uint64_t, const LivenessAnalysis &liveness,
    TranslationContext &context, const LaneLayout *, const LaneLayout *) {
  return lower_v_permlane32_swap_b32_cdna4_to_cdna3(inst, liveness, context);
}

std::vector<uint32_t> expand_global_load_lds_dwordx4_cdna4_to_cdna3(
    const Instruction &inst, uint32_t, uint64_t, const LivenessAnalysis &liveness,
    TranslationContext &context, const LaneLayout *, const LaneLayout *) {
  return lower_global_load_lds_dwordx4_cdna4_to_cdna3(inst, liveness, context);
}

std::vector<uint32_t> expand_mfma_f32_16x16x32_f16_cdna4_to_cdna3(
    const Instruction &inst, uint32_t, uint64_t, const LivenessAnalysis &liveness,
    TranslationContext &context, const LaneLayout *, const LaneLayout *) {
  return lower_wide_k_mfma_f16_cdna4_to_cdna3(inst, liveness, context,
                                              WideKMfmaShape::F32_16x16x32_F16);
}

std::vector<uint32_t> expand_mfma_f32_32x32x16_f16_cdna4_to_cdna3(
    const Instruction &inst, uint32_t, uint64_t, const LivenessAnalysis &liveness,
    TranslationContext &context, const LaneLayout *, const LaneLayout *) {
  return lower_wide_k_mfma_f16_cdna4_to_cdna3(inst, liveness, context,
                                              WideKMfmaShape::F32_32x32x16_F16);
}

constexpr uint16_t kEncVop1Dst0 = 0xFC;
constexpr uint16_t kEncVop1Dst1 = 0xFD;
constexpr uint16_t kEncVop1Dst2 = 0xFE;
constexpr uint16_t kEncVop1Dst3 = 0xFF;
constexpr uint16_t kEncVop3 = 0x1A4;
constexpr uint16_t kEncVop3pMfma = 0x1A7;
constexpr uint16_t kEncDsReadB64TrB16 = 0x1B3;
constexpr uint16_t kEncFlatGlblGlobalLoadLdsDwordx4Sc1Zero = 0x1BB;
constexpr uint16_t kEncFlatGlblGlobalLoadLdsDwordx4Sc1One = 0x1BF;

constexpr uint16_t kCdna4Op_v_mfma_f32_16x16x32_f16 = 84;
constexpr uint16_t kCdna4Op_v_mfma_f32_32x32x16_f16 = 85;
constexpr uint16_t kCdna4Op_v_permlane32_swap_b32 = 90;
constexpr uint16_t kCdna4Op_v_bitop3_b16 = 563;
constexpr uint16_t kCdna4Op_v_bitop3_b32 = 564;
constexpr uint16_t kCdna4Op_v_cvt_pk_f16_f32 = 615;
constexpr uint16_t kCdna4Op_ds_read_b64_tr_b16 = 227;
constexpr uint16_t kCdna4Op_global_load_lds_dwordx4 = 125;

// Rule table must stay sorted by (src_encoding_id, src_opcode).
const TranslationRule kExpandRules_cdna4_to_cdna3[] = {
    {kEncVop1Dst0, kCdna4Op_v_permlane32_swap_b32, RuleAction::Expand, 0, 0, nullptr,
     expand_v_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop1Dst1, kCdna4Op_v_permlane32_swap_b32, RuleAction::Expand, 0, 0, nullptr,
     expand_v_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop1Dst2, kCdna4Op_v_permlane32_swap_b32, RuleAction::Expand, 0, 0, nullptr,
     expand_v_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop1Dst3, kCdna4Op_v_permlane32_swap_b32, RuleAction::Expand, 0, 0, nullptr,
     expand_v_permlane32_swap_b32_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop3, kCdna4Op_v_bitop3_b16, RuleAction::Expand, 0, 0, nullptr,
     expand_v_bitop3_b16_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop3, kCdna4Op_v_bitop3_b32, RuleAction::Expand, 0, 0, nullptr,
     expand_v_bitop3_b32_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop3, kCdna4Op_v_cvt_pk_f16_f32, RuleAction::Expand, 0, 0, nullptr,
     expand_v_cvt_pk_f16_f32_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop3pMfma, kCdna4Op_v_mfma_f32_16x16x32_f16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_16x16x32_f16_cdna4_to_cdna3, nullptr, nullptr},
    {kEncVop3pMfma, kCdna4Op_v_mfma_f32_32x32x16_f16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_32x32x16_f16_cdna4_to_cdna3, nullptr, nullptr},
    {kEncDsReadB64TrB16, kCdna4Op_ds_read_b64_tr_b16, RuleAction::Expand, 0, 0, nullptr,
     expand_ds_read_b64_tr_b16_cdna4_to_cdna3, nullptr, nullptr},
    {kEncFlatGlblGlobalLoadLdsDwordx4Sc1Zero, kCdna4Op_global_load_lds_dwordx4, RuleAction::Expand,
     0, 0, nullptr, expand_global_load_lds_dwordx4_cdna4_to_cdna3, nullptr, nullptr},
    {kEncFlatGlblGlobalLoadLdsDwordx4Sc1One, kCdna4Op_global_load_lds_dwordx4, RuleAction::Expand,
     0, 0, nullptr, expand_global_load_lds_dwordx4_cdna4_to_cdna3, nullptr, nullptr},
};

} // namespace

// -----------------------------------------------------------------------------
// API.
// -----------------------------------------------------------------------------

std::span<const TranslationRule> cdna4_to_cdna3_expand_rules() {
  return kExpandRules_cdna4_to_cdna3;
}

} // namespace rocjitsu
