// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_rdna4.cpp
/// @brief CDNA4-to-RDNA4 handwritten semantic expansion rules.

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/hazard_tracker.h"
#include "rocjitsu/code/dbt/semantic/cdna4_to_rdna_common.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/dbt/waitcnt_translator.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include "rocjitsu/code/dbt/generated/matrix_conversions.h"

#include <bit>
#include <cassert>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

[[nodiscard]] uint32_t make_gfx12_sopp(uint8_t op, uint16_t simm16) {
  rdna4::SoppMachineInst s{};
  s.encoding = 0x17F;
  s.op = op;
  s.simm16 = simm16;
  return std::bit_cast<uint32_t>(s);
}

/// @brief Build VOP3P instruction word pair (packed math: WMMA, dot products).
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_vop3p(uint8_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2) {
  const uint32_t w0 = static_cast<uint32_t>(vdst) | (1u << 14) |
                      (static_cast<uint32_t>(op & 0x7F) << 16) | (0xCCu << 24);
  const uint32_t w1 = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18) | (3u << 27);
  return {w0, w1};
}

/// @brief Build VOP3 instruction word pair for RDNA4 VALU instructions.
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_vop3(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1 = 0, uint16_t src2 = 0) {
  const uint32_t w0 = (vdst & 0xFFu) | ((op & 0x3FFu) << 16) | (0x35u << 26);
  const uint32_t w1 = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
  return {w0, w1};
}

/// @brief Build VOP2 instruction word (xor, lshlrev, add_nc, etc.).
[[nodiscard]] constexpr uint32_t build_vop2(uint8_t op, uint8_t vdst, uint16_t src0,
                                            uint8_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
}

/// @brief Build s_mov_b64 sdst, ssrc0.
[[nodiscard]] constexpr uint32_t build_s_mov_b64(uint8_t sdst, uint16_t ssrc0) {
  rdna4::Sop1MachineInst s{};
  s.encoding = 0x17D;
  s.op = 1;
  s.sdst = sdst & 0x7F;
  s.ssrc0 = ssrc0 & 0xFF;
  return std::bit_cast<uint32_t>(s);
}

/// @brief Build s_mov_b32 sdst, literal (two-word instruction).
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_s_mov_b32_lit(uint8_t sdst,
                                                                          uint32_t literal) {
  rdna4::Sop1MachineInst s{};
  s.encoding = 0x17D;
  s.op = 0;
  s.sdst = sdst & 0x7F;
  s.ssrc0 = 0xFF;
  return {std::bit_cast<uint32_t>(s), literal};
}

/// @brief Build ds_bpermute_b32 vdst, vaddr, vdata.
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_ds_bpermute(uint8_t vdst, uint8_t vaddr,
                                                                        uint8_t vdata) {
  constexpr uint32_t kDsW0 = (0xB3u << 18) | (0x36u << 26);
  return {kDsW0, static_cast<uint32_t>(vaddr) | (static_cast<uint32_t>(vdata) << 8) |
                     (static_cast<uint32_t>(vdst) << 24)};
}

/// @brief Build a VOP1 v_mov_b32 instruction for RDNA4.
[[nodiscard]] constexpr uint32_t build_v_mov_b32(uint8_t vdst, uint16_t src0) {
  return (0x3Fu << 25) | (static_cast<uint32_t>(vdst) << 17) | (1u << 9) | (src0 & 0x1FF);
}

/// @brief Lower v_accvgpr_read_b32 to v_mov_b32 or NOP on RDNA4.
/// @details AccVGPR N is represented as unified VGPR N+256 in the current
/// descriptor translation model. If the destination already aliases that
/// unified register, a NOP preserves the source instruction slot.
std::vector<uint32_t> lower_accvgpr_read(const Instruction &inst) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));

  const uint16_t dst_vgpr = src.vdst;
  const uint16_t src_acc = src.src0;
  assert(src_acc >= 768 && src_acc <= 1023);
  const uint16_t src_unified = src_acc - 512;

  if (dst_vgpr == src_unified)
    return {build_s_nop()};

  return {build_v_mov_b32(static_cast<uint8_t>(dst_vgpr), 256 + src_unified)};
}

constexpr uint8_t kSoppWaitIdle = 0x0A;
constexpr uint8_t kOpWaitLoadcnt = 64;
constexpr uint8_t kOpWaitDscnt = 70;

constexpr uint8_t kExecLo = 126;
constexpr uint16_t kInlineConst0 = 128;
constexpr uint16_t kInlineConst2 = 130;
constexpr uint16_t kInlineConstNeg1 = 193;

constexpr uint16_t kOpMbcntLo = 0x31F;
constexpr uint16_t kOpMbcntHi = 0x320;
constexpr uint8_t kOpLshlrevB32 = 24;
constexpr uint8_t kOpXorB32 = 29;
constexpr uint8_t kOpWmmaF32_16x16x16_F16 = 64;

/// @brief Lower v_mfma_f32_16x16x16_f16 to v_wmma_f32_16x16x16_f16 on RDNA4.
/// @details WMMA Wave64 writes all 64 lanes but swaps rows 4-7 and 8-11 vs
/// MFMA layout. The ds_bpermute tail corrects the affected lanes using the
/// generated matrix conversion table.
std::vector<uint32_t> lower_mfma_f32_16x16x16_f16(const Instruction &inst,
                                                  const LivenessAnalysis &liveness) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::Vop3pMfmaMachineInst mfma{};
  std::memcpy(&mfma, raw, sizeof(mfma));

  const uint16_t vdst = mfma.vdst;
  const uint16_t src0 = mfma.src0;
  const uint16_t src1 = mfma.src1;
  const uint16_t src2 = mfma.src2;

  if (src2 >= 256)
    return {};

  assert(src0 >= 256 && src1 >= 256 && "MFMA VGPR sources expected");

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt)
    return {};
  const uint8_t kExecSave = static_cast<uint8_t>(*exec_save_opt);

  auto tmp_sgpr_opt = liveness.find_free_sgpr(&inst, kExecSave + 2);
  if (!tmp_sgpr_opt)
    return {};
  const uint8_t kTmpSgpr = static_cast<uint8_t>(*tmp_sgpr_opt);

  auto free_reg = liveness.find_free_run(&inst, 1, vdst + 4);
  if (!free_reg)
    return {};
  const uint8_t vaddr = static_cast<uint8_t>(*free_reg);

  std::vector<uint32_t> words;

  words.push_back(make_gfx12_sopp(kOpWaitLoadcnt, 0));
  words.push_back(build_s_mov_b64(kExecSave, kExecLo));

  // Compute bpermute byte addresses as lane_id * 4. HazardTracker inserts the
  // required s_delay_alu instructions between dependent VALU/SALU operations.
  using P = HazardTracker::Pipeline;
  HazardTracker hz;

  {
    auto [w0, w1] = build_vop3(kOpMbcntLo, vaddr, kInlineConstNeg1, kInlineConst0);
    hz.emit2(words, w0, w1, P::VALU);
  }
  {
    auto [w0, w1] = build_vop3(kOpMbcntHi, vaddr, kInlineConstNeg1, 256 + vaddr);
    hz.emit2(words, w0, w1, P::VALU);
  }
  hz.emit(words, build_vop2(kOpLshlrevB32, vaddr, kInlineConst2, vaddr), P::VALU);

  LanePermutation perm{};
  for (size_t i = 0; i < rocjitsu::kMatrixConversionCount; ++i) {
    if (std::string_view(rocjitsu::kMatrixConversions[i].src_mnemonic) == inst.mnemonic()) {
      perm = {rocjitsu::kMatrixConversions[i].xor_byte_mask,
              rocjitsu::kMatrixConversions[i].range_start,
              rocjitsu::kMatrixConversions[i].range_end};
      break;
    }
  }
  if (perm.xor_byte_mask != 0) {
    auto [sw0, sw1] = build_s_mov_b32_lit(kTmpSgpr, perm.xor_byte_mask);
    hz.emit2(words, sw0, sw1, P::SALU);
    uint64_t exec_mask = 0;
    for (uint8_t lane = perm.range_start; lane < perm.range_end; ++lane)
      exec_mask |= (1ULL << lane);
    auto [el, lit] = build_s_mov_b32_lit(kExecLo, static_cast<uint32_t>(exec_mask));
    hz.emit2(words, el, lit, P::None); // EXEC writes are outside HazardTracker's model.
    auto [eh, lith] = build_s_mov_b32_lit(kExecLo + 1, static_cast<uint32_t>(exec_mask >> 32));
    hz.emit2(words, eh, lith, P::None);
    hz.emit(words, build_vop2(kOpXorB32, vaddr, kTmpSgpr, vaddr), P::VALU);
  }

  words.push_back(build_s_mov_b64(kExecLo, kExecSave));

  {
    auto [w0, w1] = build_vop3p(kOpWmmaF32_16x16x16_F16, vdst, src0, src1, src2);
    words.push_back(w0);
    words.push_back(w1);
  }

  // Drain WMMA before ds_bpermute reads its VGPR outputs.
  words.push_back(pack_sopp(kSoppWaitIdle, 0));

  for (int r = 0; r < 4; ++r) {
    auto [w0, w1] = build_ds_bpermute(vdst + r, vaddr, vdst + r);
    words.push_back(w0);
    words.push_back(w1);
  }
  words.push_back(pack_sopp(kOpWaitDscnt, 0));

  words.push_back(build_s_mov_b64(kExecLo, kExecSave));

  return words;
}

std::vector<uint32_t> expand_waitcnt(const Instruction &inst, uint32_t, uint64_t,
                                     const LivenessAnalysis &, const LaneLayout *,
                                     const LaneLayout *) {
  if (!inst.raw_encoding())
    return {};
  const auto &sopp = *reinterpret_cast<const cdna4::SoppMachineInst *>(inst.raw_encoding());
  return encode_waitcnt_gfx12(decode_waitcnt_gfx9(sopp.simm16));
}

std::vector<uint32_t> expand_accvgpr_read(const Instruction &inst, uint32_t, uint64_t,
                                          const LivenessAnalysis &, const LaneLayout *,
                                          const LaneLayout *) {
  return lower_accvgpr_read(inst);
}

std::vector<uint32_t> expand_accvgpr_write(const Instruction &, uint32_t, uint64_t,
                                           const LivenessAnalysis &, const LaneLayout *,
                                           const LaneLayout *) {
  // AccVGPR writes are already represented by the unified VGPR mapping that
  // descriptor translation reserves for RDNA targets.
  return {build_s_nop()};
}

std::vector<uint32_t> expand_mfma_f32_16x16x16_f16(const Instruction &inst, uint32_t, uint64_t,
                                                   const LivenessAnalysis &liveness,
                                                   const LaneLayout *, const LaneLayout *) {
  return lower_mfma_f32_16x16x16_f16(inst, liveness);
}

// CDNA4 encoding IDs (encoding_id = w0 >> 23, from CDNA4 instruction words).
constexpr uint16_t kEncSopp = 0x17F;      // SOPP (0xBF8x -> 0x17F)
constexpr uint16_t kEncVop3 = 0x1A4;      // VOP3A (0xD2xx -> 0x1A4)
constexpr uint16_t kEncVop3pMfma = 0x1A7; // VOP3P-MFMA (0xD3xx -> 0x1A7)

// CDNA4 opcodes (from decoder: opcode_ = inst_.op).
constexpr uint16_t kCdna4Op_s_waitcnt = 12;
constexpr uint16_t kCdna4Op_v_mfma_f32_16x16x16_f16 = 77;
constexpr uint16_t kCdna4Op_v_accvgpr_read = 88;
constexpr uint16_t kCdna4Op_v_accvgpr_write = 89;
constexpr uint16_t kCdna4Op_v_lshl_add_u64 = 520;

// Table MUST be sorted by (src_encoding_id, src_opcode) for binary search.
const TranslationRule kExpandRules_cdna4_to_rdna4[] = {
    {kEncSopp, kCdna4Op_s_waitcnt, RuleAction::Expand, 0, 0, nullptr, expand_waitcnt, nullptr,
     nullptr},
    {kEncVop3, kCdna4Op_v_lshl_add_u64, RuleAction::Expand, 0, 0, nullptr,
     expand_cdna4_v_lshl_add_u64_for_rdna, nullptr, nullptr},
    {kEncVop3pMfma, kCdna4Op_v_mfma_f32_16x16x16_f16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_16x16x16_f16, &kMfmaF32_16x16x16_F16_Cdna4, &kWmmaF32_16x16x16_F16_Rdna4},
    {kEncVop3pMfma, kCdna4Op_v_accvgpr_read, RuleAction::Expand, 0, 0, nullptr, expand_accvgpr_read,
     nullptr, nullptr},
    {kEncVop3pMfma, kCdna4Op_v_accvgpr_write, RuleAction::Expand, 0, 0, nullptr,
     expand_accvgpr_write, nullptr, nullptr},
};

static_assert(rocjitsu::kMatrixConversionCount >= 2,
              "Auto-generated matrix conversion table too small");
static_assert(rocjitsu::kMatrixConversions[1].xor_byte_mask == 192,
              "v_mfma_f32_16x16x16_f16 XOR mask mismatch");
static_assert(rocjitsu::kMatrixConversions[1].range_start == 16,
              "v_mfma_f32_16x16x16_f16 range_start mismatch");
static_assert(rocjitsu::kMatrixConversions[1].range_end == 48,
              "v_mfma_f32_16x16x16_f16 range_end mismatch");

} // namespace

std::span<const TranslationRule> semantic_expand_rules_cdna4_to_rdna4() {
  return std::span<const TranslationRule>(kExpandRules_cdna4_to_rdna4);
}

} // namespace rocjitsu
