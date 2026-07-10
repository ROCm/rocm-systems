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
#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/opcodes.h"
#include "rocjitsu/isa/instruction.h"

#include "rocjitsu/code/dbt/generated/matrix_conversions.h"

#include <bit>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

[[nodiscard]] uint32_t make_gfx12_sopp(uint16_t op, uint16_t simm16) {
  rdna4::SoppMachineInst s{};
  s.encoding = rdna4::encoding::kSopp;
  s.op = op & 0x7F;
  s.simm16 = simm16;
  return std::bit_cast<uint32_t>(s);
}

/// @brief Build VOP3P instruction word pair (packed math: WMMA, dot products).
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_vop3p(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2) {
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
[[nodiscard]] constexpr uint32_t build_vop2(uint16_t op, uint8_t vdst, uint16_t src0,
                                            uint8_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
}

/// @brief Build s_mov_b64 sdst, ssrc0.
[[nodiscard]] constexpr uint32_t build_s_mov_b64(uint8_t sdst, uint16_t ssrc0) {
  rdna4::Sop1MachineInst s{};
  s.encoding = rdna4::encoding::kSop1;
  s.op = rdna4::kSMovB64;
  s.sdst = sdst & 0x7F;
  s.ssrc0 = ssrc0 & 0xFF;
  return std::bit_cast<uint32_t>(s);
}

/// @brief Build s_mov_b32 sdst, literal (two-word instruction).
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t> build_s_mov_b32_lit(uint8_t sdst,
                                                                          uint32_t literal) {
  rdna4::Sop1MachineInst s{};
  s.encoding = rdna4::encoding::kSop1;
  s.op = rdna4::kSMovB32;
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
ExpandResult lower_accvgpr_read(const Instruction &inst) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot lower AccVGPR read without a complete 64-bit raw "
                                "encoding");

  cdna4::Vop3pMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));

  const uint16_t dst_vgpr = src.vdst;
  const uint16_t src_acc = src.src0;
  if (src_acc < 768 || src_acc > 1023)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " AccVGPR read lowering expected src0 in the AccVGPR operand "
                                "range 768..1023, got " +
                                std::to_string(src_acc));
  const uint16_t src_unified = src_acc - 512;

  if (dst_vgpr == src_unified)
    return ExpandResult::success({build_s_nop()});

  return ExpandResult::success(
      {build_v_mov_b32(static_cast<uint8_t>(dst_vgpr), static_cast<uint16_t>(256 + src_unified))});
}

constexpr uint8_t kExecLo = 126;
constexpr uint16_t kInlineConst0 = 128;
constexpr uint16_t kInlineConst2 = 130;
constexpr uint16_t kInlineConstNeg1 = 193;

/// @brief Lower v_mfma_f32_16x16x16_f16 to v_wmma_f32_16x16x16_f16 on RDNA4.
/// @details WMMA Wave64 writes all 64 lanes but swaps rows 4-7 and 8-11 vs
/// MFMA layout. The ds_bpermute tail corrects the affected lanes using the
/// generated matrix conversion table.
ExpandResult lower_mfma_f32_16x16x16_f16(const Instruction &inst, const LivenessAnalysis &liveness,
                                         TranslationContext &context) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot lower MFMA without a complete 64-bit raw encoding");

  cdna4::Vop3pMfmaMachineInst mfma{};
  std::memcpy(&mfma, raw, sizeof(mfma));

  const uint16_t vdst = mfma.vdst;
  const uint16_t src0 = mfma.src0;
  const uint16_t src1 = mfma.src1;
  const uint16_t src2 = mfma.src2;

  if (src2 >= 256)
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
            " MFMA lowering only supports a VGPR accumulator source; got src2 operand " +
            std::to_string(src2),
        {"Add an MFMA lowering path for non-VGPR accumulator operands."});

  if (src0 < 256 || src1 < 256)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                    " MFMA lowering expected VGPR matrix sources, got src0=" +
                                    std::to_string(src0) + " src1=" + std::to_string(src1),
                                {"Add an MFMA lowering path for scalar or inline-constant "
                                 "matrix operands."});

  auto exec_save_opt = liveness.find_free_sgpr_pair(&inst);
  if (!exec_save_opt)
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
            " MFMA lowering could not find a free SGPR pair to save EXEC",
        {"Reduce SGPR pressure, improve liveness, or add scalar spill support for semantic "
         "lowerings."});
  const uint8_t kExecSave = static_cast<uint8_t>(*exec_save_opt);

  auto tmp_sgpr_opt = liveness.find_free_sgpr(&inst, kExecSave + 2);
  if (!tmp_sgpr_opt)
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
            " MFMA lowering could not find a temporary SGPR after the EXEC save pair",
        {"Reduce SGPR pressure, improve liveness, or add scalar spill support for semantic "
         "lowerings."});
  const uint8_t kTmpSgpr = static_cast<uint8_t>(*tmp_sgpr_opt);

  auto free_reg = liveness.find_free_run(&inst, 1, vdst + 4);
  if (!free_reg)
    return ExpandResult::failed(
        std::string(inst.mnemonic()) +
            " MFMA lowering could not find a free VGPR for ds_bpermute addresses at or above v" +
            std::to_string(vdst + 4),
        {"Reduce VGPR pressure, improve liveness, or add VGPR spill support for semantic "
         "lowerings."});
  const uint8_t vaddr = static_cast<uint8_t>(*free_reg);

  // Record feedback only after the full scratch allocation succeeds. A failed
  // lowering returns empty and should not grow descriptor resources for code
  // that was never emitted.
  context.require_sgprs(static_cast<uint32_t>(kExecSave) + 2);
  context.require_sgprs(static_cast<uint32_t>(kTmpSgpr) + 1);
  context.require_vgprs(static_cast<uint32_t>(vaddr) + 1);

  std::vector<uint32_t> words;

  words.push_back(make_gfx12_sopp(rdna4::kSWaitLoadcnt, 0));
  words.push_back(build_s_mov_b64(kExecSave, kExecLo));

  // Compute bpermute byte addresses as lane_id * 4. HazardTracker inserts the
  // required s_delay_alu instructions between dependent VALU/SALU operations.
  using P = HazardTracker::Pipeline;
  HazardTracker hz;

  {
    auto [w0, w1] = build_vop3(rdna4::kVMbcntLoU32B32Vop3, vaddr, kInlineConstNeg1, kInlineConst0);
    hz.emit2(words, w0, w1, P::VALU);
  }
  {
    auto [w0, w1] = build_vop3(rdna4::kVMbcntHiU32B32Vop3, vaddr, kInlineConstNeg1, 256 + vaddr);
    hz.emit2(words, w0, w1, P::VALU);
  }
  hz.emit(words, build_vop2(rdna4::kVLshlrevB32Vop2, vaddr, kInlineConst2, vaddr), P::VALU);

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
    hz.emit(words, build_vop2(rdna4::kVXorB32Vop2, vaddr, kTmpSgpr, vaddr), P::VALU);
  }

  words.push_back(build_s_mov_b64(kExecLo, kExecSave));

  {
    auto [w0, w1] = build_vop3p(rdna4::kVWmmaF3216x16x16F16, vdst, src0, src1, src2);
    words.push_back(w0);
    words.push_back(w1);
  }

  // Drain WMMA before ds_bpermute reads its VGPR outputs.
  words.push_back(pack_sopp(rdna4::kSWaitIdle, 0));

  for (int r = 0; r < 4; ++r) {
    auto [w0, w1] = build_ds_bpermute(vdst + r, vaddr, vdst + r);
    words.push_back(w0);
    words.push_back(w1);
  }
  words.push_back(pack_sopp(rdna4::kSWaitDscnt, 0));

  words.push_back(build_s_mov_b64(kExecLo, kExecSave));

  return ExpandResult::success(std::move(words));
}

ExpandResult expand_waitcnt(const Instruction &inst, uint32_t, uint64_t, const LivenessAnalysis &,
                            TranslationContext &, const LaneLayout *, const LaneLayout *) {
  if (!inst.raw_encoding())
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " matched the waitcnt expansion rule without raw encoding");
  if (static_cast<size_t>(inst.size()) < sizeof(cdna4::SoppMachineInst))
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " matched the waitcnt expansion rule with a truncated SOPP "
                                "encoding");
  const auto &sopp = *reinterpret_cast<const cdna4::SoppMachineInst *>(inst.raw_encoding());
  return ExpandResult::success(encode_waitcnt_gfx12(decode_waitcnt_gfx9(sopp.simm16)));
}

ExpandResult expand_accvgpr_read(const Instruction &inst, uint32_t, uint64_t,
                                 const LivenessAnalysis &, TranslationContext &, const LaneLayout *,
                                 const LaneLayout *) {
  return lower_accvgpr_read(inst);
}

ExpandResult expand_accvgpr_write(const Instruction &, uint32_t, uint64_t, const LivenessAnalysis &,
                                  TranslationContext &, const LaneLayout *, const LaneLayout *) {
  // AccVGPR writes are already represented by the unified VGPR mapping that
  // descriptor translation reserves for RDNA targets.
  return ExpandResult::success({build_s_nop()});
}

ExpandResult expand_mfma_f32_16x16x16_f16(const Instruction &inst, uint32_t, uint64_t,
                                          const LivenessAnalysis &liveness,
                                          TranslationContext &context, const LaneLayout *,
                                          const LaneLayout *) {
  return lower_mfma_f32_16x16x16_f16(inst, liveness, context);
}

// Table MUST be sorted by (src_encoding_id, src_opcode) for binary search.
const TranslationRule kExpandRules_cdna4_to_rdna4[] = {
    {cdna4::encoding::kSopp, cdna4::kSWaitcnt, RuleAction::Expand, 0, 0, nullptr, expand_waitcnt,
     nullptr, nullptr},
    {cdna4::encoding::kVop3OpHi4, cdna4::kVLshlAddU64, RuleAction::Expand, 0, 0, nullptr,
     expand_cdna4_v_lshl_add_u64_for_rdna, nullptr, nullptr},
    {cdna4::encoding::kVop3pMfma, cdna4::kVMfmaF3216x16x16F16, RuleAction::Expand, 0, 0, nullptr,
     expand_mfma_f32_16x16x16_f16, &kMfmaF32_16x16x16_F16_Cdna4, &kWmmaF32_16x16x16_F16_Rdna4},
    {cdna4::encoding::kVop3pMfma, cdna4::kVAccvgprRead, RuleAction::Expand, 0, 0, nullptr,
     expand_accvgpr_read, nullptr, nullptr},
    {cdna4::encoding::kVop3pMfma, cdna4::kVAccvgprWrite, RuleAction::Expand, 0, 0, nullptr,
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
