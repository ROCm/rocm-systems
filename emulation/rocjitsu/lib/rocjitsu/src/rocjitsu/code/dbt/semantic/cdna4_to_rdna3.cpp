// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_rdna3.cpp
/// @brief CDNA4-to-RDNA3 handwritten semantic expansion rules.

#include "rocjitsu/code/dbt/semantic/cdna4_to_rdna_common.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/instruction.h"

#include <string>
#include <vector>

namespace rocjitsu {
namespace {

// CDNA4 encoding IDs (encoding_id = w0 >> 23, from CDNA4 instruction words).
constexpr uint16_t kEncSopp = 0x17F; // SOPP (0xBF8x -> 0x17F)
constexpr uint16_t kEncVop3 = 0x1A4; // VOP3A (0xD2xx -> 0x1A4)

// CDNA4 opcodes (from decoder: opcode_ = inst_.op).
constexpr uint16_t kCdna4Op_s_waitcnt = 12;
constexpr uint16_t kCdna4Op_v_lshl_add_u64 = 520;

/// @brief Lower CDNA4 GFX9-style s_waitcnt to RDNA3 GFX11-style s_waitcnt.
/// @details The counter-bit layouts differ, so opcode substitution alone can
/// create a weaker wait. Until we add precise GFX11 re-encoding, emit a
/// conservative full-drain wait, which is slower but preserves correctness.
ExpandResult expand_waitcnt_gfx9_to_gfx11(const Instruction &inst, uint32_t, uint64_t,
                                          const LivenessAnalysis &, TranslationContext &,
                                          const LaneLayout *, const LaneLayout *) {
  constexpr uint16_t kEncSoppValue = 0x17F;
  if (inst.encoding_id() != kEncSoppValue)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " matched the waitcnt expansion rule with an unexpected encoding");

  constexpr uint32_t kRdna3SoppOp_s_waitcnt = 9;
  return ExpandResult::success({pack_sopp(kRdna3SoppOp_s_waitcnt, 0)});
}

// Table MUST be sorted by (src_encoding_id, src_opcode) for binary search.
const TranslationRule kExpandRules_cdna4_to_rdna3[] = {
    {kEncSopp, kCdna4Op_s_waitcnt, RuleAction::Expand, 0, 0, nullptr, expand_waitcnt_gfx9_to_gfx11,
     nullptr, nullptr},
    {kEncVop3, kCdna4Op_v_lshl_add_u64, RuleAction::Expand, 0, 0, nullptr,
     expand_cdna4_v_lshl_add_u64_for_rdna, nullptr, nullptr},
};

} // namespace

std::span<const TranslationRule> semantic_expand_rules_cdna4_to_rdna3() {
  return std::span<const TranslationRule>(kExpandRules_cdna4_to_rdna3);
}

} // namespace rocjitsu
