// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_rdna3.cpp
/// @brief CDNA4-to-RDNA3 handwritten semantic expansion rules.

#include "rocjitsu/code/dbt/semantic/cdna4_to_rdna_common.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/opcodes.h"
#include "rocjitsu/isa/instruction.h"

#include <string>
#include <vector>

namespace rocjitsu {
namespace {

/// @brief Lower CDNA4 GFX9-style s_waitcnt to RDNA3 GFX11-style s_waitcnt.
/// @details The counter-bit layouts differ, so opcode substitution alone can
/// create a weaker wait. Until we add precise GFX11 re-encoding, emit a
/// conservative full-drain wait, which is slower but preserves correctness.
ExpandResult expand_waitcnt_gfx9_to_gfx11(const Instruction &inst, uint32_t, uint64_t,
                                          const LivenessAnalysis &, TranslationContext &,
                                          const LaneLayout *, const LaneLayout *) {
  if (inst.encoding_id() != cdna4::encoding::kSopp)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " matched the waitcnt expansion rule with an unexpected encoding");

  return ExpandResult::success({pack_sopp(rdna3::kSWaitcnt, 0)});
}

// Table MUST be sorted by (src_encoding_id, src_opcode) for binary search.
const TranslationRule kExpandRules_cdna4_to_rdna3[] = {
    {cdna4::encoding::kSopp, cdna4::kSWaitcnt, RuleAction::Expand, 0, 0, nullptr,
     expand_waitcnt_gfx9_to_gfx11, nullptr, nullptr},
    {cdna4::encoding::kVop3OpHi4, cdna4::kVLshlAddU64, RuleAction::Expand, 0, 0, nullptr,
     expand_cdna4_v_lshl_add_u64_for_rdna, nullptr, nullptr},
};

} // namespace

std::span<const TranslationRule> semantic_expand_rules_cdna4_to_rdna3() {
  return std::span<const TranslationRule>(kExpandRules_cdna4_to_rdna3);
}

} // namespace rocjitsu
