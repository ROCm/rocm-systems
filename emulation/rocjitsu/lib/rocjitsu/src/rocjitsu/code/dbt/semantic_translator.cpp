// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic_translator.cpp
/// @brief Small dispatch facade for ISA-pair semantic expansion rules.

#include "rocjitsu/code/dbt/semantic_translator.h"

#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>

namespace rocjitsu {

namespace {

/// @brief Select the handwritten semantic rule table for one ISA pair.
/// @details Empty spans are intentional: most ISA pairs currently rely
/// entirely on generated legalization and encoding translation.
[[nodiscard]] std::span<const TranslationRule> semantic_expand_rules_for(rj_code_arch_t guest,
                                                                         rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return semantic_expand_rules_cdna4_to_rdna4();
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return semantic_expand_rules_cdna4_to_cdna3();
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return semantic_expand_rules_cdna4_to_rdna3();
  if (guest == ROCJITSU_CODE_ARCH_GFX1250 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return semantic_expand_rules_gfx1250_to_rdna4();
  return {};
}

} // namespace

SemanticTranslator::SemanticTranslator(rj_code_arch_t guest, rj_code_arch_t host)
    : expand_rules_(semantic_expand_rules_for(guest, host)), host_arch_(host) {}

ExpandResult SemanticTranslator::try_lower_expand(const Instruction &inst, uint64_t offset,
                                                  std::span<const uint8_t> source_text,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context) const {
  return try_lower_expand_with_opcode(inst, offset, source_text, liveness, context, inst.opcode());
}

ExpandResult SemanticTranslator::try_lower_expand_with_opcode(
    const Instruction &inst, uint64_t offset, std::span<const uint8_t> source_text,
    const LivenessAnalysis &liveness, TranslationContext &context, uint16_t opcode) const {
  const uint16_t eid = inst.encoding_id();
  auto try_rule = [&](uint16_t rule_opcode) -> ExpandResult {
    TranslationRule key{eid,     rule_opcode, RuleAction::Expand, 0, 0, nullptr, nullptr,
                        nullptr, nullptr};
    auto it = std::lower_bound(expand_rules_.begin(), expand_rules_.end(), key);
    if (it == expand_rules_.end() || it->src_encoding_id != eid || it->src_opcode != rule_opcode ||
        !it->expand_fn)
      return ExpandResult::not_handled();
    return it->expand_fn(inst, static_cast<uint32_t>(host_arch_), offset, source_text, liveness,
                         context, it->guest_layout, it->host_layout);
  };

  ExpandResult exact = try_rule(opcode);
  if (exact.status != ExpandStatus::NotHandled)
    return exact;

  ExpandResult wildcard = try_rule(kAnyTranslationOpcode);
  if (wildcard.status != ExpandStatus::NotHandled)
    return wildcard;
  return ExpandResult::not_handled();
}

} // namespace rocjitsu
