// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic_translator.cpp
/// @brief Small dispatch facade for ISA-pair semantic expansion rules.

#include "rocjitsu/code/dbt/semantic_translator.h"

#include "rocjitsu/isa/instruction.h"

#include <algorithm>

namespace rocjitsu {

SemanticTranslator::SemanticTranslator(std::span<const TranslationRule> expand_rules,
                                       rj_code_arch_t host_arch)
    : expand_rules_(expand_rules), host_arch_(host_arch) {
  expand_rule_keys_.reserve(expand_rules_.size());
  uint16_t max_encoding_id = 0;
  for (const TranslationRule &rule : expand_rules_) {
    expand_rule_keys_.push_back(packed_rule_key(rule.src_encoding_id, rule.src_opcode));
    max_encoding_id = std::max(max_encoding_id, rule.src_encoding_id);
  }
  if (!expand_rules_.empty()) {
    // Candidate collection scans every decoded instruction in large kernels.
    // Most encodings have no handwritten semantic rules, so this tiny bitset
    // avoids probing the sorted (encoding, opcode) table for obvious misses.
    expand_rule_encoding_bits_.assign(static_cast<size_t>(max_encoding_id / 64) + 1, 0);
    for (const TranslationRule &rule : expand_rules_)
      expand_rule_encoding_bits_[rule.src_encoding_id / 64] |= uint64_t{1}
                                                               << (rule.src_encoding_id % 64);
  }
}

const TranslationRule *SemanticTranslator::find_expand_rule(const Instruction &inst) const {
  const uint32_t key = packed_rule_key(inst.encoding_id(), inst.opcode());
  auto it = std::lower_bound(expand_rule_keys_.begin(), expand_rule_keys_.end(), key);
  if (it == expand_rule_keys_.end() || *it != key)
    return nullptr;
  const size_t index = static_cast<size_t>(it - expand_rule_keys_.begin());
  const TranslationRule &rule = expand_rules_[index];
  return rule.expand_fn ? &rule : nullptr;
}

ExpandResult SemanticTranslator::try_lower_expand(const Instruction &inst, uint64_t offset,
                                                  std::span<const uint8_t> source_text,
                                                  const LivenessAnalysis &liveness,
                                                  TranslationContext &context) const {
  const TranslationRule *rule = find_expand_rule(inst);
  if (rule != nullptr)
    return rule->expand_fn(inst, static_cast<uint32_t>(host_arch_), offset, source_text, liveness,
                           context, rule->guest_layout, rule->host_layout);
  return ExpandResult::not_handled();
}

bool SemanticTranslator::has_expand_rule(const Instruction &inst) const {
  return has_expand_rule(inst.encoding_id(), inst.opcode());
}

} // namespace rocjitsu
