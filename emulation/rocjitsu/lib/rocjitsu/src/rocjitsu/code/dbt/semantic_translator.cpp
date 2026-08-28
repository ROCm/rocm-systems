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
[[nodiscard]] std::span<const TranslationRule>
semantic_expand_rules_for(rj_code_arch_t guest, rj_code_arch_t host,
                          ProcessorRevision input_revision, ProcessorRevision output_revision) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return semantic_expand_rules_cdna4_to_rdna4();
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return semantic_expand_rules_cdna4_to_cdna3();
  // gfx1250 A0 and B0 share one architectural target ID. Select the B0-to-A0
  // profile only for that explicit revision pair.
  if (guest == ROCJITSU_CODE_ARCH_GFX1250 && host == ROCJITSU_CODE_ARCH_GFX1250 &&
      input_revision == ProcessorRevision::Gfx1250B0 &&
      output_revision == ProcessorRevision::Gfx1250A0)
    return semantic_expand_rules_gfx1250_b0_to_a0();
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return semantic_expand_rules_cdna4_to_rdna3();
  return {};
}

} // namespace

SemanticTranslator::SemanticTranslator(rj_code_arch_t guest, rj_code_arch_t host,
                                       ProcessorRevision input_revision,
                                       ProcessorRevision output_revision)
    : expand_rules_(semantic_expand_rules_for(guest, host, input_revision, output_revision)),
      host_arch_(host) {
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
  if (!has_expand_rule_encoding(inst.encoding_id()))
    return nullptr;
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

bool SemanticTranslator::expand_rule_requires_liveness(const Instruction &inst) const {
  const TranslationRule *rule = find_expand_rule(inst);
  return rule != nullptr && rule->requires_liveness;
}

} // namespace rocjitsu
