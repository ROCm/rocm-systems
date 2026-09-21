// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic_translator.cpp
/// @brief Small dispatch facade for ISA-pair semantic expansion rules.

#include "rocjitsu/code/dbt/semantic_translator.h"

#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <string_view>

namespace rocjitsu {

namespace {

/// @brief Select the handwritten rewrite registry for one ISA pair.
/// @details Most ISA pairs currently have only an opcode table and do not offer
/// complete registry-level rewrite-discharge verification.
[[nodiscard]] RewriteRegistry rewrite_registry_for(rj_code_arch_t guest, rj_code_arch_t host,
                                                   ProcessorRevision input_revision,
                                                   ProcessorRevision output_revision) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return {semantic_expand_rules_cdna4_to_rdna4(), {}};
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return {semantic_expand_rules_cdna4_to_cdna3(), {}};
  // gfx1250 A0 and B0 share one architectural target ID. Select the B0-to-A0
  // profile only for that explicit revision pair.
  if (guest == ROCJITSU_CODE_ARCH_CDNA5 && host == ROCJITSU_CODE_ARCH_CDNA5 &&
      input_revision == ProcessorRevision::Gfx1250B0 &&
      output_revision == ProcessorRevision::Gfx1250A0)
    return rewrite_registry_gfx1250_b0_to_a0();
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return {semantic_expand_rules_cdna4_to_rdna3(), {}};
  return {};
}

[[nodiscard]] ExpandResult enforce_discharge_contract(std::string_view name,
                                                      const RewriteDischarge &discharge,
                                                      ExpandResult result) {
  if (!discharge.allows(result.status)) {
    return ExpandResult::failed("rewrite registry contract violation: '" + std::string(name) +
                                "' emitted output without a residual predicate");
  }
  return result;
}

} // namespace

SemanticTranslator::SemanticTranslator(rj_code_arch_t guest, rj_code_arch_t host,
                                       ProcessorRevision input_revision,
                                       ProcessorRevision output_revision)
    : rewrite_registry_(rewrite_registry_for(guest, host, input_revision, output_revision)),
      expand_rules_(rewrite_registry_.opcode_rules),
      instruction_rewrite_rules_(rewrite_registry_.instruction_rules),
      instruction_rewrites_require_basic_block_(
          rewrite_registry_.instruction_rewrites_require_basic_block()),
      host_arch_(host) {
  expand_rule_keys_.reserve(expand_rules_.size());
  uint16_t max_encoding_id = 0;
  for (const TranslationRule &rule : expand_rules_) {
    expand_rule_keys_.push_back(packed_rule_key(rule.src_encoding_id, rule.src_opcode));
    max_encoding_id = std::max(max_encoding_id, rule.src_encoding_id);
  }
  // Every table in the tree static_asserts translation_rules_sorted(), so this
  // is the catch-all for one added later without it. Strict ordering, matching
  // that predicate: a duplicated key would leave the second rule unreachable
  // behind the first.
  assert(std::adjacent_find(expand_rule_keys_.begin(), expand_rule_keys_.end(),
                            std::greater_equal<>{}) == expand_rule_keys_.end() &&
         "semantic rule tables must stay strictly sorted by (encoding id, opcode)");
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
  if (rule != nullptr) {
    auto result = rule->expand_fn(inst, static_cast<uint32_t>(host_arch_), offset, source_text,
                                  liveness, context, rule->guest_layout, rule->host_layout);
    return enforce_discharge_contract(inst.mnemonic(), rule->discharge, std::move(result));
  }
  return ExpandResult::not_handled();
}

ExpandResult SemanticTranslator::try_lower_instruction_rewrite(const Instruction &inst,
                                                               uint64_t offset,
                                                               std::span<const uint8_t> source_text,
                                                               const LivenessAnalysis &liveness,
                                                               TranslationContext &context) const {
  for (const RegisteredInstructionRewrite &rule : instruction_rewrite_rules_) {
    if (!rule.applies(inst))
      continue;
    auto result = rule.lower(inst, offset, source_text, liveness, context);
    if (result.status == ExpandStatus::NotHandled) {
      return ExpandResult::failed("registered instruction rewrite '" + std::string(rule.name) +
                                  "' matched but declined lowering");
    }
    return enforce_discharge_contract(rule.name, rule.discharge, std::move(result));
  }
  return ExpandResult::not_handled();
}

bool SemanticTranslator::has_expand_rule(const Instruction &inst) const {
  return has_expand_rule(inst.encoding_id(), inst.opcode());
}

bool SemanticTranslator::has_instruction_rewrite(const Instruction &inst) const {
  return std::ranges::any_of(instruction_rewrite_rules_,
                             [&](const RegisteredInstructionRewrite &rule) {
                               return rule.applies != nullptr && rule.applies(inst);
                             });
}

bool SemanticTranslator::rewrite_requires_liveness(const Instruction &inst) const {
  const TranslationRule *rule = find_expand_rule(inst);
  if (rule != nullptr && rule->requires_liveness)
    return true;
  return std::ranges::any_of(instruction_rewrite_rules_,
                             [&](const RegisteredInstructionRewrite &candidate) {
                               return candidate.requires_liveness && candidate.applies != nullptr &&
                                      candidate.applies(inst);
                             });
}

bool SemanticTranslator::residual_rewrite_applies(const Instruction &inst) const {
  return residual_rewrite_applies(inst, RewriteDischargeContext::Instruction) ||
         residual_rewrite_applies(inst, RewriteDischargeContext::BasicBlock);
}

bool SemanticTranslator::instruction_local_residual_rewrite_applies(const Instruction &inst) const {
  return residual_rewrite_applies(inst, RewriteDischargeContext::Instruction);
}

bool SemanticTranslator::residual_rewrite_needs_basic_block(const Instruction &inst) const {
  if (instruction_rewrites_require_basic_block_)
    return true;
  const TranslationRule *rule = find_expand_rule(inst);
  if (rule != nullptr && rule->discharge.disposition == RewriteDischargeDisposition::Checked &&
      rule->discharge.context == RewriteDischargeContext::BasicBlock) {
    return true;
  }
  return false;
}

bool SemanticTranslator::residual_rewrite_applies(const Instruction &inst,
                                                  RewriteDischargeContext context) const {
  const TranslationRule *rule = find_expand_rule(inst);
  if (rule != nullptr && rule->discharge.check != nullptr &&
      rule->discharge.disposition == RewriteDischargeDisposition::Checked &&
      rule->discharge.context == context && rule->discharge.check(inst)) {
    return true;
  }
  return std::ranges::any_of(
      instruction_rewrite_rules_, [&](const RegisteredInstructionRewrite &candidate) {
        return candidate.discharge.disposition == RewriteDischargeDisposition::Checked &&
               candidate.discharge.context == context && candidate.discharge.check != nullptr &&
               candidate.discharge.check(inst);
      });
}

} // namespace rocjitsu
