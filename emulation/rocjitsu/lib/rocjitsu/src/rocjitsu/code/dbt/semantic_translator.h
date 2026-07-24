// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic_translator.h
/// @brief Semantic translation for ISA-level behavioral differences.
///
/// @details Handles instructions and ABI conventions whose semantics change
/// across ISA generations, as opposed to the encoding translator which handles
/// pure binary format differences. Current semantic translations include:
///
/// - **Waitcnt lowering**: re-encode or conservatively expand a source
///   s_waitcnt when the host has a different wait-counter model.
/// - **Pair-specific instruction lowering**: MFMA→WMMA, AccVGPR elimination,
///   and other one-to-many target sequences. Kernel-entry descriptor ABI
///   prologues are built by KernelDescriptorTranslator and reached through a
///   CodeObjectPatcher descriptor-entry redirect.
///
/// The translator runs per instruction before the encoding translator. It
/// performs a binary search over the selected ISA-pair rule table, then invokes
/// the matched ExpandFn to produce replacement words that BinaryTranslator
/// writes in-place or through a code cave.
///
/// Rules are data-driven and live under code/dbt/semantic/. Adding a new
/// handwritten semantic rule means adding one entry to the relevant ISA-pair
/// table, not modifying BinaryTranslator's ISA-agnostic loop.

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/processor_revision.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

class Instruction;

/// @brief Result of a successful semantic translation: the source byte range
/// and the target instruction words that replace it.
struct SemanticReplacement {
  uint64_t start_offset = 0;          ///< First byte of the matched source range.
  uint64_t end_offset = 0;            ///< One past the last byte of the source range.
  std::vector<uint32_t> target_words; ///< Replacement instruction words for the host ISA.

  /// @brief Whether this replacement represents a successful match.
  [[nodiscard]] bool matched() const { return !target_words.empty(); }
};

/// @brief Semantic translator for cross-ISA behavioral differences.
///
/// @details All expansion rules (waitcnt, MFMA→WMMA, AccVGPR, etc.) are
/// registered as TranslationRule entries with RuleAction::Expand. The
/// try_lower_expand() method looks up rules by (encoding_id, opcode) via
/// binary search.
class SemanticTranslator {
public:
  SemanticTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                     ProcessorRevision input_revision, ProcessorRevision output_revision);

  /// @brief Try to expand/lower an instruction via the expand rules table.
  /// @param inst        The decoded instruction.
  /// @param offset      Byte offset of the instruction in .text.
  /// @param source_text Full source .text bytes for rules that need modifier/literal payloads.
  /// @param liveness    Kernel-scoped live-before data used for scratch register allocation.
  /// @returns Structured expansion status and replacement words.
  [[nodiscard]] ExpandResult try_lower_expand(const Instruction &inst, uint64_t offset,
                                              std::span<const uint8_t> source_text,
                                              const LivenessAnalysis &liveness,
                                              TranslationContext &context) const;

  /// @brief Whether @p inst has a registered semantic expansion rule.
  ///
  /// @details A true result means the instruction may query instruction-level
  /// liveness during try_lower_expand(). The rule can still decline expansion
  /// after inspecting operands or payload bits.
  [[nodiscard]] bool has_expand_rule(const Instruction &inst) const;
  [[nodiscard]] bool has_expand_rule(uint16_t encoding_id, uint16_t opcode) const {
    if (!has_expand_rule_encoding(encoding_id))
      return false;
    return std::binary_search(expand_rule_keys_.begin(), expand_rule_keys_.end(),
                              packed_rule_key(encoding_id, opcode));
  }

  [[nodiscard]] bool has_rules() const { return !expand_rules_.empty(); }

private:
  [[nodiscard]] static constexpr uint32_t packed_rule_key(uint16_t encoding_id, uint16_t opcode) {
    return (static_cast<uint32_t>(encoding_id) << 16) | opcode;
  }

  [[nodiscard]] bool has_expand_rule_encoding(uint16_t encoding_id) const {
    const size_t word_index = encoding_id / 64;
    return word_index < expand_rule_encoding_bits_.size() &&
           (expand_rule_encoding_bits_[word_index] & (uint64_t{1} << (encoding_id % 64))) != 0;
  }

  [[nodiscard]] const TranslationRule *find_expand_rule(const Instruction &inst) const;

  std::span<const TranslationRule> expand_rules_;   ///< Sorted by (src_encoding_id, src_opcode).
  std::vector<uint32_t> expand_rule_keys_;          ///< Packed keys parallel to expand_rules_.
  std::vector<uint64_t> expand_rule_encoding_bits_; ///< Cheap encoding prefilter for hot scans.
  rj_code_arch_t host_arch_;
};

} // namespace rocjitsu
