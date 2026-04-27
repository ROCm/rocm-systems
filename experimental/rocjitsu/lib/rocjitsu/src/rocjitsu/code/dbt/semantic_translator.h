// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic_translator.h
/// @brief Semantic translation for ISA-level behavioral differences.
///
/// @details Handles instructions and ABI conventions whose semantics change
/// across ISA generations, as opposed to the encoding translator which handles
/// pure binary format differences. Current semantic translations:
///
/// - **Waitcnt splitting**: GFX9 monolithic s_waitcnt → GFX12 split
///   s_wait_loadcnt / s_wait_storecnt_dscnt / s_wait_kmcnt / s_wait_expcnt
/// - **Workgroup ID delivery**: GFX9 delivers workgroup IDs via SGPRs,
///   RDNA4 delivers them via TTMP registers (TTMP9 for X, TTMP7 for Y/Z)
///
/// The translator runs per-basic-block before the per-instruction encoding
/// loop. It scans for anchor instructions (identified by InstFlags), applies
/// the first matching rule, and produces a SemanticReplacement that the binary
/// translator writes in-place or via code caves.
///
/// Rules are data-driven: each SemanticRule is a (name, anchor_flags,
/// translate_fn) tuple. Adding a new rule means adding one entry to the
/// per-pair rule table.

#pragma once

#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

class BasicBlock;
class Instruction;

/// @brief Decoded wait-counter values from a GFX9 s_waitcnt simm16 field.
struct WaitcntValues {
  uint8_t vmcnt = 0x3F;   ///< VM count (loads + stores on GFX9). Sentinel: 0x3F.
  uint8_t lgkmcnt = 0x0F; ///< LDS/GDS/Kmem count. Sentinel: 0x0F.
  uint8_t expcnt = 0x07;  ///< Export count. Sentinel: 0x07.
};

/// @brief Decode a GFX9 s_waitcnt simm16 field into individual counter values.
[[nodiscard]] WaitcntValues decode_waitcnt_gfx9(uint16_t simm16);

/// @brief Encode wait-counter values as GFX12 split s_wait_* instruction words.
[[nodiscard]] std::vector<uint32_t> encode_waitcnt_gfx12(const WaitcntValues &vals);

/// @brief Result of a successful semantic translation: the source byte range
/// and the target instruction words that replace it.
struct SemanticReplacement {
  uint64_t start_offset = 0;          ///< First byte of the matched source range.
  uint64_t end_offset = 0;            ///< One past the last byte of the source range.
  std::vector<uint32_t> target_words; ///< Replacement instruction words for the host ISA.

  /// @brief Whether this replacement represents a successful match.
  [[nodiscard]] bool matched() const { return !target_words.empty(); }
};

/// @brief A single semantic translation rule.
///
/// @details Each rule identifies anchor instructions via anchor_flags (tested
/// against Instruction::flags()) and provides a translate function that
/// attempts to match and produce the replacement. Returns a SemanticReplacement
/// with matched()==true on success, or an empty replacement on failure.
struct SemanticRule {
  const char *name;      ///< Human-readable rule name for diagnostics.
  uint64_t anchor_flags; ///< Required InstFlags bits on the anchor instruction.

  /// @brief Attempt to translate an anchor instruction.
  ///
  /// @param anchor        The decoded guest instruction that triggered this rule.
  /// @param anchor_offset Byte offset of the anchor within the .text section.
  /// @param host_arch     Target ISA architecture.
  /// @returns SemanticReplacement with matched()==true on success, empty on failure.
  using TranslateFn = SemanticReplacement (*)(const Instruction &anchor, uint64_t anchor_offset,
                                              rj_code_arch_t host_arch);
  TranslateFn translate; ///< The translate function for this rule.
};

/// @brief Per-basic-block semantic translator.
///
/// @details Constructed once per BinaryTranslator with a (guest_arch, host_arch)
/// pair. Selects the appropriate rule table for the pair. The translate() method
/// scans a basic block's instructions, tests each against the rule table's
/// anchor_flags, and calls the translate function for matching rules.
class SemanticTranslator {
public:
  /// @brief Construct a translator for the given (guest, host) ISA pair.
  /// @param guest_arch  Source ISA architecture.
  /// @param host_arch   Target ISA architecture.
  SemanticTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch);

  /// @brief Scan a basic block for instructions requiring semantic translation.
  ///
  /// @param block  The decoded basic block to scan.
  /// @returns A list of non-overlapping replacements, ordered by start_offset.
  [[nodiscard]] std::vector<SemanticReplacement> translate(BasicBlock &block) const;

  /// @brief Rewrite workgroup_id SGPR references to TTMP registers.
  /// On RDNA4, workgroup IDs are delivered via TTMP registers, not SGPRs.
  /// This pass substitutes the SGPR operand in each matching instruction.
  [[nodiscard]] std::vector<SemanticReplacement>
  rewrite_workgroup_ids(BasicBlock &block,
                        std::span<const CodeObjectPatcher::WorkGroupIdInfo> wg_info,
                        std::span<const uint8_t> translated_text) const;

  /// @brief Try to lower an instruction marked as Action::Expand.
  /// @returns Replacement instruction words on success, empty vector if unhandled.
  [[nodiscard]] std::vector<uint32_t> try_lower_expand(const Instruction &inst) const;

  /// @brief Whether any semantic rules exist for this (guest, host) pair.
  [[nodiscard]] bool has_rules() const { return !rules_.empty(); }

private:
  std::span<const SemanticRule> rules_;        ///< Rule table for this (guest, host) pair.
  [[maybe_unused]] rj_code_arch_t guest_arch_; ///< Source ISA.
  rj_code_arch_t host_arch_;                   ///< Target ISA.
};

} // namespace rocjitsu
