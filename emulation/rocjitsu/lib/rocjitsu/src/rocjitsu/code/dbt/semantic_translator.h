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
/// - **Instruction lowering**: MFMA/AccVGPR and other one-to-many target
///   sequences. Kernel-entry descriptor ABI prologues are built by
///   KernelDescriptorTranslator and reached through a CodeObjectPatcher
///   descriptor-entry redirect.
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

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/dbt/translation_rule.h"
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

/// @brief Semantic translator for cross-ISA behavioral differences.
///
/// @details All expansion rules (waitcnt, MFMA→WMMA, AccVGPR, etc.) are
/// registered as TranslationRule entries with RuleAction::Expand. The
/// try_lower_expand() method looks up rules by opcode via binary search.
class SemanticTranslator {
public:
  SemanticTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch);

  /// @brief Try to expand/lower an instruction via the expand rules table.
  /// @param inst      The decoded instruction.
  /// @param offset    Byte offset of the instruction in .text.
  /// @param liveness  Kernel-scoped live-before data used for scratch register allocation.
  /// @returns Replacement instruction words on success, empty vector if no rule matches.
  [[nodiscard]] std::vector<uint32_t> try_lower_expand(const Instruction &inst, uint64_t offset,
                                                       const LivenessAnalysis &liveness) const;

  [[nodiscard]] bool has_rules() const { return !expand_rules_.empty(); }

private:
  std::span<const TranslationRule> expand_rules_; ///< Sorted by src_opcode.
  rj_code_arch_t host_arch_;
};

} // namespace rocjitsu
