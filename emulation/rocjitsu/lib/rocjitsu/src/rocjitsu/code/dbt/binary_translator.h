// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file binary_translator.h
/// @brief ISA-agnostic binary translator for cross-ISA GPU code object translation.
///
/// @details Translates an AmdGpuCodeObject from a guest ISA to a host ISA using a
/// two-tier architecture:
///
/// 1. **Semantic translator** — scans each basic block for instructions whose
///    semantics change across ISA generations (waitcnt, barriers, MFMA, AccVGPR)
///    and replaces them via data-driven rules. Handles the ~20% of instructions
///    where per-instruction encoding translation is insufficient.
///
/// 2. **Per-instruction encoding translation** — for all remaining instructions,
///    looks up the legalization action (Identity/Substitute/Lower/Expand) and
///    applies the generated decode→neutral→encode pipeline.
///
/// ISA-pair-specific logic is isolated behind function pointers and rule tables
/// selected at construction time. The translation loop itself contains no
/// ISA-specific branches.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/instruction_list.h"
#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

class AmdGpuCodeObject;
class CodeObjectPatcher;
class SemanticTranslator;
class Instruction;
struct InstructionLegalization;

struct CaveBranchIslandState {
  bool initialized = false;
  size_t next_long_search_index = 0;
  std::vector<uint64_t> body_offsets;
  std::vector<bool> used;
};

/// @brief Encoding translation function type.
///
/// @details Dispatches to the generated per-pair translate function
/// (e.g., translate_encoding_cdna4_to_rdna4). Decodes guest encoding fields
/// into ISA-neutral structs, re-encodes into host encoding with coherency
/// remapping, and returns the translated instruction words.
///
/// @param encoding_id  Guest encoding format ID (bits [31:23] of word 0).
/// @param w0           Guest instruction word 0.
/// @param w1           Guest instruction word 1 (0 if single-word).
/// @param w2           Guest instruction word 2 (0 if ≤64-bit).
/// @param dst_op       Target opcode (from legalization table).
/// @returns TranslationResult with the encoded host instruction words.
using EncodingTranslateFn = TranslationResult (*)(uint32_t encoding_id, uint32_t w0, uint32_t w1,
                                                  uint32_t w2, uint16_t dst_op);

/// @brief Legalization lookup function type.
///
/// @details Queries the generated per-pair legalization table for a
/// (encoding_id, opcode) pair. Returns the InstructionLegalization entry
/// describing the action (Identity/Substitute/Lower/Expand) and target opcode.
///
/// @param encoding_id  Guest encoding format ID.
/// @param opcode       Guest opcode within the encoding format.
/// @returns Pointer to the legalization entry, or nullptr if not found.
using LegalizationLookupFn = const InstructionLegalization *(*)(uint16_t encoding_id,
                                                                uint16_t opcode);

/// @brief One source instruction trace event emitted by BinaryTranslator.
///
/// @details Offsets are .text-relative. When emitted_in_cave is true,
/// target_offset points into the logical .text continuation that later becomes
/// `.rj_translations`; subtracting the original .text size gives the offset in
/// that cave section. source_words and target_words are only valid for the
/// duration of the callback; callers that need to retain them must copy the
/// spans.
struct TranslationTraceEvent {
  uint64_t source_offset = 0;
  uint32_t source_size = 0;
  std::span<const uint32_t> source_words;
  const InstructionLegalization *legalization = nullptr;
  bool copied_original = false;
  bool semantic_lowering = false;
  bool changed = false;
  bool emitted_in_cave = false;
  uint64_t target_offset = 0;
  std::span<const uint32_t> target_words;
};

using TranslationTraceCallback = std::function<void(const TranslationTraceEvent &)>;

/// @brief Optional controls for DBT translation.
struct BinaryTranslatorOptions {
  /// @brief Force liveness-based VGPR scratch allocation above a debug floor.
  ///
  /// @details This debug mode leaves normal liveness dataflow untouched, but
  /// makes find_free_run() skip VGPRs below this floor. It is useful when
  /// investigating register clobbers caused by overly optimistic liveness.
  std::optional<uint16_t> debug_min_free_vgpr;

  /// @brief Keep scanning instructions after recoverable translation failures.
  ///
  /// @details This is a diagnostics-only mode. The translator preserves the
  /// original instruction at each failed source location and continues so one run
  /// can report multiple missing EXPAND rules or resource-limit failures. If any
  /// error diagnostic is collected, the final code object is still left unchanged
  /// because the partially translated text is only useful for finding failures,
  /// not for execution.
  bool debug_continue_after_failure = false;
};

/// @brief Result of translating a code object.
struct TranslatedCodeObject {
  std::vector<uint8_t> elf_bytes;                        ///< Translated ELF for the host ISA.
  rj_code_arch_t host_arch = ROCJITSU_CODE_ARCH_INVALID; ///< Host ISA architecture.
  std::vector<std::string> warnings;                     ///< Non-fatal translation warnings.
  std::vector<TranslationDiagnostic> diagnostics;        ///< Structured translation diagnostics.

  [[nodiscard]] bool ok() const { return !has_error_diagnostic(diagnostics); }
};

/// @brief Top-level dynamic binary translator.
///
/// @details Translates an AmdGpuCodeObject from guest_arch to host_arch by:
///   1. Decoding all instructions via the existing Decoder::create() factory.
///   2. Running the semantic translator per-block for special-case translations.
///   3. Translating remaining instructions via legalization + encoding translate.
///   4. Re-emitting a valid ELF for host_arch via CodeObjectPatcher.
///
/// The default path preserves original .text instruction addresses by writing
/// same-size replacements in place or by branching to code caves. For large
/// kernels where dense size-growing replacements cannot reach a cave, the
/// translator may instead append a fully translated executable copy and redirect
/// kernel descriptors to that copy after fixing direct branch offsets.
class BinaryTranslator {
public:
  /// @brief Construct a translator for the given (guest, host) ISA pair.
  /// @param guest_arch    Source ISA architecture.
  /// @param host_arch     Target ISA architecture.
  /// @param target_mach   EF_AMDGPU_MACH value for the target GPU stepping.
  ///                      0 = auto-detect from host_arch (default GFX1200 for RDNA4).
  BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch, uint32_t target_mach = 0,
                   BinaryTranslatorOptions options = {});
  ~BinaryTranslator();

  /// @brief Install an optional callback for per-instruction debugging.
  void set_trace_callback(TranslationTraceCallback callback);

  /// @brief Translate a decoded code object.
  /// @param obj  The guest code object to translate.
  /// @returns TranslatedCodeObject with the host ELF bytes and diagnostics.
  [[nodiscard]] TranslatedCodeObject translate(const AmdGpuCodeObject &obj);

private:
  struct CaveChainState {
    bool active = false;
    uint64_t return_target = 0;
    uint64_t trailer_body_offset = 0;
  };

  /// @brief Apply a single semantic replacement to the translated text.
  ///
  /// @details If the replacement fits within the source byte range, writes
  /// in-place and pads any leftover source words. If it expands, writes a
  /// branch stub in-place and writes the replacement body + return branch to a
  /// nearby local cave when possible, otherwise appends it to .rj_translations.
  ///
  /// @param repl    The semantic replacement to apply.
  /// @param text    The translated text buffer (same size as original .text).
  /// @param patcher The code object patcher for cave body accumulation.
  /// @param local_caves Ranges in .text already reserved for local cave bodies.
  /// @param protected_ranges Decoded reachable code ranges that must not be
  ///                         repurposed as local caves.
  /// @param allow_unreachable_text_caves Whether decoded-unreachable, non-padding
  ///                                     text may be repurposed for cave bodies.
  /// @param long_return_sgpr_pair Optional dead SGPR pair used to materialize a
  ///                              far return when the appended cave is outside
  ///                              SOPP branch range.
  /// @param long_return_scc_sgpr Optional dead SGPR used to preserve SCC across
  ///                             a far return to an s_cbranch_scc* target.
  /// @param cave_chain Per-basic-block state for joining adjacent far-cave
  ///                   replacements behind one source branch.
  /// @param cave_branch_islands Per-code-object state for reserved branch
  ///                            islands at the front of the appended cave.
  /// @returns true if the replacement was applied safely; false if an expanding
  ///          replacement could not be branched to/from the code cave.
  [[nodiscard]] bool apply_semantic(const struct SemanticReplacement &repl,
                                    std::vector<uint8_t> &text, CodeObjectPatcher &patcher,
                                    std::vector<std::pair<uint64_t, uint64_t>> &local_caves,
                                    std::span<const std::pair<uint64_t, uint64_t>> protected_ranges,
                                    bool allow_unreachable_text_caves,
                                    std::optional<uint16_t> long_return_sgpr_pair = std::nullopt,
                                    std::optional<uint16_t> long_return_scc_sgpr = std::nullopt,
                                    CaveChainState *cave_chain = nullptr,
                                    CaveBranchIslandState *cave_branch_islands = nullptr);

  /// @brief Translate a single instruction via the encoding translation pipeline.
  ///
  /// @details Extracts raw encoding words, calls the per-pair encoding translate
  /// function, and writes the result into the translated text at the given offset.
  /// Falls back to copying the original encoding if translation produces no output.
  ///
  /// @param inst       The decoded guest instruction.
  /// @param offset     Byte offset of the instruction within .text.
  /// @param text       The translated text buffer.
  /// @param dst_opcode Target opcode from the legalization table.
  /// @param patcher    The code object patcher for expanded instruction bodies.
  /// @param orig_text   The original .text bytes used to preserve trailing literals.
  /// @param local_caves Ranges in .text already reserved for local cave bodies.
  /// @param protected_ranges Decoded reachable code ranges that must not be
  ///                         repurposed as local caves.
  /// @param allow_unreachable_text_caves Whether decoded-unreachable, non-padding
  ///                                     text may be repurposed for cave bodies.
  /// @param rdna4_grid_x_sgpr SGPR holding the entry-captured RDNA4 GridX value,
  ///                          or -1 when no remap is needed.
  /// @param rdna4_grid_yz_sgpr SGPR holding the entry-captured packed RDNA4
  ///                           GridY/Z value, or -1 when no remap is needed.
  /// @param long_return_sgpr_pair Optional dead SGPR pair used to materialize a
  ///                              far return when the appended cave is outside
  ///                              SOPP branch range.
  /// @param long_return_scc_sgpr Optional dead SGPR used to preserve SCC across
  ///                             a far return to an s_cbranch_scc* target.
  /// @param cave_chain Per-basic-block state for joining adjacent far-cave
  ///                   replacements behind one source branch.
  /// @param cave_branch_islands Per-code-object state for reserved branch
  ///                            islands at the front of the appended cave.
  /// @returns true if the instruction was translated or copied safely; false if
  ///          the translated encoding expanded and could not be branched through
  ///          the code cave.
  [[nodiscard]] bool handle_encoding(
      const Instruction &inst, uint64_t offset, std::vector<uint8_t> &text, uint16_t dst_opcode,
      CodeObjectPatcher &patcher, std::span<const uint8_t> orig_text,
      std::vector<std::pair<uint64_t, uint64_t>> &local_caves,
      std::span<const std::pair<uint64_t, uint64_t>> protected_ranges,
      bool allow_unreachable_text_caves, int16_t rdna4_grid_x_sgpr, int16_t rdna4_grid_yz_sgpr,
      InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
      std::span<BasicBlock *const> scope_blocks,
      std::optional<uint16_t> long_return_sgpr_pair = std::nullopt,
      std::optional<uint16_t> long_return_scc_sgpr = std::nullopt,
      CaveChainState *cave_chain = nullptr, CaveBranchIslandState *cave_branch_islands = nullptr);

  /// @brief Translate one instruction to host instruction words.
  ///
  /// @details Used by the expanded-copy path where size-growing replacements
  /// can be emitted inline and later direct-branch fixups rewrite the copied
  /// branch immediates.
  [[nodiscard]] std::vector<uint32_t> translate_instruction_words(
      const Instruction &inst, uint64_t offset, const class LivenessAnalysis &liveness,
      std::span<const uint8_t> orig_text, int16_t rdna4_grid_x_sgpr, int16_t rdna4_grid_yz_sgpr);

  /// @brief Decode remapped guest words and translate them with @p inst's liveness.
  ///
  /// @details High-bank virtualization may rewrite only operand encodings and
  /// then hand the instruction back to the normal semantic translator.
  [[nodiscard]] std::vector<uint32_t> translate_remapped_guest_instruction_words(
      const Instruction &inst, class LivenessAnalysis &liveness,
      std::span<const uint32_t> guest_words, int16_t rdna4_grid_x_sgpr, int16_t rdna4_grid_yz_sgpr);

  rj_code_arch_t guest_arch_;                               ///< Source ISA.
  rj_code_arch_t host_arch_;                                ///< Target ISA.
  uint32_t target_mach_;                                    ///< ELF MACH flag for target stepping.
  TranslationTraceCallback trace_callback_;                 ///< Optional debug trace callback.
  BinaryTranslatorOptions options_;                         ///< Optional translation controls.
  EncodingTranslateFn encoding_translate_;                  ///< Per-pair encoding translator.
  LegalizationLookupFn legalization_lookup_;                ///< Per-pair legalization table.
  std::unique_ptr<SemanticTranslator> semantic_translator_; ///< Per-pair semantic rule engine.
  std::vector<std::string> *warnings_ = nullptr;            ///< Active result warnings.
  std::vector<TranslationDiagnostic> *diagnostics_ = nullptr; ///< Active structured diagnostics.
};

} // namespace rocjitsu
