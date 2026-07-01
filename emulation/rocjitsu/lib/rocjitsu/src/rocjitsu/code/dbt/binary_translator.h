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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

class AmdGpuCodeObject;
class SemanticTranslator;
class Instruction;
struct InstructionLegalization;
struct KernelTextLayout;

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
/// @details Offsets are .text-relative in the relocated output. When
/// emitted_in_cave is true, target_offset points into the current kernel's
/// private local cave inside the rewritten .text. source_words and target_words
/// are only valid for the duration of the callback; callers that need to retain
/// them must copy the spans.
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
  std::vector<TranslationDiagnostic> diagnostics;        ///< Translation warnings/errors.

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
/// DBT relocates each kernel into a fresh .text layout instead of appending a
/// global `.rj_translations` cave. Each descriptor entry gets a private emitted
/// body plus a local cave immediately after that body. Since explicit branches
/// may move by a different delta than their targets, direct PC-relative branch
/// immediates are patched through the kernel-local source-to-target block
/// placement map. Fallthrough is preserved by emitting reachable blocks in
/// original .text order.
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
  /// @brief Apply a single semantic replacement to the translated text.
  ///
  /// @details If the replacement fits within the source byte range, writes
  /// in-place and pads any leftover source words. If it expands, writes a
  /// branch stub in-place and appends the replacement body + return branch to
  /// the current kernel's local cave.
  ///
  /// @param repl                 The semantic replacement to apply.
  /// @param text                 The relocated .text buffer under construction.
  /// @param layout               Current kernel layout for local cave accounting.
  /// @param orig_text            Original guest .text used to classify fallthrough padding.
  /// @param source_return_offset Original .text offset reached after the replaced instruction.
  /// @returns true if the replacement was applied safely; false if an expanding
  ///          replacement could not be branched to/from the code cave.
  [[nodiscard]] bool apply_semantic(const struct SemanticReplacement &repl,
                                    std::vector<uint8_t> &text, KernelTextLayout &layout,
                                    std::span<const uint8_t> orig_text,
                                    uint64_t source_return_offset);

  /// @brief Translate a single instruction via the encoding translation pipeline.
  ///
  /// @details Extracts raw encoding words, calls the per-pair encoding translate
  /// function, and writes the result into the translated text at the given offset.
  /// Falls back to copying the original encoding if translation produces no output.
  ///
  /// @param inst       The decoded guest instruction.
  /// @param offset     Source byte offset of the instruction within original .text.
  /// @param target_offset Relocated byte offset of the instruction in output .text.
  /// @param text       The translated text buffer.
  /// @param dst_opcode Target opcode from the legalization table.
  /// @param layout     Current kernel layout for local cave accounting.
  /// @param orig_text   The original .text bytes used to preserve trailing literals.
  /// @returns true if the instruction was translated or copied safely; false if
  ///          the translated encoding expanded and could not be branched through
  ///          the code cave.
  [[nodiscard]] bool handle_encoding(const Instruction &inst, uint64_t offset,
                                     uint64_t target_offset, std::vector<uint8_t> &text,
                                     uint16_t dst_opcode, KernelTextLayout &layout,
                                     std::span<const uint8_t> orig_text,
                                     const InstructionLegalization *leg);

  rj_code_arch_t guest_arch_;                               ///< Source ISA.
  rj_code_arch_t host_arch_;                                ///< Target ISA.
  uint32_t target_mach_;                                    ///< ELF MACH flag for target stepping.
  TranslationTraceCallback trace_callback_;                 ///< Optional debug trace callback.
  BinaryTranslatorOptions options_;                         ///< Optional translation controls.
  EncodingTranslateFn encoding_translate_;                  ///< Per-pair encoding translator.
  LegalizationLookupFn legalization_lookup_;                ///< Per-pair legalization table.
  std::unique_ptr<SemanticTranslator> semantic_translator_; ///< Per-pair semantic rule engine.
  std::vector<TranslationDiagnostic> *diagnostics_ = nullptr; ///< Active result diagnostics.
};

} // namespace rocjitsu
