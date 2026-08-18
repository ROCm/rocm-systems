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
#include <unordered_set>
#include <vector>

#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/processor_revision.h"
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
/// @details Offsets are .text-relative in the relocated output. The
/// emitted_in_cave flag is kept for older tooling but cursor-based relocation
/// emits expanded instructions inline, so new translations leave it false.
/// source_words and target_words are only valid for the duration of the
/// callback; callers that need to retain them must copy the spans.
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
  /// @brief Input silicon revision used to determine translation direction.
  ///
  /// @details The command-line translation tool requires this when the input
  /// architecture is gfx1250. Other architectures leave it unspecified.
  ProcessorRevision input_revision = ProcessorRevision::Unspecified;

  /// @brief Output silicon revision used to determine translation direction.
  ///
  /// @details The command-line translation tool requires this when the output
  /// architecture is gfx1250. Other architectures leave it unspecified.
  ProcessorRevision output_revision = ProcessorRevision::Unspecified;

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

  /// @brief Preserve a failed kernel body and keep translating independent kernels.
  ///
  /// @details This is intended for load-time DBT of large code objects where not
  /// every kernel symbol is necessarily dispatched. A failed kernel is replaced
  /// by a minimal target-ISA `s_endpgm` stub, while the diagnostic is reported as a
  /// skipped-kernel warning. The symbol remains loadable so other kernels in the
  /// same code object are not blocked by one untranslated kernel.
  bool skip_failed_kernels = false;

  /// @brief Audit registered rewrites that remain actionable in final output.
  ///
  /// @details The audit is available only for translation profiles with registered semantic or
  /// operand-level residual checks. Requesting it for any other profile produces an error
  /// diagnostic. Runtime translation does not enable this development check by default.
  bool verify_rewrite_discharge = false;
};

/// @brief Result of translating a code object.
struct TranslatedCodeObject {
  std::vector<uint8_t> elf_bytes;                        ///< Translated ELF for the host ISA.
  rj_code_arch_t host_arch = ROCJITSU_CODE_ARCH_INVALID; ///< Host ISA architecture.
  std::vector<TranslationDiagnostic> diagnostics;        ///< Translation warnings/errors.
  bool rewrite_discharge_checked = false;                ///< Final output scan was attempted.
  bool rewrite_discharge_verified = false; ///< No registered rewrite remained actionable.

  /// @brief True if translation produced no error diagnostics.
  ///
  /// @details Note that ok() can be true while the artifact is NOT dispatchable:
  /// skip_failed_kernels reports a KernelSkipped *warning*, not an error. Use
  /// dispatchable() before emitting or executing the ELF.
  [[nodiscard]] bool ok() const { return !has_error_diagnostic(diagnostics); }

  /// @brief True if the artifact is safe to emit for execution.
  ///
  /// @details False when any kernel was replaced by a non-dispatchable no-op stub
  /// (has_skipped_kernel). A skipped kernel's `s_endpgm` stub completes normally
  /// without producing the kernel's outputs and would silently produce wrong results if
  /// dispatched, so code-object output paths and the CLI must refuse it, matching
  /// the HSA hook that rejects such a load.
  [[nodiscard]] bool dispatchable() const { return ok() && !has_skipped_kernel(diagnostics); }
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
/// body whose translated instruction sizes may differ from the source sizes.
/// Since explicit branches may move by a different delta than their targets,
/// direct PC-relative branch immediates and recovered indirect transfer windows
/// are patched through the kernel-local source-to-target block placement map.
/// Fallthrough is preserved by emitting reachable blocks in original .text order.
class BinaryTranslator {
public:
  /// @brief Construct a translator for the given (guest, host) ISA pair.
  /// @param guest_arch    Source ISA architecture.
  /// @param host_arch     Target ISA architecture.
  /// @param target_mach   EF_AMDGPU_MACH value for the target processor.
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
  /// @brief Translate without final-output validation.
  /// @details Keeping the translation body in a separate call ensures its
  /// analysis state is destroyed before translate() audits the resulting ELF.
  [[nodiscard]] TranslatedCodeObject translate_impl(const AmdGpuCodeObject &obj);

  /// @brief Whether this translator is running the gfx1250 B0-to-A0 profile.
  [[nodiscard]] bool is_gfx1250_b0_to_a0() const;

  void verify_rewrite_discharge(TranslatedCodeObject &result) const;

  /// @brief Return the generated or revision-specific legalization for an instruction.
  [[nodiscard]] const InstructionLegalization *lookup_legalization(const Instruction &inst) const;

  /// @brief Translate a single instruction via the encoding translation pipeline.
  ///
  /// @details Extracts raw encoding words, calls the per-pair encoding translate
  /// function, and appends the result to the translated text cursor.
  /// Falls back to copying the original encoding if translation produces no output.
  ///
  /// @param inst       The decoded guest instruction.
  /// @param offset     Source byte offset of the instruction within original .text.
  /// @param text       The translated text buffer.
  /// @param dst_opcode Target opcode from the legalization table.
  /// @param orig_text   The original .text bytes used to preserve trailing literals.
  /// @returns true if the instruction was translated or copied safely.
  [[nodiscard]] bool handle_encoding(const Instruction &inst, uint64_t offset,
                                     std::vector<uint8_t> &text, uint16_t dst_opcode,
                                     std::span<const uint8_t> orig_text, bool collect_target_words,
                                     bool &copied_original, bool &changed,
                                     std::vector<uint32_t> &target_words);

  rj_code_arch_t guest_arch_;                               ///< Source ISA.
  rj_code_arch_t host_arch_;                                ///< Target ISA.
  uint32_t target_mach_;                                    ///< ELF MACH flag for target processor.
  TranslationTraceCallback trace_callback_;                 ///< Optional debug trace callback.
  BinaryTranslatorOptions options_;                         ///< Optional translation controls.
  EncodingTranslateFn encoding_translate_;                  ///< Per-pair encoding translator.
  LegalizationLookupFn legalization_lookup_;                ///< Per-pair legalization table.
  std::unique_ptr<SemanticTranslator> semantic_translator_; ///< Per-pair semantic rule engine.

  /// @brief Deferred-family mnemonics already reported in this translation.
  ///
  /// @details Cleared at the start of every translate() call. Scoping the
  /// suppression to one code object keeps the report informative without
  /// letting the first object that uses a deferred mnemonic hide the same gap
  /// in every object loaded after it.
  std::unordered_set<std::string> reported_deferred_families_;
};

} // namespace rocjitsu
