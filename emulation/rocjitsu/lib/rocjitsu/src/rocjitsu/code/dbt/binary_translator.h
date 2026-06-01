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
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

class AmdGpuCodeObject;
class CodeObjectPatcher;
class SemanticTranslator;
class Instruction;
struct InstructionLegalization;

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

/// @brief Result of translating a code object.
struct TranslatedCodeObject {
  std::vector<uint8_t> elf_bytes;    ///< Translated ELF for the host ISA.
  rj_code_arch_t host_arch;          ///< Host ISA architecture.
  std::vector<std::string> warnings; ///< Non-fatal translation issues.
};

/// @brief Top-level dynamic binary translator.
///
/// @details Translates an AmdGpuCodeObject from guest_arch to host_arch by:
///   1. Decoding all instructions via the existing Decoder::create() factory.
///   2. Running the semantic translator per-block for special-case translations.
///   3. Translating remaining instructions via legalization + encoding translate.
///   4. Re-emitting a valid ELF for host_arch via CodeObjectPatcher.
///
/// Because every in-place replacement preserves the original instruction's size
/// (same-size for Identity/Substitute/Lower; branch stub for code caves),
/// no instruction in .text ever shifts. Branch offsets remain valid — no global
/// branch fixup pass is required.
class BinaryTranslator {
public:
  /// @brief Construct a translator for the given (guest, host) ISA pair.
  /// @param guest_arch    Source ISA architecture.
  /// @param host_arch     Target ISA architecture.
  /// @param target_mach   EF_AMDGPU_MACH value for the target GPU stepping.
  ///                      0 = auto-detect from host_arch (default GFX1200 for RDNA4).
  BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch, uint32_t target_mach = 0);
  ~BinaryTranslator();

  /// @brief Translate a decoded code object.
  /// @param obj  The guest code object to translate.
  /// @returns TranslatedCodeObject with the host ELF bytes and any warnings.
  [[nodiscard]] TranslatedCodeObject translate(const AmdGpuCodeObject &obj);

private:
  /// @brief Apply a single semantic replacement to the translated text.
  ///
  /// @details If the replacement fits within the source byte range, writes
  /// in-place and pads any leftover source words. If it expands, writes a
  /// branch stub in-place and appends the replacement body + return branch to
  /// the .rj_translations code cave via the patcher.
  ///
  /// @param repl    The semantic replacement to apply.
  /// @param text    The translated text buffer (same size as original .text).
  /// @param patcher The code object patcher for cave body accumulation.
  /// @returns true if the replacement was applied safely; false if an expanding
  ///          replacement could not be branched to/from the code cave.
  [[nodiscard]] bool apply_semantic(const struct SemanticReplacement &repl,
                                    std::vector<uint8_t> &text, CodeObjectPatcher &patcher);

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
  /// @returns true if the instruction was translated or copied safely; false if
  ///          the translated encoding expanded and could not be branched through
  ///          the code cave.
  [[nodiscard]] bool handle_encoding(const Instruction &inst, uint64_t offset,
                                     std::vector<uint8_t> &text, uint16_t dst_opcode,
                                     CodeObjectPatcher &patcher,
                                     std::span<const uint8_t> orig_text);

  rj_code_arch_t guest_arch_;                               ///< Source ISA.
  rj_code_arch_t host_arch_;                                ///< Target ISA.
  uint32_t target_mach_;                                    ///< ELF MACH flag for target stepping.
  EncodingTranslateFn encoding_translate_;                  ///< Per-pair encoding translator.
  LegalizationLookupFn legalization_lookup_;                ///< Per-pair legalization table.
  std::unique_ptr<SemanticTranslator> semantic_translator_; ///< Per-pair semantic rule engine.
  std::vector<std::string> *warnings_ = nullptr;            ///< Points to active result's warnings.
};

} // namespace rocjitsu
