// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_translate.h
/// @brief Internal entry point for the rj_dbt_translate CLI and tests.

#ifndef ROCJITSU_TOOLS_DBT_TRANSLATE_H_
#define ROCJITSU_TOOLS_DBT_TRANSLATE_H_

#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/processor_revision.h"
#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/rj_code.h"
#include "tool_result.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::tools {

enum class DisassemblyMode {
  None,
  Source,
  Translated,
  Both,
};

struct CodeSectionReport {
  std::string name;
  size_t size_bytes = 0;
  size_t instruction_count = 0;
  size_t decode_failure_count = 0;

  // Store the first failing offset so the text report points developers at a
  // concrete location without retaining every failed decode in memory.
  bool has_first_decode_failure = false;
  size_t first_decode_failure_offset = 0;
  std::string first_decode_failure_message;
};

struct CodeObjectReport {
  bool available = false;
  bool decoder_available = false;
  std::vector<CodeSectionReport> sections;
};

struct InstructionTranslationReport {
  uint64_t source_offset = 0;
  uint32_t source_size = 0;
  std::vector<uint32_t> source_words;
  std::string source_instruction;
  bool has_legalization = false;
  Action action = Action::Identity;
  bool copied_original = false;
  bool semantic_lowering = false;
  bool changed = false;
  bool emitted_in_cave = false;
  uint64_t target_offset = 0;
  std::vector<uint32_t> target_words;
  std::vector<std::string> target_instructions;

  /// @brief What actually happened to this instruction, as one value.
  ///
  /// @details The label and the action counters must agree, and reconstructing
  /// the rules for that at each consumer is how they drift apart. The one case
  /// that is not simply `action` is a semantic rule carrying an instruction with
  /// no legalization entry naming it: that is an expansion, and reporting it as
  /// a re-encode would hide the rule that rewrote the instruction.
  ///
  /// @returns The effective legalization action, or nullopt when the instruction
  ///          was copied verbatim or re-encoded without a rule.
  [[nodiscard]] std::optional<Action> effective_action() const {
    if (copied_original)
      return std::nullopt;
    if (!has_legalization)
      return semantic_lowering ? std::optional<Action>(Action::Expand) : std::nullopt;
    return action;
  }
};

struct TranslateOptions {
  std::string input_path;

  rj_code_target_id_t input_target = ROCJITSU_CODE_TARGET_GFX950;
  uint32_t code_object_index = 0;

  rj_code_arch_t guest_arch = ROCJITSU_CODE_ARCH_CDNA4;
  rj_code_arch_t host_arch = ROCJITSU_CODE_ARCH_RDNA4;
  uint32_t target_mach = 0;
  ProcessorRevision input_revision = ProcessorRevision::Unspecified;
  ProcessorRevision output_revision = ProcessorRevision::Unspecified;

  bool collect_diagnostics = false;
  std::optional<uint16_t> debug_min_free_vgpr;
  bool debug_continue_after_failure = false;
  bool skip_failed_kernels = false;
  /// @brief Rerun a same-architecture translation and require identical ELF bytes.
  bool verify_idempotence = false;
  /// @brief Require every audited semantic rewrite to be discharged in final output.
  bool verify_rewrite_discharge = false;
  DisassemblyMode disassembly = DisassemblyMode::None;
};

struct TranslateOutput {
  std::vector<uint8_t> elf_bytes;
  /// 64-bit FNV-1a identity of the exact selected source code-object bytes.
  uint64_t source_code_object_id = 0;
  rj_code_arch_t host_arch = ROCJITSU_CODE_ARCH_INVALID;
  uint32_t target_mach = 0;
  ProcessorRevision input_revision = ProcessorRevision::Unspecified;
  ProcessorRevision output_revision = ProcessorRevision::Unspecified;
  CodeObjectReport source_report;
  CodeObjectReport translated_report;
  std::vector<InstructionTranslationReport> instruction_translations;
  std::vector<TranslationDiagnostic> diagnostics;
  /// @brief Diagnostics produced by the verification translation only.
  ///
  /// @details These do not affect ok() or dispatchable(): elf_bytes is the
  /// already-validated first-pass output.
  std::vector<TranslationDiagnostic> idempotence_diagnostics;
  std::string disassembly;
  /// @brief True when the requested second translation was attempted.
  bool idempotence_checked = false;
  /// @brief True when the requested second translation matched the first output.
  bool idempotence_verified = false;
  /// @brief True when the requested final-output applicability scan was attempted.
  bool rewrite_discharge_checked = false;
  /// @brief True when no registered rewrite remained actionable.
  bool rewrite_discharge_verified = false;

  /// @brief True if translation produced no error diagnostics.
  [[nodiscard]] bool ok() const { return !has_error_diagnostic(diagnostics); }

  /// @brief True if elf_bytes is safe to emit for execution.
  ///
  /// @details False when a kernel was replaced by a non-dispatchable no-op stub
  /// (has_skipped_kernel): its `s_endpgm` completes normally without producing the
  /// kernel's outputs and would silently produce wrong results. Executable emitters must
  /// gate on this, not just ok() -- a KernelSkipped diagnostic is only a warning.
  [[nodiscard]] bool dispatchable() const { return ok() && !has_skipped_kernel(diagnostics); }
};

/// @brief Validate translation option combinations.
/// @returns An error message when the request is invalid, or nullopt otherwise.
[[nodiscard]] std::optional<std::string_view>
translation_request_error(const TranslateOptions &options);

namespace detail {

/// @brief Non-owning executable-section data used by idempotence comparison.
struct ExecutableSectionBytes {
  std::string_view name;
  std::span<const uint8_t> bytes;
};

[[nodiscard]] std::string describe_byte_difference(std::span<const uint8_t> first,
                                                   std::span<const uint8_t> second,
                                                   std::string_view location);

[[nodiscard]] std::string
find_idempotence_difference(std::span<const ExecutableSectionBytes> first_sections,
                            std::span<const ExecutableSectionBytes> second_sections,
                            std::span<const uint8_t> first_elf,
                            std::span<const uint8_t> second_elf);

} // namespace detail

/// @brief Translate one AMDGPU code object using the DBT pipeline.
///
/// This is an internal repo-facing API. It is deliberately small and mirrors the
/// CLI behavior so tests can avoid process management when they need direct
/// access to translated bytes or structured diagnostics.
[[nodiscard]] ToolResult<TranslateOutput> translate_code_object(const TranslateOptions &options);

} // namespace rocjitsu::tools

#endif // ROCJITSU_TOOLS_DBT_TRANSLATE_H_
