// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_translate.h
/// @brief Internal entry point for the rj_dbt_translate CLI and tests.

#ifndef ROCJITSU_TOOLS_DBT_TRANSLATE_H_
#define ROCJITSU_TOOLS_DBT_TRANSLATE_H_

#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/rj_code.h"
#include "tools/tool_result.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::tools {

enum class InputKind {
  Auto,
  Executable,
  CodeObject,
};

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
};

struct TranslateOptions {
  std::string input_path;
  std::string output_path;

  InputKind input_kind = InputKind::Auto;
  rj_code_target_id_t input_target = ROCJITSU_CODE_TARGET_GFX950;
  uint32_t code_object_index = 0;

  rj_code_arch_t guest_arch = ROCJITSU_CODE_ARCH_CDNA4;
  rj_code_arch_t host_arch = ROCJITSU_CODE_ARCH_RDNA4;
  uint32_t target_mach = 0;

  bool validate_host_decode = false;
  bool fail_on_warning = false;
  bool collect_diagnostics = false;
  DisassemblyMode disassembly = DisassemblyMode::None;

  // Optional exact section-name filter for disassembly output. This keeps
  // inspection output focused on the rewritten region a developer cares about,
  // such as .rj_translations, without changing the translated ELF.
  std::string disassembly_section;
};

struct TranslateOutput {
  std::vector<uint8_t> elf_bytes;
  rj_code_arch_t host_arch = ROCJITSU_CODE_ARCH_INVALID;
  uint32_t target_mach = 0;
  CodeObjectReport source_report;
  CodeObjectReport translated_report;
  std::vector<InstructionTranslationReport> instruction_translations;
  std::string disassembly;
};

/// @brief Translate one AMDGPU code object using the DBT pipeline.
///
/// This is an internal repo-facing API. It is deliberately small and mirrors the
/// CLI behavior so tests can avoid process management when they need direct
/// access to translated bytes or structured diagnostics.
[[nodiscard]] ToolResult<TranslateOutput> translate_code_object(const TranslateOptions &options);

} // namespace rocjitsu::tools

#endif // ROCJITSU_TOOLS_DBT_TRANSLATE_H_
