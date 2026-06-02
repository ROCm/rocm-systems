// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "tools/dbt_translate.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/executable.h"

#include <array>
#include <charconv>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

using namespace rocjitsu;
using namespace rocjitsu::tools;

namespace {

constexpr int kUsageError = 1;
constexpr int kOutputError = 4;

enum class OutputMode {
  Disasm,
  CodeObject,
  Diff,
};

struct TargetInfo {
  std::string_view name;
  rj_code_arch_t arch;
  uint32_t mach;
  rj_code_target_id_t code_object_target;
};

constexpr std::array<TargetInfo, 4> kTargetInfos = {{
    {"gfx942", ROCJITSU_CODE_ARCH_CDNA3, EF_AMDGPU_MACH_AMDGCN_GFX942, ROCJITSU_CODE_TARGET_GFX942},
    {"gfx950", ROCJITSU_CODE_ARCH_CDNA4, EF_AMDGPU_MACH_AMDGCN_GFX950, ROCJITSU_CODE_TARGET_GFX950},
    {"gfx1200", ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1200,
     ROCJITSU_CODE_TARGET_GFX1200},
    {"gfx1201", ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201,
     ROCJITSU_CODE_TARGET_GFX1201},
}};

struct CliOptions {
  TranslateOptions translate;
  std::string input_target_name;
  std::string output_target_name;
  OutputMode output_mode = OutputMode::Disasm;
  bool list_code_objects = false;
  bool show_help = false;
  bool saw_input = false;
  bool saw_input_target = false;
  bool saw_output_target = false;
};

void print_supported_targets(std::ostream &os) {
  for (size_t i = 0; i < kTargetInfos.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << kTargetInfos[i].name;
  }
}

void print_help() {
  std::cout
      << "Usage: rj_dbt_translate INPUT --input-target TARGET --output-target TARGET "
         "[options]\n\n"
      << "Options:\n"
      << "  --input-target TARGET           Input LLVM machine, e.g. gfx950\n"
      << "  --output-target TARGET          Output LLVM machine, e.g. gfx1200\n"
      << "  --code-object-index N           Code-object index for executable input (default: 0)\n"
      << "  --output-mode MODE              disasm, code-object, or diff (default: disasm)\n"
      << "  --debug-conservative-liveness N Only allocate free VGPR scratch at or above N\n"
      << "  --debug-continue-after-failure Continue collecting diagnostics after failures\n"
      << "  --list-code-objects             List extractable code objects and exit\n"
      << "  --help                          Show this help\n\n"
      << "Supported target names: ";
  print_supported_targets(std::cout);
  std::cout << ".\n";
}

[[nodiscard]] bool parse_u32(std::string_view text, uint32_t &value) {
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
    base = 16;
  }
  auto *begin = text.data();
  auto *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value, base);
  return ec == std::errc{} && ptr == end;
}

[[nodiscard]] std::optional<TargetInfo> parse_target_info(std::string_view value) {
  for (const TargetInfo &target : kTargetInfos) {
    if (value == target.name)
      return target;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<OutputMode> parse_output_mode(std::string_view value) {
  if (value == "disasm")
    return OutputMode::Disasm;
  if (value == "code-object")
    return OutputMode::CodeObject;
  if (value == "diff")
    return OutputMode::Diff;
  return std::nullopt;
}

[[nodiscard]] bool require_value(int argc, char **argv, int &index, std::string_view flag,
                                 std::string_view &value) {
  if (index + 1 >= argc) {
    std::cerr << "missing value for " << flag << "\n";
    return false;
  }
  ++index;
  value = argv[index];
  return true;
}

[[nodiscard]] bool parse_args(int argc, char **argv, CliOptions &options) {
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    std::string_view value;

    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return true;
    }
    if (arg == "--list-code-objects") {
      options.list_code_objects = true;
      continue;
    }
    if (arg == "--debug-conservative-liveness") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      uint32_t min_free_vgpr = 0;
      if (!parse_u32(value, min_free_vgpr) || min_free_vgpr > UINT16_MAX) {
        std::cerr << "invalid minimum free VGPR for " << arg << ": " << value << "\n";
        return false;
      }
      options.translate.debug_min_free_vgpr = static_cast<uint16_t>(min_free_vgpr);
      continue;
    }
    if (arg == "--debug-continue-after-failure") {
      options.translate.debug_continue_after_failure = true;
      continue;
    }

    if (arg == "--input-target") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto target = parse_target_info(value);
      if (!target) {
        std::cerr << "invalid input target: " << value << "\n";
        return false;
      }
      options.translate.guest_arch = target->arch;
      options.translate.input_target = target->code_object_target;
      options.input_target_name = std::string(target->name);
      options.saw_input_target = true;
    } else if (arg == "--output-target") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto target = parse_target_info(value);
      if (!target) {
        std::cerr << "invalid output target: " << value << "\n";
        return false;
      }
      options.translate.host_arch = target->arch;
      options.translate.target_mach = target->mach;
      options.output_target_name = std::string(target->name);
      options.saw_output_target = true;
    } else if (arg == "--code-object-index") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!parse_u32(value, options.translate.code_object_index)) {
        std::cerr << "invalid code-object index: " << value << "\n";
        return false;
      }
    } else if (arg == "--output-mode") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto mode = parse_output_mode(value);
      if (!mode) {
        std::cerr << "invalid output mode: " << value << "\n";
        return false;
      }
      options.output_mode = *mode;
    } else {
      if (!arg.empty() && arg.front() != '-') {
        if (options.saw_input) {
          std::cerr << "unexpected positional argument: " << arg << "\n";
          return false;
        }
        options.translate.input_path = std::string(arg);
        options.saw_input = true;
        continue;
      }
      std::cerr << "unknown option: " << arg << "\n";
      return false;
    }
  }

  return true;
}

struct ReportTotals {
  size_t size_bytes = 0;
  size_t instruction_count = 0;
  size_t decode_failure_count = 0;
};

[[nodiscard]] ReportTotals report_totals(const CodeObjectReport &report) {
  ReportTotals totals;
  for (const auto &section : report.sections) {
    totals.size_bytes += section.size_bytes;
    totals.instruction_count += section.instruction_count;
    totals.decode_failure_count += section.decode_failure_count;
  }
  return totals;
}

[[nodiscard]] size_t text_section_size(const CodeObjectReport &report) {
  for (const auto &section : report.sections) {
    if (section.name == ".text")
      return section.size_bytes;
  }
  return 0;
}

[[nodiscard]] std::string hex_offset(uint64_t value) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(4) << std::setfill('0') << value;
  return os.str();
}

[[nodiscard]] std::string words_text(const std::vector<uint32_t> &words) {
  std::ostringstream os;
  for (size_t i = 0; i < words.size(); ++i) {
    if (i != 0)
      os << ' ';
    os << std::hex << std::setw(8) << std::setfill('0') << words[i];
  }
  return os.str();
}

[[nodiscard]] const char *diagnostic_severity_name(DiagnosticSeverity severity) {
  switch (severity) {
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  }
  return "diagnostic";
}

[[nodiscard]] const char *diagnostic_kind_name(DiagnosticKind kind) {
  switch (kind) {
  case DiagnosticKind::UnsupportedGuestArch:
    return "unsupported-guest-arch";
  case DiagnosticKind::KernelDescriptor:
    return "kernel-descriptor";
  case DiagnosticKind::Legalization:
    return "legalization";
  case DiagnosticKind::ExpandMissing:
    return "expand-missing";
  case DiagnosticKind::ExpandFailed:
    return "expand-failed";
  case DiagnosticKind::ResourceLimit:
    return "resource-limit";
  }
  return "unknown";
}

void print_diagnostic(std::ostream &os, const TranslationDiagnostic &diagnostic) {
  os << diagnostic_severity_name(diagnostic.severity) << ": "
     << diagnostic_kind_name(diagnostic.kind);
  if (diagnostic.guest_offset)
    os << " .text+" << hex_offset(*diagnostic.guest_offset);
  if (!diagnostic.mnemonic.empty())
    os << " " << diagnostic.mnemonic;
  os << ": " << diagnostic.message << "\n";
  for (const auto &item : diagnostic.required_work)
    os << "  required: " << item << "\n";
}

[[nodiscard]] const char *legalization_action_name(Action action) {
  switch (action) {
  case Action::Identity:
    return "identity";
  case Action::Substitute:
    return "substitute";
  case Action::Lower:
    return "lower";
  case Action::Expand:
    return "expand";
  case Action::Illegal:
    return "illegal";
  }
  return "unknown";
}

[[nodiscard]] std::string translation_action_text(const InstructionTranslationReport &translation) {
  if (translation.copied_original)
    return "copy_original";
  if (!translation.has_legalization)
    return "encode";

  std::string text = legalization_action_name(translation.action);
  if (translation.semantic_lowering)
    text += " semantic";
  return text;
}

[[nodiscard]] bool should_show_translation(const InstructionTranslationReport &translation) {
  if (translation.copied_original ||
      (translation.has_legalization && translation.action == Action::Identity))
    return translation.changed || translation.emitted_in_cave;
  return true;
}

[[nodiscard]] size_t count_action(const std::vector<InstructionTranslationReport> &translations,
                                  Action action) {
  size_t count = 0;
  for (const auto &translation : translations) {
    if (!translation.copied_original && translation.has_legalization &&
        translation.action == action)
      ++count;
  }
  return count;
}

[[nodiscard]] size_t
count_runtime_encode(const std::vector<InstructionTranslationReport> &translations) {
  size_t count = 0;
  for (const auto &translation : translations) {
    if (!translation.copied_original && !translation.has_legalization)
      ++count;
  }
  return count;
}

[[nodiscard]] size_t
count_copied_original(const std::vector<InstructionTranslationReport> &translations) {
  size_t count = 0;
  for (const auto &translation : translations) {
    if (translation.copied_original)
      ++count;
  }
  return count;
}

[[nodiscard]] size_t
count_semantic_lowerings(const std::vector<InstructionTranslationReport> &translations) {
  size_t count = 0;
  for (const auto &translation : translations) {
    if (translation.semantic_lowering)
      ++count;
  }
  return count;
}

[[nodiscard]] std::string target_location(const InstructionTranslationReport &translation,
                                          size_t text_size) {
  if (translation.emitted_in_cave && translation.target_offset >= text_size)
    return ".rj_translations+" + hex_offset(translation.target_offset - text_size);
  return ".text+" + hex_offset(translation.target_offset);
}

void print_code_object_report(std::ostream &os, std::string_view label,
                              const CodeObjectReport &report) {
  os << "\n" << label << ":\n";
  if (!report.available) {
    os << "  unavailable\n";
    return;
  }

  const ReportTotals totals = report_totals(report);
  os << "  decoder: " << (report.decoder_available ? "available" : "unavailable") << "\n";
  os << "  sections: " << report.sections.size() << " bytes=" << totals.size_bytes
     << " instructions=" << totals.instruction_count
     << " decode_failures=" << totals.decode_failure_count << "\n";

  for (const auto &section : report.sections) {
    os << "  section " << section.name << " bytes=" << section.size_bytes
       << " instructions=" << section.instruction_count
       << " decode_failures=" << section.decode_failure_count;
    if (section.has_first_decode_failure) {
      os << " first_failure=0x" << std::hex << section.first_decode_failure_offset << std::dec
         << " reason=" << section.first_decode_failure_message;
    }
    os << "\n";
  }
}

void print_instruction_translation_report(std::ostream &os, const TranslateOutput &output) {
  const auto &translations = output.instruction_translations;
  size_t shown = 0;
  size_t changed = 0;
  for (const auto &translation : translations) {
    if (translation.changed)
      ++changed;
    if (should_show_translation(translation))
      ++shown;
  }

  os << "\ninstruction_translations:\n";
  os << "  total=" << translations.size() << " changed=" << changed << " shown=" << shown << "\n";
  os << "  actions:" << " copy_original=" << count_copied_original(translations)
     << " encode=" << count_runtime_encode(translations)
     << " identity=" << count_action(translations, Action::Identity)
     << " substitute=" << count_action(translations, Action::Substitute)
     << " lower=" << count_action(translations, Action::Lower)
     << " expand=" << count_action(translations, Action::Expand)
     << " illegal=" << count_action(translations, Action::Illegal)
     << " semantic=" << count_semantic_lowerings(translations) << "\n";

  const size_t text_size = text_section_size(output.source_report);
  for (const auto &translation : translations) {
    if (!should_show_translation(translation))
      continue;

    os << "  " << hex_offset(translation.source_offset) << " "
       << translation_action_text(translation) << " .text+" << hex_offset(translation.source_offset)
       << " -> " << target_location(translation, text_size) << "\n";
    if (!translation.source_words.empty())
      os << "    source_words: " << words_text(translation.source_words) << "\n";
    os << "    source: " << translation.source_instruction << "\n";
    if (!translation.target_words.empty())
      os << "    target_words: " << words_text(translation.target_words) << "\n";
    for (const auto &target_instruction : translation.target_instructions)
      os << "    target: " << target_instruction << "\n";
  }
}

void print_text_report(std::ostream &os, const CliOptions &options,
                       const ToolResult<TranslateOutput> &result) {
  const TranslateOutput &output = result.value;

  os << "rj_dbt_translate: " << (result.ok() ? "ok" : "failed") << "\n";
  os << "input: " << options.translate.input_path << "\n";
  os << "input_target: " << options.input_target_name << "\n";
  os << "output_target: " << options.output_target_name << "\n";
  os << "output_elf_bytes: " << output.elf_bytes.size() << "\n";

  print_code_object_report(os, "source", output.source_report);
  print_code_object_report(os, "translated", output.translated_report);
  print_instruction_translation_report(os, output);

  os << "\ndiagnostics: " << output.diagnostics.size() << "\n";
  os << "errors: " << result.errors.size() << "\n";
}

int list_code_objects(const CliOptions &options) {
  if (options.translate.input_path.empty()) {
    std::cerr << "input path is required with --list-code-objects\n";
    return kUsageError;
  }

  Executable executable(options.translate.input_path);
  if (executable.is_valid()) {
    for (const TargetInfo &target : kTargetInfos) {
      std::cout << target.name << ": " << executable.num_code_objects(target.code_object_target)
                << "\n";
    }
    return 0;
  }

  AmdGpuCodeObject code_object(options.translate.input_path);
  if (!code_object.is_valid()) {
    std::cerr << "failed to parse input as executable or AMDGPU code object\n";
    return 2;
  }

  std::cout << "standalone-code-object: target-id=" << static_cast<int>(code_object.target_id())
            << "\n";
  return 0;
}

[[nodiscard]] bool emit_output(const CliOptions &options, const TranslateOutput &output) {
  if (options.output_mode == OutputMode::CodeObject) {
    std::cout.write(reinterpret_cast<const char *>(output.elf_bytes.data()),
                    static_cast<std::streamsize>(output.elf_bytes.size()));
    return static_cast<bool>(std::cout);
  }

  std::cout << output.disassembly;
  return static_cast<bool>(std::cout);
}

} // namespace

int main(int argc, char **argv) {
  CliOptions options;
  if (!parse_args(argc, argv, options)) {
    print_help();
    return kUsageError;
  }

  if (options.show_help) {
    print_help();
    return 0;
  }

  if (options.list_code_objects)
    return list_code_objects(options);

  if (!options.saw_input || !options.saw_input_target || !options.saw_output_target) {
    std::cerr << "input path, --input-target, and --output-target are required\n";
    print_help();
    return kUsageError;
  }

  options.translate.collect_diagnostics = options.output_mode == OutputMode::Diff;
  options.translate.disassembly = options.output_mode == OutputMode::Disasm
                                      ? DisassemblyMode::Translated
                                      : DisassemblyMode::None;

  auto result = translate_code_object(options.translate);

  for (const auto &diagnostic : result.value.diagnostics)
    print_diagnostic(std::cerr, diagnostic);

  if (options.output_mode == OutputMode::Diff)
    print_text_report(std::cout, options, result);

  if (!result.ok()) {
    for (const auto &error : result.errors)
      std::cerr << "error: " << error.message << "\n";
    return result.errors.front().exit_code;
  }

  if (options.output_mode != OutputMode::Diff && !emit_output(options, result.value)) {
    std::cerr << "error: failed to write output\n";
    return kOutputError;
  }

  return 0;
}
