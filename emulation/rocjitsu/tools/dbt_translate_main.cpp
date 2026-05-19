// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "tools/dbt_translate.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/executable.h"

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

enum class ReportMode {
  None,
  Text,
};

struct CliOptions {
  TranslateOptions translate;
  ReportMode report = ReportMode::None;
  bool list_code_objects = false;
  bool print_warnings = false;
  bool show_help = false;
  bool saw_guest_arch = false;
  bool saw_host_arch = false;
};

void print_help() {
  std::cout
      << "Usage: rj_dbt_translate --input PATH --guest-arch ARCH --host-arch ARCH "
         "[options]\n\n"
      << "Options:\n"
      << "  --input PATH                    Input host object, fat binary, or AMDGPU code object\n"
      << "  --output PATH                   Output translated AMDGPU code object\n"
      << "  --guest-arch ARCH               Source arch: cdna1..cdna4, rdna1..rdna4\n"
      << "  --host-arch ARCH                Target arch: cdna1..cdna4, rdna1..rdna4\n"
      << "  --input-kind KIND               auto, executable, or code-object (default: auto)\n"
      << "  --input-target TARGET           gfx942 or gfx950 (default: gfx950)\n"
      << "  --code-object-index N           Code-object index for executable input (default: 0)\n"
      << "  --target-mach VALUE             auto, gfx1100, gfx1200, gfx1201, or numeric\n"
      << "  --validate-host-decode          Fail if translated code does not decode as host ISA\n"
      << "  --fail-on-warning               Return nonzero when translation emits warnings\n"
      << "  --print-warnings                Print translation warnings to stderr\n"
      << "  --print-disasm MODE             source, translated, or both\n"
      << "  --report MODE                   Print compact translation report: text\n"
      << "  --section NAME                  Restrict disassembly output to one section\n"
      << "  --list-code-objects             List extractable code objects and exit\n"
      << "  --help                          Show this help\n\n"
      << "At least one of --output, --print-disasm, or --report is required for "
         "translation mode.\n";
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

[[nodiscard]] std::optional<rj_code_arch_t> parse_arch(std::string_view value) {
  if (value == "cdna1")
    return ROCJITSU_CODE_ARCH_CDNA1;
  if (value == "cdna2")
    return ROCJITSU_CODE_ARCH_CDNA2;
  if (value == "cdna3")
    return ROCJITSU_CODE_ARCH_CDNA3;
  if (value == "cdna4")
    return ROCJITSU_CODE_ARCH_CDNA4;
  if (value == "rdna1")
    return ROCJITSU_CODE_ARCH_RDNA1;
  if (value == "rdna2")
    return ROCJITSU_CODE_ARCH_RDNA2;
  if (value == "rdna3")
    return ROCJITSU_CODE_ARCH_RDNA3;
  if (value == "rdna3_5" || value == "rdna3.5")
    return ROCJITSU_CODE_ARCH_RDNA3_5;
  if (value == "rdna4")
    return ROCJITSU_CODE_ARCH_RDNA4;
  return std::nullopt;
}

[[nodiscard]] std::optional<rj_code_target_id_t> parse_target(std::string_view value) {
  if (value == "gfx942")
    return ROCJITSU_CODE_TARGET_GFX942;
  if (value == "gfx950")
    return ROCJITSU_CODE_TARGET_GFX950;
  return std::nullopt;
}

[[nodiscard]] std::optional<InputKind> parse_input_kind(std::string_view value) {
  if (value == "auto")
    return InputKind::Auto;
  if (value == "executable")
    return InputKind::Executable;
  if (value == "code-object")
    return InputKind::CodeObject;
  return std::nullopt;
}

[[nodiscard]] std::optional<DisassemblyMode> parse_disassembly_mode(std::string_view value) {
  if (value == "source")
    return DisassemblyMode::Source;
  if (value == "translated")
    return DisassemblyMode::Translated;
  if (value == "both")
    return DisassemblyMode::Both;
  return std::nullopt;
}

[[nodiscard]] std::optional<ReportMode> parse_report_mode(std::string_view value) {
  if (value == "text")
    return ReportMode::Text;
  return std::nullopt;
}

[[nodiscard]] std::optional<uint32_t> parse_mach(std::string_view value) {
  if (value == "auto")
    return 0;
  if (value == "gfx1100")
    return EF_AMDGPU_MACH_AMDGCN_GFX1100;
  if (value == "gfx1200")
    return EF_AMDGPU_MACH_AMDGCN_GFX1200;
  if (value == "gfx1201")
    return EF_AMDGPU_MACH_AMDGCN_GFX1201;

  uint32_t parsed = 0;
  if (parse_u32(value, parsed))
    return parsed;
  return std::nullopt;
}

[[nodiscard]] const char *arch_name(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return "cdna1";
  case ROCJITSU_CODE_ARCH_CDNA2:
    return "cdna2";
  case ROCJITSU_CODE_ARCH_CDNA3:
    return "cdna3";
  case ROCJITSU_CODE_ARCH_CDNA4:
    return "cdna4";
  case ROCJITSU_CODE_ARCH_RDNA1:
    return "rdna1";
  case ROCJITSU_CODE_ARCH_RDNA2:
    return "rdna2";
  case ROCJITSU_CODE_ARCH_RDNA3:
    return "rdna3";
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return "rdna3_5";
  case ROCJITSU_CODE_ARCH_RDNA4:
    return "rdna4";
  default:
    return "invalid";
  }
}

[[nodiscard]] std::string mach_name(uint32_t mach) {
  if (mach == 0)
    return "auto";
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1100)
    return "gfx1100";
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1200)
    return "gfx1200";
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1201)
    return "gfx1201";

  std::ostringstream os;
  os << "0x" << std::hex << mach;
  return os.str();
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
    if (arg == "--validate-host-decode") {
      options.translate.validate_host_decode = true;
      continue;
    }
    if (arg == "--fail-on-warning") {
      options.translate.fail_on_warning = true;
      continue;
    }
    if (arg == "--print-warnings") {
      options.print_warnings = true;
      continue;
    }
    if (arg == "--list-code-objects") {
      options.list_code_objects = true;
      continue;
    }

    if (arg == "--input") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.translate.input_path = std::string(value);
    } else if (arg == "--output") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.translate.output_path = std::string(value);
    } else if (arg == "--guest-arch") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto arch = parse_arch(value);
      if (!arch) {
        std::cerr << "invalid guest arch: " << value << "\n";
        return false;
      }
      options.translate.guest_arch = *arch;
      options.saw_guest_arch = true;
    } else if (arg == "--host-arch") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto arch = parse_arch(value);
      if (!arch) {
        std::cerr << "invalid host arch: " << value << "\n";
        return false;
      }
      options.translate.host_arch = *arch;
      options.saw_host_arch = true;
    } else if (arg == "--input-kind") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto kind = parse_input_kind(value);
      if (!kind) {
        std::cerr << "invalid input kind: " << value << "\n";
        return false;
      }
      options.translate.input_kind = *kind;
    } else if (arg == "--input-target") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto target = parse_target(value);
      if (!target) {
        std::cerr << "invalid input target: " << value << "\n";
        return false;
      }
      options.translate.input_target = *target;
    } else if (arg == "--code-object-index") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!parse_u32(value, options.translate.code_object_index)) {
        std::cerr << "invalid code-object index: " << value << "\n";
        return false;
      }
    } else if (arg == "--target-mach") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto mach = parse_mach(value);
      if (!mach) {
        std::cerr << "invalid target mach: " << value << "\n";
        return false;
      }
      options.translate.target_mach = *mach;
    } else if (arg == "--print-disasm") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto mode = parse_disassembly_mode(value);
      if (!mode) {
        std::cerr << "invalid disassembly mode: " << value << "\n";
        return false;
      }
      options.translate.disassembly = *mode;
    } else if (arg == "--report") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      auto mode = parse_report_mode(value);
      if (!mode) {
        std::cerr << "invalid report mode: " << value << "\n";
        return false;
      }
      options.report = *mode;
      options.translate.collect_diagnostics = true;
    } else if (arg == "--section") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.translate.disassembly_section = std::string(value);
    } else {
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

[[nodiscard]] std::string
translation_action_text(const InstructionTranslationReport &translation) {
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

[[nodiscard]] size_t count_copied_original(
    const std::vector<InstructionTranslationReport> &translations) {
  size_t count = 0;
  for (const auto &translation : translations) {
    if (translation.copied_original)
      ++count;
  }
  return count;
}

[[nodiscard]] size_t count_semantic_lowerings(
    const std::vector<InstructionTranslationReport> &translations) {
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

void print_code_object_report(std::string_view label, const CodeObjectReport &report) {
  std::cout << "\n" << label << ":\n";
  if (!report.available) {
    std::cout << "  unavailable\n";
    return;
  }

  const ReportTotals totals = report_totals(report);
  std::cout << "  decoder: " << (report.decoder_available ? "available" : "unavailable")
            << "\n";
  std::cout << "  sections: " << report.sections.size() << " bytes=" << totals.size_bytes
            << " instructions=" << totals.instruction_count
            << " decode_failures=" << totals.decode_failure_count << "\n";

  for (const auto &section : report.sections) {
    std::cout << "  section " << section.name << " bytes=" << section.size_bytes
              << " instructions=" << section.instruction_count
              << " decode_failures=" << section.decode_failure_count;
    if (section.has_first_decode_failure) {
      std::cout << " first_failure=0x" << std::hex
                << section.first_decode_failure_offset << std::dec << " reason="
                << section.first_decode_failure_message;
    }
    std::cout << "\n";
  }
}

void print_instruction_translation_report(const TranslateOutput &output) {
  const auto &translations = output.instruction_translations;
  size_t shown = 0;
  size_t changed = 0;
  for (const auto &translation : translations) {
    if (translation.changed)
      ++changed;
    if (should_show_translation(translation))
      ++shown;
  }

  std::cout << "\ninstruction_translations:\n";
  std::cout << "  total=" << translations.size() << " changed=" << changed
            << " shown=" << shown << "\n";
  std::cout << "  actions:"
            << " copy_original=" << count_copied_original(translations)
            << " encode=" << count_runtime_encode(translations)
            << " identity=" << count_action(translations, Action::Identity)
            << " substitute=" << count_action(translations, Action::Substitute)
            << " lower=" << count_action(translations, Action::Lower)
            << " expand=" << count_action(translations, Action::Expand)
            << " illegal=" << count_action(translations, Action::Illegal)
            << " semantic=" << count_semantic_lowerings(translations)
            << "\n";

  const size_t text_size = text_section_size(output.source_report);
  for (const auto &translation : translations) {
    if (!should_show_translation(translation))
      continue;

    std::cout << "  " << hex_offset(translation.source_offset) << " "
              << translation_action_text(translation) << " .text+"
              << hex_offset(translation.source_offset) << " -> "
              << target_location(translation, text_size) << "\n";
    std::cout << "    source: " << translation.source_instruction << "\n";
    if (!translation.target_words.empty())
      std::cout << "    target_words: " << words_text(translation.target_words) << "\n";
    for (const auto &target_instruction : translation.target_instructions)
      std::cout << "    target: " << target_instruction << "\n";
  }
}

void print_text_report(const CliOptions &options, const ToolResult<TranslateOutput> &result) {
  const TranslateOutput &output = result.value;

  std::cout << "rj_dbt_translate: " << (result.ok() ? "ok" : "failed") << "\n";
  std::cout << "input: " << options.translate.input_path << "\n";
  if (!options.translate.output_path.empty())
    std::cout << "output: " << options.translate.output_path << "\n";
  std::cout << "guest_arch: " << arch_name(options.translate.guest_arch) << "\n";
  std::cout << "host_arch: " << arch_name(output.host_arch) << "\n";
  std::cout << "target_mach: " << mach_name(output.target_mach) << "\n";
  std::cout << "output_elf_bytes: " << output.elf_bytes.size() << "\n";

  print_code_object_report("source", output.source_report);
  print_code_object_report("translated", output.translated_report);
  print_instruction_translation_report(output);

  std::cout << "\nwarnings: " << result.warnings.size() << "\n";
  std::cout << "errors: " << result.errors.size() << "\n";
}

int list_code_objects(const CliOptions &options) {
  if (options.translate.input_path.empty()) {
    std::cerr << "--input is required with --list-code-objects\n";
    return kUsageError;
  }

  Executable executable(options.translate.input_path);
  if (executable.is_valid()) {
    std::cout << "gfx942: " << executable.num_code_objects(ROCJITSU_CODE_TARGET_GFX942) << "\n";
    std::cout << "gfx950: " << executable.num_code_objects(ROCJITSU_CODE_TARGET_GFX950) << "\n";
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

  if (options.translate.input_path.empty() || !options.saw_guest_arch || !options.saw_host_arch) {
    std::cerr << "--input, --guest-arch, and --host-arch are required\n";
    print_help();
    return kUsageError;
  }

  if (options.translate.output_path.empty() &&
      options.translate.disassembly == DisassemblyMode::None &&
      options.report == ReportMode::None) {
    std::cerr << "at least one of --output, --print-disasm, or --report is required\n";
    print_help();
    return kUsageError;
  }

  auto result = translate_code_object(options.translate);

  if (options.print_warnings || !result.ok()) {
    for (const auto &warning : result.warnings)
      std::cerr << "warning: " << warning << "\n";
  }

  if (options.report == ReportMode::Text)
    print_text_report(options, result);

  if (options.report == ReportMode::Text && !result.value.disassembly.empty())
    std::cout << "\n";

  if (!result.value.disassembly.empty())
    std::cout << result.value.disassembly;

  if (!result.ok()) {
    for (const auto &error : result.errors)
      std::cerr << "error: " << error.message << "\n";
    return result.errors.front().exit_code;
  }

  return 0;
}
