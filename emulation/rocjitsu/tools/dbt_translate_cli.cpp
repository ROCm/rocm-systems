// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dbt_translate_cli.h"
#include "cli_choices.h"
#include "util/cli.h"

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
namespace cli = util::cli;

namespace {

constexpr int kUsageError = 1;
// Matches the translation-failure exit code used by translate_code_object so a
// non-dispatchable (skipped-kernel) artifact reports the same class of failure.
constexpr int kTranslationError = 3;
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
  bool operator==(const TargetInfo &) const = default;
};

// \NPI new GPU: add a {"gfxNNNN", ARCH, EF_MACH, TARGET} row here
// (and bump the  std::array size) so the DBT tool can translate to/from it.
constexpr std::array<TargetInfo, 6> kTargetInfos = {{
    {"gfx942", ROCJITSU_CODE_ARCH_CDNA3, EF_AMDGPU_MACH_AMDGCN_GFX942, ROCJITSU_CODE_TARGET_GFX942},
    {"gfx950", ROCJITSU_CODE_ARCH_CDNA4, EF_AMDGPU_MACH_AMDGCN_GFX950, ROCJITSU_CODE_TARGET_GFX950},
    {"gfx1200", ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1200,
     ROCJITSU_CODE_TARGET_GFX1200},
    {"gfx1201", ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201,
     ROCJITSU_CODE_TARGET_GFX1201},
    // gfx1250 A0 and B0 share the same ELF machine and structural ISA. Separate
    // revision options select the translation direction.
    {"gfx1250", ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1250,
     ROCJITSU_CODE_TARGET_GFX1250},
    {"gfx1251", ROCJITSU_CODE_ARCH_CDNA5, EF_AMDGPU_MACH_AMDGCN_GFX1251,
     ROCJITSU_CODE_TARGET_GFX1251},
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
  bool saw_input_revision = false;
  bool saw_output_revision = false;
  bool show_all_translations = false;
};

void print_supported_targets(std::ostream &os) {
  for (size_t i = 0; i < kTargetInfos.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << kTargetInfos[i].name;
  }
}

[[nodiscard]] bool parse_u32(std::string_view text, uint32_t &value) {
  if (text.empty())
    return false;
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

[[nodiscard]] std::string_view revision_name(ProcessorRevision revision) {
  switch (revision) {
  case ProcessorRevision::Gfx1250A0:
    return "a0";
  case ProcessorRevision::Gfx1250B0:
    return "b0";
  case ProcessorRevision::Unspecified:
    return "unspecified";
  }
  return "unspecified";
}

// The DBT interface owns its allowed values and optional policy; the shared
// helper only provides table lookup and formatting.
template <typename T> class ChoiceFlag final : public cli::Flag {
public:
  ChoiceFlag(cli::FlagInfo info, std::vector<NamedChoice<T>> choices, std::optional<T> initial = {})
      : Flag(std::move(info)), choices_(std::move(choices)), value_(std::move(initial)) {}
  const std::optional<T> &value() const { return value_; }
  util::Result set_value(std::optional<T> value, util::DiagnosticEmitter diagnostic) {
    if (value && !choice_name<T>(choices(), *value))
      return diagnostic.emit() << "unsupported choice";
    value_ = std::move(value);
    return util::Result::success();
  }
  std::string format_value() const override {
    if (value_) {
      std::optional<std::string_view> name = choice_name<T>(choices(), *value_);
      if (name)
        return std::string(*name);
    }
    return "unset";
  }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    std::optional<T> choice = find_choice<T>(choices(), text);
    if (choice)
      return set_value(*choice, diagnostic);
    return diagnostic.emit() << "unsupported value '" << text << "'";
  }
  std::span<const NamedChoice<T>> choices() const { return {choices_.data(), choices_.size()}; }
  std::vector<NamedChoice<T>> choices_;
  std::optional<T> value_;
};

class DbtUIntFlag final : public cli::Flag {
public:
  DbtUIntFlag(cli::FlagInfo info, uint32_t maximum = UINT32_MAX,
              std::optional<uint32_t> initial = 0)
      : Flag(std::move(info)), value_(initial), maximum_(maximum) {}
  std::optional<uint32_t> value() const { return value_; }
  util::Result set_value(std::optional<uint32_t> value, util::DiagnosticEmitter diagnostic) {
    if (value && *value > maximum_)
      return diagnostic.emit() << "expected an integer at most " << maximum_;
    value_ = value;
    return util::Result::success();
  }
  std::string format_value() const override { return value_ ? std::to_string(*value_) : "unset"; }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    uint32_t parsed = 0;
    if (!parse_u32(text, parsed))
      return diagnostic.emit() << "expected a decimal or hexadecimal uint32";
    return set_value(parsed, diagnostic);
  }
  std::optional<uint32_t> value_;
  uint32_t maximum_;
};

struct DbtFlags {
  std::shared_ptr<ChoiceFlag<TargetInfo>> input_target;
  std::shared_ptr<ChoiceFlag<TargetInfo>> output_target;
  std::shared_ptr<ChoiceFlag<ProcessorRevision>> input_revision;
  std::shared_ptr<ChoiceFlag<ProcessorRevision>> output_revision;
  std::shared_ptr<ChoiceFlag<OutputMode>> output_mode;
  std::shared_ptr<DbtUIntFlag> code_object_index = std::make_shared<DbtUIntFlag>(cli::FlagInfo{
      "--code-object-index", {}, "Code-object index for executable input (default: 0)"});
  std::shared_ptr<DbtUIntFlag> debug_min_free_vgpr = std::make_shared<DbtUIntFlag>(
      cli::FlagInfo{"--debug-conservative-liveness",
                    {},
                    "Only allocate free VGPR scratch at or above this index"},
      UINT16_MAX, std::nullopt);
  std::shared_ptr<cli::BoolFlag> keep_going = std::make_shared<cli::BoolFlag>(
      cli::FlagInfo{"--debug-continue-after-failure",
                    {},
                    "Continue collecting diagnostics after failures",
                    cli::ValueMode::Switch});
  std::shared_ptr<cli::BoolFlag> skip_failed = std::make_shared<cli::BoolFlag>(
      cli::FlagInfo{"--skip-failed-kernels",
                    {},
                    "Preserve failed kernels and continue other kernels",
                    cli::ValueMode::Switch});
  std::shared_ptr<cli::BoolFlag> idempotence = std::make_shared<cli::BoolFlag>(
      cli::FlagInfo{"--verify-idempotence",
                    {},
                    "Require a same-architecture second pass to be unchanged",
                    cli::ValueMode::Switch});
  std::shared_ptr<cli::BoolFlag> discharge = std::make_shared<cli::BoolFlag>(
      cli::FlagInfo{"--verify-rewrite-discharge",
                    {},
                    "Require no registered rewrite trigger in final output",
                    cli::ValueMode::Switch});
  std::shared_ptr<cli::BoolFlag> show_all_translations = std::make_shared<cli::BoolFlag>(
      cli::FlagInfo{"--show-all-translations",
                    {},
                    "Include unchanged identity mappings in diff output",
                    cli::ValueMode::Switch});
  std::shared_ptr<cli::BoolFlag> list_code_objects = std::make_shared<cli::BoolFlag>(cli::FlagInfo{
      "--list-code-objects", {}, "List extractable code objects and exit", cli::ValueMode::Switch});
  std::shared_ptr<cli::BoolFlag> help = std::make_shared<cli::BoolFlag>(
      cli::FlagInfo{"--help", {"-h"}, "Show help", cli::ValueMode::Switch});

  DbtFlags() {
    std::vector<NamedChoice<TargetInfo>> targets;
    for (const auto &target : kTargetInfos)
      targets.emplace_back(target.name, target);
    input_target = std::make_shared<ChoiceFlag<TargetInfo>>(
        cli::FlagInfo{"--input-target", {}, "Input LLVM machine, e.g. gfx950"}, targets);
    output_target = std::make_shared<ChoiceFlag<TargetInfo>>(
        cli::FlagInfo{"--output-target", {}, "Output LLVM machine, e.g. gfx1200"}, targets);
    const std::vector<NamedChoice<ProcessorRevision>> revisions{
        {"a0", ProcessorRevision::Gfx1250A0}, {"b0", ProcessorRevision::Gfx1250B0}};
    input_revision = std::make_shared<ChoiceFlag<ProcessorRevision>>(
        cli::FlagInfo{"--input-revision", {}, "Input revision: a0 or b0 (gfx1250 only)"},
        revisions);
    output_revision = std::make_shared<ChoiceFlag<ProcessorRevision>>(
        cli::FlagInfo{"--output-revision", {}, "Output revision: a0 or b0 (gfx1250 only)"},
        revisions);
    const std::vector<NamedChoice<OutputMode>> modes{{"disasm", OutputMode::Disasm},
                                                     {"code-object", OutputMode::CodeObject},
                                                     {"diff", OutputMode::Diff}};
    output_mode = std::make_shared<ChoiceFlag<OutputMode>>(
        cli::FlagInfo{"--output-mode", {}, "disasm, code-object, or diff (default: disasm)"}, modes,
        OutputMode::Disasm);
  }

  util::Result register_flags(cli::Command &command, util::DiagnosticEmitter diagnostic) const {
    if (command
            .add_flags("Translation",
                       {input_target, output_target, input_revision, output_revision,
                        code_object_index, output_mode},
                       diagnostic)
            .failed() ||
        command
            .add_flags("Validation and diagnostics",
                       {debug_min_free_vgpr, keep_going, skip_failed, idempotence, discharge,
                        show_all_translations},
                       diagnostic)
            .failed())
      return util::Result::failure();
    return command.add_flags("Information", {list_code_objects, help}, diagnostic);
  }
};

void print_help(const cli::Command &command) {
  std::cout << cli::format_help(command)
            << "\nINPUT, --input-target and --output-target are required for translation.\n"
            << "gfx1250 b0-to-a0 selects its translation profile; equal revisions are identity.\n"
            << "Supported target names: ";
  print_supported_targets(std::cout);
  std::cout << ".\n";
}

// Build the translator's existing domain request from typed values at dispatch.
// The registered flag objects remain alive throughout the invocation.
[[nodiscard]] bool parse_args(int argc, char **argv, cli::Command &command, const DbtFlags &flags,
                              CliOptions &options) {
  auto report = [](std::string_view message) { std::cerr << message << '\n'; };
  util::DiagnosticEmitter diagnostic(report);
  if (flags.register_flags(command, diagnostic).failed())
    return false;
  const std::vector<std::string_view> arguments(argv + 1, argv + argc);
  auto parsed = cli::parse(command, arguments, diagnostic);
  if (parsed.failed())
    return false;
  options.show_help = flags.help->value();
  if (options.show_help)
    return true;
  if (parsed.value().positional.size() > 1) {
    std::cerr << "unexpected positional argument: " << parsed.value().positional[1] << '\n';
    return false;
  }
  options.saw_input =
      !parsed.value().positional.empty() && !parsed.value().positional.front().empty();
  if (options.saw_input)
    options.translate.input_path = parsed.value().positional.front();
  if (const auto &target = flags.input_target->value()) {
    options.translate.guest_arch = target->arch;
    options.translate.input_target = target->code_object_target;
    options.input_target_name = target->name;
    options.saw_input_target = true;
  }
  if (const auto &target = flags.output_target->value()) {
    options.translate.host_arch = target->arch;
    options.translate.target_mach = target->mach;
    options.output_target_name = target->name;
    options.saw_output_target = true;
  }
  options.saw_input_revision = flags.input_revision->value().has_value();
  options.saw_output_revision = flags.output_revision->value().has_value();
  options.translate.input_revision =
      flags.input_revision->value().value_or(ProcessorRevision::Unspecified);
  options.translate.output_revision =
      flags.output_revision->value().value_or(ProcessorRevision::Unspecified);
  options.translate.code_object_index = flags.code_object_index->value().value_or(0);
  if (flags.debug_min_free_vgpr->value())
    options.translate.debug_min_free_vgpr =
        static_cast<uint16_t>(*flags.debug_min_free_vgpr->value());
  options.translate.debug_continue_after_failure = flags.keep_going->value();
  options.translate.skip_failed_kernels = flags.skip_failed->value();
  options.translate.verify_idempotence = flags.idempotence->value();
  options.translate.verify_rewrite_discharge = flags.discharge->value();
  options.output_mode = *flags.output_mode->value();
  options.show_all_translations = flags.show_all_translations->value();
  options.list_code_objects = flags.list_code_objects->value();
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
  case DiagnosticKind::DataOnly:
    return "data-only";
  case DiagnosticKind::NothingToTranslate:
    return "nothing-to-translate";
  case DiagnosticKind::ResourceLimit:
    return "resource-limit";
  case DiagnosticKind::KernelSkipped:
    return "kernel-skipped";
  case DiagnosticKind::ResidualRewrite:
    return "residual-rewrite";
  }
  return "unknown";
}

void print_diagnostic(std::ostream &os, const TranslationDiagnostic &diagnostic,
                      std::string_view prefix = {}) {
  if (!prefix.empty())
    os << prefix << ' ';
  os << diagnostic_severity_name(diagnostic.severity) << ": "
     << diagnostic_kind_name(diagnostic.kind);
  if (diagnostic.guest_offset)
    os << " .text+" << hex_offset(*diagnostic.guest_offset);
  if (diagnostic.output_offset)
    os << " output:.text+" << hex_offset(*diagnostic.output_offset);
  if (!diagnostic.mnemonic.empty())
    os << " " << diagnostic.mnemonic;
  os << ": " << diagnostic.message << "\n";
  for (const auto &item : diagnostic.required_work) {
    os << "  ";
    if (!prefix.empty())
      os << prefix << ' ';
    os << "required: " << item << "\n";
  }
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

  // An unclassified semantic lowering deliberately reads the same as a
  // classified expansion: whether a legalization entry also named the opcode is
  // internal to the profile, and the action the reader sees is the same one.
  const auto action = translation.effective_action();
  if (!action)
    return "encode";

  std::string text = legalization_action_name(*action);
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
  return static_cast<size_t>(std::ranges::count_if(
      translations, [action](const InstructionTranslationReport &translation) {
        return translation.effective_action() == action;
      }));
}

/// @brief Instructions re-encoded with no rule: not copied, and no action.
[[nodiscard]] size_t
count_runtime_encode(const std::vector<InstructionTranslationReport> &translations) {
  return static_cast<size_t>(
      std::ranges::count_if(translations, [](const InstructionTranslationReport &translation) {
        return !translation.copied_original && !translation.effective_action();
      }));
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

[[nodiscard]] std::string target_location(const InstructionTranslationReport &translation) {
  if (translation.emitted_in_cave)
    return ".text(local-cave)+" + hex_offset(translation.target_offset);
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

void print_instruction_translation_report(std::ostream &os, const TranslateOutput &output,
                                          bool show_all_translations) {
  const auto &translations = output.instruction_translations;
  size_t shown = 0;
  size_t changed = 0;
  for (const auto &translation : translations) {
    if (translation.changed)
      ++changed;
    if (show_all_translations || should_show_translation(translation))
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

  for (const auto &translation : translations) {
    if (!show_all_translations && !should_show_translation(translation))
      continue;

    os << "  " << hex_offset(translation.source_offset) << " "
       << translation_action_text(translation) << " .text+" << hex_offset(translation.source_offset)
       << " -> " << target_location(translation) << "\n";
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
  os << "source_code_object_id: fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
     << output.source_code_object_id << std::dec << std::setfill(' ') << "\n";
  if (options.saw_input_revision)
    os << "input_revision: " << revision_name(output.input_revision) << "\n";
  os << "output_target: " << options.output_target_name << "\n";
  if (options.saw_output_revision)
    os << "output_revision: " << revision_name(output.output_revision) << "\n";
  os << "output_elf_bytes: " << output.elf_bytes.size() << "\n";
  if (options.translate.verify_idempotence) {
    const std::string_view status = !output.idempotence_checked   ? "not-checked"
                                    : output.idempotence_verified ? "verified"
                                                                  : "not-verified";
    os << "idempotence: " << status << "\n";
  }
  if (options.translate.verify_rewrite_discharge) {
    const std::string_view status = !output.rewrite_discharge_checked   ? "not-checked"
                                    : output.rewrite_discharge_verified ? "verified"
                                                                        : "not-verified";
    os << "rewrite_discharge: " << status << "\n";
  }

  print_code_object_report(os, "source", output.source_report);
  print_code_object_report(os, "translated", output.translated_report);
  print_instruction_translation_report(os, output, options.show_all_translations);

  os << "\ndiagnostics: " << output.diagnostics.size() << "\n";
  os << "idempotence_diagnostics: " << output.idempotence_diagnostics.size() << "\n";
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

int rocjitsu::tools::detail::run_dbt_translate_cli(int argc, char **argv,
                                                   TranslateCodeObjectFn translate) {
  cli::Command command{"rj_dbt_translate"};
  command.set_positional_usage("INPUT");
  DbtFlags flags;
  CliOptions options;
  if (!parse_args(argc, argv, command, flags, options)) {
    print_help(command);
    return kUsageError;
  }

  if (options.show_help) {
    print_help(command);
    return 0;
  }

  if (options.list_code_objects && options.translate.verify_idempotence) {
    std::cerr << "--verify-idempotence cannot be combined with --list-code-objects\n";
    return kUsageError;
  }
  if (options.list_code_objects && options.translate.verify_rewrite_discharge) {
    std::cerr << "--verify-rewrite-discharge cannot be combined with --list-code-objects\n";
    return kUsageError;
  }

  if (options.list_code_objects)
    return list_code_objects(options);

  if (!options.saw_input || !options.saw_input_target || !options.saw_output_target) {
    std::cerr << "input path, --input-target, and --output-target are required\n";
    print_help(command);
    return kUsageError;
  }

  if (const std::optional<std::string_view> request_error =
          translation_request_error(options.translate)) {
    std::cerr << *request_error << "\n";
    return kUsageError;
  }

  options.translate.collect_diagnostics = options.output_mode == OutputMode::Diff;
  options.translate.disassembly = options.output_mode == OutputMode::Disasm
                                      ? DisassemblyMode::Translated
                                      : DisassemblyMode::None;

  auto result = translate(options.translate);

  for (const auto &diagnostic : result.value.diagnostics)
    print_diagnostic(std::cerr, diagnostic);
  for (const auto &diagnostic : result.value.idempotence_diagnostics)
    print_diagnostic(std::cerr, diagnostic, "idempotence");

  if (options.output_mode == OutputMode::Diff)
    print_text_report(std::cout, options, result);

  if (!result.ok()) {
    for (const auto &error : result.errors)
      std::cerr << "error: " << error.message << "\n";
    return result.errors.front().exit_code;
  }

  // A skipped kernel is a warning, so result.ok() stays true, but its s_endpgm
  // stub completes normally without producing the kernel's outputs and would
  // silently produce wrong results if dispatched. Refuse to emit an executable
  // code object in that case (the HSA hook rejects the same load). Diff/Disasm
  // are diagnostic-only inspection modes and remain available.
  if (options.output_mode == OutputMode::CodeObject && !result.value.dispatchable()) {
    std::cerr << "error: translation skipped one or more kernels; the code object is not "
                 "dispatchable and will not be written\n";
    return kTranslationError;
  }

  if (options.output_mode != OutputMode::Diff && !emit_output(options, result.value)) {
    std::cerr << "error: failed to write output\n";
    return kOutputError;
  }

  return 0;
}
