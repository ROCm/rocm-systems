// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "tools/dbt_translate.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <exception>
#include <iomanip>
#include <memory>
#include <span>
#include <sstream>
#include <utility>

namespace rocjitsu::tools {

namespace {

constexpr int kInputError = 2;
constexpr int kTranslationError = 3;
constexpr int kValidationError = 5;

void add_error(ToolResult<TranslateOutput> &result, int exit_code, std::string message) {
  result.errors.push_back({exit_code, std::move(message)});
}

struct CodeObjectInspection {
  CodeObjectReport report;
  std::string disassembly;
};

void record_decode_failure(CodeSectionReport &section_report, size_t byte_offset,
                           std::string message) {
  ++section_report.decode_failure_count;
  if (!section_report.has_first_decode_failure) {
    section_report.has_first_decode_failure = true;
    section_report.first_decode_failure_offset = byte_offset;
    section_report.first_decode_failure_message = std::move(message);
  }
}

[[nodiscard]] CodeObjectInspection inspect_code_object(const AmdGpuCodeObject &obj,
                                                       rj_code_arch_t arch,
                                                       const std::string &label,
                                                       bool include_disassembly) {
  CodeObjectInspection inspection;
  inspection.report.available = true;

  std::ostringstream os;
  if (include_disassembly)
    os << "--- " << label << " ---\n";

  auto decoder = Decoder::create(arch);
  inspection.report.decoder_available = decoder != nullptr;
  if (!decoder) {
    if (include_disassembly)
      os << "decoder unavailable\n";
    for (const auto *section : obj.code_sections()) {
      CodeSectionReport section_report;
      section_report.name = section->name();
      section_report.size_bytes = section->size();
      inspection.report.sections.push_back(std::move(section_report));
    }
    inspection.disassembly = os.str();
    return inspection;
  }

  for (const auto *section : obj.code_sections()) {
    CodeSectionReport section_report;
    section_report.name = section->name();
    section_report.size_bytes = section->size();

    if (include_disassembly)
      os << "section " << section->name() << " size=" << section->size() << "\n";

    const auto *words = reinterpret_cast<const uint32_t *>(section->data());
    const size_t word_count = section->size() / sizeof(uint32_t);
    size_t pc = 0;

    while (pc < word_count) {
      try {
        std::unique_ptr<Instruction> inst(decoder->decode(&words[pc]));
        if (!inst) {
          record_decode_failure(section_report, pc * sizeof(uint32_t), "decode returned null");
          if (include_disassembly) {
            os << "  0x" << std::hex << std::setw(4) << std::setfill('0') << pc * 4
               << ": <decode returned null>\n"
               << std::dec << std::setfill(' ');
          }
          ++pc;
          continue;
        }

        const uint32_t inst_words = inst->size() / sizeof(uint32_t);
        ++section_report.instruction_count;
        if (include_disassembly) {
          os << "  0x" << std::hex << std::setw(4) << std::setfill('0') << pc * 4 << ": "
             << std::dec << std::setfill(' ') << inst->disassemble() << " [";
          for (uint32_t i = 0; i < inst_words && pc + i < word_count; ++i) {
            if (i != 0)
              os << ' ';
            os << std::hex << std::setw(8) << std::setfill('0') << words[pc + i];
          }
          os << std::dec << std::setfill(' ') << "]\n";
        }
        pc += inst_words == 0 ? 1 : inst_words;
      } catch (const std::exception &e) {
        record_decode_failure(section_report, pc * sizeof(uint32_t), e.what());
        if (include_disassembly) {
          os << "  0x" << std::hex << std::setw(4) << std::setfill('0') << pc * 4
             << ": <decode error: " << e.what() << ">\n"
             << std::dec << std::setfill(' ');
        }
        ++pc;
      }
    }

    inspection.report.sections.push_back(std::move(section_report));
  }

  inspection.disassembly = os.str();
  return inspection;
}

[[nodiscard]] std::string format_offset(size_t offset) {
  std::ostringstream os;
  os << "0x" << std::hex << offset;
  return os.str();
}

[[nodiscard]] bool validate_host_decode(const CodeObjectReport &report, std::string &error) {
  if (!report.available) {
    error = "translated output was not inspected";
    return false;
  }
  if (!report.decoder_available) {
    error = "host decoder unavailable";
    return false;
  }

  size_t inst_count = 0;
  size_t failures = 0;
  const CodeSectionReport *first_failed_section = nullptr;
  for (const auto &section : report.sections) {
    inst_count += section.instruction_count;
    failures += section.decode_failure_count;
    if (first_failed_section == nullptr && section.has_first_decode_failure)
      first_failed_section = &section;
  }

  if (inst_count == 0) {
    error = "translated output contains no decodable host instructions";
    return false;
  }
  if (failures != 0) {
    error = std::to_string(failures) + " translated instructions failed host decode";
    if (first_failed_section != nullptr) {
      error += " (first failure in " + first_failed_section->name + " at " +
               format_offset(first_failed_section->first_decode_failure_offset) + ": " +
               first_failed_section->first_decode_failure_message + ")";
    }
    return false;
  }

  return true;
}

[[nodiscard]] std::string disassemble_source_instruction(const AmdGpuCodeObject &obj,
                                                         uint64_t offset, rj_code_arch_t arch) {
  auto decoder = Decoder::create(arch);
  if (!decoder)
    return "<decoder unavailable>";

  const auto &text_sections = obj.text_sections();
  if (text_sections.empty())
    return "<source section unavailable>";

  // BinaryTranslator trace offsets are relative to the original .text section.
  // Use text_sections() instead of code_sections() so the DBT cave never shifts
  // source locations when a translated object is inspected again.
  const auto *text = text_sections.front();
  if (offset % sizeof(uint32_t) != 0)
    return "<source offset unaligned>";
  if (offset + sizeof(uint32_t) > text->size())
    return "<source offset out of range>";

  const auto *words = reinterpret_cast<const uint32_t *>(text->data());
  try {
    std::unique_ptr<Instruction> inst(decoder->decode(&words[offset / sizeof(uint32_t)]));
    if (!inst)
      return "<decode returned null>";
    return inst->disassemble();
  } catch (const std::exception &e) {
    return std::string("<decode error: ") + e.what() + ">";
  }
}

[[nodiscard]] std::vector<std::string> disassemble_words(std::span<const uint32_t> words,
                                                         rj_code_arch_t arch) {
  std::vector<std::string> lines;
  auto decoder = Decoder::create(arch);
  if (!decoder) {
    lines.push_back("<decoder unavailable>");
    return lines;
  }

  size_t pc = 0;
  while (pc < words.size()) {
    try {
      std::unique_ptr<Instruction> inst(decoder->decode(&words[pc]));
      if (!inst) {
        lines.push_back("<decode returned null>");
        ++pc;
        continue;
      }

      lines.push_back(inst->disassemble());
      const uint32_t inst_words = inst->size() / sizeof(uint32_t);
      pc += inst_words == 0 ? 1 : inst_words;
    } catch (const std::exception &e) {
      lines.push_back(std::string("<decode error: ") + e.what() + ">");
      ++pc;
    }
  }

  return lines;
}

[[nodiscard]] InstructionTranslationReport
build_instruction_report(const TranslationTraceEvent &trace, const AmdGpuCodeObject &source,
                         rj_code_arch_t guest_arch, rj_code_arch_t host_arch) {
  InstructionTranslationReport report;
  report.source_offset = trace.source_offset;
  report.source_size = trace.source_size;
  report.source_words.assign(trace.source_words.begin(), trace.source_words.end());
  report.source_instruction =
      disassemble_source_instruction(source, trace.source_offset, guest_arch);
  report.has_legalization = trace.legalization != nullptr;
  report.action = trace.legalization ? trace.legalization->action : Action::Identity;
  report.copied_original = trace.copied_original;
  report.semantic_lowering = trace.semantic_lowering;
  report.changed = trace.changed;
  report.emitted_in_cave = trace.emitted_in_cave;
  report.target_offset = trace.target_offset;
  report.target_words.assign(trace.target_words.begin(), trace.target_words.end());
  report.target_instructions = disassemble_words(report.target_words, host_arch);
  return report;
}

struct SelectedInput {
  std::unique_ptr<Executable> executable;
  std::unique_ptr<AmdGpuCodeObject> direct_code_object;
  const AmdGpuCodeObject *code_object = nullptr;
};

[[nodiscard]] SelectedInput select_input(const TranslateOptions &options, std::string &error) {
  SelectedInput selected;

  selected.executable = std::make_unique<Executable>(options.input_path);
  if (selected.executable->is_valid()) {
    if (options.input_target == ROCJITSU_CODE_TARGET_INVALID) {
      error = "input target is not selectable from executable inputs: " + options.input_path;
      selected.executable.reset();
      return selected;
    }

    selected.code_object =
        selected.executable->code_object(options.input_target, options.code_object_index);
    if (selected.code_object != nullptr)
      return selected;

    error = "failed to select requested code object from executable: " + options.input_path;
    selected.executable.reset();
    return selected;
  }

  // Try executable inputs first, then standalone AMDGPU code-object ELFs. This
  // keeps the CLI convenient for both HIP objects and already-extracted DBT
  // outputs without exposing a mode flag.
  selected.executable.reset();
  selected.direct_code_object = std::make_unique<AmdGpuCodeObject>(options.input_path);
  if (!selected.direct_code_object->is_valid()) {
    error = "failed to parse input as an AMDGPU code object: " + options.input_path;
    selected.direct_code_object.reset();
    return selected;
  }

  selected.code_object = selected.direct_code_object.get();
  return selected;
}

} // namespace

ToolResult<TranslateOutput> translate_code_object(const TranslateOptions &options) {
  ToolResult<TranslateOutput> output;
  output.value.host_arch = options.host_arch;
  output.value.target_mach =
      options.target_mach ? options.target_mach : elf_mach_for_arch(options.host_arch);

  if (options.input_path.empty()) {
    add_error(output, kInputError, "input path is required");
    return output;
  }

  std::string error;
  SelectedInput input = select_input(options, error);
  if (input.code_object == nullptr) {
    add_error(output, kInputError, error.empty() ? "failed to load input" : error);
    return output;
  }

  const bool need_report = options.collect_diagnostics;
  const bool need_source_disassembly = options.disassembly == DisassemblyMode::Source ||
                                       options.disassembly == DisassemblyMode::Both;
  const bool need_translated_disassembly = options.disassembly == DisassemblyMode::Translated ||
                                           options.disassembly == DisassemblyMode::Both;

  if (need_report || need_source_disassembly) {
    auto source_inspection = inspect_code_object(*input.code_object, options.guest_arch, "source",
                                                 need_source_disassembly);
    output.value.source_report = std::move(source_inspection.report);
    output.value.disassembly += source_inspection.disassembly;
  }

  try {
    BinaryTranslatorOptions translator_options;
    translator_options.debug_min_free_vgpr = options.debug_min_free_vgpr;
    translator_options.debug_continue_after_failure = options.debug_continue_after_failure;
    BinaryTranslator translator(options.guest_arch, options.host_arch, options.target_mach,
                                translator_options);
    if (need_report) {
      output.value.instruction_translations.clear();
      translator.set_trace_callback([&](const TranslationTraceEvent &trace) {
        output.value.instruction_translations.push_back(build_instruction_report(
            trace, *input.code_object, options.guest_arch, options.host_arch));
      });
    }
    auto translated = translator.translate(*input.code_object);
    output.value.elf_bytes = std::move(translated.elf_bytes);
    output.value.host_arch = translated.host_arch;
    output.value.target_mach =
        options.target_mach ? options.target_mach : elf_mach_for_arch(output.value.host_arch);
    output.value.diagnostics = std::move(translated.diagnostics);
  } catch (const std::exception &e) {
    add_error(output, kTranslationError, std::string("translation threw exception: ") + e.what());
    return output;
  }

  if (has_error_diagnostic(output.value.diagnostics)) {
    add_error(output, kTranslationError, "translation failed");
    return output;
  }

  if (output.value.elf_bytes.empty()) {
    add_error(output, kTranslationError, "translation produced an empty ELF image");
    return output;
  }

  AmdGpuCodeObject translated_obj(output.value.elf_bytes.data(), output.value.elf_bytes.size());
  if (!translated_obj.is_valid()) {
    add_error(output, kTranslationError, "translation produced an invalid AMDGPU code object");
    return output;
  }

  {
    auto translated_inspection = inspect_code_object(translated_obj, options.host_arch,
                                                     "translated", need_translated_disassembly);
    output.value.translated_report = std::move(translated_inspection.report);
    output.value.disassembly += translated_inspection.disassembly;
  }

  if (!validate_host_decode(output.value.translated_report, error)) {
    add_error(output, kValidationError, error);
    return output;
  }

  return output;
}

} // namespace rocjitsu::tools
