// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/gfx1250_b0_to_a0_library.h"
#include "rocjitsu/isa/isa_family.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/code_object_identity.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/gfx1250_b0_to_a0_diagnostics.h"

#include <string>
#include <utility>
#include <vector>

namespace {

constexpr char kSeverityWarning[] = "warning";
constexpr char kSeverityError[] = "error";
constexpr char kKindApiInvalidArgument[] = "api-invalid-argument";
constexpr char kKindInputInvalidCodeObject[] = "input-invalid-code-object";
constexpr char kKindResultNotDispatchable[] = "result-not-dispatchable";
constexpr char kKindTranslationException[] = "translation-exception";
constexpr char kKindTranslatorUnsupportedGuestArch[] = "translator-unsupported-guest-arch";
constexpr char kKindTranslatorKernelDescriptor[] = "translator-kernel-descriptor";
constexpr char kKindTranslatorLegalization[] = "translator-legalization";
constexpr char kKindTranslatorExpandMissing[] = "translator-expand-missing";
constexpr char kKindTranslatorExpandFailed[] = "translator-expand-failed";
constexpr char kKindTranslatorDataOnly[] = "translator-data-only";
constexpr char kKindTranslatorNothingToTranslate[] = "translator-nothing-to-translate";
constexpr char kKindTranslatorResourceLimit[] = "translator-resource-limit";
constexpr char kKindTranslatorKernelSkipped[] = "translator-kernel-skipped";
constexpr char kKindTranslatorResidualRewrite[] = "translator-residual-rewrite";

const char *diagnostic_severity_name(rocjitsu::DiagnosticSeverity severity) noexcept {
  switch (severity) {
  case rocjitsu::DiagnosticSeverity::Warning:
    return kSeverityWarning;
  case rocjitsu::DiagnosticSeverity::Error:
    return kSeverityError;
  }
  return "unknown";
}

const char *diagnostic_kind_name(rocjitsu::DiagnosticKind kind) noexcept {
  switch (kind) {
  case rocjitsu::DiagnosticKind::UnsupportedGuestArch:
    return kKindTranslatorUnsupportedGuestArch;
  case rocjitsu::DiagnosticKind::KernelDescriptor:
    return kKindTranslatorKernelDescriptor;
  case rocjitsu::DiagnosticKind::Legalization:
    return kKindTranslatorLegalization;
  case rocjitsu::DiagnosticKind::ExpandMissing:
    return kKindTranslatorExpandMissing;
  case rocjitsu::DiagnosticKind::ExpandFailed:
    return kKindTranslatorExpandFailed;
  case rocjitsu::DiagnosticKind::DataOnly:
    return kKindTranslatorDataOnly;
  case rocjitsu::DiagnosticKind::NothingToTranslate:
    return kKindTranslatorNothingToTranslate;
  case rocjitsu::DiagnosticKind::ResourceLimit:
    return kKindTranslatorResourceLimit;
  case rocjitsu::DiagnosticKind::KernelSkipped:
    return kKindTranslatorKernelSkipped;
  case rocjitsu::DiagnosticKind::ResidualRewrite:
    return kKindTranslatorResidualRewrite;
  case rocjitsu::DiagnosticKind::InternalError:
    return kKindTranslationException;
  }
  return "unknown";
}

void invoke_diagnostic_callback(rocjitsu::Gfx1250B0ToA0DiagnosticCallback callback,
                                const rocjitsu::Gfx1250B0ToA0DiagnosticView &diagnostic,
                                void *user_data) noexcept {
  if (callback == nullptr)
    return;
  callback(&diagnostic, user_data);
}

void emit_diagnostic(rocjitsu::Gfx1250B0ToA0DiagnosticCallback callback, void *user_data,
                     const char *severity, const char *kind, const char *message) noexcept {
  if (callback == nullptr)
    return;
  const rocjitsu::Gfx1250B0ToA0DiagnosticView view{
      severity, kind, 0, 0, "", message, 0,
  };
  invoke_diagnostic_callback(callback, view, user_data);
}

void emit_diagnostics_impl(
    rocjitsu::Gfx1250B0ToA0DiagnosticCallback callback, void *user_data,
    const std::vector<rocjitsu::TranslationDiagnostic> &diagnostics) noexcept {
  if (callback == nullptr)
    return;
  for (const rocjitsu::TranslationDiagnostic &diagnostic : diagnostics) {
    const rocjitsu::Gfx1250B0ToA0DiagnosticView view{
        diagnostic_severity_name(diagnostic.severity),
        diagnostic_kind_name(diagnostic.kind),
        diagnostic.guest_offset.has_value(),
        diagnostic.guest_offset.value_or(0),
        diagnostic.mnemonic.c_str(),
        diagnostic.message.c_str(),
        0,
    };
    invoke_diagnostic_callback(callback, view, user_data);
    for (const std::string &item : diagnostic.required_work) {
      const rocjitsu::Gfx1250B0ToA0DiagnosticView required{
          view.severity, view.kind, view.has_guest_offset, view.guest_offset, view.mnemonic,
          item.c_str(),  1,
      };
      invoke_diagnostic_callback(callback, required, user_data);
    }
  }
}

} // namespace

void rocjitsu::emit_gfx1250_b0_to_a0_diagnostics(
    rocjitsu::Gfx1250B0ToA0DiagnosticCallback callback, void *user_data,
    const std::vector<TranslationDiagnostic> &diagnostics) noexcept {
  emit_diagnostics_impl(callback, user_data, diagnostics);
}

rj_status_t rocjitsu::translate_gfx1250_b0_to_a0(
    std::span<const uint8_t> source_elf, std::vector<uint8_t> &translated_elf,
    Gfx1250B0ToA0TranslationInfo &info, Gfx1250B0ToA0DiagnosticCallback diagnostic_callback,
    void *user_data) noexcept {
  translated_elf.clear();
  info = {};
  if (source_elf.empty()) {
    emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindApiInvalidArgument,
                    "translation received an empty code object");
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  }

  info.source_code_object_id = stable_code_object_id(source_elf.data(), source_elf.size());
  AmdGpuCodeObject source(source_elf.data(), source_elf.size());
  if (!source.is_valid() || source.target_id() != ROCJITSU_CODE_TARGET_GFX1250) {
    emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindInputInvalidCodeObject,
                    "source is not a valid gfx1250 AMDGPU code object");
    return ROCJITSU_STATUS_INVALID_CODE_OBJECT;
  }

  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  BinaryTranslator translator(IsaFamily::Cdna5, IsaFamily::Cdna5, 0, options);
  translator.set_trace_callback([&](const TranslationTraceEvent &trace) {
    if (trace.changed)
      ++info.changed_instruction_count;
  });
  TranslatedCodeObject result = translator.translate(source);
  // Successful translations can still report unsupported pass-through behavior.
  // Keep those warnings and every required-work item visible to hook diagnostics.
  emit_gfx1250_b0_to_a0_diagnostics(diagnostic_callback, user_data, result.diagnostics);
  // Internal failures describe this attempt; rejected input can be cached.
  const rj_status_t status = gfx1250_b0_to_a0_translation_status(
      result.diagnostics, !result.elf_bytes.empty(), result.dispatchable());
  if (status != ROCJITSU_STATUS_SUCCESS) {
    if (result.diagnostics.empty())
      emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindResultNotDispatchable,
                      "translation produced no dispatchable code object");
    return status;
  }

  translated_elf = std::move(result.elf_bytes);
  return ROCJITSU_STATUS_SUCCESS;
}
