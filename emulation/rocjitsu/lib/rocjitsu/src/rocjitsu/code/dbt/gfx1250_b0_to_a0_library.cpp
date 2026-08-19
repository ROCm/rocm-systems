// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/code_object_identity.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/gfx1250_b0_to_a0_diagnostics.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr char kSeverityWarning[] = "warning";
constexpr char kSeverityError[] = "error";
constexpr char kKindApiInvalidArgument[] = "api-invalid-argument";
constexpr char kKindInputInvalidCodeObject[] = "input-invalid-code-object";
constexpr char kKindResultNotDispatchable[] = "result-not-dispatchable";
constexpr char kKindOutputAllocationFailed[] = "output-allocation-failed";
constexpr char kKindTranslationOutOfMemory[] = "translation-out-of-memory";
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
  }
  return "unknown";
}

void invoke_diagnostic_callback(rj_gfx1250_b0_to_a0_diagnostic_callback_t callback,
                                const rj_gfx1250_b0_to_a0_diagnostic_t &diagnostic,
                                void *user_data) noexcept {
  if (callback == nullptr)
    return;
  try {
    callback(&diagnostic, user_data);
  } catch (...) {
    // A reporting sink must not let an exception cross the C API boundary or
    // change the translation result it is observing.
  }
}

void emit_diagnostic(rj_gfx1250_b0_to_a0_diagnostic_callback_t callback, void *user_data,
                     const char *severity, const char *kind, const char *message) noexcept {
  if (callback == nullptr)
    return;
  const rj_gfx1250_b0_to_a0_diagnostic_t view{
      severity, kind, 0, 0, "", message, 0,
  };
  invoke_diagnostic_callback(callback, view, user_data);
}

void emit_diagnostics_impl(
    rj_gfx1250_b0_to_a0_diagnostic_callback_t callback, void *user_data,
    const std::vector<rocjitsu::TranslationDiagnostic> &diagnostics) noexcept {
  if (callback == nullptr)
    return;
  for (const rocjitsu::TranslationDiagnostic &diagnostic : diagnostics) {
    const rj_gfx1250_b0_to_a0_diagnostic_t view{
        diagnostic_severity_name(diagnostic.severity),
        diagnostic_kind_name(diagnostic.kind),
        diagnostic.guest_offset.has_value() ? 1 : 0,
        diagnostic.guest_offset.value_or(0),
        diagnostic.mnemonic.c_str(),
        diagnostic.message.c_str(),
        0,
    };
    invoke_diagnostic_callback(callback, view, user_data);
    for (const std::string &item : diagnostic.required_work) {
      const rj_gfx1250_b0_to_a0_diagnostic_t required{
          view.severity, view.kind, view.has_guest_offset, view.guest_offset, view.mnemonic,
          item.c_str(),  1,
      };
      invoke_diagnostic_callback(callback, required, user_data);
    }
  }
}

} // namespace

void rocjitsu::emit_gfx1250_b0_to_a0_diagnostics(
    rj_gfx1250_b0_to_a0_diagnostic_callback_t callback, void *user_data,
    const std::vector<TranslationDiagnostic> &diagnostics) noexcept {
  emit_diagnostics_impl(callback, user_data, diagnostics);
}

rj_status_t
rj_gfx1250_b0_to_a0_translate(const void *source_elf, size_t source_size, uint8_t **translated_elf,
                              size_t *translated_size, rj_gfx1250_b0_to_a0_translation_info_t *info,
                              rj_gfx1250_b0_to_a0_diagnostic_callback_t diagnostic_callback,
                              void *user_data) {
  if (translated_elf)
    *translated_elf = nullptr;
  if (translated_size)
    *translated_size = 0;
  if (info)
    *info = {};

  if (!source_elf || source_size == 0 || !translated_elf || !translated_size || !info) {
    emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindApiInvalidArgument,
                    "translation received an invalid input or output argument");
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  }

  info->source_code_object_id = rocjitsu::stable_code_object_id(source_elf, source_size);

  try {
    const auto *source_bytes = static_cast<const uint8_t *>(source_elf);
    rocjitsu::AmdGpuCodeObject source(source_bytes, source_size);
    if (!source.is_valid() || source.target_id() != ROCJITSU_CODE_TARGET_GFX1250) {
      emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindInputInvalidCodeObject,
                      "source is not a valid gfx1250 AMDGPU code object");
      return ROCJITSU_STATUS_INVALID_CODE_OBJECT;
    }

    rocjitsu::BinaryTranslatorOptions options;
    options.input_revision = rocjitsu::ProcessorRevision::Gfx1250B0;
    options.output_revision = rocjitsu::ProcessorRevision::Gfx1250A0;
    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_CDNA5, 0,
                                          options);
    translator.set_trace_callback([&](const rocjitsu::TranslationTraceEvent &trace) {
      if (trace.changed)
        ++info->changed_instruction_count;
    });
    auto result = translator.translate(source);
    // Diagnostics are reported for every outcome, not only failures. A
    // translation can succeed and still have something to say -- a family passed
    // through unchanged because its A0 handling is not implemented yet is
    // exactly the kind of gap someone triaging a misbehaving kernel needs to
    // see, and it never reaches the caller if reporting is conditional on the
    // object being undispatchable.
    rocjitsu::emit_gfx1250_b0_to_a0_diagnostics(diagnostic_callback, user_data, result.diagnostics);
    // A translation that ran to completion and produced nothing dispatchable is a
    // statement about the input, not about this run: repeating it reaches the same
    // conclusion. Reporting it as such is what lets a caller distinguish it from
    // the environmental failures below and cache the verdict rather than
    // rediscovering it. ROCJITSU_STATUS_ERROR is left to mean the translator threw,
    // which carries no such promise.
    if (result.elf_bytes.empty() || !result.dispatchable()) {
      if (result.diagnostics.empty())
        emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindResultNotDispatchable,
                        "translation produced no dispatchable code object");
      return ROCJITSU_STATUS_INVALID_CODE_OBJECT;
    }

    auto *output = static_cast<uint8_t *>(std::malloc(result.elf_bytes.size()));
    if (!output) {
      emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindOutputAllocationFailed,
                      "could not allocate the translated code object");
      return ROCJITSU_STATUS_OUT_OF_RESOURCES;
    }

    std::memcpy(output, result.elf_bytes.data(), result.elf_bytes.size());
    *translated_elf = output;
    *translated_size = result.elf_bytes.size();
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindTranslationOutOfMemory,
                    "translation ran out of memory");
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (const std::exception &error) {
    emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindTranslationException,
                    error.what());
    return ROCJITSU_STATUS_ERROR;
  } catch (...) {
    emit_diagnostic(diagnostic_callback, user_data, kSeverityError, kKindTranslationException,
                    "translation threw an unknown exception");
    return ROCJITSU_STATUS_ERROR;
  }
}

void rj_gfx1250_b0_to_a0_free(void *translated_elf) { std::free(translated_elf); }
