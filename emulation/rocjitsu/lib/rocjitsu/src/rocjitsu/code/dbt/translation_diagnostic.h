// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translation_diagnostic.h
/// @brief Structured diagnostics reported by the DBT pipeline.

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief Severity for a translation diagnostic.
enum class DiagnosticSeverity {
  Warning,
  Error,
};

/// @brief Broad subsystem or failure class for a translation diagnostic.
enum class DiagnosticKind {
  UnsupportedGuestArch,
  KernelDescriptor,
  Legalization,
  ExpandMissing,
  ExpandFailed,
  DataOnly,
  NothingToTranslate,
  ResourceLimit,
  KernelSkipped,
  ResidualRewrite,
};

/// @brief One user/developer-facing DBT diagnostic.
///
/// @details Translation diagnostics use @c guest_offset and @c mnemonic to point
/// at the original guest instruction. Final-output verification instead uses
/// @c output_offset. Whole-image failures such as descriptor translation can
/// leave both offsets empty. @c required_work is intentionally a short checklist
/// for EXPAND failures so missing lowerings document the next implementation
/// steps instead of only reporting that translation failed.
struct TranslationDiagnostic {
  DiagnosticSeverity severity = DiagnosticSeverity::Warning;
  DiagnosticKind kind = DiagnosticKind::Legalization;
  std::optional<uint64_t> guest_offset;
  std::optional<uint64_t> output_offset;
  std::string mnemonic;
  std::string message;
  std::vector<std::string> required_work;
};

[[nodiscard]] inline bool
has_error_diagnostic(const std::vector<TranslationDiagnostic> &diagnostics) {
  return std::ranges::any_of(diagnostics, [](const TranslationDiagnostic &diagnostic) {
    return diagnostic.severity == DiagnosticSeverity::Error;
  });
}

[[nodiscard]] inline bool has_diagnostic_kind(const std::vector<TranslationDiagnostic> &diagnostics,
                                              DiagnosticKind kind) {
  return std::ranges::any_of(diagnostics, [kind](const TranslationDiagnostic &diagnostic) {
    return diagnostic.kind == kind;
  });
}

/// @brief True if any kernel was replaced by a non-dispatchable no-op stub.
///
/// @details skip_failed_kernels reports a KernelSkipped *warning* (not an error),
/// so a code object with a skipped kernel still passes has_error_diagnostic. The
/// stub contains only `s_endpgm`, so it completes normally without producing the
/// kernel's outputs. Dispatching it would therefore silently produce wrong results. Any
/// consumer that emits or dispatches the translated artifact must treat this as
/// non-dispatchable, matching the HSA hook which refuses such a load.
[[nodiscard]] inline bool
has_skipped_kernel(const std::vector<TranslationDiagnostic> &diagnostics) {
  return has_diagnostic_kind(diagnostics, DiagnosticKind::KernelSkipped);
}

} // namespace rocjitsu
