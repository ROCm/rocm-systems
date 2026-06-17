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
  ResourceLimit,
};

/// @brief One user/developer-facing DBT diagnostic.
///
/// @details Instruction diagnostics use @c guest_offset and @c mnemonic to point
/// at the original guest instruction. Whole-image failures such as descriptor
/// translation can leave those fields empty. @c required_work is intentionally a
/// short checklist for EXPAND failures so missing lowerings document the next
/// implementation steps instead of only reporting that translation failed.
struct TranslationDiagnostic {
  DiagnosticSeverity severity = DiagnosticSeverity::Warning;
  DiagnosticKind kind = DiagnosticKind::Legalization;
  std::optional<uint64_t> guest_offset;
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

} // namespace rocjitsu
