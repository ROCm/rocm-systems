// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::consan::hook {

enum class ReportDiagnosticKind : uint8_t {
  Failure,
  Evidence,
  Summary,
  Detail,
};

struct ReportDiagnostic {
  ReportDiagnosticKind kind = ReportDiagnosticKind::Detail;
  std::string text;
};

/// Mode-owned rendering projected back to the common renderer. The summary
/// fragment contains only active-mode fields; evidence precedes the common
/// summary and detail follows it.
struct ConflictRendering {
  uint32_t effective_diagnostic_count = 0;
  std::string summary_fields;
  std::vector<ReportDiagnostic> evidence;
  std::vector<ReportDiagnostic> details;
};

inline constexpr uint32_t kReportDetailLimit = 4;

[[nodiscard]] constexpr uint32_t report_detail_count(uint32_t visible_count) {
  return visible_count < kReportDetailLimit ? visible_count : kReportDetailLimit;
}

[[nodiscard]] std::string format_report_text(const char *format, ...);

} // namespace rocjitsu::consan::hook
