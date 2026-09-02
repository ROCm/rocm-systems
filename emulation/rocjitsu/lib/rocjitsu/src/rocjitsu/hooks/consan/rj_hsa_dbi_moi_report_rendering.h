// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::consan_hook {

enum class AutoMoiReportDiagnosticKind : uint8_t {
  Failure,
  Evidence,
  Summary,
  Detail,
};

struct AutoMoiReportDiagnostic {
  AutoMoiReportDiagnosticKind kind = AutoMoiReportDiagnosticKind::Detail;
  std::string text;
};

/// Mode-owned rendering projected back to the common renderer. The summary
/// fragment contains only active-mode fields; evidence precedes the common
/// summary and detail follows it.
struct AutoMoiModeRendering {
  uint32_t effective_diagnostic_count = 0;
  std::string summary_fields;
  std::vector<AutoMoiReportDiagnostic> evidence;
  std::vector<AutoMoiReportDiagnostic> details;
};

inline constexpr uint32_t kAutoMoiReportDetailLimit = 4;

[[nodiscard]] constexpr uint32_t auto_moi_report_detail_count(uint32_t visible_count) {
  return visible_count < kAutoMoiReportDetailLimit ? visible_count : kAutoMoiReportDetailLimit;
}

[[nodiscard]] std::string format_auto_moi_report_text(const char *format, ...);

} // namespace rocjitsu::consan_hook
