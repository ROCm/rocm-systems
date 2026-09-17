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

[[nodiscard]] std::string format_report_text(const char *format, ...);

} // namespace rocjitsu::consan::hook
