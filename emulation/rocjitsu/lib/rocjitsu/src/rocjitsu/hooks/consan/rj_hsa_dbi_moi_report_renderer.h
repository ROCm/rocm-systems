// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_analyzer.h"

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

/// Fully typed input to diagnostic rendering. The renderer never observes HSA
/// handles or raw report bytes and therefore cannot reinterpret evidence.
struct AutoMoiReportRenderInput {
  const AutoMoiReportPipelineInput &pipeline;
  const AutoMoiDecodedReport &decoded;
  const AutoMoiReportSummary &summary;
  const AutoMoiModeAnalysis *mode_analysis = nullptr;
};

inline constexpr uint32_t kAutoMoiReportDetailLimit = 4;

[[nodiscard]] constexpr uint32_t auto_moi_report_detail_count(uint32_t visible_count) {
  return visible_count < kAutoMoiReportDetailLimit ? visible_count : kAutoMoiReportDetailLimit;
}

[[nodiscard]] std::vector<AutoMoiReportDiagnostic>
render_auto_moi_report(const AutoMoiReportRenderInput &input);

} // namespace rocjitsu::consan_hook
