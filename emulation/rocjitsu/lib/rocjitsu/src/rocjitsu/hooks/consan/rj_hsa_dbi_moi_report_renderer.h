// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_analyzer.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_rendering.h"

#include <vector>

namespace rocjitsu::consan_hook {

/// Fully typed input to diagnostic rendering. The renderer never observes HSA
/// handles or raw report bytes and therefore cannot reinterpret evidence.
struct AutoMoiReportRenderInput {
  const AutoMoiReportPipelineInput &pipeline;
  const AutoMoiDecodedReport &decoded;
  const AutoMoiReportSummary &summary;
  const AutoMoiModeAnalysis *mode_analysis = nullptr;
};

[[nodiscard]] std::vector<AutoMoiReportDiagnostic>
render_auto_moi_report(const AutoMoiReportRenderInput &input);

} // namespace rocjitsu::consan_hook
