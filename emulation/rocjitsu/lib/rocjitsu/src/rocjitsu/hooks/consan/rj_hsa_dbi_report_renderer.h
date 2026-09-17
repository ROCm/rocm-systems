// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_conflict_analysis.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_rendering.h"

#include <vector>

namespace rocjitsu::consan::hook {

/// Fully typed input to diagnostic rendering. The renderer never observes HSA
/// handles or raw report bytes and therefore cannot reinterpret evidence.
struct ReportRenderInput {
  const ReportPipelineInput &pipeline;
  const DecodedReport &decoded;
  const ReportSummary &summary;
  const ConflictAnalysis *conflict_analysis = nullptr;
};

[[nodiscard]] std::vector<ReportDiagnostic> render_report(const ReportRenderInput &input);

} // namespace rocjitsu::consan::hook
