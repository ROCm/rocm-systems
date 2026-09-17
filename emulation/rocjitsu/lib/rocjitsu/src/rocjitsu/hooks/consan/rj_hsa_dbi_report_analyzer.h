// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_conflict_analysis.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_decoder.h"

namespace rocjitsu::consan::hook {

/// ConSan conflict analysis and its complete summary projection.
struct ReportAnalysis {
  ReportSummary summary;
  ConflictAnalysis conflicts;
};

[[nodiscard]] ReportAnalysis analyze_report(const ReportPipelineInput &input,
                                            const DecodedReport &decoded);

} // namespace rocjitsu::consan::hook
