// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_report_analyzer.h"

#include <utility>

namespace rocjitsu::consan::hook {

ReportAnalysis analyze_report(const ReportPipelineInput &input, const DecodedReport &decoded) {
  ReportSummary summary = decoded.summary;
  auto analysis =
      analyze_conflicts(decoded.records.evidence, decoded.records.synchronization_evidence_complete,
                        input.conflict_example_limit, input.allow_uniform_lds_stores);
  accumulate_analysis(summary, analysis);
  return {summary, std::move(analysis)};
}

} // namespace rocjitsu::consan::hook
