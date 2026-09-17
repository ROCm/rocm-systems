// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_report_pipeline.h"

#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_analyzer.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_renderer.h"

namespace rocjitsu::consan::hook {

ReportPipelineResult process_report(const ReportPipelineInput &input,
                                    const ReportSnapshot &snapshot, ReportSummary summary) {
  if (const auto *metadata = input.static_metadata
                                 ? (*input.static_metadata ? &**input.static_metadata : nullptr)
                                 : nullptr;
      metadata && metadata->malformed) {
    ++summary.static_mapping_malformed_count;
  }
  const DecodedReport decoded = decode_report(input, snapshot, summary);
  summary = decoded.summary;

  std::optional<ReportAnalysis> analysis;
  if (decoded.complete()) {
    analysis = analyze_report(input, decoded);
    summary = analysis->summary;
  }

  const std::vector<ReportDiagnostic> diagnostics =
      render_report({input, decoded, summary, analysis ? &analysis->conflicts : nullptr});
  for (const ReportDiagnostic &diagnostic : diagnostics)
    log_message(kLogInfo, "%s", diagnostic.text.c_str());
  const auto *conflicts = analysis ? &analysis->conflicts : nullptr;
  return {.summary = summary,
          .complete = decoded.complete(),
          .conflict_example_count =
              conflicts ? static_cast<uint32_t>(conflicts->examples.size()) : 0u};
}

ReportSummary summarize_report(const ReportPipelineInput &input, const ReportSnapshot &snapshot,
                               ReportSummary summary) {
  return process_report(input, snapshot, summary).summary;
}

} // namespace rocjitsu::consan::hook
