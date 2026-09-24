// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_report_pipeline.h"

#include "rocjitsu/hooks/consan/rj_hsa_dbi_conflict_analysis.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_renderer.h"

namespace rocjitsu::consan::hook {

ReportPipelineResult process_report(const ReportPipelineInput &input,
                                    const ReportSnapshot &snapshot, ReportSummary summary) {
  if (const auto *metadata = input.static_metadata; metadata && metadata->malformed) {
    ++summary.static_mapping_malformed_count;
  }
  const DecodedReport decoded = decode_report(input, snapshot, summary);
  summary = decoded.summary;

  std::optional<ConflictAnalysis> analysis;
  if (decoded.complete()) {
    analysis = analyze_conflicts(
        decoded.records.evidence, decoded.records.synchronization_evidence_complete,
        input.conflict_example_limit, input.allow_uniform_lds_stores,
        decoded.publications.status == PublicationDecodeStatus::Disabled ? nullptr
                                                                         : &decoded.publications);
    accumulate_analysis(summary, *analysis);
  }

  const std::vector<ReportDiagnostic> diagnostics =
      render_report({input, decoded, summary, analysis ? &*analysis : nullptr});
  for (const ReportDiagnostic &diagnostic : diagnostics)
    log_message(kLogInfo, "%s", diagnostic.text.c_str());
  const auto *conflicts = analysis ? &*analysis : nullptr;
  return {.summary = summary,
          .complete = decoded.complete(),
          .conflict_example_count =
              conflicts ? static_cast<uint32_t>(conflicts->examples.size()) : 0u};
}

} // namespace rocjitsu::consan::hook
