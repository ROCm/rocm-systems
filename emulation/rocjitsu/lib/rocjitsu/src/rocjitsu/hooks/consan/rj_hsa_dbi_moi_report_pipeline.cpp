// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_pipeline.h"

#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_analyzer.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_renderer.h"

namespace rocjitsu::consan_hook {

AutoMoiReportSummary summarize_auto_moi_report(const AutoMoiReportPipelineInput &input,
                                               const AutoMoiReportSnapshot &snapshot,
                                               AutoMoiReportSummary summary) {
  if (const auto *sampled = input.static_metadata
                                ? std::get_if<AutoMoiSampledStaticMetadata>(input.static_metadata)
                                : nullptr;
      sampled && sampled->malformed) {
    ++summary.sampled_static_mapping_malformed_count;
  }
  if (const auto *inline_compact =
          input.static_metadata
              ? std::get_if<AutoMoiInlineCompactStaticMetadata>(input.static_metadata)
              : nullptr;
      inline_compact && inline_compact->malformed) {
    ++summary.inline_malformed_count;
  }
  const AutoMoiDecodedReport decoded = decode_auto_moi_report(input, snapshot, summary);
  summary = decoded.summary;

  std::optional<AutoMoiReportAnalysis> analysis;
  if (decoded.complete()) {
    analysis = analyze_auto_moi_report(input, decoded);
    summary = analysis->summary;
  }

  const std::vector<AutoMoiReportDiagnostic> diagnostics =
      render_auto_moi_report({input, decoded, summary, analysis ? &analysis->mode : nullptr});
  for (const AutoMoiReportDiagnostic &diagnostic : diagnostics)
    log_message(kLogInfo, "%s", diagnostic.text.c_str());
  return summary;
}

} // namespace rocjitsu::consan_hook
