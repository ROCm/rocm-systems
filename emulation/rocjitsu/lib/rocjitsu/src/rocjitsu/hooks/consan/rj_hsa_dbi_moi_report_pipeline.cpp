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

  AutoMoiSampledConflictAnalysis sampled_analysis;
  AutoMoiRecordReplayAnalysis record_replay_analysis;
  const AutoMoiSampledConflictAnalysis *sampled_analysis_ptr = nullptr;
  const AutoMoiRecordReplayAnalysis *record_replay_analysis_ptr = nullptr;
  if (decoded.complete()) {
    const auto *record_replay_metadata =
        input.static_metadata
            ? std::get_if<AutoMoiRecordReplayStaticMetadata>(input.static_metadata)
            : nullptr;
    sampled_analysis = analyze_auto_moi_sampled_conflicts(
        decoded.sampled, decoded.sampled_synchronization_evidence_complete);
    record_replay_analysis = analyze_auto_moi_record_replay(
        decoded.header, decoded.engine, decoded.replay_access_records, decoded.barrier_records,
        decoded.atomic_records, decoded.fence_records,
        record_replay_metadata ? record_replay_metadata->mappings
                               : std::span<const AutoMoiRecordReplayStaticMapping>{},
        record_replay_metadata && record_replay_metadata->malformed,
        input.layout.record_replay_logical_access_range_count,
        input.layout.record_replay_address_group_headroom);
    const RecordReplayPressureTelemetry &pressure = record_replay_analysis.pressure;
    summary.record_replay_bank_saturation_count =
        record_replay_bank_saturation_count(decoded.header, decoded.engine);
    summary.record_replay_invalid_site_token_count = pressure.invalid_site_token_count;
    summary.replay_conflict_count = record_replay_analysis.effective_conflict ? 1u : 0u;
    summary.replay_diagnostic_count = record_replay_analysis.effective_diagnostic_count;
    summary.replay_dropped_access_count = record_replay_analysis.replay.dropped_access_count;
    summary.replay_dropped_barrier_count = record_replay_analysis.replay.dropped_barrier_count;
    summary.replay_unsupported_access_count =
        record_replay_analysis.replay.unsupported_access_count;
    summary.replay_unsupported_atomic_count =
        record_replay_analysis.replay.unsupported_atomic_count;
    summary.replay_unsupported_fence_count = record_replay_analysis.replay.unsupported_fence_count;
    summary.replay_metadata_full_count = record_replay_analysis.replay.metadata_full ? 1u : 0u;
    summary.replay_diagnostic_capacity_exhausted_count =
        record_replay_analysis.replay.diagnostic_capacity_exhausted ? 1u : 0u;
    summary.sampled_conflict_count = sampled_analysis.conflict_count;
    sampled_analysis_ptr = &sampled_analysis;
    record_replay_analysis_ptr = &record_replay_analysis;
  }

  const std::vector<AutoMoiReportDiagnostic> diagnostics = render_auto_moi_report(
      {input, decoded, summary, sampled_analysis_ptr, record_replay_analysis_ptr});
  for (const AutoMoiReportDiagnostic &diagnostic : diagnostics)
    log_message(kLogInfo, "%s", diagnostic.text.c_str());
  return summary;
}

} // namespace rocjitsu::consan_hook
