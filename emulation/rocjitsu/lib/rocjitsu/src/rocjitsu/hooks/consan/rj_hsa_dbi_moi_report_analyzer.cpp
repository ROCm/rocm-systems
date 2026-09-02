// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_analyzer.h"

#include <span>
#include <utility>

namespace rocjitsu::consan_hook {

AutoMoiReportAnalysis analyze_auto_moi_report(const AutoMoiReportPipelineInput &input,
                                              const AutoMoiDecodedReport &decoded) {
  AutoMoiReportSummary summary = decoded.summary;
  switch (decoded.engine) {
  case ConSanMoiEngine::RecordReplay: {
    const auto *metadata =
        input.static_metadata
            ? std::get_if<AutoMoiRecordReplayStaticMetadata>(input.static_metadata)
            : nullptr;
    AutoMoiRecordReplayAnalysis analysis = analyze_auto_moi_record_replay(
        decoded.header, decoded.replay_access_records, decoded.barrier_records,
        decoded.atomic_records, decoded.fence_records,
        metadata ? metadata->mappings : std::span<const AutoMoiRecordReplayStaticMapping>{},
        metadata && metadata->malformed, input.layout.record_replay_logical_access_range_count,
        input.layout.record_replay_address_group_headroom);
    accumulate_auto_moi_record_replay_analysis(summary, analysis);
    return {summary, std::move(analysis)};
  }
  case ConSanMoiEngine::InlineShadow:
    return {summary, AutoMoiInlineShadowAnalysis{}};
  case ConSanMoiEngine::Sampled: {
    AutoMoiSampledConflictAnalysis analysis = analyze_auto_moi_sampled_conflicts(
        decoded.sampled, decoded.sampled_synchronization_evidence_complete);
    accumulate_auto_moi_sampled_analysis(summary, analysis);
    return {summary, std::move(analysis)};
  }
  }
  return {summary, AutoMoiInlineShadowAnalysis{}};
}

bool auto_moi_analysis_matches_engine(const AutoMoiModeAnalysis &analysis, ConSanMoiEngine engine) {
  switch (engine) {
  case ConSanMoiEngine::RecordReplay:
    return std::holds_alternative<AutoMoiRecordReplayAnalysis>(analysis);
  case ConSanMoiEngine::InlineShadow:
    return std::holds_alternative<AutoMoiInlineShadowAnalysis>(analysis);
  case ConSanMoiEngine::Sampled:
    return std::holds_alternative<AutoMoiSampledConflictAnalysis>(analysis);
  }
  return false;
}

} // namespace rocjitsu::consan_hook
