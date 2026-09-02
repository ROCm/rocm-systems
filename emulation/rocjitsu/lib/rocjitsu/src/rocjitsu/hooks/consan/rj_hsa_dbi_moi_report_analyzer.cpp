// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_analyzer.h"

#include <span>
#include <type_traits>
#include <utility>

namespace rocjitsu::consan_hook {

AutoMoiReportAnalysis analyze_auto_moi_report(const AutoMoiReportPipelineInput &input,
                                              const AutoMoiDecodedReport &decoded) {
  return std::visit(
      [&](const auto &decoded_mode) -> AutoMoiReportAnalysis {
        using DecodedMode = std::remove_cvref_t<decltype(decoded_mode)>;
        AutoMoiReportSummary summary = decoded.summary;
        if constexpr (std::is_same_v<DecodedMode, AutoMoiRecordReplayDecodedReport>) {
          const auto *metadata =
              input.static_metadata
                  ? std::get_if<AutoMoiRecordReplayStaticMetadata>(input.static_metadata)
                  : nullptr;
          AutoMoiRecordReplayAnalysis analysis = analyze_auto_moi_record_replay(
              decoded.header, decoded_mode.access_records, decoded.barrier_records,
              decoded.atomic_records, decoded.fence_records,
              metadata ? metadata->mappings : std::span<const AutoMoiRecordReplayStaticMapping>{},
              metadata && metadata->malformed,
              input.layout.record_replay_logical_access_range_count,
              input.layout.record_replay_address_group_headroom);
          accumulate_auto_moi_record_replay_analysis(summary, analysis);
          return {summary, std::move(analysis)};
        } else if constexpr (std::is_same_v<DecodedMode, AutoMoiInlineShadowDecodedReport>) {
          return {summary, AutoMoiInlineShadowAnalysis{}};
        } else {
          AutoMoiSampledConflictAnalysis analysis = analyze_auto_moi_sampled_conflicts(
              decoded_mode.evidence, decoded_mode.synchronization_evidence_complete);
          accumulate_auto_moi_sampled_analysis(summary, analysis);
          return {summary, std::move(analysis)};
        }
      },
      decoded.mode);
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
