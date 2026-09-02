// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_inline_shadow_report_analyzer.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_record_replay_report_analyzer.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_sampled_report_analyzer.h"

#include <variant>

namespace rocjitsu::consan_hook {

using AutoMoiModeAnalysis = std::variant<AutoMoiRecordReplayAnalysis, AutoMoiInlineShadowAnalysis,
                                         AutoMoiSampledConflictAnalysis>;

/// One mode-selected analysis and its complete summary projection. The common
/// analyzer is a composition point only; each alternative owns its algorithm
/// and summary fields in its mode-named component.
struct AutoMoiReportAnalysis {
  AutoMoiReportSummary summary;
  AutoMoiModeAnalysis mode;
};

[[nodiscard]] AutoMoiReportAnalysis analyze_auto_moi_report(const AutoMoiReportPipelineInput &input,
                                                            const AutoMoiDecodedReport &decoded);

[[nodiscard]] bool auto_moi_analysis_matches_engine(const AutoMoiModeAnalysis &analysis,
                                                    ConSanMoiEngine engine);

} // namespace rocjitsu::consan_hook
