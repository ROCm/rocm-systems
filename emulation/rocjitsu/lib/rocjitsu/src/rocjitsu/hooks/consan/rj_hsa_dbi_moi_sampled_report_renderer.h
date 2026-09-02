// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_rendering.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_sampled_report_analyzer.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_sampled_report_decoder.h"

namespace rocjitsu::consan_hook {

[[nodiscard]] AutoMoiModeRendering render_auto_moi_sampled_report(
    const AutoMoiReportPipelineInput &input, const ConSanMoiReportHeader &header,
    const AutoMoiReportSummary &summary, const AutoMoiSampledDecodedReport &mode,
    const AutoMoiSampledConflictAnalysis &analysis);

} // namespace rocjitsu::consan_hook
