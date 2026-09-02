// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_record_replay_report_analyzer.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_record_replay_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_rendering.h"

namespace rocjitsu::consan_hook {

[[nodiscard]] AutoMoiModeRendering render_auto_moi_record_replay_report(
    const AutoMoiReportPipelineInput &input, const ConSanMoiReportHeader &header,
    const AutoMoiReportSummary &summary, const AutoMoiRecordReplayDecodedReport &mode,
    const AutoMoiRecordReplayAnalysis &analysis);

} // namespace rocjitsu::consan_hook
