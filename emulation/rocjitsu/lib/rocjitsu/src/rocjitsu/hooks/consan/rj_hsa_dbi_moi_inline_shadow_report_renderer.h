// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_inline_shadow_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_rendering.h"

namespace rocjitsu::consan_hook {

[[nodiscard]] AutoMoiModeRendering render_auto_moi_inline_shadow_report(
    const AutoMoiReportPipelineInput &input, const ConSanMoiReportHeader &header,
    const AutoMoiReportSummary &summary, const AutoMoiInlineShadowDecodedReport &mode);

} // namespace rocjitsu::consan_hook
