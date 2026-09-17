// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_conflict_analysis.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_evidence_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_rendering.h"

namespace rocjitsu::consan::hook {

[[nodiscard]] ConflictRendering render_conflicts(const ReportPipelineInput &input,
                                                 const ReportHeader &header,
                                                 const ReportSummary &summary,
                                                 const DecodedEvidence &mode,
                                                 const ConflictAnalysis &analysis);

} // namespace rocjitsu::consan::hook
