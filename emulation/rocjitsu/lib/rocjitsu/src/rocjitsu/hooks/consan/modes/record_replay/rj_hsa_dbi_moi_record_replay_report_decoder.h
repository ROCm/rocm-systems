// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_pipeline.h"

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu::consan_hook {

struct AutoMoiRecordReplayDecodedReport {
  std::vector<ConSanMoiAccessRecord> access_records;
};

[[nodiscard]] AutoMoiRecordReplayDecodedReport
decode_auto_moi_record_replay_report(std::span<const ConSanMoiAccessRecord> records,
                                     uint32_t publication_hint, AutoMoiReportSummary &summary);

} // namespace rocjitsu::consan_hook
