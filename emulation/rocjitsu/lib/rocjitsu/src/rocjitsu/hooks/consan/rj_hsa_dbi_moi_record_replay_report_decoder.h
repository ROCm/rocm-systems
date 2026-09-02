// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_pipeline.h"

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu::consan_hook {

struct CompactRecordReplayAccessRecords {
  uint32_t committed_record_count = 0;
  std::vector<ConSanMoiAccessRecord> replay_records;
};

[[nodiscard]] CompactRecordReplayAccessRecords
compact_record_replay_access_records(std::span<const ConSanMoiAccessRecord> records,
                                     uint32_t publication_hint);

struct AutoMoiRecordReplayDecodedReport {
  std::vector<ConSanMoiAccessRecord> access_records;
};

} // namespace rocjitsu::consan_hook
