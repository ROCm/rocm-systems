// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_record_replay_report_decoder.h"

#include <algorithm>

namespace rocjitsu::consan_hook {

AutoMoiRecordReplayDecodedReport
decode_auto_moi_record_replay_report(std::span<const ConSanMoiAccessRecord> records,
                                     uint32_t publication_hint, AutoMoiReportSummary &summary) {
  AutoMoiRecordReplayDecodedReport result;
  result.access_records.reserve(std::min<size_t>(records.size(), publication_hint));
  for (const ConSanMoiAccessRecord &record : records) {
    if (record.access_kind != static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty))
      ++summary.visible_access_record_count;
    if (!consan_moi_access_record_is_unpublished(record))
      result.access_records.push_back(record);
  }
  return result;
}

} // namespace rocjitsu::consan_hook
