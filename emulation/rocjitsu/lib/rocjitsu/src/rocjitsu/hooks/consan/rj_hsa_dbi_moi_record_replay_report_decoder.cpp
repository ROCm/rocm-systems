// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_record_replay_report_decoder.h"

#include <algorithm>

namespace rocjitsu::consan_hook {

CompactRecordReplayAccessRecords
compact_record_replay_access_records(std::span<const ConSanMoiAccessRecord> records,
                                     uint32_t publication_hint) {
  CompactRecordReplayAccessRecords result;
  result.replay_records.reserve(std::min<size_t>(records.size(), publication_hint));
  for (const ConSanMoiAccessRecord &record : records) {
    if (record.access_kind != static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty))
      ++result.committed_record_count;
    if (!consan_moi_access_record_is_unpublished(record))
      result.replay_records.push_back(record);
  }
  return result;
}

} // namespace rocjitsu::consan_hook
