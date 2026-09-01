// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

namespace rocjitsu {

[[nodiscard]] ConSanMoiSampledSyncPublishResult
consan_moi_sampled_publish_sync_metadata(std::span<ConSanMoiSampledSyncMetadataPacked> slots,
                                         uint32_t selected_slot,
                                         const ConSanMoiSampledSyncMetadata &metadata);

[[nodiscard]] ConSanMoiSampledPublishResult
consan_moi_sampled_publish_access_records(const ConSanMoiReportHeader &header,
                                          std::span<const ConSanMoiAccessRecord> access_records,
                                          std::span<uint64_t> sampled_watchpoint_entries);

[[nodiscard]] ConSanMoiSampledPublishResult
consan_moi_sampled_publish_causal_windows(const ConSanMoiReportHeader &header,
                                          std::span<const ConSanMoiAccessRecord> access_records,
                                          uint32_t selection_stride, uint32_t selection_offset,
                                          std::span<uint64_t> sampled_watchpoint_entries,
                                          std::span<ConSanMoiSampledCausalWindow> causal_windows);

[[nodiscard]] ConSanMoiSampledReplayResult
consan_moi_sampled_replay_entries(ConSanMoiReportHeader &header,
                                  std::span<const uint64_t> sampled_watchpoint_entries,
                                  std::span<ConSanMoiDiagnosticRecord> diagnostic_records);

[[nodiscard]] ConSanMoiSampledReplayResult
consan_moi_sampled_replay_snapshots(ConSanMoiReportHeader &header,
                                    std::span<const ConSanMoiSampledSnapshotWords> snapshots,
                                    std::span<ConSanMoiDiagnosticRecord> diagnostic_records);

[[nodiscard]] ConSanMoiSampledReplayResult consan_moi_sampled_replay_causal_windows(
    ConSanMoiReportHeader &header, std::span<const uint64_t> sampled_watchpoint_entries,
    std::span<const ConSanMoiSampledCausalWindow> causal_windows,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records);

[[nodiscard]] ConSanMoiSampledClaimResult
consan_moi_sampled_begin_causal_claim(std::span<ConSanMoiSampledCausalWindow> windows,
                                      const ConSanMoiSampledCausalKey &key);

[[nodiscard]] bool consan_moi_sampled_commit_causal_claim(ConSanMoiSampledCausalWindow &window,
                                                          const ConSanMoiSampledCausalKey &key,
                                                          uint32_t first_entry,
                                                          uint32_t entry_count);

[[nodiscard]] bool consan_moi_sampled_abort_causal_claim(ConSanMoiSampledCausalWindow &window);

} // namespace rocjitsu
