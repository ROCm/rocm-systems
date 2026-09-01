// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

namespace rocjitsu {

/// Independent host reference for the device-side causal-window selector.
[[nodiscard]] constexpr uint64_t consan_moi_sampled_causal_mix(uint64_t hash, uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  hash ^= hash >> 30u;
  hash *= 0xbf58476d1ce4e5b9ull;
  hash ^= hash >> 27u;
  hash *= 0x94d049bb133111ebull;
  return hash ^ (hash >> 31u);
}

[[nodiscard]] constexpr bool consan_moi_sampled_causal_window_selected(
    uint64_t generation, uint64_t dispatch_id, uint32_t workgroup_x, uint32_t workgroup_y,
    uint32_t workgroup_z, uint32_t epoch, uint32_t stride, uint32_t offset) {
  if (stride == 0 || stride > 1024u || (stride & (stride - 1u)) != 0 || offset >= stride)
    return false;
  uint64_t hash = consan_moi_sampled_causal_mix(0x243f6a8885a308d3ull, generation);
  hash = consan_moi_sampled_causal_mix(hash, dispatch_id);
  hash = consan_moi_sampled_causal_mix(hash, workgroup_x);
  hash = consan_moi_sampled_causal_mix(hash, workgroup_y);
  hash = consan_moi_sampled_causal_mix(hash, workgroup_z);
  hash = consan_moi_sampled_causal_mix(hash, epoch);
  return (hash & (stride - 1u)) == offset;
}

/// Independent host encoder for the complete device-published Sampled
/// watchpoint ABI. The byte-count field primitive remains shared with the
/// production emitter.
[[nodiscard]] constexpr uint64_t
pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind kind, uint32_t owner_id,
                                         uint32_t epoch, uint32_t generation, uint32_t start_byte,
                                         uint32_t byte_count, bool consumed = false) {
  return consan_moi_sampled_watchpoint::valid_mask |
         (consumed ? consan_moi_sampled_watchpoint::consumed_mask : uint64_t{0}) |
         ((static_cast<uint64_t>(kind) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::access_kind_bits))
          << consan_moi_sampled_watchpoint::access_kind_shift) |
         ((static_cast<uint64_t>(owner_id) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::owner_bits))
          << consan_moi_sampled_watchpoint::owner_shift) |
         ((static_cast<uint64_t>(epoch) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::epoch_bits))
          << consan_moi_sampled_watchpoint::epoch_shift) |
         ((static_cast<uint64_t>(generation) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::generation_bits))
          << consan_moi_sampled_watchpoint::generation_shift) |
         ((static_cast<uint64_t>(start_byte) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::start_byte_bits))
          << consan_moi_sampled_watchpoint::start_byte_shift) |
         ((static_cast<uint64_t>(encode_consan_moi_sampled_byte_count(byte_count)) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::count_bits))
          << consan_moi_sampled_watchpoint::count_shift);
}

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
