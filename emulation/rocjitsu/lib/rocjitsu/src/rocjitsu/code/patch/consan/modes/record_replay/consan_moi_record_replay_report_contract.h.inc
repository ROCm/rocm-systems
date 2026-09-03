// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Resolve the wave-wide outcome represented by a Record/Replay atomic event.
/// Mixed-lane compare-exchange outcomes cannot be represented by this ABI.
[[nodiscard]] std::optional<ConSanMoiAtomicOutcome>
consan_moi_record_replay_resolved_atomic_outcome(const ConSanMoiRecordReplayAtomicEvent &record);

/// Recognize untouched fixed-capacity report slots before replay.
[[nodiscard]] bool consan_moi_record_replay_atomic_event_is_unpublished(
    const ConSanMoiRecordReplayAtomicEvent &record);
[[nodiscard]] bool
consan_moi_record_replay_fence_event_is_unpublished(const ConSanMoiRecordReplayFenceEvent &record);

/// Derive the explicit fixed-capacity Record/Replay report geometry.
[[nodiscard]] constexpr ConSanMoiReportBufferLayout
consan_moi_report_buffer_layout_for_bytes(uint64_t report_buffer_size, bool include_barriers,
                                          bool include_atomics = false,
                                          bool include_fences = false) {
  ConSanMoiReportBufferLayout layout;
  layout.engine = ConSanMoiEngine::RecordReplay;
  if (report_buffer_size < sizeof(ConSanMoiReportHeader) ||
      report_buffer_size > std::numeric_limits<size_t>::max())
    return layout;

  const uint64_t payload_bytes = report_buffer_size - sizeof(ConSanMoiReportHeader);
  const uint64_t bytes_per_capacity = sizeof(ConSanMoiAccessRecord) +
                                      (include_barriers ? sizeof(ConSanMoiBarrierRecord) : 0u) +
                                      (include_atomics ? sizeof(ConSanMoiAtomicRecord) : 0u) +
                                      (include_fences ? sizeof(ConSanMoiFenceRecord) : 0u);
  const uint32_t capacity = consan_moi_clamp_u32_capacity(payload_bytes / bytes_per_capacity);
  layout.access_record_capacity = capacity;
  layout.barrier_record_capacity = include_barriers ? capacity : 0u;
  layout.atomic_record_capacity = include_atomics ? capacity : 0u;
  layout.fence_record_capacity = include_fences ? capacity : 0u;
  layout.barrier_records_offset =
      layout.access_records_offset +
      static_cast<size_t>(layout.access_record_capacity) * sizeof(ConSanMoiAccessRecord);
  layout.atomic_records_offset =
      layout.barrier_records_offset +
      static_cast<size_t>(layout.barrier_record_capacity) * sizeof(ConSanMoiBarrierRecord);
  layout.fence_records_offset =
      layout.atomic_records_offset +
      static_cast<size_t>(layout.atomic_record_capacity) * sizeof(ConSanMoiAtomicRecord);
  layout.diagnostic_records_offset =
      layout.fence_records_offset +
      static_cast<size_t>(layout.fence_record_capacity) * sizeof(ConSanMoiFenceRecord);
  layout.exact_shadow_entries_offset = layout.diagnostic_records_offset;
  layout.inline_atomic_release_slots_offset = layout.exact_shadow_entries_offset;
  layout.inline_acquired_epoch_token_slots_offset = layout.exact_shadow_entries_offset;
  layout.inline_causal_snapshots_offset = layout.exact_shadow_entries_offset;
  layout.sampled_watchpoints_offset = layout.exact_shadow_entries_offset;
  layout.sampled_sync_metadata_offset = layout.sampled_watchpoints_offset;
  layout.required_bytes = layout.diagnostic_records_offset;
  layout.valid = layout.required_bytes <= report_buffer_size;
  return layout;
}
