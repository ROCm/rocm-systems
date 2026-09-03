// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Derive the explicit direct-Sampled report geometry.
[[nodiscard]] constexpr ConSanMoiReportBufferLayout
consan_moi_direct_sampled_report_buffer_layout_for_bytes(uint64_t report_buffer_size) {
  ConSanMoiReportBufferLayout layout;
  layout.engine = ConSanMoiEngine::Sampled;
  if (report_buffer_size < sizeof(ConSanMoiReportHeader) ||
      report_buffer_size > std::numeric_limits<size_t>::max())
    return layout;

  const uint64_t payload_bytes = report_buffer_size - sizeof(ConSanMoiReportHeader);
  const uint64_t paired_capacity =
      payload_bytes /
      (sizeof(ConSanMoiSampledCausalWindow) + sizeof(uint64_t) +
       sizeof(ConSanMoiSampledSyncMetadataPacked) + sizeof(ConSanMoiSampledPendingAcquireSlot));
  layout.sampled_watchpoint_capacity = consan_moi_clamp_u32_capacity(paired_capacity);
  layout.sampled_causal_window_capacity = layout.sampled_watchpoint_capacity;
  layout.sampled_sync_metadata_capacity = layout.sampled_watchpoint_capacity;
  layout.sampled_pending_acquire_capacity = layout.sampled_watchpoint_capacity;
  layout.barrier_records_offset = layout.access_records_offset;
  layout.atomic_records_offset = layout.access_records_offset;
  layout.diagnostic_records_offset = layout.access_records_offset;
  layout.exact_shadow_entries_offset = layout.access_records_offset;
  layout.inline_atomic_release_slots_offset = layout.access_records_offset;
  layout.inline_acquired_epoch_token_slots_offset = layout.access_records_offset;
  layout.inline_causal_snapshots_offset = layout.access_records_offset;
  layout.sampled_causal_windows_offset = layout.access_records_offset;
  layout.sampled_watchpoints_offset = layout.sampled_causal_windows_offset +
                                      static_cast<size_t>(layout.sampled_causal_window_capacity) *
                                          sizeof(ConSanMoiSampledCausalWindow);
  layout.sampled_sync_metadata_offset =
      layout.sampled_watchpoints_offset +
      static_cast<size_t>(layout.sampled_watchpoint_capacity) * sizeof(uint64_t);
  layout.sampled_pending_acquires_offset =
      layout.sampled_sync_metadata_offset +
      static_cast<size_t>(layout.sampled_sync_metadata_capacity) *
          sizeof(ConSanMoiSampledSyncMetadataPacked);
  layout.required_bytes = layout.sampled_pending_acquires_offset +
                          static_cast<size_t>(layout.sampled_pending_acquire_capacity) *
                              sizeof(ConSanMoiSampledPendingAcquireSlot);
  layout.valid = layout.required_bytes <= report_buffer_size;
  return layout;
}
