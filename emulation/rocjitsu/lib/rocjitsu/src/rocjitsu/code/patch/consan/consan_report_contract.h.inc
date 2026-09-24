// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Derive the explicit direct-ConSan report geometry.
[[nodiscard]] constexpr ReportBufferLayout
direct_report_buffer_layout_for_bytes(uint64_t report_buffer_size) {
  ReportBufferLayout layout;
  if (report_buffer_size < sizeof(ReportHeader) ||
      report_buffer_size > std::numeric_limits<size_t>::max())
    return layout;

  const uint64_t payload_bytes = report_buffer_size - sizeof(ReportHeader);
  const uint64_t paired_capacity =
      payload_bytes / (sizeof(CausalWindow) + sizeof(uint64_t) + sizeof(SyncMetadataPacked) +
                       sizeof(PendingAcquireSlot));
  layout.watchpoint_capacity = clamp_u32_capacity(paired_capacity);
  layout.causal_window_capacity = layout.watchpoint_capacity;
  layout.sync_metadata_capacity = layout.watchpoint_capacity;
  layout.pending_acquire_capacity = layout.watchpoint_capacity;
  layout.causal_windows_offset = sizeof(ReportHeader);
  layout.watchpoints_offset =
      layout.causal_windows_offset +
      static_cast<size_t>(layout.causal_window_capacity) * sizeof(CausalWindow);
  layout.sync_metadata_offset = layout.watchpoints_offset +
                                static_cast<size_t>(layout.watchpoint_capacity) * sizeof(uint64_t);
  layout.pending_acquires_offset =
      layout.sync_metadata_offset +
      static_cast<size_t>(layout.sync_metadata_capacity) * sizeof(SyncMetadataPacked);
  layout.required_bytes =
      layout.pending_acquires_offset +
      static_cast<size_t>(layout.pending_acquire_capacity) * sizeof(PendingAcquireSlot);
  layout.publication_events_offset = layout.required_bytes;
  layout.valid = layout.required_bytes <= report_buffer_size;
  return layout;
}
