[[nodiscard]] constexpr size_t consan_moi_report_buffer_min_bytes(
    uint32_t access_record_capacity, uint32_t diagnostic_capacity,
    uint32_t exact_shadow_entry_capacity, uint32_t sampled_watchpoint_capacity,
    uint32_t barrier_record_capacity = 0, uint32_t atomic_record_capacity = 0,
    uint32_t fence_record_capacity = 0) {
  return sizeof(ConSanMoiReportHeader) +
         static_cast<size_t>(access_record_capacity) * sizeof(ConSanMoiAccessRecord) +
         static_cast<size_t>(barrier_record_capacity) * sizeof(ConSanMoiBarrierRecord) +
         static_cast<size_t>(atomic_record_capacity) * sizeof(ConSanMoiAtomicRecord) +
         static_cast<size_t>(fence_record_capacity) * sizeof(ConSanMoiFenceRecord) +
         static_cast<size_t>(diagnostic_capacity) * sizeof(ConSanMoiDiagnosticRecord) +
         static_cast<size_t>(exact_shadow_entry_capacity) * sizeof(uint64_t) +
         static_cast<size_t>(sampled_watchpoint_capacity) *
             (sizeof(uint64_t) + sizeof(ConSanMoiSampledSyncMetadataPacked) +
              sizeof(ConSanMoiSampledPendingAcquireSlot));
}

[[nodiscard]] constexpr uint32_t consan_moi_clamp_u32_capacity(uint64_t capacity) {
  return capacity > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                         : static_cast<uint32_t>(capacity);
}

[[nodiscard]] constexpr bool
consan_moi_report_layout_matches_header(const ConSanMoiReportHeader &header,
                                        const ConSanMoiReportBufferLayout &layout,
                                        ConSanMoiEngine engine, uint64_t report_buffer_size) {
  const uint32_t expected_layout_flags =
      engine == ConSanMoiEngine::InlineShadow ? kConSanMoiReportKnownLayoutFlags : 0u;
  return consan_moi_report_header_is_current(header) && layout.valid &&
         layout.required_bytes <= report_buffer_size &&
         header.engine == static_cast<uint32_t>(engine) &&
         header.layout_flags == expected_layout_flags &&
         header.record_replay_dispatch_token_capacity ==
             layout.record_replay_dispatch_token_capacity &&
         header.access_record_capacity == layout.access_record_capacity &&
         header.barrier_record_capacity == layout.barrier_record_capacity &&
         header.atomic_record_capacity == layout.atomic_record_capacity &&
         header.fence_record_capacity == layout.fence_record_capacity &&
         header.diagnostic_capacity == layout.diagnostic_capacity &&
         header.exact_shadow_entry_capacity == layout.exact_shadow_entry_capacity &&
         header.inline_atomic_release_capacity == layout.inline_atomic_release_capacity &&
         header.inline_acquired_epoch_token_capacity ==
             layout.inline_acquired_epoch_token_capacity &&
         header.inline_causal_snapshot_capacity == layout.inline_causal_snapshot_capacity &&
         header.sampled_watchpoint_capacity == layout.sampled_watchpoint_capacity &&
         header.sampled_causal_window_capacity == layout.sampled_causal_window_capacity &&
         header.sampled_sync_metadata_capacity == layout.sampled_sync_metadata_capacity &&
         header.sampled_pending_acquire_capacity == layout.sampled_pending_acquire_capacity;
}
