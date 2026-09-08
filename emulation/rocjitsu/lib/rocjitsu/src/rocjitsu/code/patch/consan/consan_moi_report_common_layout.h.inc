struct alignas(8) ConSanMoiDiagnosticRecord {
  uint32_t kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::None);
  uint32_t backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
  // Report generation for ordinary records; native dispatch identity for
  // InlineShadow atomic-tracking diagnostics.
  uint64_t generation = 0;
  uint32_t epoch = 0;
  // Epoch of the prior exact-shadow access. InlineShadow atomic-tracking
  // diagnostics use this with the token's consumer segment epoch to prove
  // that an acquire preceded that specific access.
  uint32_t first_epoch = 0;
  uint32_t first_owner_id = 0;
  uint32_t second_owner_id = 0;
  // Record/Replay host diagnostics retain the current event index here so
  // byte-range provenance can be joined to the exact dynamic record.
  // InlineShadow atomic-tracking diagnostics instead retain the persistent
  // workgroup key so teardown can qualify a token publication that completed
  // after the device-side diagnostic snapshot.
  uint32_t reserved = 0;
  uint64_t first_lane_mask = 0;
  uint64_t second_lane_mask = 0;
  uint32_t first_instruction_offset = 0;
  uint32_t second_instruction_offset = 0;
  uint32_t first_lds_byte_offset = 0;
  uint32_t first_lds_byte_count = 0;
  uint32_t second_lds_byte_offset = 0;
  uint32_t second_lds_byte_count = 0;
  uint32_t first_access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
  uint32_t second_access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
};

/// Creates a production header from the exact heterogeneous layout. Focused
/// tests that deliberately construct partial headers own their convenience
/// builder outside the production contract.
[[nodiscard]] constexpr ConSanMoiReportHeader
make_consan_moi_report_header_for_layout(uint64_t generation, uint64_t dispatch_id,
                                         const ConSanMoiReportBufferLayout &layout) {
  ConSanMoiReportHeader header;
  header.header_size = sizeof(ConSanMoiReportHeader);
  header.generation = generation;
  header.dispatch_id = dispatch_id;
  header.engine = static_cast<uint32_t>(layout.engine);
  header.record_replay_dispatch_token_capacity = layout.record_replay_dispatch_token_capacity;
  header.access_record_capacity = layout.access_record_capacity;
  header.barrier_record_capacity = layout.barrier_record_capacity;
  header.atomic_record_capacity = layout.atomic_record_capacity;
  header.fence_record_capacity = layout.fence_record_capacity;
  header.diagnostic_capacity = layout.diagnostic_capacity;
  header.exact_shadow_entry_capacity = layout.exact_shadow_entry_capacity;
  header.inline_atomic_release_capacity = layout.inline_atomic_release_capacity;
  header.inline_acquired_epoch_token_capacity = layout.inline_acquired_epoch_token_capacity;
  header.inline_causal_snapshot_capacity = layout.inline_causal_snapshot_capacity;
  header.sampled_watchpoint_capacity = layout.sampled_watchpoint_capacity;
  header.sampled_causal_window_capacity = layout.sampled_causal_window_capacity;
  header.sampled_sync_metadata_capacity = layout.sampled_sync_metadata_capacity;
  header.sampled_pending_acquire_capacity = layout.sampled_pending_acquire_capacity;
  header.layout_flags = layout.layout_flags;
  return header;
}
