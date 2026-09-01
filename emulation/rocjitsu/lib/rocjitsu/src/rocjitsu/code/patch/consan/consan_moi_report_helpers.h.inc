/// Resolve the wave-wide outcome represented by a Record/Replay atomic event.
/// Mixed-lane compare-exchange outcomes cannot be represented by this ABI.
[[nodiscard]] std::optional<ConSanMoiAtomicOutcome>
consan_moi_record_replay_resolved_atomic_outcome(const ConSanMoiRecordReplayAtomicEvent &record);

/// Recognize untouched fixed-capacity report slots before replay.
[[nodiscard]] bool consan_moi_record_replay_atomic_event_is_unpublished(
    const ConSanMoiRecordReplayAtomicEvent &record);
[[nodiscard]] bool
consan_moi_record_replay_fence_event_is_unpublished(const ConSanMoiRecordReplayFenceEvent &record);

[[nodiscard]] constexpr ConSanMoiReportHeader make_consan_moi_report_header(
    uint64_t generation, uint64_t dispatch_id, uint32_t access_record_capacity,
    uint32_t diagnostic_capacity, uint32_t exact_shadow_entry_capacity,
    uint32_t sampled_watchpoint_capacity, uint32_t barrier_record_capacity = 0,
    uint32_t atomic_record_capacity = 0, uint32_t inline_atomic_release_capacity = 0,
    uint32_t fence_record_capacity = 0, uint32_t inline_acquired_epoch_token_capacity = 0,
    uint32_t inline_causal_snapshot_capacity = 0,
    ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay) {
  ConSanMoiReportHeader header;
  header.header_size = sizeof(ConSanMoiReportHeader);
  header.generation = generation;
  header.dispatch_id = dispatch_id;
  header.engine = static_cast<uint32_t>(engine);
  header.access_record_capacity = access_record_capacity;
  header.barrier_record_capacity = barrier_record_capacity;
  header.atomic_record_capacity = atomic_record_capacity;
  header.diagnostic_capacity = diagnostic_capacity;
  header.exact_shadow_entry_capacity = exact_shadow_entry_capacity;
  header.sampled_watchpoint_capacity = sampled_watchpoint_capacity;
  header.inline_atomic_release_capacity = inline_atomic_release_capacity;
  header.inline_acquired_epoch_token_capacity = inline_acquired_epoch_token_capacity;
  header.inline_causal_snapshot_capacity = inline_causal_snapshot_capacity;
  header.sampled_causal_window_capacity = sampled_watchpoint_capacity;
  header.sampled_sync_metadata_capacity = sampled_watchpoint_capacity;
  header.sampled_pending_acquire_capacity = sampled_watchpoint_capacity;
  header.fence_record_capacity = fence_record_capacity;
  if (engine == ConSanMoiEngine::InlineShadow)
    header.layout_flags = kConSanMoiReportKnownLayoutFlags;
  return header;
}

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

inline constexpr uint32_t kConSanMoiInlineShadowDefaultDiagnosticCapacity = 4;
inline constexpr uint32_t kConSanMoiInlineShadowDiagnosticHeadroomPerAccess = 64;
inline constexpr uint32_t kConSanMoiInlineShadowConservativeExactShadowEntries = 16u * 1024u;
inline constexpr uint32_t kConSanMoiInlineShadowAtomicReleaseSlotCapacity = 64;
inline constexpr uint32_t kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity = 1024;
// A release publisher holds its direct-mapped object slot while scanning the
// full token table. Leave enough bounded, backed-off polls for that ordinary
// critical section before a competing publisher or consumer reports a
// genuinely stuck odd version.
inline constexpr uint32_t kConSanMoiInlineMetadataPublicationRetryLimit = 65536;
inline constexpr uint32_t kConSanMoiInlineMetadataPublicationSleepDelay = 16;

inline constexpr uint32_t kConSanMoiInlineTokenWorkgroupMultiplier = 0x9e3779b1u;
inline constexpr uint32_t kConSanMoiInlineTokenConsumerMultiplier = 0x85ebca6bu;
inline constexpr uint32_t kConSanMoiInlineTokenProducerMultiplier = 0xc2b2ae35u;
inline constexpr uint32_t kConSanMoiInlineTokenConsumerEpochMultiplier = 0x27d4eb2fu;
inline constexpr uint32_t kConSanMoiInlineTokenReleaseSequenceSalt = 0x9e3779b9u;
inline constexpr uint32_t kConSanMoiInlineTokenAvalancheMultiplier0 = 0x7feb352du;
inline constexpr uint32_t kConSanMoiInlineTokenAvalancheMultiplier1 = 0x846ca68bu;

enum class ConSanMoiInlineAcquiredTokenState : uint8_t {
  Empty,
  Stable,
  Publishing,
  Changed,
  Malformed,
};

struct ConSanMoiInlineAcquiredTokenSnapshot {
  uint32_t version_before = 0;
  ConSanMoiInlineAcquiredEpochTokenSlot payload{};
  uint32_t version_after = 0;
};

struct ConSanMoiInlineAcquiredTokenClassification {
  ConSanMoiInlineAcquiredTokenState state = ConSanMoiInlineAcquiredTokenState::Empty;
  ConSanMoiInlineAcquiredEpochTokenSlot token{};
};

/// Classifies one complete token read. A token is usable only when the
/// same nonzero even version brackets all 56 bytes of identity and provenance.
[[nodiscard]] constexpr ConSanMoiInlineAcquiredTokenClassification
consan_moi_inline_classify_acquired_token(ConSanMoiInlineAcquiredTokenSnapshot snapshot) {
  const auto &token = snapshot.payload;
  const bool payload_empty = token.version == 0 && token.consumer_owner_id == 0 &&
                             token.producer_owner_id == 0 && token.producer_epoch_plus_one == 0 &&
                             token.workgroup_key == 0 && token.kind == 0 &&
                             token.dispatch_id == 0 && token.source_release_address == 0 &&
                             token.source_release_version == 0 &&
                             token.consumer_epoch_plus_one == 0 && token.reservation_version == 0;
  if (snapshot.version_before != snapshot.version_after)
    return {.state = ConSanMoiInlineAcquiredTokenState::Changed};
  if (snapshot.version_before == 0)
    return {.state = payload_empty ? ConSanMoiInlineAcquiredTokenState::Empty
                                   : ConSanMoiInlineAcquiredTokenState::Malformed};
  if ((snapshot.version_before & 1u) != 0)
    return {.state = ConSanMoiInlineAcquiredTokenState::Publishing};
  if (token.version != snapshot.version_before || token.consumer_owner_id == 0 ||
      token.producer_owner_id == 0 || token.consumer_owner_id == token.producer_owner_id ||
      !consan_moi_inline_causal_epoch_is_valid(token.producer_epoch_plus_one) ||
      token.workgroup_key == 0 ||
      token.kind > static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence) ||
      token.dispatch_id == 0 || token.source_release_address == 0 ||
      !consan_moi_inline_release_version_is_ready(token.source_release_version) ||
      !consan_moi_inline_causal_epoch_is_valid(token.consumer_epoch_plus_one) ||
      token.reservation_version != 0)
    return {.state = ConSanMoiInlineAcquiredTokenState::Malformed};
  return {.state = ConSanMoiInlineAcquiredTokenState::Stable, .token = token};
}

[[nodiscard]] constexpr ConSanMoiReportBufferLayout
consan_moi_inline_shadow_report_buffer_layout_for_bytes(
    uint64_t report_buffer_size,
    uint32_t requested_diagnostic_capacity = kConSanMoiInlineShadowDefaultDiagnosticCapacity) {
  ConSanMoiReportBufferLayout layout;
  layout.engine = ConSanMoiEngine::InlineShadow;
  layout.inline_exact_dispatch_bank_count = kConSanMoiInlineMaximumDispatchBankCount;
  if (report_buffer_size < sizeof(ConSanMoiReportHeader) ||
      report_buffer_size > std::numeric_limits<size_t>::max())
    return layout;

  const uint64_t payload_bytes = report_buffer_size - sizeof(ConSanMoiReportHeader);
  uint32_t diagnostic_capacity = consan_moi_clamp_u32_capacity(std::min<uint64_t>(
      requested_diagnostic_capacity, payload_bytes / sizeof(ConSanMoiDiagnosticRecord)));
  constexpr uint64_t metadata_bytes_per_minimum_slot =
      sizeof(ConSanMoiInlineAtomicReleaseSlot) + sizeof(ConSanMoiInlineCausalSnapshot) +
      sizeof(ConSanMoiInlineAcquiredEpochTokenSlot);
  constexpr uint64_t minimum_metadata_bytes =
      metadata_bytes_per_minimum_slot + sizeof(ConSanMoiInlineExactShadowSlot);
  while (diagnostic_capacity != 0 && payload_bytes - static_cast<uint64_t>(diagnostic_capacity) *
                                                         sizeof(ConSanMoiDiagnosticRecord) <
                                         minimum_metadata_bytes) {
    --diagnostic_capacity;
  }

  const uint64_t diagnostic_bytes =
      static_cast<uint64_t>(diagnostic_capacity) * sizeof(ConSanMoiDiagnosticRecord);
  const uint64_t post_diagnostic_bytes = payload_bytes - diagnostic_bytes;
  constexpr uint64_t full_metadata_bytes =
      static_cast<uint64_t>(kConSanMoiInlineShadowAtomicReleaseSlotCapacity) *
          (sizeof(ConSanMoiInlineAtomicReleaseSlot) + sizeof(ConSanMoiInlineCausalSnapshot)) +
      static_cast<uint64_t>(kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity) *
          sizeof(ConSanMoiInlineAcquiredEpochTokenSlot);
  const bool full_metadata =
      post_diagnostic_bytes >= full_metadata_bytes + sizeof(ConSanMoiInlineExactShadowSlot);
  const uint32_t inline_atomic_release_capacity =
      full_metadata ? kConSanMoiInlineShadowAtomicReleaseSlotCapacity
      : post_diagnostic_bytes >= minimum_metadata_bytes ? 1u
                                                        : 0u;
  const uint32_t inline_acquired_epoch_token_capacity =
      full_metadata ? kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity
                    : inline_atomic_release_capacity;
  const uint64_t inline_atomic_release_slot_bytes =
      static_cast<uint64_t>(inline_atomic_release_capacity) *
      sizeof(ConSanMoiInlineAtomicReleaseSlot);
  const uint64_t inline_acquired_epoch_token_slot_bytes =
      static_cast<uint64_t>(inline_acquired_epoch_token_capacity) *
      sizeof(ConSanMoiInlineAcquiredEpochTokenSlot);
  const uint64_t inline_causal_snapshot_bytes =
      static_cast<uint64_t>(inline_atomic_release_capacity) * sizeof(ConSanMoiInlineCausalSnapshot);
  layout.diagnostic_capacity = diagnostic_capacity;
  layout.inline_atomic_release_capacity = inline_atomic_release_capacity;
  layout.inline_acquired_epoch_token_capacity = inline_acquired_epoch_token_capacity;
  layout.inline_causal_snapshot_capacity = inline_atomic_release_capacity;
  layout.exact_shadow_entry_capacity = consan_moi_clamp_u32_capacity(
      (payload_bytes - diagnostic_bytes - inline_atomic_release_slot_bytes -
       inline_causal_snapshot_bytes - inline_acquired_epoch_token_slot_bytes) /
      sizeof(ConSanMoiInlineExactShadowSlot));
  layout.diagnostic_records_offset = sizeof(ConSanMoiReportHeader);
  layout.exact_shadow_entries_offset =
      layout.diagnostic_records_offset +
      static_cast<size_t>(layout.diagnostic_capacity) * sizeof(ConSanMoiDiagnosticRecord);
  layout.inline_atomic_release_slots_offset =
      layout.exact_shadow_entries_offset + static_cast<size_t>(layout.exact_shadow_entry_capacity) *
                                               sizeof(ConSanMoiInlineExactShadowSlot);
  layout.inline_causal_snapshots_offset =
      layout.inline_atomic_release_slots_offset + inline_atomic_release_slot_bytes;
  layout.inline_acquired_epoch_token_slots_offset =
      layout.inline_causal_snapshots_offset + inline_causal_snapshot_bytes;
  layout.sampled_watchpoints_offset =
      layout.inline_acquired_epoch_token_slots_offset + inline_acquired_epoch_token_slot_bytes;
  layout.sampled_sync_metadata_offset = layout.sampled_watchpoints_offset;
  layout.required_bytes = layout.sampled_watchpoints_offset;
  layout.barrier_records_offset = layout.access_records_offset;
  layout.atomic_records_offset = layout.access_records_offset;
  layout.valid = layout.required_bytes <= report_buffer_size;
  return layout;
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
