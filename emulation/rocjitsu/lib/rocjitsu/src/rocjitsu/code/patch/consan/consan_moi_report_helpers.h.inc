[[nodiscard]] ConSanMoiRecordReplayTraceHeader consan_moi_compact_record_replay_trace(
    uint64_t generation, uint64_t dispatch_id,
    std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<ConSanMoiRecordReplayPcEntry> dictionary,
    std::span<ConSanMoiRecordReplayWorkgroupRun> workgroup_runs,
    std::span<ConSanMoiRecordReplayCompactEvent> events);

[[nodiscard]] ConSanMoiRecordReplayTraceHeader consan_moi_compact_record_replay_trace(
    uint64_t generation, uint64_t dispatch_id,
    std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<const ConSanMoiRecordReplayFenceEvent> fence_events,
    std::span<ConSanMoiRecordReplayPcEntry> dictionary,
    std::span<ConSanMoiRecordReplayWorkgroupRun> workgroup_runs,
    std::span<ConSanMoiRecordReplayCompactEvent> events);

/// Plans a deterministic prefix of complete per-workgroup barrier epochs from
/// an RR1 compact trace. Every selected event receives its output-window index;
/// omitted events retain kConSanMoiRecordReplayUncapturedEvent. Invalid trace
/// structure produces no selection.
[[nodiscard]] ConSanMoiRecordReplayCaptureResult consan_moi_plan_record_replay_capture(
    const ConSanMoiRecordReplayTraceHeader &header,
    std::span<const ConSanMoiRecordReplayPcEntry> dictionary,
    std::span<const ConSanMoiRecordReplayWorkgroupRun> workgroup_runs,
    std::span<const ConSanMoiRecordReplayCompactEvent> events,
    ConSanMoiRecordReplayCaptureLimits limits,
    std::span<ConSanMoiRecordReplayCaptureWindow> windows,
    std::span<uint32_t> event_window_indices);

/// Reconstructs and replays exactly the selected complete RR2 windows in
/// global event order. Static access/atomic semantics come from the RR1 PC
/// dictionary; dynamic workgroup, owner, epoch, lane, range, and address data
/// come from the compact events and run table.
[[nodiscard]] ConSanMoiRecordReplayWindowResult consan_moi_replay_record_replay_capture(
    const ConSanMoiRecordReplayTraceHeader &header,
    std::span<const ConSanMoiRecordReplayPcEntry> dictionary,
    std::span<const ConSanMoiRecordReplayWorkgroupRun> workgroup_runs,
    std::span<const ConSanMoiRecordReplayCompactEvent> events,
    std::span<const ConSanMoiRecordReplayCaptureWindow> windows,
    std::span<const uint32_t> event_window_indices,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries);

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
inline constexpr uint32_t kConSanMoiInlineShadowConservativeExactShadowEntries = 16u * 1024u;
inline constexpr uint32_t kConSanMoiInlineShadowAtomicReleaseSlotCapacity = 64;
inline constexpr uint32_t kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity = 64;

[[nodiscard]] constexpr uint32_t
consan_moi_inline_atomic_release_slot_index(uint64_t atomic_address, uint32_t table_capacity) {
  if (table_capacity == 0 || (table_capacity & (table_capacity - 1u)) != 0)
    return 0;
  const uint32_t low = static_cast<uint32_t>(atomic_address);
  const uint32_t high = static_cast<uint32_t>(atomic_address >> 32u);
  uint32_t hash = (low >> 2u) ^ high;
  hash ^= hash >> 16u;
  hash ^= hash >> 8u;
  hash ^= hash >> 4u;
  hash *= 0x85ebca6bu;
  return hash & (table_capacity - 1u);
}

enum class ConSanMoiInlineAtomicLookup : uint8_t {
  Empty,
  ExactSameOwner,
  ExactOtherOwner,
  Collision,
};

inline constexpr uint32_t kConSanMoiInlineTokenWorkgroupMultiplier = 0x9e3779b1u;
inline constexpr uint32_t kConSanMoiInlineTokenConsumerMultiplier = 0x85ebca6bu;
inline constexpr uint32_t kConSanMoiInlineTokenProducerMultiplier = 0xc2b2ae35u;
inline constexpr uint32_t kConSanMoiInlineTokenReleaseSequenceSalt = 0x9e3779b9u;
inline constexpr uint32_t kConSanMoiInlineTokenAvalancheMultiplier0 = 0x7feb352du;
inline constexpr uint32_t kConSanMoiInlineTokenAvalancheMultiplier1 = 0x846ca68bu;

/// @brief Reference contract for the direct-mapped inline atomic table.
[[nodiscard]] constexpr ConSanMoiInlineAtomicLookup
consan_moi_inline_atomic_release_lookup(const ConSanMoiInlineAtomicReleaseSlot &slot,
                                        uint64_t atomic_address, uint32_t owner_id,
                                        uint32_t workgroup_key = 0) {
  if (slot.version == 0)
    return ConSanMoiInlineAtomicLookup::Empty;
  if (!consan_moi_inline_release_version_is_ready(slot.version))
    return ConSanMoiInlineAtomicLookup::Collision;
  if (slot.atomic_address != atomic_address || slot.workgroup_key != workgroup_key)
    return ConSanMoiInlineAtomicLookup::Collision;
  return slot.owner_id == owner_id ? ConSanMoiInlineAtomicLookup::ExactSameOwner
                                   : ConSanMoiInlineAtomicLookup::ExactOtherOwner;
}

[[nodiscard]] constexpr uint32_t consan_moi_inline_acquired_epoch_token_slot_index(
    uint32_t workgroup_key, uint32_t consumer_owner_id, uint32_t producer_owner_id,
    uint32_t table_capacity, bool release_sequence = false) {
  if (table_capacity == 0 || (table_capacity & (table_capacity - 1u)) != 0)
    return 0;
  uint32_t mixed = workgroup_key * kConSanMoiInlineTokenWorkgroupMultiplier;
  mixed ^= consumer_owner_id * kConSanMoiInlineTokenConsumerMultiplier;
  mixed ^= producer_owner_id * kConSanMoiInlineTokenProducerMultiplier;
  if (release_sequence)
    mixed ^= kConSanMoiInlineTokenReleaseSequenceSalt;
  mixed ^= mixed >> 16u;
  mixed *= kConSanMoiInlineTokenAvalancheMultiplier0;
  mixed ^= mixed >> 15u;
  mixed *= kConSanMoiInlineTokenAvalancheMultiplier1;
  mixed ^= mixed >> 16u;
  return mixed & (table_capacity - 1u);
}

enum class ConSanMoiInlineAcquiredEpochTokenLookup : uint8_t {
  Empty,
  Exact,
  Collision,
};

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

[[nodiscard]] constexpr bool consan_moi_inline_acquired_token_identity_matches(
    const ConSanMoiInlineAcquiredEpochTokenSlot &actual,
    const ConSanMoiInlineAcquiredEpochTokenSlot &expected) {
  return actual.consumer_owner_id == expected.consumer_owner_id &&
         actual.producer_owner_id == expected.producer_owner_id &&
         actual.producer_epoch_plus_one == expected.producer_epoch_plus_one &&
         actual.workgroup_key == expected.workgroup_key && actual.kind == expected.kind &&
         actual.dispatch_id == expected.dispatch_id &&
         actual.source_release_address == expected.source_release_address &&
         actual.source_release_version == expected.source_release_version &&
         actual.consumer_epoch_plus_one == expected.consumer_epoch_plus_one &&
         actual.reservation_version == expected.reservation_version;
}

struct ConSanMoiInlineAcquiredTokenTransactionResult {
  bool committed = false;
  bool collision = false;
  bool duplicate_destination = false;
  bool claim_failed = false;
  bool invalid = false;
};

/// Reference all-or-nothing transaction for a direct token and its inherited
/// companions. Every destination is validated and reserved odd before any
/// payload is changed. Any failure restores every reservation byte-for-byte.
[[nodiscard]] constexpr ConSanMoiInlineAcquiredTokenTransactionResult
consan_moi_inline_publish_acquired_token_transaction(
    std::span<ConSanMoiInlineAcquiredEpochTokenSlot> table,
    std::span<const ConSanMoiInlineAcquiredEpochTokenSlot> desired,
    std::span<const bool> simulated_claim_success = {}) {
  ConSanMoiInlineAcquiredTokenTransactionResult result;
  if (desired.empty() || desired.size() > 5 || table.empty() ||
      (table.size() & (table.size() - 1u)) != 0 ||
      (!simulated_claim_success.empty() && simulated_claim_success.size() != desired.size())) {
    result.invalid = true;
    return result;
  }
  std::array<size_t, 5> destinations{};
  std::array<ConSanMoiInlineAcquiredEpochTokenSlot, 5> originals{};
  for (size_t i = 0; i < desired.size(); ++i) {
    const auto &token = desired[i];
    if (token.version != 0 || token.consumer_owner_id == 0 || token.producer_owner_id == 0 ||
        token.consumer_owner_id == token.producer_owner_id ||
        !consan_moi_inline_causal_epoch_is_valid(token.producer_epoch_plus_one) ||
        token.workgroup_key == 0 ||
        token.kind > static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence) ||
        token.dispatch_id == 0 || token.source_release_address == 0 ||
        !consan_moi_inline_release_version_is_ready(token.source_release_version) ||
        !consan_moi_inline_causal_epoch_is_valid(token.consumer_epoch_plus_one) ||
        token.reservation_version != 0) {
      result.invalid = true;
      return result;
    }
    destinations[i] = consan_moi_inline_acquired_epoch_token_slot_index(
        token.workgroup_key, token.consumer_owner_id, token.producer_owner_id, table.size(),
        token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence));
    for (size_t prior = 0; prior < i; ++prior) {
      if (destinations[prior] == destinations[i]) {
        result.duplicate_destination = true;
        return result;
      }
    }
    originals[i] = table[destinations[i]];
    const auto classified = consan_moi_inline_classify_acquired_token(
        {originals[i].version, originals[i], originals[i].version});
    if (classified.state != ConSanMoiInlineAcquiredTokenState::Empty &&
        classified.state != ConSanMoiInlineAcquiredTokenState::Stable) {
      result.collision = true;
      return result;
    }
    if (classified.state == ConSanMoiInlineAcquiredTokenState::Stable &&
        (classified.token.consumer_owner_id != token.consumer_owner_id ||
         classified.token.producer_owner_id != token.producer_owner_id ||
         classified.token.workgroup_key != token.workgroup_key ||
         classified.token.dispatch_id != token.dispatch_id)) {
      result.collision = true;
      return result;
    }
  }
  size_t reserved = 0;
  for (; reserved < desired.size(); ++reserved) {
    if (!simulated_claim_success.empty() && !simulated_claim_success[reserved]) {
      result.claim_failed = true;
      break;
    }
    auto &slot = table[destinations[reserved]];
    slot.version = originals[reserved].version == 0 ? 1u : originals[reserved].version + 1u;
  }
  if (reserved != desired.size()) {
    while (reserved != 0) {
      --reserved;
      table[destinations[reserved]] = originals[reserved];
    }
    return result;
  }
  for (size_t i = 0; i < desired.size(); ++i) {
    const uint32_t committed_version = originals[i].version == 0 ? 2u : originals[i].version + 2u;
    table[destinations[i]] = desired[i];
    table[destinations[i]].version = committed_version;
  }
  result.committed = true;
  return result;
}

[[nodiscard]] constexpr ConSanMoiInlineAcquiredEpochTokenLookup
consan_moi_inline_acquired_epoch_token_lookup(const ConSanMoiInlineAcquiredEpochTokenSlot &slot,
                                              uint32_t workgroup_key, uint32_t consumer_owner_id,
                                              uint32_t producer_owner_id) {
  if (slot.version == 0) {
    return slot.consumer_owner_id == 0 && slot.producer_owner_id == 0 &&
                   slot.producer_epoch_plus_one == 0 && slot.workgroup_key == 0 && slot.kind == 0 &&
                   slot.dispatch_id == 0 && slot.source_release_address == 0 &&
                   slot.source_release_version == 0 && slot.consumer_epoch_plus_one == 0 &&
                   slot.reservation_version == 0
               ? ConSanMoiInlineAcquiredEpochTokenLookup::Empty
               : ConSanMoiInlineAcquiredEpochTokenLookup::Collision;
  }
  if (!consan_moi_inline_release_version_is_ready(slot.version))
    return ConSanMoiInlineAcquiredEpochTokenLookup::Collision;
  if (slot.consumer_owner_id == 0 || slot.producer_owner_id == 0 ||
      slot.consumer_owner_id == slot.producer_owner_id ||
      !consan_moi_inline_causal_epoch_is_valid(slot.producer_epoch_plus_one) ||
      slot.workgroup_key == 0 ||
      slot.kind > static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence) ||
      slot.dispatch_id == 0 || slot.source_release_address == 0 ||
      !consan_moi_inline_release_version_is_ready(slot.source_release_version) ||
      !consan_moi_inline_causal_epoch_is_valid(slot.consumer_epoch_plus_one) ||
      slot.reservation_version != 0)
    return ConSanMoiInlineAcquiredEpochTokenLookup::Collision;
  if (slot.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence) ||
      slot.workgroup_key != workgroup_key || slot.consumer_owner_id != consumer_owner_id ||
      slot.producer_owner_id != producer_owner_id)
    return ConSanMoiInlineAcquiredEpochTokenLookup::Collision;
  return ConSanMoiInlineAcquiredEpochTokenLookup::Exact;
}

[[nodiscard]] constexpr ConSanMoiReportBufferLayout
consan_moi_inline_shadow_report_buffer_layout_for_bytes(
    uint64_t report_buffer_size,
    uint32_t requested_diagnostic_capacity = kConSanMoiInlineShadowDefaultDiagnosticCapacity) {
  ConSanMoiReportBufferLayout layout;
  layout.inline_exact_dispatch_bank_count = kConSanMoiInlineMaximumDispatchBankCount;
  if (report_buffer_size < sizeof(ConSanMoiReportHeader) ||
      report_buffer_size > std::numeric_limits<size_t>::max())
    return layout;

  const uint64_t payload_bytes = report_buffer_size - sizeof(ConSanMoiReportHeader);
  uint32_t diagnostic_capacity = consan_moi_clamp_u32_capacity(std::min<uint64_t>(
      requested_diagnostic_capacity, payload_bytes / sizeof(ConSanMoiDiagnosticRecord)));
  constexpr uint64_t metadata_bytes_per_slot = sizeof(ConSanMoiInlineAtomicReleaseSlot) +
                                               sizeof(ConSanMoiInlineCausalSnapshot) +
                                               sizeof(ConSanMoiInlineAcquiredEpochTokenSlot);
  constexpr uint64_t minimum_metadata_bytes =
      metadata_bytes_per_slot + sizeof(ConSanMoiInlineExactShadowSlot);
  while (diagnostic_capacity != 0 && payload_bytes - static_cast<uint64_t>(diagnostic_capacity) *
                                                         sizeof(ConSanMoiDiagnosticRecord) <
                                         minimum_metadata_bytes) {
    --diagnostic_capacity;
  }

  const uint64_t diagnostic_bytes =
      static_cast<uint64_t>(diagnostic_capacity) * sizeof(ConSanMoiDiagnosticRecord);
  const uint64_t post_diagnostic_bytes = payload_bytes - diagnostic_bytes;
  const uint32_t inline_metadata_capacity =
      post_diagnostic_bytes >=
              static_cast<uint64_t>(kConSanMoiInlineShadowAtomicReleaseSlotCapacity) *
                      metadata_bytes_per_slot +
                  sizeof(ConSanMoiInlineExactShadowSlot)
          ? kConSanMoiInlineShadowAtomicReleaseSlotCapacity
      : post_diagnostic_bytes >= minimum_metadata_bytes ? 1u
                                                        : 0u;
  const uint64_t inline_atomic_release_slot_bytes =
      static_cast<uint64_t>(inline_metadata_capacity) * sizeof(ConSanMoiInlineAtomicReleaseSlot);
  const uint64_t inline_acquired_epoch_token_slot_bytes =
      static_cast<uint64_t>(inline_metadata_capacity) *
      sizeof(ConSanMoiInlineAcquiredEpochTokenSlot);
  const uint64_t inline_causal_snapshot_bytes =
      static_cast<uint64_t>(inline_metadata_capacity) * sizeof(ConSanMoiInlineCausalSnapshot);
  layout.diagnostic_capacity = diagnostic_capacity;
  layout.inline_atomic_release_capacity = inline_metadata_capacity;
  layout.inline_acquired_epoch_token_capacity = inline_metadata_capacity;
  layout.inline_causal_snapshot_capacity = inline_metadata_capacity;
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

/// Returns whether an engine-specific report layout contains every region
/// required by the enabled instrumentation. Direct Sampled synchronization
/// uses paired sampled metadata, not the generic Record/Replay record arrays.
[[nodiscard]] constexpr bool
consan_moi_report_layout_has_required_capacities(const ConSanMoiReportBufferLayout &layout,
                                                 ConSanMoiEngine engine, bool track_barriers,
                                                 bool track_atomics) {
  if (!layout.valid)
    return false;
  if (engine == ConSanMoiEngine::Sampled)
    return layout.sampled_watchpoint_capacity != 0 && layout.sampled_causal_window_capacity != 0 &&
           layout.sampled_sync_metadata_capacity != 0 &&
           layout.sampled_pending_acquire_capacity != 0;
  if (engine == ConSanMoiEngine::InlineShadow)
    return layout.diagnostic_capacity != 0 && layout.exact_shadow_entry_capacity != 0 &&
           layout.inline_exact_dispatch_bank_count != 0 &&
           (layout.inline_exact_dispatch_bank_count &
            (layout.inline_exact_dispatch_bank_count - 1u)) == 0u &&
           (!track_atomics || layout.inline_atomic_release_capacity != 0);
  const bool dispatch_table_valid = (layout.record_replay_dispatch_token_capacity == 0u &&
                                     layout.record_replay_access_dispatch_bank_count == 1u) ||
                                    (layout.record_replay_dispatch_token_capacity != 0u &&
                                     layout.record_replay_dispatch_token_capacity <=
                                         kConSanMoiRecordReplayMaximumDispatchTokenCount &&
                                     (layout.record_replay_dispatch_token_capacity &
                                      (layout.record_replay_dispatch_token_capacity - 1u)) == 0u);
  const uint64_t anticipated_identity_count =
      static_cast<uint64_t>(layout.record_replay_logical_access_range_count) *
      layout.record_replay_access_dispatch_bank_count *
      layout.record_replay_access_owner_bank_count *
      kConSanMoiRecordReplayMaximumAddressGroupsPerWave;
  const bool access_table_valid =
      layout.record_replay_dispatch_token_capacity == 0u ||
      (layout.record_replay_logical_access_range_count != 0u &&
       anticipated_identity_count <= layout.access_record_capacity &&
       (layout.access_record_capacity & (layout.access_record_capacity - 1u)) == 0u);
  return layout.access_record_capacity != 0 && dispatch_table_valid && access_table_valid &&
         layout.record_replay_access_dispatch_bank_count != 0 &&
         (layout.record_replay_access_dispatch_bank_count &
          (layout.record_replay_access_dispatch_bank_count - 1u)) == 0u &&
         layout.record_replay_access_owner_bank_count != 0 &&
         (layout.record_replay_access_owner_bank_count &
          (layout.record_replay_access_owner_bank_count - 1u)) == 0u &&
         (!track_barriers || layout.barrier_record_capacity != 0) &&
         (!track_atomics || layout.atomic_record_capacity != 0);
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
