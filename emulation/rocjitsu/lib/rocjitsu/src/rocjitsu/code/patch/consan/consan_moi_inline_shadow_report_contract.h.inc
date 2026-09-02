// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

/// Derive the explicit InlineShadow report geometry.
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
