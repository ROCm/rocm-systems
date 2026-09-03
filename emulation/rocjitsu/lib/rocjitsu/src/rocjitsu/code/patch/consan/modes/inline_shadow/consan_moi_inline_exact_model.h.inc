// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

inline constexpr uint32_t kConSanMoiInlineExactInitialVersion = 0;
inline constexpr uint32_t kConSanMoiInlineExactMaxReadyVersion =
    std::numeric_limits<uint32_t>::max() - 1u;

enum class ConSanMoiInlineExactSnapshotState : uint8_t {
  Empty,
  Stable,
  Publishing,
  ChangedDuringRead,
  Malformed,
};

/// Words observed by a reader in acquire order. The version is deliberately
/// sampled around every payload field: accepting a sidecar or one version read
/// would permit a packed access from one publisher to be paired with another
/// dispatch.
struct ConSanMoiInlineExactSnapshotWords {
  uint32_t version_before = 0;
  uint64_t packed_access = 0;
  uint64_t dispatch_id = 0;
  uint32_t byte_provenance = 0;
  uint32_t version_after = 0;
};

struct ConSanMoiInlineExactSnapshot {
  ConSanMoiInlineExactSnapshotState state = ConSanMoiInlineExactSnapshotState::Empty;
  ConSanMoiExactShadowEntry entry;
  ConSanMoiExactByteCellProvenance byte_provenance;
  uint64_t dispatch_id = 0;
  uint32_t version = 0;
};

[[nodiscard]] constexpr ConSanMoiInlineExactSnapshot
classify_consan_moi_inline_exact_snapshot(const ConSanMoiInlineExactSnapshotWords &words) {
  if (words.version_before != words.version_after) {
    return {.state = ConSanMoiInlineExactSnapshotState::ChangedDuringRead,
            .entry = {},
            .byte_provenance = {},
            .dispatch_id = 0,
            .version = 0};
  }
  if ((words.version_after & 1u) != 0) {
    return {.state = ConSanMoiInlineExactSnapshotState::Publishing,
            .entry = {},
            .byte_provenance = {},
            .dispatch_id = 0,
            .version = words.version_after};
  }
  if (words.version_after == kConSanMoiInlineExactInitialVersion) {
    if (words.packed_access == 0 && words.dispatch_id == 0 && words.byte_provenance == 0)
      return {.state = ConSanMoiInlineExactSnapshotState::Empty,
              .entry = {},
              .byte_provenance = {},
              .dispatch_id = 0,
              .version = 0};
    return {.state = ConSanMoiInlineExactSnapshotState::Malformed,
            .entry = {},
            .byte_provenance = {},
            .dispatch_id = 0,
            .version = 0};
  }
  const ConSanMoiExactShadowEntry entry = decode_consan_moi_exact_shadow_entry(words.packed_access);
  const ConSanMoiExactByteCellProvenance byte_provenance =
      decode_consan_moi_exact_byte_cell_provenance(words.byte_provenance);
  if (!byte_provenance.valid || words.packed_access == 0 || words.dispatch_id == 0 ||
      entry.kind < ConSanMoiShadowAccessKind::Read ||
      entry.kind > ConSanMoiShadowAccessKind::Atomic || entry.generation == 0) {
    return {.state = ConSanMoiInlineExactSnapshotState::Malformed,
            .entry = {},
            .byte_provenance = {},
            .dispatch_id = 0,
            .version = words.version_after};
  }
  return {.state = ConSanMoiInlineExactSnapshotState::Stable,
          .entry = entry,
          .byte_provenance = byte_provenance,
          .dispatch_id = words.dispatch_id,
          .version = words.version_after};
}

[[nodiscard]] constexpr uint32_t consan_moi_inline_acquired_epoch_token_value(uint32_t epoch) {
  return epoch >= consan_moi_exact_shadow::max_epoch ? consan_moi_exact_shadow::max_epoch
                                                     : epoch + 1u;
}

/// Qualify one directed edge in a deferred diagnostic. The diagnostic retains
/// both access epochs and the persistent workgroup key, so a stable final token
/// can safely prove that this consumer observed this producer segment even
/// when the token transaction completed after the device-side lookup.
[[nodiscard]] constexpr bool consan_moi_inline_stable_token_orders_deferred(
    const ConSanMoiInlineAcquiredEpochTokenSlot &token, uint64_t dispatch_id,
    uint32_t workgroup_key, uint32_t consumer_owner_id, uint32_t producer_owner_id,
    uint32_t consumer_epoch, uint32_t producer_epoch) {
  if (dispatch_id == 0 || workgroup_key == 0 || consumer_owner_id == 0 || producer_owner_id == 0 ||
      consumer_owner_id == producer_owner_id ||
      consumer_epoch >= consan_moi_exact_shadow::max_epoch ||
      producer_epoch > consan_moi_exact_shadow::max_epoch || token.version == 0 ||
      (token.version & 1u) != 0 || token.source_release_address == 0 ||
      !consan_moi_inline_release_version_is_ready(token.source_release_version) ||
      !consan_moi_inline_causal_epoch_is_valid(token.consumer_epoch_plus_one) ||
      !consan_moi_inline_causal_epoch_is_valid(token.producer_epoch_plus_one) ||
      token.reservation_version != 0)
    return false;
  if (token.dispatch_id != dispatch_id || token.workgroup_key != workgroup_key ||
      token.consumer_owner_id != consumer_owner_id ||
      token.producer_owner_id != producer_owner_id ||
      token.consumer_epoch_plus_one >
          consan_moi_inline_acquired_epoch_token_value(consumer_epoch) ||
      token.producer_epoch_plus_one < consan_moi_inline_acquired_epoch_token_value(producer_epoch))
    return false;
  return token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct) ||
         token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited);
}

struct ConSanMoiDeferredDiagnosticFilterResult {
  std::vector<uint32_t> visible_indices;
  uint32_t qualified_count = 0;
  uint32_t effective_diagnostic_count = 0;
};

/// Filters only diagnostics whose access pair is ordered in either direction
/// by a stable acquired token. The total count includes dropped records, so
/// the effective count retains overflow evidence while subtracting qualified
/// visible records. Incomplete token evidence fails closed.
[[nodiscard]] inline ConSanMoiDeferredDiagnosticFilterResult
consan_moi_filter_deferred_inline_diagnostics(
    std::span<const ConSanMoiDiagnosticRecord> visible_diagnostics, uint32_t total_diagnostic_count,
    std::span<const ConSanMoiInlineAcquiredEpochTokenSlot> stable_tokens,
    bool token_evidence_complete) {
  ConSanMoiDeferredDiagnosticFilterResult result;
  result.visible_indices.reserve(visible_diagnostics.size());
  for (uint32_t i = 0; i < visible_diagnostics.size(); ++i) {
    const auto &diagnostic = visible_diagnostics[i];
    const bool ordered =
        token_evidence_complete &&
        diagnostic.kind == static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict) &&
        diagnostic.backend == static_cast<uint32_t>(ConSanMoiEngine::InlineShadow) &&
        diagnostic.reserved != 0 && std::ranges::any_of(stable_tokens, [&](const auto &token) {
          return consan_moi_inline_stable_token_orders_deferred(
                     token, diagnostic.generation, diagnostic.reserved, diagnostic.second_owner_id,
                     diagnostic.first_owner_id, diagnostic.epoch, diagnostic.first_epoch) ||
                 consan_moi_inline_stable_token_orders_deferred(
                     token, diagnostic.generation, diagnostic.reserved, diagnostic.first_owner_id,
                     diagnostic.second_owner_id, diagnostic.first_epoch, diagnostic.epoch);
        });
    if (ordered)
      ++result.qualified_count;
    else
      result.visible_indices.push_back(i);
  }
  result.effective_diagnostic_count = total_diagnostic_count >= result.qualified_count
                                          ? total_diagnostic_count - result.qualified_count
                                          : 0;
  return result;
}
