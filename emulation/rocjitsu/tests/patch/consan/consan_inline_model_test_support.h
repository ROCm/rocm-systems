// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_inline_model_test_support.h
/// @brief Host reference models for testing InlineShadow GPU transactions.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>

namespace rocjitsu {

enum class ConSanMoiInlineWorkgroupKeyShape : uint8_t {
  OneDimensional,
  TwoDimensional,
  ThreeDimensional,
};

struct ConSanMoiInlineWorkgroupKey {
  bool valid = false;
  uint32_t value = 0;
};

/// Test oracle for the exact nonzero workgroup identity encoded by the
/// InlineShadow emitter. Each shape is injective inside its declared bounds;
/// coordinates that do not fit are unsupported rather than truncated or
/// hashed.
[[nodiscard]] constexpr ConSanMoiInlineWorkgroupKey
consan_moi_inline_workgroup_key(uint32_t x, uint32_t y, uint32_t z,
                                ConSanMoiInlineWorkgroupKeyShape shape) {
  uint32_t packed = 0;
  switch (shape) {
  case ConSanMoiInlineWorkgroupKeyShape::OneDimensional:
    if (y != 0 || z != 0 || x >= consan_moi_exact_shadow::max_generation)
      return {};
    packed = x;
    break;
  case ConSanMoiInlineWorkgroupKeyShape::TwoDimensional:
    if (z != 0 || x >= (1u << 10u) || y >= (1u << 10u))
      return {};
    packed = x | (y << 10u);
    if (packed == consan_moi_exact_shadow::max_generation)
      return {};
    break;
  case ConSanMoiInlineWorkgroupKeyShape::ThreeDimensional:
    if (x >= (1u << 8u) || y >= (1u << 6u) || z >= (1u << 6u))
      return {};
    packed = x | (y << 8u) | (z << 14u);
    if (packed == consan_moi_exact_shadow::max_generation)
      return {};
    break;
  }
  return {/*valid=*/true, /*value=*/packed + 1u};
}

/// Test oracle for the byte-provenance word emitted by InlineShadow.
[[nodiscard]] constexpr uint32_t
pack_consan_moi_exact_byte_cell_provenance(uint32_t byte_mask, uint32_t representative_lane) {
  const ConSanMoiExactByteCellProvenance provenance =
      consan_moi_exact_byte_cell_provenance_from_mask(byte_mask, representative_lane);
  if (!provenance.valid)
    return 0;
  return byte_mask |
         (static_cast<uint32_t>(provenance.byte_offset)
          << consan_moi_exact_byte_cell::byte_offset_shift) |
         (static_cast<uint32_t>(provenance.byte_offset + provenance.byte_count - 1u)
          << consan_moi_exact_byte_cell::byte_end_minus_one_shift) |
         ((representative_lane + 1u) << consan_moi_exact_byte_cell::lane_plus_one_shift);
}

/// Test oracle for decomposing one possibly unaligned access into the emitted
/// exact byte-cell masks.
[[nodiscard]] constexpr uint32_t consan_moi_exact_byte_mask_for_cell(uint32_t access_byte_offset,
                                                                     uint32_t access_byte_count,
                                                                     uint32_t relative_cell_index) {
  if (access_byte_count == 0u)
    return 0u;
  const uint32_t start = access_byte_offset & (consan_moi_shadow_cell::granule_bytes - 1u);
  const uint64_t relative_cell_start =
      static_cast<uint64_t>(relative_cell_index) * consan_moi_shadow_cell::granule_bytes;
  const uint64_t relative_cell_end = relative_cell_start + consan_moi_shadow_cell::granule_bytes;
  const uint64_t access_end = static_cast<uint64_t>(start) + access_byte_count;
  const uint64_t overlap_start = std::max<uint64_t>(start, relative_cell_start);
  const uint64_t overlap_end = std::min(access_end, relative_cell_end);
  if (overlap_start >= overlap_end)
    return 0u;
  const uint32_t cell_offset = static_cast<uint32_t>(overlap_start - relative_cell_start);
  const uint32_t cell_count = static_cast<uint32_t>(overlap_end - overlap_start);
  return ((1u << cell_count) - 1u) << cell_offset;
}

/// Test oracle for the packed-cell predicate emitted by InlineShadow.
[[nodiscard]] constexpr bool
consan_moi_exact_byte_cells_conflict(const ConSanMoiExactShadowEntry &current_access,
                                     const ConSanMoiExactByteCellProvenance &current_byte,
                                     const ConSanMoiExactShadowEntry &prior_access,
                                     const ConSanMoiExactByteCellProvenance &prior_byte) {
  if (!current_byte.valid || !prior_byte.valid ||
      (current_byte.byte_mask & prior_byte.byte_mask) == 0u ||
      current_access.epoch != prior_access.epoch ||
      current_access.generation != prior_access.generation ||
      !consan_moi_shadow_kind_conflicts(current_access.kind, prior_access.kind)) {
    return false;
  }
  if (current_access.owner_id != prior_access.owner_id)
    return true;
  return current_access.instruction_offset == prior_access.instruction_offset &&
         current_byte.representative_lane != prior_byte.representative_lane;
}

/// Test oracle for the packed-entry predicate emitted by InlineShadow.
[[nodiscard]] constexpr bool
consan_moi_exact_shadow_entries_conflict(const ConSanMoiExactShadowEntry &current,
                                         const ConSanMoiExactShadowEntry &prior) {
  return !consan_moi_shadow_kind_is_empty(prior.kind) && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         consan_moi_shadow_kind_conflicts(current.kind, prior.kind);
}

enum class ConSanMoiInlineReleaseClaim : uint8_t {
  Empty,
  ExactReady,
  InvalidIdentity,
  Publishing,
  UnstableRead,
  Collision,
  VersionExhausted,
};

struct ConSanMoiInlineVersionedReleaseState {
  uint32_t version = 0;
  ConSanMoiInlineVersionedReleaseIdentity identity;
};

struct ConSanMoiInlineReleaseClaimResult {
  ConSanMoiInlineReleaseClaim claim = ConSanMoiInlineReleaseClaim::Collision;
  uint32_t expected_version = 0;
  uint32_t publishing_version = 0;

  [[nodiscard]] constexpr bool can_publish() const {
    return claim == ConSanMoiInlineReleaseClaim::Empty ||
           claim == ConSanMoiInlineReleaseClaim::ExactReady;
  }
};

[[nodiscard]] constexpr ConSanMoiInlineReleaseClaimResult
consan_moi_inline_plan_release_claim(const ConSanMoiInlineVersionedReleaseState &state,
                                     const ConSanMoiInlineVersionedReleaseIdentity &identity,
                                     uint32_t version_after_identity_read) {
  if (!identity.valid())
    return {ConSanMoiInlineReleaseClaim::InvalidIdentity, state.version, 0};
  if (state.version == 0)
    return {ConSanMoiInlineReleaseClaim::Empty, /*expected_version=*/0,
            /*publishing_version=*/1};
  if (consan_moi_inline_release_version_is_publishing(state.version))
    return {ConSanMoiInlineReleaseClaim::Publishing, state.version, 0};
  if (version_after_identity_read != state.version)
    return {ConSanMoiInlineReleaseClaim::UnstableRead, state.version, 0};
  if (state.identity != identity)
    return {ConSanMoiInlineReleaseClaim::Collision, state.version, 0};
  if (state.version >= std::numeric_limits<uint32_t>::max() - 1u)
    return {ConSanMoiInlineReleaseClaim::VersionExhausted, state.version, 0};
  return {ConSanMoiInlineReleaseClaim::ExactReady, state.version, state.version + 1u};
}

[[nodiscard]] constexpr bool
consan_moi_inline_release_claim_is_well_formed(const ConSanMoiInlineReleaseClaimResult &claim) {
  if (!claim.can_publish() ||
      !consan_moi_inline_release_version_is_publishing(claim.publishing_version) ||
      claim.publishing_version == std::numeric_limits<uint32_t>::max() ||
      claim.publishing_version != claim.expected_version + 1u)
    return false;
  if (claim.claim == ConSanMoiInlineReleaseClaim::Empty)
    return claim.expected_version == 0;
  return claim.claim == ConSanMoiInlineReleaseClaim::ExactReady &&
         consan_moi_inline_release_version_is_ready(claim.expected_version);
}

[[nodiscard]] constexpr bool
consan_moi_inline_release_claim_cas_succeeded(const ConSanMoiInlineReleaseClaimResult &claim,
                                              uint32_t returned_version) {
  return consan_moi_inline_release_claim_is_well_formed(claim) &&
         returned_version == claim.expected_version;
}

[[nodiscard]] constexpr uint32_t
consan_moi_inline_commit_release_version(const ConSanMoiInlineReleaseClaimResult &claim) {
  return consan_moi_inline_release_claim_is_well_formed(claim) ? claim.publishing_version + 1u : 0u;
}

[[nodiscard]] constexpr uint32_t
consan_moi_inline_restore_release_version(const ConSanMoiInlineReleaseClaimResult &claim) {
  return consan_moi_inline_release_claim_is_well_formed(claim) ? claim.expected_version : 0u;
}

enum class ConSanMoiInlineReleaseTransactionEvent : uint8_t {
  PriorSnapshot,
  Reserve,
  GuestAtomic,
  AcquireImport,
  Metadata,
  CausalSnapshot,
  CommitReady,
  RestorePrior,
  PoisonCoverage,
};

/// Reference ordering contract for the versioned release critical section.
/// This host-only oracle checks the GPU instruction sequence in focused tests;
/// it is not a second production implementation of the transaction.
[[nodiscard]] constexpr bool consan_moi_inline_release_transaction_is_sound(
    std::span<const ConSanMoiInlineReleaseTransactionEvent> events, bool claim_succeeded,
    bool dynamic_acquire_semantics, bool release_outcome, bool outcome_dependent_release) {
  const auto position = [&](ConSanMoiInlineReleaseTransactionEvent event) -> size_t {
    const auto found = std::find(events.begin(), events.end(), event);
    return found == events.end() ? events.size() : static_cast<size_t>(found - events.begin());
  };
  const auto exactly_once = [&](ConSanMoiInlineReleaseTransactionEvent event) {
    return std::count(events.begin(), events.end(), event) == 1;
  };
  if (!exactly_once(ConSanMoiInlineReleaseTransactionEvent::GuestAtomic))
    return false;
  const size_t guest = position(ConSanMoiInlineReleaseTransactionEvent::GuestAtomic);
  if (!claim_succeeded) {
    const size_t poison = position(ConSanMoiInlineReleaseTransactionEvent::PoisonCoverage);
    const size_t prior = position(ConSanMoiInlineReleaseTransactionEvent::PriorSnapshot);
    return exactly_once(ConSanMoiInlineReleaseTransactionEvent::PoisonCoverage) && poison < guest &&
           (prior == events.size() ||
            (exactly_once(ConSanMoiInlineReleaseTransactionEvent::PriorSnapshot) &&
             prior < poison)) &&
           position(ConSanMoiInlineReleaseTransactionEvent::Reserve) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::AcquireImport) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::Metadata) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::CausalSnapshot) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::CommitReady) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::RestorePrior) == events.size();
  }
  if (!exactly_once(ConSanMoiInlineReleaseTransactionEvent::Reserve) ||
      position(ConSanMoiInlineReleaseTransactionEvent::Reserve) >= guest ||
      position(ConSanMoiInlineReleaseTransactionEvent::PoisonCoverage) != events.size())
    return false;
  const size_t prior = position(ConSanMoiInlineReleaseTransactionEvent::PriorSnapshot);
  const size_t acquire = position(ConSanMoiInlineReleaseTransactionEvent::AcquireImport);
  if (dynamic_acquire_semantics != (acquire != events.size()) ||
      dynamic_acquire_semantics != (prior != events.size()) ||
      (dynamic_acquire_semantics &&
       (!exactly_once(ConSanMoiInlineReleaseTransactionEvent::PriorSnapshot) ||
        !exactly_once(ConSanMoiInlineReleaseTransactionEvent::AcquireImport) ||
        prior >= position(ConSanMoiInlineReleaseTransactionEvent::Reserve) || acquire <= guest)))
    return false;
  if (!release_outcome) {
    const size_t restore = position(ConSanMoiInlineReleaseTransactionEvent::RestorePrior);
    return outcome_dependent_release &&
           exactly_once(ConSanMoiInlineReleaseTransactionEvent::RestorePrior) && restore > guest &&
           (!dynamic_acquire_semantics || restore > acquire) &&
           position(ConSanMoiInlineReleaseTransactionEvent::Metadata) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::CausalSnapshot) == events.size() &&
           position(ConSanMoiInlineReleaseTransactionEvent::CommitReady) == events.size();
  }
  const size_t metadata = position(ConSanMoiInlineReleaseTransactionEvent::Metadata);
  const size_t snapshot = position(ConSanMoiInlineReleaseTransactionEvent::CausalSnapshot);
  const size_t commit = position(ConSanMoiInlineReleaseTransactionEvent::CommitReady);
  const bool metadata_ordered = outcome_dependent_release || dynamic_acquire_semantics
                                    ? metadata > guest && snapshot > guest
                                    : metadata < guest && snapshot < guest;
  return exactly_once(ConSanMoiInlineReleaseTransactionEvent::Metadata) &&
         exactly_once(ConSanMoiInlineReleaseTransactionEvent::CausalSnapshot) &&
         exactly_once(ConSanMoiInlineReleaseTransactionEvent::CommitReady) && metadata_ordered &&
         (!dynamic_acquire_semantics || (metadata > acquire && snapshot > acquire)) &&
         commit > guest && commit > metadata && commit > snapshot &&
         position(ConSanMoiInlineReleaseTransactionEvent::RestorePrior) == events.size();
}

/// Stable-read token-table view used by release-time reference capture. The
/// same version must bracket every field; only nonzero even versions are ready.
struct alignas(8) ConSanMoiInlineCausalTokenView {
  uint32_t version_before = 0;
  uint32_t version_after = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_key = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
  ConSanMoiInlineTokenEvidenceKind kind = ConSanMoiInlineTokenEvidenceKind::Direct;
  uint64_t source_release_address = 0;
  uint32_t source_release_version = 0;
  uint32_t consumer_epoch_plus_one = 0;
  uint32_t reservation_version = 0;
};
/// Captures the exact causal frontier owned by `releaser_owner_id`. Unrelated
/// ready tokens are ignored. Multiple valid witnesses for the same ancestor
/// coalesce to its maximum epoch. Any malformed table entry, cyclic ancestry,
/// incomplete scan, or fifth distinct matching ancestor marks the complete
/// snapshot nonauthorizing.
[[nodiscard]] constexpr ConSanMoiInlineCausalSnapshot
consan_moi_inline_capture_causal_snapshot(std::span<const ConSanMoiInlineCausalTokenView> tokens,
                                          uint64_t dispatch_id, uint32_t workgroup_key,
                                          uint32_t releaser_owner_id, bool source_complete = true) {
  ConSanMoiInlineCausalSnapshot snapshot;
  if (!source_complete)
    snapshot.flags |=
        consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::SourceIncomplete);
  if (dispatch_id == 0 || workgroup_key == 0 || releaser_owner_id == 0)
    snapshot.flags |=
        consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);

  for (const auto &token : tokens) {
    if (token.version_before == 0 && token.version_after == 0 && token.dispatch_id == 0 &&
        token.workgroup_key == 0 && token.consumer_owner_id == 0 && token.producer_owner_id == 0 &&
        token.producer_epoch_plus_one == 0 && token.source_release_address == 0 &&
        token.source_release_version == 0 &&
        token.kind == ConSanMoiInlineTokenEvidenceKind::Direct &&
        token.consumer_epoch_plus_one == 0 && token.reservation_version == 0)
      continue;
    if (token.version_before != token.version_after || token.version_before == 0 ||
        (token.version_before & 1u) != 0 || token.dispatch_id == 0 || token.workgroup_key == 0 ||
        token.consumer_owner_id == 0 || token.producer_owner_id == 0 ||
        !consan_moi_inline_causal_epoch_is_valid(token.producer_epoch_plus_one) ||
        (token.kind != ConSanMoiInlineTokenEvidenceKind::Direct &&
         token.kind != ConSanMoiInlineTokenEvidenceKind::Inherited &&
         token.kind != ConSanMoiInlineTokenEvidenceKind::ReleaseSequence) ||
        token.source_release_address == 0 ||
        !consan_moi_inline_release_version_is_ready(token.source_release_version) ||
        !consan_moi_inline_causal_epoch_is_valid(token.consumer_epoch_plus_one) ||
        token.reservation_version != 0) {
      snapshot.flags |=
          consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);
      continue;
    }
    if (token.dispatch_id != dispatch_id || token.workgroup_key != workgroup_key ||
        token.consumer_owner_id != releaser_owner_id)
      continue;
    if (token.producer_owner_id == releaser_owner_id) {
      snapshot.flags |=
          consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed);
      continue;
    }

    uint32_t insertion = 0;
    while (insertion < snapshot.entry_count &&
           insertion < kConSanMoiInlineCausalSnapshotEntryCapacity &&
           snapshot.entries[insertion].ancestor_owner_id < token.producer_owner_id)
      ++insertion;
    if (insertion < snapshot.entry_count &&
        snapshot.entries[insertion].ancestor_owner_id == token.producer_owner_id) {
      snapshot.entries[insertion].ancestor_epoch_plus_one = std::max(
          snapshot.entries[insertion].ancestor_epoch_plus_one, token.producer_epoch_plus_one);
      continue;
    }
    if (snapshot.entry_count >= kConSanMoiInlineCausalSnapshotEntryCapacity) {
      snapshot.flags |= consan_moi_inline_causal_snapshot_flag(
          ConSanMoiInlineCausalSnapshotFlag::CapacityOverflow);
      continue;
    }
    for (uint32_t move = snapshot.entry_count; move > insertion; --move)
      snapshot.entries[move] = snapshot.entries[move - 1u];
    snapshot.entries[insertion] = {token.producer_owner_id, token.producer_epoch_plus_one};
    ++snapshot.entry_count;
  }
  return snapshot;
}
struct ConSanMoiInlineStableReleaseEvidence {
  ConSanMoiInlineVersionedReleaseIdentity identity;
  uint32_t version_before = 0;
  uint32_t version_after = 0;
  uint32_t releaser_owner_id = 0;
  uint32_t releaser_epoch_plus_one = 0;
  ConSanMoiInlineCausalSnapshot snapshot;
};

struct ConSanMoiInlineTokenEvidence {
  uint64_t dispatch_id = 0;
  uint32_t workgroup_key = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
  ConSanMoiInlineTokenEvidenceKind kind = ConSanMoiInlineTokenEvidenceKind::Direct;
  ConSanMoiInlineVersionedReleaseIdentity source_release;
  uint32_t source_release_version = 0;
};

struct ConSanMoiInlineAccessEvidence {
  uint64_t dispatch_id = 0;
  uint32_t workgroup_key = 0;
  uint32_t owner_id = 0;
  uint32_t access_count = 0;
};

struct ConSanMoiInlineEvidenceCounters {
  uint64_t dispatch_id = 0;
  uint32_t dispatch_coverage_count = 0;
  uint32_t undercoverage_count = 0;
  uint32_t overflow_count = 0;
  uint32_t unsupported_count = 0;
  uint32_t consan_diagnostic_count = 0;
  uint32_t matched_source_diagnostic_count = 0;
};

enum class ConSanMoiInlineQualificationSemantics : uint8_t {
  Ordered,
  NativeRelaxed,
};

struct ConSanMoiInlineQualificationExpectation {
  ConSanMoiInlineQualificationSemantics semantics = ConSanMoiInlineQualificationSemantics::Ordered;
  ConSanMoiInlineVersionedReleaseIdentity release_identity;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t required_ancestor_owner_id = 0;
  uint32_t required_ancestor_epoch_plus_one = 0;
};

enum class ConSanMoiInlineQualificationFailure : uint32_t {
  None = 0,
  InvalidExpectation = 1u << 0u,
  DispatchCoverage = 1u << 1u,
  Undercoverage = 1u << 2u,
  Overflow = 1u << 3u,
  Unsupported = 1u << 4u,
  ProducerAccess = 1u << 5u,
  ConsumerAccess = 1u << 6u,
  StableRelease = 1u << 7u,
  DirectToken = 1u << 8u,
  InheritedToken = 1u << 9u,
  UnexpectedAuthorization = 1u << 10u,
  MissingSourceDiagnostic = 1u << 11u,
  UnexpectedDiagnostic = 1u << 12u,
  MalformedEvidence = 1u << 13u,
};

struct ConSanMoiInlineQualificationResult {
  uint32_t failures = 0;

  [[nodiscard]] constexpr bool accepted() const { return failures == 0; }
  [[nodiscard]] constexpr bool has(ConSanMoiInlineQualificationFailure failure) const {
    return (failures & static_cast<uint32_t>(failure)) != 0;
  }
};

[[nodiscard]] constexpr ConSanMoiInlineQualificationResult consan_moi_inline_qualify_token_evidence(
    const ConSanMoiInlineQualificationExpectation &expected,
    std::span<const ConSanMoiInlineStableReleaseEvidence> releases,
    std::span<const ConSanMoiInlineTokenEvidence> tokens,
    std::span<const ConSanMoiInlineAccessEvidence> accesses,
    const ConSanMoiInlineEvidenceCounters &counters) {
  ConSanMoiInlineQualificationResult result;
  const auto fail = [&](ConSanMoiInlineQualificationFailure failure) {
    result.failures |= static_cast<uint32_t>(failure);
  };
  const bool requires_ancestor = expected.required_ancestor_owner_id != 0;
  if (!expected.release_identity.valid() || expected.producer_owner_id == 0 ||
      !consan_moi_inline_causal_epoch_is_valid(expected.producer_epoch_plus_one) ||
      expected.consumer_owner_id == 0 || expected.consumer_owner_id == expected.producer_owner_id ||
      (requires_ancestor != (expected.required_ancestor_epoch_plus_one != 0)) ||
      (requires_ancestor &&
       (expected.required_ancestor_owner_id == expected.producer_owner_id ||
        expected.required_ancestor_owner_id == expected.consumer_owner_id ||
        !consan_moi_inline_causal_epoch_is_valid(expected.required_ancestor_epoch_plus_one))))
    fail(ConSanMoiInlineQualificationFailure::InvalidExpectation);

  if (counters.dispatch_id != expected.release_identity.dispatch_id ||
      counters.dispatch_coverage_count == 0)
    fail(ConSanMoiInlineQualificationFailure::DispatchCoverage);
  if (counters.undercoverage_count != 0)
    fail(ConSanMoiInlineQualificationFailure::Undercoverage);
  if (counters.overflow_count != 0)
    fail(ConSanMoiInlineQualificationFailure::Overflow);
  if (counters.unsupported_count != 0)
    fail(ConSanMoiInlineQualificationFailure::Unsupported);

  for (const auto &access : accesses) {
    if (access.dispatch_id == expected.release_identity.dispatch_id &&
        access.workgroup_key == expected.release_identity.workgroup_key &&
        (access.owner_id == 0 || access.access_count == 0))
      fail(ConSanMoiInlineQualificationFailure::MalformedEvidence);
  }
  const auto has_access = [&](uint32_t owner_id) {
    return std::any_of(accesses.begin(), accesses.end(), [&](const auto &access) {
      return access.dispatch_id == expected.release_identity.dispatch_id &&
             access.workgroup_key == expected.release_identity.workgroup_key &&
             access.owner_id == owner_id && access.access_count != 0;
    });
  };
  if (!has_access(expected.producer_owner_id))
    fail(ConSanMoiInlineQualificationFailure::ProducerAccess);
  if (!has_access(expected.consumer_owner_id))
    fail(ConSanMoiInlineQualificationFailure::ConsumerAccess);

  const ConSanMoiInlineStableReleaseEvidence *matching_release = nullptr;
  uint32_t relevant_release_count = 0;
  for (const auto &release : releases) {
    if (release.identity.dispatch_id != expected.release_identity.dispatch_id ||
        release.identity.workgroup_key != expected.release_identity.workgroup_key)
      continue;
    ++relevant_release_count;
    const bool structurally_valid =
        release.identity.valid() &&
        consan_moi_inline_release_snapshot_is_stable(release.version_before,
                                                     release.version_after) &&
        release.releaser_owner_id != 0 &&
        consan_moi_inline_causal_epoch_is_valid(release.releaser_epoch_plus_one) &&
        consan_moi_inline_validate_causal_snapshot(release.snapshot, release.releaser_owner_id) ==
            ConSanMoiInlineCausalSnapshotStatus::Usable;
    if (!structurally_valid) {
      fail(ConSanMoiInlineQualificationFailure::MalformedEvidence);
      continue;
    }
    if (release.identity == expected.release_identity &&
        release.releaser_owner_id == expected.producer_owner_id &&
        release.releaser_epoch_plus_one == expected.producer_epoch_plus_one) {
      if (matching_release != nullptr)
        fail(ConSanMoiInlineQualificationFailure::MalformedEvidence);
      matching_release = &release;
    }
  }

  uint32_t relevant_token_count = 0;
  uint32_t direct_token_count = 0;
  uint32_t inherited_token_count = 0;
  for (const auto &token : tokens) {
    if (token.dispatch_id != expected.release_identity.dispatch_id ||
        token.workgroup_key != expected.release_identity.workgroup_key)
      continue;
    ++relevant_token_count;
    const bool structurally_valid =
        token.consumer_owner_id != 0 && token.producer_owner_id != 0 &&
        token.consumer_owner_id != token.producer_owner_id &&
        consan_moi_inline_causal_epoch_is_valid(token.producer_epoch_plus_one) &&
        (token.kind == ConSanMoiInlineTokenEvidenceKind::Direct ||
         token.kind == ConSanMoiInlineTokenEvidenceKind::Inherited ||
         token.kind == ConSanMoiInlineTokenEvidenceKind::ReleaseSequence) &&
        token.source_release.valid() && token.source_release.dispatch_id == token.dispatch_id &&
        token.source_release.workgroup_key == token.workgroup_key &&
        consan_moi_inline_release_version_is_ready(token.source_release_version);
    if (!structurally_valid) {
      fail(ConSanMoiInlineQualificationFailure::MalformedEvidence);
      continue;
    }
    const bool expected_source = matching_release != nullptr &&
                                 token.source_release == matching_release->identity &&
                                 token.source_release_version == matching_release->version_after;
    if (expected_source && token.kind == ConSanMoiInlineTokenEvidenceKind::Direct &&
        token.consumer_owner_id == expected.consumer_owner_id &&
        token.producer_owner_id == expected.producer_owner_id &&
        token.producer_epoch_plus_one == expected.producer_epoch_plus_one)
      ++direct_token_count;
    if (expected_source && token.kind == ConSanMoiInlineTokenEvidenceKind::Inherited &&
        token.consumer_owner_id == expected.consumer_owner_id && requires_ancestor &&
        token.producer_owner_id == expected.required_ancestor_owner_id &&
        token.producer_epoch_plus_one == expected.required_ancestor_epoch_plus_one)
      ++inherited_token_count;
  }

  if (expected.semantics == ConSanMoiInlineQualificationSemantics::NativeRelaxed) {
    if (relevant_release_count != 0 || relevant_token_count != 0)
      fail(ConSanMoiInlineQualificationFailure::UnexpectedAuthorization);
    if (counters.matched_source_diagnostic_count == 0)
      fail(ConSanMoiInlineQualificationFailure::MissingSourceDiagnostic);
    if (counters.consan_diagnostic_count != counters.matched_source_diagnostic_count)
      fail(ConSanMoiInlineQualificationFailure::UnexpectedDiagnostic);
    return result;
  }

  if (matching_release == nullptr)
    fail(ConSanMoiInlineQualificationFailure::StableRelease);
  if (direct_token_count != 1)
    fail(ConSanMoiInlineQualificationFailure::DirectToken);
  if (requires_ancestor) {
    bool snapshot_contains_ancestor = false;
    if (matching_release != nullptr) {
      snapshot_contains_ancestor = std::any_of(
          matching_release->snapshot.entries.begin(),
          matching_release->snapshot.entries.begin() + matching_release->snapshot.entry_count,
          [&](const auto &entry) {
            return entry.ancestor_owner_id == expected.required_ancestor_owner_id &&
                   entry.ancestor_epoch_plus_one == expected.required_ancestor_epoch_plus_one;
          });
    }
    if (!snapshot_contains_ancestor || inherited_token_count != 1)
      fail(ConSanMoiInlineQualificationFailure::InheritedToken);
  }
  if (counters.consan_diagnostic_count != 0 || counters.matched_source_diagnostic_count != 0)
    fail(ConSanMoiInlineQualificationFailure::UnexpectedDiagnostic);
  return result;
}

struct ConSanMoiInlineCausalImportEntry {
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
};

enum class ConSanMoiInlineCausalImportStatus : uint8_t {
  Usable,
  InvalidConsumer,
  UnstableRelease,
  IdentityMismatch,
  CapacityOverflow,
  SourceIncomplete,
  MalformedSnapshot,
  DestinationCollision,
  DestinationCapacityExhausted,
};

struct ConSanMoiInlineCausalImportPlan {
  ConSanMoiInlineCausalImportStatus status = ConSanMoiInlineCausalImportStatus::MalformedSnapshot;
  uint32_t entry_count = 0;
  std::array<ConSanMoiInlineCausalImportEntry, kConSanMoiInlineCausalSnapshotEntryCapacity + 1u>
      entries{};

  [[nodiscard]] constexpr bool authoritative() const {
    return status == ConSanMoiInlineCausalImportStatus::Usable;
  }
};

/// Produces direct `(consumer,releaser)` first, then immutable inherited
/// `(consumer,ancestor)` tokens. The caller must preflight every destination
/// and pass collision/capacity state here before publishing any token; any
/// failure returns zero authorizing entries and poisons suppression for the
/// dispatch.
[[nodiscard]] constexpr ConSanMoiInlineCausalImportPlan consan_moi_inline_plan_causal_import(
    const ConSanMoiInlineStableReleaseSnapshot &release,
    const ConSanMoiInlineVersionedReleaseIdentity &expected_identity, uint32_t consumer_owner_id,
    bool destination_collision = false, bool destination_capacity_exhausted = false) {
  ConSanMoiInlineCausalImportPlan result;
  if (consumer_owner_id == 0) {
    result.status = ConSanMoiInlineCausalImportStatus::InvalidConsumer;
    return result;
  }
  if (!consan_moi_inline_release_snapshot_is_stable(release.version_before,
                                                    release.version_after)) {
    result.status = ConSanMoiInlineCausalImportStatus::UnstableRelease;
    return result;
  }
  if (!expected_identity.valid() || release.identity != expected_identity) {
    result.status = ConSanMoiInlineCausalImportStatus::IdentityMismatch;
    return result;
  }
  if (release.releaser_owner_id == 0 ||
      !consan_moi_inline_causal_epoch_is_valid(release.releaser_epoch_plus_one)) {
    result.status = ConSanMoiInlineCausalImportStatus::MalformedSnapshot;
    return result;
  }
  switch (consan_moi_inline_validate_causal_snapshot(release.snapshot, release.releaser_owner_id)) {
  case ConSanMoiInlineCausalSnapshotStatus::CapacityOverflow:
    result.status = ConSanMoiInlineCausalImportStatus::CapacityOverflow;
    return result;
  case ConSanMoiInlineCausalSnapshotStatus::SourceIncomplete:
    result.status = ConSanMoiInlineCausalImportStatus::SourceIncomplete;
    return result;
  case ConSanMoiInlineCausalSnapshotStatus::Malformed:
    result.status = ConSanMoiInlineCausalImportStatus::MalformedSnapshot;
    return result;
  case ConSanMoiInlineCausalSnapshotStatus::Usable:
    break;
  }
  if (destination_capacity_exhausted) {
    result.status = ConSanMoiInlineCausalImportStatus::DestinationCapacityExhausted;
    return result;
  }
  if (destination_collision) {
    result.status = ConSanMoiInlineCausalImportStatus::DestinationCollision;
    return result;
  }
  result.status = ConSanMoiInlineCausalImportStatus::Usable;
  if (release.releaser_owner_id != consumer_owner_id)
    result.entries[result.entry_count++] = {release.releaser_owner_id,
                                            release.releaser_epoch_plus_one};
  for (uint32_t i = 0; i < release.snapshot.entry_count; ++i) {
    const auto &entry = release.snapshot.entries[i];
    if (entry.ancestor_owner_id == consumer_owner_id)
      continue;
    result.entries[result.entry_count++] = {entry.ancestor_owner_id, entry.ancestor_epoch_plus_one};
  }
  return result;
}

struct ConSanMoiInlineAcquiredEpochTokenPublishResult {
  bool updated = false;
  bool collision = false;
  bool invalid_identity = false;
};

/// Reference publication contract for one already-indexed direct-mapped slot.
/// Exact updates are monotonic; a collision or invalid identity leaves the
/// slot untouched so it can never manufacture acquired order.
[[nodiscard]] constexpr ConSanMoiInlineAcquiredEpochTokenPublishResult
consan_moi_inline_publish_acquired_epoch_token(ConSanMoiInlineAcquiredEpochTokenSlot &slot,
                                               uint32_t workgroup_key, uint32_t consumer_owner_id,
                                               uint32_t producer_owner_id, uint32_t producer_epoch,
                                               uint32_t consumer_epoch, uint64_t dispatch_id,
                                               ConSanMoiInlineTokenEvidenceKind kind,
                                               uint64_t source_release_address,
                                               uint32_t source_release_version) {
  ConSanMoiInlineAcquiredEpochTokenPublishResult result;
  if (workgroup_key == 0 || consumer_owner_id == 0 || producer_owner_id == 0 ||
      consumer_owner_id == producer_owner_id || dispatch_id == 0 || source_release_address == 0 ||
      !consan_moi_inline_release_version_is_ready(source_release_version) ||
      consumer_epoch >= consan_moi_exact_shadow::max_epoch ||
      (kind != ConSanMoiInlineTokenEvidenceKind::Direct &&
       kind != ConSanMoiInlineTokenEvidenceKind::Inherited &&
       kind != ConSanMoiInlineTokenEvidenceKind::ReleaseSequence)) {
    result.invalid_identity = true;
    return result;
  }
  const uint32_t token_value = consan_moi_inline_acquired_epoch_token_value(producer_epoch);
  const uint32_t consumer_token_value =
      consan_moi_inline_acquired_epoch_token_value(consumer_epoch);
  const auto lookup = consan_moi_inline_acquired_epoch_token_lookup(
      slot, workgroup_key, consumer_owner_id, producer_owner_id);
  if (lookup == ConSanMoiInlineAcquiredEpochTokenLookup::Collision) {
    result.collision = true;
    return result;
  }
  if (lookup == ConSanMoiInlineAcquiredEpochTokenLookup::Exact &&
      slot.consumer_epoch_plus_one != consumer_token_value) {
    result.collision = true;
    return result;
  }
  if (lookup == ConSanMoiInlineAcquiredEpochTokenLookup::Exact &&
      slot.producer_epoch_plus_one >= token_value &&
      slot.consumer_epoch_plus_one >= consumer_token_value)
    return result;
  slot = {.version = 2,
          .consumer_owner_id = consumer_owner_id,
          .producer_owner_id = producer_owner_id,
          .producer_epoch_plus_one = token_value,
          .workgroup_key = workgroup_key,
          .kind = static_cast<uint32_t>(kind),
          .dispatch_id = dispatch_id,
          .source_release_address = source_release_address,
          .source_release_version = source_release_version,
          .consumer_epoch_plus_one = consumer_token_value};
  result.updated = true;
  return result;
}

[[nodiscard]] constexpr bool
consan_moi_inline_acquired_epoch_orders(const ConSanMoiInlineAcquiredEpochTokenSlot &slot,
                                        const ConSanMoiExactShadowEntry &current,
                                        const ConSanMoiExactShadowEntry &prior) {
  if (current.generation == 0 || current.generation != prior.generation)
    return false;
  if (consan_moi_inline_acquired_epoch_token_lookup(slot, current.generation, current.owner_id,
                                                    prior.owner_id) !=
      ConSanMoiInlineAcquiredEpochTokenLookup::Exact)
    return false;
  return current.epoch < consan_moi_exact_shadow::max_epoch &&
         slot.consumer_epoch_plus_one <=
             consan_moi_inline_acquired_epoch_token_value(current.epoch) &&
         slot.producer_epoch_plus_one >= consan_moi_inline_acquired_epoch_token_value(prior.epoch);
}

[[nodiscard]] constexpr bool consan_moi_inline_acquired_epoch_orders_pair(
    const ConSanMoiInlineAcquiredEpochTokenSlot &current_after_prior,
    const ConSanMoiInlineAcquiredEpochTokenSlot &prior_after_current,
    const ConSanMoiExactShadowEntry &current, const ConSanMoiExactShadowEntry &prior) {
  return consan_moi_inline_acquired_epoch_orders(current_after_prior, current, prior) ||
         consan_moi_inline_acquired_epoch_orders(prior_after_current, prior, current);
}

[[nodiscard]] constexpr bool consan_moi_inline_stable_token_orders(
    const ConSanMoiInlineAcquiredTokenSnapshot &token_words,
    const ConSanMoiInlineReleaseSnapshotWords & /*source_words*/, uint64_t dispatch_id,
    const ConSanMoiExactShadowEntry &current, const ConSanMoiExactShadowEntry &prior) {
  const auto classified_token = consan_moi_inline_classify_acquired_token(token_words);
  if (classified_token.state != ConSanMoiInlineAcquiredTokenState::Stable || dispatch_id == 0 ||
      current.generation == 0 || current.generation != prior.generation)
    return false;
  const auto &token = classified_token.token;
  if (token.dispatch_id != dispatch_id || token.workgroup_key != current.generation ||
      token.consumer_owner_id != current.owner_id || token.producer_owner_id != prior.owner_id ||
      current.epoch >= consan_moi_exact_shadow::max_epoch ||
      token.consumer_epoch_plus_one > consan_moi_inline_acquired_epoch_token_value(current.epoch) ||
      token.producer_epoch_plus_one < consan_moi_inline_acquired_epoch_token_value(prior.epoch))
    return false;
  return token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct) ||
         token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited);
}

} // namespace rocjitsu
