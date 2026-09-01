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

} // namespace rocjitsu
