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

} // namespace rocjitsu
