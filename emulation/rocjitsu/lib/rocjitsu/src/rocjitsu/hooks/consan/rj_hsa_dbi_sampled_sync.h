// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_HOOKS_RJ_HSA_DBI_SAMPLED_SYNC_H_
#define ROCJITSU_HOOKS_RJ_HSA_DBI_SAMPLED_SYNC_H_

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <cstdint>

namespace rocjitsu {

struct ConSanMoiSampledPendingJoinResult {
  ConSanMoiSampledPendingAcquireState state = ConSanMoiSampledPendingAcquireState::Empty;
  ConSanMoiSampledSyncDecodeResult sync;
};

/// Joins a stable deferred-acquire snapshot to its later stable access window.
/// The pending record retains the source epoch; ordinary per-window metadata
/// uses the later window epoch after every retained identity field matches.
[[nodiscard]] inline ConSanMoiSampledPendingJoinResult
consan_moi_sampled_join_pending_acquire(const ConSanMoiSampledPendingAcquireView &pending,
                                        const ConSanMoiSampledCausalWindow &window,
                                        uint64_t packed_watchpoint, uint32_t selected_slot) {
  const ConSanMoiSampledWatchpointEntry watchpoint =
      decode_consan_moi_sampled_watchpoint_entry(packed_watchpoint);
  const ConSanMoiSampledPendingAcquireKey key{
      .generation = window.generation,
      .dispatch_id = window.dispatch_id,
      .workgroup_x = window.workgroup_x,
      .workgroup_y = window.workgroup_y,
      .workgroup_z = window.workgroup_z,
      .owner_id = watchpoint.owner_id,
      .source_epoch = pending.slot.source_epoch,
      .selected_slot = selected_slot,
  };
  const ConSanMoiSampledPendingAcquireState state =
      classify_consan_moi_sampled_pending_acquire(pending, key, window.epoch);
  if (state != ConSanMoiSampledPendingAcquireState::Ready)
    return {.state = state, .sync = {}};
  if (!consan_moi_sampled_atomic_attachment_matches(window, packed_watchpoint, selected_slot,
                                                    {.generation = window.generation,
                                                     .dispatch_id = window.dispatch_id,
                                                     .workgroup_x = window.workgroup_x,
                                                     .workgroup_y = window.workgroup_y,
                                                     .workgroup_z = window.workgroup_z,
                                                     .epoch = window.epoch,
                                                     .owner_id = watchpoint.owner_id}))
    return {.state = ConSanMoiSampledPendingAcquireState::IdentityMismatch, .sync = {}};

  ConSanMoiSampledSyncDecodeResult decoded =
      decode_consan_moi_sampled_sync_metadata(pending.slot.metadata);
  decoded.metadata.epoch_before = window.epoch;
  decoded.metadata.epoch_after = window.epoch;
  return {.state = state, .sync = decoded};
}

/// ABI v1 reports collision/capacity failures only as aggregate counters. A
/// nonzero counter cannot be attributed away from a candidate pair, so no
/// ordering suppression is sound for that report.
[[nodiscard]] inline bool consan_moi_sampled_sync_report_is_complete(
    uint32_t dropped_window_count, uint32_t unsupported_sync_count, uint32_t malformed_sync_count,
    uint32_t pending_acquire_collision_count = 0u, uint32_t pending_acquire_malformed_count = 0u) {
  return dropped_window_count == 0u && unsupported_sync_count == 0u && malformed_sync_count == 0u &&
         pending_acquire_collision_count == 0u && pending_acquire_malformed_count == 0u;
}

/// Returns true only when two stable, per-window atomic metadata records prove
/// a release-to-acquire ordering candidate for accesses in the same workgroup.
/// The caller remains responsible for establishing that the two windows belong
/// to that same workgroup. Slot order is not event order, so either argument
/// may carry the statically associated release or acquire side.
/// Correctness also relies on the device publisher's contract that a Release
/// record is attached only to an access before that edge and an Acquire record
/// only to an access after that edge; this host predicate cannot reconstruct
/// program order from hashed slot indices.
///
/// Wavefront scope is intentionally rejected: Sampled ABI v1 does not retain
/// enough wave identity to prove that two owners share a wave. Exact range
/// equality avoids composing unrelated or merely overlapping atomic objects.
[[nodiscard]] inline bool consan_moi_sampled_atomic_pair_orders_same_workgroup(
    const ConSanMoiSampledSyncDecodeResult &first, const ConSanMoiSampledSyncDecodeResult &second) {
  using Classification = ConSanMoiSampledSyncClassification;
  using Kind = ConSanMoiSampledSyncKind;
  using Outcome = ConSanMoiSampledSyncOutcome;
  using Role = ConSanMoiSampledSyncRole;
  using Scope = ConSanMoiSampledSyncScope;

  if (first.classification != Classification::Valid ||
      second.classification != Classification::Valid)
    return false;
  const ConSanMoiSampledSyncMetadata &first_metadata = first.metadata;
  const ConSanMoiSampledSyncMetadata &second_metadata = second.metadata;
  if (first_metadata.kind != Kind::Atomic || second_metadata.kind != Kind::Atomic ||
      first_metadata.address != second_metadata.address ||
      first_metadata.byte_count != second_metadata.byte_count ||
      first_metadata.epoch_before != first_metadata.epoch_after ||
      second_metadata.epoch_before != second_metadata.epoch_after ||
      first_metadata.epoch_after != second_metadata.epoch_before)
    return false;

  constexpr uint32_t release_bit = static_cast<uint32_t>(Role::Release);
  constexpr uint32_t acquire_bit = static_cast<uint32_t>(Role::Acquire);
  // The conflict reader has already matched workgroup coordinates. Both
  // scopes must therefore cover at least that workgroup. ABI v1 cannot prove
  // the identity needed to accept wavefront-scoped synchronization.
  if (first_metadata.scope < Scope::Workgroup || second_metadata.scope < Scope::Workgroup)
    return false;

  const auto has_release = [&](const ConSanMoiSampledSyncMetadata &metadata) {
    // A failed compare-exchange performs no release operation.
    return (static_cast<uint32_t>(metadata.role) & release_bit) != 0u &&
           metadata.outcome != Outcome::CasFailure;
  };
  const auto has_acquire = [&](const ConSanMoiSampledSyncMetadata &metadata) {
    // A compare-exchange failure may still perform its permitted acquire
    // ordering, so its outcome does not disqualify the acquire half.
    return (static_cast<uint32_t>(metadata.role) & acquire_bit) != 0u;
  };
  return (has_release(first_metadata) && has_acquire(second_metadata)) ||
         (has_release(second_metadata) && has_acquire(first_metadata));
}

} // namespace rocjitsu

#endif // ROCJITSU_HOOKS_RJ_HSA_DBI_SAMPLED_SYNC_H_
