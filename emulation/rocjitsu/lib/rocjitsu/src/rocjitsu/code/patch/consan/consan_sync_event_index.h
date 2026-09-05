// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_PATCH_CONSAN_SYNC_EVENT_INDEX_H
#define ROCJITSU_CODE_PATCH_CONSAN_SYNC_EVENT_INDEX_H

#include "rocjitsu/code/patch/consan/consan.h"

#include <span>
#include <vector>

namespace rocjitsu {

/// Unique logical sequence membership for one event. A null sequence with an
/// ambiguous flag is distinct from a missing map entry so policy can retain
/// the reason that a graph edge was rejected.
struct SyncSequenceMembership {
  const ConSanSyncSequence *sequence = nullptr;
  bool ambiguous = false;
};

using SyncSequenceMembershipIndex = std::vector<SyncSequenceMembership>;

[[nodiscard]] SyncSequenceMembershipIndex
build_sync_sequence_membership_index(std::span<const ConSanSyncSequence> sequences,
                                     size_t event_count);

[[nodiscard]] const SyncSequenceMembership *
find_sync_sequence_membership(const SyncSequenceMembershipIndex &index, ConSanSyncEventId event);

/// Verify that every declared sequence member resolves to the same ordered,
/// bounded event in the immutable program inventory.
[[nodiscard]] bool sequence_has_exact_members(const SynchronizationInventoryView &inventory,
                                              const ConSanSyncSequence &sequence);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_PATCH_CONSAN_SYNC_EVENT_INDEX_H
