// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_PATCH_CONSAN_SYNC_EVENT_INDEX_H
#define ROCJITSU_CODE_PATCH_CONSAN_SYNC_EVENT_INDEX_H

#include "rocjitsu/code/patch/consan/consan.h"

#include <span>
#include <vector>

namespace rocjitsu::consan {

[[nodiscard]] std::vector<SyncSequenceMembership>
build_sync_sequence_membership_index(std::span<const SyncSequence> sequences, size_t event_count);

/// Verify that every declared sequence member resolves to the same ordered,
/// bounded event in the immutable program inventory.
[[nodiscard]] bool sequence_has_exact_members(const SynchronizationInventoryView &inventory,
                                              const SyncSequence &sequence);

} // namespace rocjitsu::consan

#endif // ROCJITSU_CODE_PATCH_CONSAN_SYNC_EVENT_INDEX_H
