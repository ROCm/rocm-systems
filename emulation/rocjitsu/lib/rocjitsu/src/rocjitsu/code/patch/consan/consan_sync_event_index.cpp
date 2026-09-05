// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

namespace rocjitsu {
bool sequence_has_exact_members(const SynchronizationInventoryView &inventory,
                                const ConSanSyncSequence &sequence) {
  uint64_t prior_end = sequence.begin_text_offset;
  for (const ConSanSyncEventId member : sequence.member_event_ids) {
    const ConSanSyncEvent *event = inventory.find_event(member);
    const ConSanProgramSite *source = event == nullptr ? nullptr : inventory.source(*event);
    if (event == nullptr || source == nullptr ||
        source->container.name != sequence.container_name ||
        source->container.is_kernel() != sequence.in_kernel || event->text_offset() < prior_end ||
        event->text_offset() < sequence.begin_text_offset ||
        event->text_offset() + source->size() > sequence.end_text_offset) {
      return false;
    }
    prior_end = event->text_offset() + source->size();
  }
  return !sequence.member_event_ids.empty();
}

SyncSequenceMembershipIndex
build_sync_sequence_membership_index(std::span<const ConSanSyncSequence> sequences,
                                     size_t event_count) {
  SyncSequenceMembershipIndex result(event_count);
  for (const ConSanSyncSequence &sequence : sequences) {
    for (const ConSanSyncEventId member : sequence.member_event_ids) {
      if (!member.valid() || member.ordinal >= result.size())
        continue;
      SyncSequenceMembership &entry = result[member.ordinal];
      if (entry.sequence == nullptr && !entry.ambiguous)
        entry.sequence = &sequence;
      else {
        entry.sequence = nullptr;
        entry.ambiguous = true;
      }
    }
  }
  return result;
}

const SyncSequenceMembership *
find_sync_sequence_membership(const SyncSequenceMembershipIndex &index, ConSanSyncEventId event) {
  return event.valid() && event.ordinal < index.size() ? &index[event.ordinal] : nullptr;
}

} // namespace rocjitsu
