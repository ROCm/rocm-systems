// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

#include <functional>
#include <string>

namespace rocjitsu {
namespace {

template <typename FindEvent>
[[nodiscard]] bool sequence_has_exact_members_impl(const ConSanSyncSequence &sequence,
                                                   FindEvent find_event) {
  if (sequence.member_semantic_ids.size() != sequence.member_event_identities.size())
    return false;
  uint64_t prior_end = sequence.begin_text_offset;
  for (size_t index = 0; index < sequence.member_semantic_ids.size(); ++index) {
    SemanticSiteId event_identity = sequence.member_semantic_ids[index];
    event_identity.domain = ConSanSemanticSiteDomain::SynchronizationEvent;
    const ConSanSyncEvent *event = find_event(event_identity);
    if (event == nullptr || event->container_name != sequence.container_name ||
        event->identity != sequence.member_event_identities[index] ||
        event->in_kernel != sequence.in_kernel || event->text_offset < prior_end ||
        event->text_offset < sequence.begin_text_offset ||
        event->text_offset + event->size > sequence.end_text_offset) {
      return false;
    }
    prior_end = event->text_offset + event->size;
  }
  return !sequence.member_semantic_ids.empty();
}

} // namespace

size_t SyncEventSemanticIdHash::operator()(const SemanticSiteId &identity) const noexcept {
  size_t hash = std::hash<std::string>{}(identity.physical.code_object.fingerprint);
  const auto mix = [&](uint64_t value) {
    hash ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  mix(identity.physical.code_object.byte_size);
  mix(identity.physical.code_object.collision_verifier);
  mix(identity.physical.original_text_offset);
  mix(static_cast<uint8_t>(identity.domain));
  mix(identity.member_ordinal);
  mix(identity.range_ordinal);
  return hash;
}

SyncEventSemanticIndex
build_sync_event_semantic_index(std::span<const ConSanSyncEvent> sync_events) {
  SyncEventSemanticIndex index;
  index.reserve(sync_events.size());
  for (const ConSanSyncEvent &event : sync_events)
    index.emplace(event.semantic_id, &event);
  return index;
}

const ConSanSyncEvent *find_sequence_member_event(const SyncEventSemanticIndex &index,
                                                  SemanticSiteId identity) {
  identity.domain = ConSanSemanticSiteDomain::SynchronizationEvent;
  const auto event = index.find(identity);
  return event == index.end() ? nullptr : event->second;
}

bool sequence_has_exact_members(const SynchronizationInventoryView &inventory,
                                const ConSanSyncSequence &sequence) {
  return sequence_has_exact_members_impl(
      sequence, [&](SemanticSiteId identity) { return inventory.find_event(identity); });
}

bool sequence_has_exact_members(const SyncEventSemanticIndex &events,
                                const ConSanSyncSequence &sequence) {
  return sequence_has_exact_members_impl(sequence, [&](SemanticSiteId identity) {
    return find_sequence_member_event(events, identity);
  });
}

} // namespace rocjitsu
