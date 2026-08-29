// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

#include <functional>
#include <string>

namespace rocjitsu {

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

} // namespace rocjitsu
