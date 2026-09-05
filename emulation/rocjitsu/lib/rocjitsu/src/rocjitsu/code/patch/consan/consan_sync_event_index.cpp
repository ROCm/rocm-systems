// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

#include <functional>
#include <string>

namespace rocjitsu {
namespace {

template <typename FindEvent, typename FindSource>
[[nodiscard]] bool sequence_has_exact_members_impl(const ConSanSyncSequence &sequence,
                                                   FindEvent find_event, FindSource find_source) {
  uint64_t prior_end = sequence.begin_text_offset;
  for (const SemanticSiteId &member_identity : sequence.member_semantic_ids) {
    const SemanticSiteId event_identity =
        member_identity.in_domain(ConSanSemanticSiteDomain::SynchronizationEvent);
    const ConSanSyncEvent *event = find_event(event_identity);
    const ConSanProgramSite *source = event == nullptr ? nullptr : find_source(*event);
    if (event == nullptr || event->container_name != sequence.container_name || source == nullptr ||
        event->in_kernel != sequence.in_kernel || event->text_offset() < prior_end ||
        event->text_offset() < sequence.begin_text_offset ||
        event->text_offset() + source->size() > sequence.end_text_offset) {
      return false;
    }
    prior_end = event->text_offset() + source->size();
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
build_sync_event_semantic_index(std::span<const ConSanSyncEvent> sync_events,
                                std::span<const ConSanProgramSite> program_sites) {
  SyncEventSemanticIndex index;
  index.events.reserve(sync_events.size());
  index.program_sites = program_sites;
  for (const ConSanSyncEvent &event : sync_events) {
    const auto [entry, inserted] = index.events.emplace(event.semantic_id, &event);
    if (!inserted)
      entry->second = nullptr;
  }
  return index;
}

const ConSanSyncEvent *find_sequence_member_event(const SyncEventSemanticIndex &index,
                                                  SemanticSiteId identity) {
  identity = identity.in_domain(ConSanSemanticSiteDomain::SynchronizationEvent);
  const auto event = index.events.find(identity);
  return event == index.events.end() ? nullptr : event->second;
}

SyncSequenceMembershipIndex
build_sync_sequence_membership_index(std::span<const ConSanSyncSequence> sequences) {
  size_t member_count = 0;
  for (const ConSanSyncSequence &sequence : sequences)
    member_count += sequence.member_semantic_ids.size();
  SyncSequenceMembershipIndex result;
  result.reserve(member_count);
  for (const ConSanSyncSequence &sequence : sequences) {
    for (const SemanticSiteId &member : sequence.member_semantic_ids) {
      const SemanticSiteId event = member.in_domain(ConSanSemanticSiteDomain::SynchronizationEvent);
      const auto [entry, inserted] =
          result.try_emplace(event, SyncSequenceMembership{&sequence, false});
      if (!inserted) {
        entry->second.sequence = nullptr;
        entry->second.ambiguous = true;
      }
    }
  }
  return result;
}

bool sequence_has_exact_members(const SynchronizationInventoryView &inventory,
                                const ConSanSyncSequence &sequence) {
  return sequence_has_exact_members_impl(
      sequence, [&](SemanticSiteId identity) { return inventory.find_event(identity); },
      [&](const ConSanSyncEvent &event) { return inventory.source(event); });
}

bool sequence_has_exact_members(const SyncEventSemanticIndex &events,
                                const ConSanSyncSequence &sequence) {
  return sequence_has_exact_members_impl(
      sequence,
      [&](SemanticSiteId identity) { return find_sequence_member_event(events, identity); },
      [&](const ConSanSyncEvent &event) -> const ConSanProgramSite * {
        return event.source_site.valid() && event.source_site.ordinal < events.program_sites.size()
                   ? &events.program_sites[event.source_site.ordinal]
                   : nullptr;
      });
}

} // namespace rocjitsu
