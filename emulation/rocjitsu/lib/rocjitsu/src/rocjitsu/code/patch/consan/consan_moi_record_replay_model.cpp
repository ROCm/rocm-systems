// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_record_replay_model.cpp
/// @brief Record/Replay host report normalization and runtime analysis.

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <algorithm>
#include <array>
#include <bit>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace rocjitsu {

[[nodiscard]] std::optional<ConSanMoiAtomicOutcome>
consan_moi_record_replay_resolved_atomic_outcome(const ConSanMoiRecordReplayAtomicEvent &record) {
  if (record.operation == ConSanMoiAtomicOperation::Rmw)
    return record.outcome == ConSanMoiAtomicOutcome::NotApplicable
               ? std::optional{ConSanMoiAtomicOutcome::NotApplicable}
               : std::nullopt;
  if (record.operation != ConSanMoiAtomicOperation::CompareExchange)
    return std::nullopt;
  if (record.outcome == ConSanMoiAtomicOutcome::Success ||
      record.outcome == ConSanMoiAtomicOutcome::Failure)
    return record.outcome;
  if (record.outcome != ConSanMoiAtomicOutcome::Unavailable || record.lane_mask == 0 ||
      (record.success_lane_mask & ~record.lane_mask) != 0)
    return std::nullopt;
  if (record.success_lane_mask == 0)
    return ConSanMoiAtomicOutcome::Failure;
  if (record.success_lane_mask == record.lane_mask)
    return ConSanMoiAtomicOutcome::Success;
  // Record/replay ordering is currently wave-owned. A mixed lane outcome
  // cannot be collapsed to a wave release without ordering failed lanes.
  return std::nullopt;
}

bool consan_moi_record_replay_atomic_event_is_unpublished(
    const ConSanMoiRecordReplayAtomicEvent &record) {
  return record.generation == 0 && record.workgroup_x == 0 && record.workgroup_y == 0 &&
         record.workgroup_z == 0 && record.owner_id == 0 && record.atomic_address == 0 &&
         record.instruction_offset == 0 && record.event_index == 0 && record.epoch == 0 &&
         (static_cast<uint32_t>(record.kind) == 0 ||
          record.kind == ConSanMoiAtomicEventKind::Release) &&
         record.scope == 0 && record.semantics == 0 &&
         (static_cast<uint32_t>(record.operation) == 0 ||
          record.operation == ConSanMoiAtomicOperation::Rmw) &&
         record.outcome == ConSanMoiAtomicOutcome::NotApplicable && record.lane_mask == 0 &&
         record.success_lane_mask == 0;
}

bool consan_moi_record_replay_fence_event_is_unpublished(
    const ConSanMoiRecordReplayFenceEvent &record) {
  return record.generation == 0 && record.workgroup_x == 0 && record.workgroup_y == 0 &&
         record.workgroup_z == 0 && record.owner_id == 0 && record.instruction_offset == 0 &&
         record.event_index == 0 && record.epoch == 0 &&
         (static_cast<uint32_t>(record.kind) == 0 ||
          record.kind == ConSanMoiFenceEventKind::Release) &&
         record.scope == 0 && record.semantics == 0 && record.communication_token == 0;
}

bool consan_moi_acquired_epoch_orders(std::span<const ConSanMoiAcquiredEpochToken> tokens,
                                      const ConSanMoiExactShadowEntry &current,
                                      const ConSanMoiExactShadowEntry &prior) {
  const uint32_t required_token = consan_moi_acquired_epoch_token_value(prior.epoch);
  if (required_token == 0)
    return false;
  for (const ConSanMoiAcquiredEpochToken &token : tokens) {
    if (!token.valid)
      continue;
    if (token.generation != current.generation || token.generation != prior.generation)
      continue;
    if (token.consumer_owner_id != current.owner_id || token.producer_owner_id != prior.owner_id)
      continue;
    if (token.producer_epoch_plus_one >= required_token)
      return true;
  }
  return false;
}

namespace {

struct ConSanMoiCausalEpochComponent {
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
};

void merge_causal_epoch_component(std::vector<ConSanMoiCausalEpochComponent> &components,
                                  uint32_t owner_id, uint32_t epoch) {
  for (ConSanMoiCausalEpochComponent &component : components) {
    if (component.owner_id != owner_id)
      continue;
    component.epoch = std::max(component.epoch, epoch);
    return;
  }
  components.push_back({owner_id, epoch});
}

[[nodiscard]] std::vector<ConSanMoiCausalEpochComponent>
collect_causal_epoch_components(std::span<const ConSanMoiAcquiredEpochToken> tokens,
                                uint64_t generation, uint32_t publisher_owner_id,
                                uint32_t publisher_epoch) {
  std::vector<ConSanMoiCausalEpochComponent> components;
  components.reserve(tokens.size() + 1u);
  merge_causal_epoch_component(components, publisher_owner_id, publisher_epoch);
  for (const ConSanMoiAcquiredEpochToken &token : tokens) {
    if (!token.valid || token.generation != generation ||
        token.consumer_owner_id != publisher_owner_id || token.producer_epoch_plus_one == 0)
      continue;
    merge_causal_epoch_component(components, token.producer_owner_id,
                                 token.producer_epoch_plus_one - 1u);
  }
  return components;
}

[[nodiscard]] ConSanMoiAtomicSyncResult
record_acquired_epoch_components(std::span<ConSanMoiAcquiredEpochToken> tokens,
                                 std::span<const ConSanMoiCausalEpochComponent> components,
                                 uint64_t generation, uint32_t consumer_owner_id,
                                 uint32_t acquire_instruction_offset) {
  std::vector<ConSanMoiAcquiredEpochToken *> targets(components.size());
  std::vector<ConSanMoiAcquiredEpochToken *> empty_slots;
  empty_slots.reserve(tokens.size());
  for (ConSanMoiAcquiredEpochToken &token : tokens) {
    if (!token.valid) {
      empty_slots.push_back(&token);
      continue;
    }
    for (size_t index = 0; index < components.size(); ++index) {
      if (token.generation == generation && token.consumer_owner_id == consumer_owner_id &&
          token.producer_owner_id == components[index].owner_id) {
        targets[index] = &token;
        break;
      }
    }
  }

  size_t next_empty_slot = 0;
  for (ConSanMoiAcquiredEpochToken *&target : targets) {
    if (target != nullptr)
      continue;
    if (next_empty_slot == empty_slots.size())
      return {/*metadata_full=*/true, /*updated_record_count=*/0};
    target = empty_slots[next_empty_slot++];
  }

  ConSanMoiAtomicSyncResult result;
  for (size_t index = 0; index < components.size(); ++index) {
    ConSanMoiAcquiredEpochToken &token = *targets[index];
    const uint32_t producer_epoch_plus_one =
        consan_moi_acquired_epoch_token_value(components[index].epoch);
    if (!token.valid) {
      token = ConSanMoiAcquiredEpochToken{
          /*valid=*/true,          generation,
          consumer_owner_id,       components[index].owner_id,
          producer_epoch_plus_one, acquire_instruction_offset,
      };
      ++result.updated_record_count;
      continue;
    }
    if (producer_epoch_plus_one > token.producer_epoch_plus_one) {
      token.producer_epoch_plus_one = producer_epoch_plus_one;
      token.acquire_instruction_offset = acquire_instruction_offset;
      ++result.updated_record_count;
    }
  }
  return result;
}

} // namespace

ConSanMoiAtomicSyncResult
consan_moi_record_replay_atomic_release(std::span<ConSanMoiAtomicReleaseRecord> release_records,
                                        uint64_t generation, uint64_t atomic_address,
                                        uint32_t producer_owner_id, uint32_t producer_epoch,
                                        uint32_t release_instruction_offset) {
  return consan_moi_record_replay_atomic_release(
      release_records, std::span<const ConSanMoiAcquiredEpochToken>{}, generation, atomic_address,
      producer_owner_id, producer_epoch, release_instruction_offset);
}

ConSanMoiAtomicSyncResult consan_moi_record_replay_atomic_release(
    std::span<ConSanMoiAtomicReleaseRecord> release_records,
    std::span<const ConSanMoiAcquiredEpochToken> acquired_epoch_tokens, uint64_t generation,
    uint64_t atomic_address, uint32_t producer_owner_id, uint32_t producer_epoch,
    uint32_t release_instruction_offset) {
  const std::vector<ConSanMoiCausalEpochComponent> components = collect_causal_epoch_components(
      acquired_epoch_tokens, generation, producer_owner_id, producer_epoch);
  std::vector<ConSanMoiAtomicReleaseRecord *> targets(components.size());
  std::vector<ConSanMoiAtomicReleaseRecord *> empty_slots;
  empty_slots.reserve(release_records.size());
  for (ConSanMoiAtomicReleaseRecord &record : release_records) {
    if (!record.valid) {
      empty_slots.push_back(&record);
      continue;
    }
    for (size_t index = 0; index < components.size(); ++index) {
      if (record.generation == generation && record.atomic_address == atomic_address &&
          record.producer_owner_id == components[index].owner_id) {
        targets[index] = &record;
        break;
      }
    }
  }

  size_t next_empty_slot = 0;
  for (ConSanMoiAtomicReleaseRecord *&target : targets) {
    if (target != nullptr)
      continue;
    if (next_empty_slot == empty_slots.size())
      return {/*metadata_full=*/true, /*updated_record_count=*/0};
    target = empty_slots[next_empty_slot++];
  }

  ConSanMoiAtomicSyncResult result;
  for (size_t index = 0; index < components.size(); ++index) {
    ConSanMoiAtomicReleaseRecord &record = *targets[index];
    const ConSanMoiCausalEpochComponent component = components[index];
    if (!record.valid) {
      record = ConSanMoiAtomicReleaseRecord{
          /*valid=*/true,     generation,      atomic_address,
          component.owner_id, component.epoch, release_instruction_offset,
      };
      ++result.updated_record_count;
      continue;
    }
    if (component.epoch > record.producer_epoch) {
      record.producer_epoch = component.epoch;
      record.release_instruction_offset = release_instruction_offset;
      ++result.updated_record_count;
    }
  }
  return result;
}

ConSanMoiAtomicSyncResult consan_moi_record_replay_atomic_acquire(
    std::span<const ConSanMoiAtomicReleaseRecord> release_records,
    std::span<ConSanMoiAcquiredEpochToken> acquired_epoch_tokens, uint64_t generation,
    uint64_t atomic_address, uint32_t consumer_owner_id, uint32_t acquire_instruction_offset) {
  std::vector<ConSanMoiCausalEpochComponent> components;
  components.reserve(release_records.size());
  for (const ConSanMoiAtomicReleaseRecord &release : release_records) {
    if (!release.valid)
      continue;
    if (release.generation != generation || release.atomic_address != atomic_address ||
        release.producer_owner_id == consumer_owner_id)
      continue;
    merge_causal_epoch_component(components, release.producer_owner_id, release.producer_epoch);
  }
  return record_acquired_epoch_components(acquired_epoch_tokens, components, generation,
                                          consumer_owner_id, acquire_instruction_offset);
}

ConSanMoiSparseExactByteShadow::ConSanMoiSparseExactByteShadow(uint64_t byte_capacity,
                                                               uint32_t maximum_access_count)
    : byte_capacity_(byte_capacity), maximum_access_count_(maximum_access_count) {}

void ConSanMoiSparseExactByteShadow::retire_before_epoch(uint64_t generation,
                                                         uint32_t first_live_epoch) {
  for (auto current = cross_owner_intervals_.begin(); current != cross_owner_intervals_.end();) {
    if (current->first.generation == generation && current->first.epoch < first_live_epoch)
      current = cross_owner_intervals_.erase(current);
    else
      ++current;
  }
  for (auto current = same_site_intervals_.begin(); current != same_site_intervals_.end();) {
    if (current->first.epoch_kind.generation == generation &&
        current->first.epoch_kind.epoch < first_live_epoch) {
      current = same_site_intervals_.erase(current);
    } else {
      ++current;
    }
  }

  std::vector<bool> referenced(provenance_.size());
  const auto mark_referenced = [&](const IntervalMap &intervals) {
    for (const auto &[begin, interval] : intervals) {
      (void)begin;
      if (interval.provenance_index != 0 && interval.provenance_index <= provenance_.size())
        referenced[interval.provenance_index - 1u] = true;
    }
  };
  for (const auto &[epoch_kind, owners] : cross_owner_intervals_) {
    (void)epoch_kind;
    for (const auto &[owner_id, intervals] : owners) {
      (void)owner_id;
      mark_referenced(intervals);
    }
  }
  for (const auto &[same_site, lane_groups] : same_site_intervals_) {
    (void)same_site;
    for (const auto &[lane_mask, intervals] : lane_groups) {
      (void)lane_mask;
      mark_referenced(intervals);
    }
  }

  std::vector<uint32_t> remapped_indices(provenance_.size() + 1u);
  std::vector<ConSanMoiExactByteAccess> retained;
  retained.reserve(provenance_.size());
  for (size_t index = 0; index < provenance_.size(); ++index) {
    if (!referenced[index])
      continue;
    remapped_indices[index + 1u] = static_cast<uint32_t>(retained.size()) + 1u;
    retained.push_back(provenance_[index]);
  }
  const auto remap_intervals = [&](IntervalMap &intervals) {
    for (auto &[begin, interval] : intervals) {
      (void)begin;
      interval.provenance_index = remapped_indices[interval.provenance_index];
    }
  };
  for (auto &[epoch_kind, owners] : cross_owner_intervals_) {
    (void)epoch_kind;
    for (auto &[owner_id, intervals] : owners) {
      (void)owner_id;
      remap_intervals(intervals);
    }
  }
  for (auto &[same_site, lane_groups] : same_site_intervals_) {
    (void)same_site;
    for (auto &[lane_mask, intervals] : lane_groups) {
      (void)lane_mask;
      remap_intervals(intervals);
    }
  }
  provenance_ = std::move(retained);
}

ConSanMoiExactByteAccessResult ConSanMoiSparseExactByteShadow::access(
    const ConSanMoiExactByteAccess &current,
    std::span<const ConSanMoiAcquiredEpochToken> acquired_epoch_tokens) {
  ConSanMoiExactByteAccessResult result;
  if (current.lds_byte_count == 0 || consan_moi_shadow_kind_is_empty(current.kind))
    return result;

  const uint64_t byte_end = static_cast<uint64_t>(current.lds_byte_offset) + current.lds_byte_count;
  if (byte_end > byte_capacity_) {
    result.capacity_exhausted = true;
    return result;
  }

  const ConSanMoiExactShadowEntry current_entry{
      current.kind,
      current.owner_id,
      current.epoch,
      static_cast<uint32_t>(current.generation),
      current.instruction_offset,
  };

  std::optional<uint32_t> newest_conflict_index;
  bool provenance_valid = true;
  const auto consider_intervals = [&](const IntervalMap &intervals) {
    auto overlap = intervals.upper_bound(current.lds_byte_offset);
    if (overlap != intervals.begin()) {
      auto prior_interval = std::prev(overlap);
      if (prior_interval->second.end > current.lds_byte_offset)
        overlap = prior_interval;
    }
    for (; overlap != intervals.end() && overlap->first < byte_end; ++overlap) {
      if (overlap->second.end <= current.lds_byte_offset)
        continue;
      const uint32_t prior_index = overlap->second.provenance_index;
      if (prior_index == 0 || prior_index > provenance_.size()) {
        result.capacity_exhausted = true;
        provenance_valid = false;
        return false;
      }
      const ConSanMoiExactByteAccess &prior = provenance_[prior_index - 1u];
      if (!consan_moi_exact_byte_accesses_conflict(current, prior))
        continue;
      const ConSanMoiExactShadowEntry prior_entry{
          prior.kind,
          prior.owner_id,
          prior.epoch,
          static_cast<uint32_t>(prior.generation),
          prior.instruction_offset,
      };
      if (consan_moi_acquired_epoch_orders(acquired_epoch_tokens, current_entry, prior_entry))
        continue;
      if (!newest_conflict_index || prior_index > *newest_conflict_index)
        newest_conflict_index = prior_index;
    }
    return true;
  };
  constexpr std::array access_kinds{
      ConSanMoiShadowAccessKind::Read,
      ConSanMoiShadowAccessKind::Write,
      ConSanMoiShadowAccessKind::ReadWrite,
      ConSanMoiShadowAccessKind::Atomic,
  };
  for (const ConSanMoiShadowAccessKind prior_kind : access_kinds) {
    if (!consan_moi_shadow_kind_conflicts(current.kind, prior_kind))
      continue;
    const EpochKindClass epoch_kind{
        current.generation,
        current.epoch,
        prior_kind,
    };
    const auto prior_owners = cross_owner_intervals_.find(epoch_kind);
    if (prior_owners != cross_owner_intervals_.end()) {
      for (const auto &[owner_id, intervals] : prior_owners->second) {
        if (owner_id == current.owner_id)
          continue;
        if (!consider_intervals(intervals))
          break;
      }
    }
    if (!provenance_valid)
      break;
    if (!current.exact_address_group)
      continue;
    const SameSiteClass same_site{
        epoch_kind,
        current.owner_id,
        current.instruction_offset,
    };
    const auto prior_groups = same_site_intervals_.find(same_site);
    if (prior_groups != same_site_intervals_.end()) {
      for (const auto &[lane_mask, intervals] : prior_groups->second) {
        (void)lane_mask;
        if (!consider_intervals(intervals))
          break;
      }
    }
    if (!provenance_valid)
      break;
  }
  if (newest_conflict_index) {
    result.conflict = true;
    result.prior = provenance_[*newest_conflict_index - 1u];
    return result;
  }
  if (!provenance_valid)
    return result;

  if (provenance_.size() >= maximum_access_count_) {
    result.capacity_exhausted = true;
    return result;
  }
  provenance_.push_back(current);
  const uint32_t provenance_index = static_cast<uint32_t>(provenance_.size());
  const auto update_intervals = [&](IntervalMap &intervals) {
    auto replace_begin = intervals.lower_bound(current.lds_byte_offset);
    if (replace_begin != intervals.begin()) {
      auto prior_interval = std::prev(replace_begin);
      if (prior_interval->second.end > current.lds_byte_offset)
        replace_begin = prior_interval;
    }
    std::optional<std::pair<uint64_t, Interval>> left_remainder;
    std::optional<std::pair<uint64_t, Interval>> right_remainder;
    auto replace_end = replace_begin;
    for (; replace_end != intervals.end() && replace_end->first < byte_end; ++replace_end) {
      if (replace_end->second.end <= current.lds_byte_offset)
        continue;
      if (replace_end->first < current.lds_byte_offset) {
        left_remainder =
            std::pair{replace_end->first,
                      Interval{current.lds_byte_offset, replace_end->second.provenance_index}};
      }
      if (replace_end->second.end > byte_end) {
        right_remainder = std::pair{
            byte_end, Interval{replace_end->second.end, replace_end->second.provenance_index}};
      }
    }
    intervals.erase(replace_begin, replace_end);
    if (left_remainder)
      intervals.insert_or_assign(left_remainder->first, left_remainder->second);
    intervals.insert_or_assign(current.lds_byte_offset, Interval{byte_end, provenance_index});
    if (right_remainder)
      intervals.insert_or_assign(right_remainder->first, right_remainder->second);
  };
  update_intervals(cross_owner_intervals_[EpochKindClass{
      current.generation,
      current.epoch,
      current.kind,
  }][current.owner_id]);
  if (current.exact_address_group) {
    update_intervals(same_site_intervals_[SameSiteClass{
        EpochKindClass{current.generation, current.epoch, current.kind},
        current.owner_id,
        current.instruction_offset,
    }][current.lane_mask]);
  }
  return result;
}

ConSanMoiRecordReplayResult
consan_moi_record_replay_access_records(ConSanMoiReportHeader &header,
                                        std::span<const ConSanMoiAccessRecord> access_records,
                                        std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
                                        std::span<uint64_t> exact_shadow_entries) {
  return consan_moi_record_replay_access_records(header, access_records,
                                                 std::span<const ConSanMoiBarrierRecord>{},
                                                 diagnostic_records, exact_shadow_entries);
}

ConSanMoiRecordReplayResult
consan_moi_record_replay_access_records(ConSanMoiReportHeader &header,
                                        std::span<const ConSanMoiAccessRecord> access_records,
                                        std::span<const ConSanMoiBarrierRecord> barrier_records,
                                        std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
                                        std::span<uint64_t> exact_shadow_entries) {
  return consan_moi_record_replay_access_records(
      header, access_records, barrier_records, std::span<const ConSanMoiRecordReplayAtomicEvent>{},
      diagnostic_records, exact_shadow_entries);
}

ConSanMoiRecordReplayResult consan_moi_record_replay_access_records(
    ConSanMoiReportHeader &header, std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries) {
  return consan_moi_record_replay_access_records(
      header, access_records, barrier_records, atomic_events,
      std::span<const ConSanMoiRecordReplayFenceEvent>{}, diagnostic_records, exact_shadow_entries);
}

ConSanMoiRecordReplayResult consan_moi_record_replay_access_records(
    ConSanMoiReportHeader &header, std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<const ConSanMoiRecordReplayFenceEvent> fence_events,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries) {
  auto span_size_u32 = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };
  auto decode_access_kind = [](uint32_t value) -> std::optional<ConSanMoiShadowAccessKind> {
    switch (static_cast<ConSanMoiShadowAccessKind>(value)) {
    case ConSanMoiShadowAccessKind::Read:
    case ConSanMoiShadowAccessKind::Write:
    case ConSanMoiShadowAccessKind::ReadWrite:
    case ConSanMoiShadowAccessKind::Atomic:
      return static_cast<ConSanMoiShadowAccessKind>(value);
    case ConSanMoiShadowAccessKind::Empty:
      return std::nullopt;
    }
    return std::nullopt;
  };
  ConSanMoiRecordReplayResult replay;
  const uint32_t access_count = std::min({header.access_record_count, header.access_record_capacity,
                                          span_size_u32(access_records.size())});
  replay.published_access_count = static_cast<uint32_t>(
      std::count_if(access_records.begin(), access_records.begin() + access_count,
                    [](const ConSanMoiAccessRecord &record) {
                      return !consan_moi_access_record_is_unpublished(record);
                    }));
  replay.dropped_access_count =
      header.access_record_count > access_count ? header.access_record_count - access_count : 0;
  const uint32_t barrier_count =
      std::min({header.barrier_record_count, header.barrier_record_capacity,
                span_size_u32(barrier_records.size())});
  replay.dropped_barrier_count =
      header.barrier_record_count > barrier_count ? header.barrier_record_count - barrier_count : 0;
  const uint32_t diagnostic_capacity =
      std::min(header.diagnostic_capacity, span_size_u32(diagnostic_records.size()));

  struct ReplayWorkgroupState {
    struct ReportedDiagnosticKey {
      uint32_t kind = 0;
      uint32_t first_site = 0;
      uint32_t second_site = 0;
      uint64_t overlap_begin = 0;
      uint64_t overlap_end = 0;

      auto operator<=>(const ReportedDiagnosticKey &) const = default;
    };

    ReplayWorkgroupState(uint64_t exact_byte_capacity, uint32_t access_capacity,
                         size_t synchronization_capacity)
        : exact_byte_shadow(exact_byte_capacity, access_capacity),
          synchronization_metadata_capacity(synchronization_capacity) {}

    uint64_t generation = 0;
    uint32_t workgroup_x = 0;
    uint32_t workgroup_y = 0;
    uint32_t workgroup_z = 0;
    std::vector<uint32_t> owner_epochs;
    ConSanMoiSparseExactByteShadow exact_byte_shadow;
    std::vector<uint64_t> exported_exact_shadow_entries;
    std::vector<ConSanMoiAtomicReleaseRecord> atomic_release_records;
    std::vector<ConSanMoiAcquiredEpochToken> acquired_epoch_tokens;
    size_t synchronization_metadata_capacity = 0;
    std::set<ReportedDiagnosticKey> reported_diagnostics;
    bool in_barrier_run = false;
  };
  std::vector<ReplayWorkgroupState> workgroups;
  const auto synchronization_metadata_capacity_for_workgroup =
      [&](uint64_t generation, uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
        size_t event_count = 0;
        for (const ConSanMoiRecordReplayAtomicEvent &record : atomic_events) {
          const uint64_t record_generation =
              record.generation != 0 ? record.generation : header.generation;
          if (!consan_moi_record_replay_atomic_event_is_unpublished(record) &&
              record_generation == generation && record.workgroup_x == workgroup_x &&
              record.workgroup_y == workgroup_y && record.workgroup_z == workgroup_z)
            ++event_count;
        }
        for (const ConSanMoiRecordReplayFenceEvent &record : fence_events) {
          const uint64_t record_generation =
              record.generation != 0 ? record.generation : header.generation;
          if (!consan_moi_record_replay_fence_event_is_unpublished(record) &&
              record_generation == generation && record.workgroup_x == workgroup_x &&
              record.workgroup_y == workgroup_y && record.workgroup_z == workgroup_z)
            ++event_count;
        }

        // Each event can introduce one owner, and a causal frontier can retain
        // one component for every ordered producer/consumer pair. Reserving
        // only one slot per event loses valid acquire-release chains once a
        // fourth owner imports three predecessors. Bound each workgroup by the
        // square of its own event count so total storage follows the observed
        // synchronization distribution rather than multiplying a process-wide
        // maximum by every workgroup.
        if (event_count != 0 && event_count > std::numeric_limits<size_t>::max() / event_count) {
          replay.metadata_full = true;
          return event_count;
        }
        return event_count * event_count;
      };
  std::optional<size_t> first_workgroup_index;
  auto find_workgroup_state = [&](uint64_t generation, uint32_t workgroup_x, uint32_t workgroup_y,
                                  uint32_t workgroup_z) -> ReplayWorkgroupState & {
    for (ReplayWorkgroupState &state : workgroups) {
      if (state.generation == generation && state.workgroup_x == workgroup_x &&
          state.workgroup_y == workgroup_y && state.workgroup_z == workgroup_z)
        return state;
    }

    const size_t synchronization_metadata_capacity =
        synchronization_metadata_capacity_for_workgroup(generation, workgroup_x, workgroup_y,
                                                        workgroup_z);
    ReplayWorkgroupState state(static_cast<uint64_t>(exact_shadow_entries.size()) *
                                   consan_moi_exact_shadow::granule_bytes,
                               replay.published_access_count, synchronization_metadata_capacity);
    state.generation = generation;
    state.workgroup_x = workgroup_x;
    state.workgroup_y = workgroup_y;
    state.workgroup_z = workgroup_z;
    state.owner_epochs.resize(consan_moi_exact_shadow::max_owner + 1u);
    if (!first_workgroup_index)
      state.exported_exact_shadow_entries.resize(exact_shadow_entries.size());
    workgroups.push_back(std::move(state));
    if (!first_workgroup_index)
      first_workgroup_index = workgroups.size() - 1u;
    return workgroups.back();
  };
  const auto reconcile_owner_epoch = [](ReplayWorkgroupState &state, uint32_t owner_id,
                                        uint32_t recorded_epoch) {
    uint32_t &epoch = state.owner_epochs[owner_id];
    epoch = std::max(epoch, recorded_epoch);
    return epoch;
  };
  const auto advance_owner_epoch = [](ReplayWorkgroupState &state, uint32_t owner_id) {
    uint32_t &epoch = state.owner_epochs[owner_id];
    if (epoch < consan_moi_exact_shadow::max_epoch)
      ++epoch;
  };

  // Atomic and fence synchronization share one sparse causal clock. The
  // event-count-squared value above is a fail-closed capacity bound, not a
  // mandate to allocate and scan that many empty entries. Grow by at most one
  // component per possible producer owner for the current operation and trim
  // unused tail slots immediately afterward.
  constexpr size_t kMaximumCausalComponents = consan_moi_exact_shadow::max_owner + 1u;
  const auto grow_causal_metadata = [=](auto &records, size_t capacity) {
    if (records.size() >= capacity)
      return;
    const size_t growth = std::min(kMaximumCausalComponents, capacity - records.size());
    records.resize(records.size() + growth);
  };
  const auto trim_unused_causal_metadata = [](auto &records) {
    while (!records.empty() && !records.back().valid)
      records.pop_back();
  };
  const auto bounded_metadata_count = [](size_t size) {
    return size > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                       : static_cast<uint32_t>(size);
  };
  const auto replay_atomic_release = [&](ReplayWorkgroupState &state, uint64_t generation,
                                         uint64_t address, uint32_t producer_owner_id,
                                         uint32_t producer_epoch, uint32_t instruction_offset) {
    grow_causal_metadata(state.atomic_release_records, state.synchronization_metadata_capacity);
    const ConSanMoiAtomicSyncResult result = consan_moi_record_replay_atomic_release(
        state.atomic_release_records, state.acquired_epoch_tokens, generation, address,
        producer_owner_id, producer_epoch, instruction_offset);
    trim_unused_causal_metadata(state.atomic_release_records);
    replay.maximum_atomic_release_metadata_count =
        std::max(replay.maximum_atomic_release_metadata_count,
                 bounded_metadata_count(state.atomic_release_records.size()));
    return result;
  };
  const auto replay_atomic_acquire = [&](ReplayWorkgroupState &state, uint64_t generation,
                                         uint64_t address, uint32_t consumer_owner_id,
                                         uint32_t instruction_offset) {
    grow_causal_metadata(state.acquired_epoch_tokens, state.synchronization_metadata_capacity);
    const ConSanMoiAtomicSyncResult result = consan_moi_record_replay_atomic_acquire(
        state.atomic_release_records, state.acquired_epoch_tokens, generation, address,
        consumer_owner_id, instruction_offset);
    trim_unused_causal_metadata(state.acquired_epoch_tokens);
    replay.maximum_acquired_epoch_metadata_count =
        std::max(replay.maximum_acquired_epoch_metadata_count,
                 bounded_metadata_count(state.acquired_epoch_tokens.size()));
    return result;
  };

  const auto make_access_conflict_diagnostic = [](const ConSanMoiExactByteAccess &first,
                                                  const ConSanMoiExactByteAccess &second) {
    ConSanMoiDiagnosticRecord diagnostic;
    diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict);
    diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
    diagnostic.generation = second.generation;
    diagnostic.epoch = second.epoch;
    diagnostic.first_owner_id = first.owner_id;
    diagnostic.second_owner_id = second.owner_id;
    diagnostic.first_lane_mask = first.lane_mask;
    diagnostic.second_lane_mask = second.lane_mask;
    diagnostic.first_instruction_offset = first.instruction_offset;
    diagnostic.second_instruction_offset = second.instruction_offset;
    diagnostic.first_lds_byte_offset = first.lds_byte_offset;
    diagnostic.first_lds_byte_count = first.lds_byte_count;
    diagnostic.second_lds_byte_offset = second.lds_byte_offset;
    diagnostic.second_lds_byte_count = second.lds_byte_count;
    diagnostic.first_access_kind = static_cast<uint32_t>(first.kind);
    diagnostic.second_access_kind = static_cast<uint32_t>(second.kind);
    return diagnostic;
  };

  const auto make_metadata_full_diagnostic = [](const ConSanMoiExactByteAccess &access) {
    ConSanMoiDiagnosticRecord diagnostic;
    diagnostic.kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::MetadataFull);
    diagnostic.backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
    diagnostic.generation = access.generation;
    diagnostic.epoch = access.epoch;
    diagnostic.second_owner_id = access.owner_id;
    diagnostic.second_lane_mask = access.lane_mask;
    diagnostic.second_instruction_offset = access.instruction_offset;
    diagnostic.second_lds_byte_offset = access.lds_byte_offset;
    diagnostic.second_lds_byte_count = access.lds_byte_count;
    diagnostic.second_access_kind = static_cast<uint32_t>(access.kind);
    return diagnostic;
  };

  const auto make_intra_wave_write_diagnostic =
      [&](const ConSanMoiExactByteAccess &access) -> std::optional<ConSanMoiDiagnosticRecord> {
    if (!consan_moi_shadow_kind_conflicts(access.kind, access.kind) ||
        std::popcount(access.lane_mask) < 2) {
      return std::nullopt;
    }
    ConSanMoiExactByteAccess first = access;
    ConSanMoiExactByteAccess second = access;
    first.lane_mask = access.lane_mask & (~access.lane_mask + uint64_t{1});
    second.lane_mask = access.lane_mask ^ first.lane_mask;
    return make_access_conflict_diagnostic(first, second);
  };

  const auto append_diagnostic = [&](ReplayWorkgroupState &state,
                                     const ConSanMoiDiagnosticRecord &diagnostic,
                                     uint32_t event_index) {
    const uint32_t first_site =
        std::min(diagnostic.first_instruction_offset, diagnostic.second_instruction_offset);
    const uint32_t second_site =
        std::max(diagnostic.first_instruction_offset, diagnostic.second_instruction_offset);
    const uint64_t first_end =
        static_cast<uint64_t>(diagnostic.first_lds_byte_offset) + diagnostic.first_lds_byte_count;
    const uint64_t second_end =
        static_cast<uint64_t>(diagnostic.second_lds_byte_offset) + diagnostic.second_lds_byte_count;
    const uint64_t overlap_begin =
        std::max(diagnostic.first_lds_byte_offset, diagnostic.second_lds_byte_offset);
    const uint64_t overlap_end = std::min(first_end, second_end);
    const ReplayWorkgroupState::ReportedDiagnosticKey key{diagnostic.kind, first_site, second_site,
                                                          overlap_begin, overlap_end};
    if (!state.reported_diagnostics.insert(key).second)
      return;
    if (header.diagnostic_count < diagnostic_capacity) {
      ConSanMoiDiagnosticRecord published = diagnostic;
      published.reserved = event_index;
      diagnostic_records[header.diagnostic_count] = published;
      ++header.diagnostic_count;
      ++replay.emitted_diagnostic_count;
    } else {
      replay.diagnostic_capacity_exhausted = true;
    }
  };

  struct ReplayEvent {
    enum class Kind {
      Access,
      Barrier,
      Atomic,
      Fence,
    };

    uint32_t event_index = 0;
    uint32_t input_order = 0;
    uint32_t record_index = 0;
    Kind kind = Kind::Access;
  };
  std::vector<ReplayEvent> events;
  events.reserve(static_cast<size_t>(replay.published_access_count) + barrier_count +
                 atomic_events.size() + fence_events.size());
  for (uint32_t i = 0; i < access_count; ++i) {
    if (!consan_moi_access_record_is_unpublished(access_records[i]))
      events.push_back({access_records[i].event_index, i, i, ReplayEvent::Kind::Access});
  }
  for (uint32_t i = 0; i < barrier_count; ++i)
    events.push_back(
        {barrier_records[i].event_index, access_count + i, i, ReplayEvent::Kind::Barrier});
  for (uint32_t i = 0; i < span_size_u32(atomic_events.size()); ++i) {
    if (consan_moi_record_replay_atomic_event_is_unpublished(atomic_events[i]))
      continue;
    events.push_back({atomic_events[i].event_index, access_count + barrier_count + i, i,
                      ReplayEvent::Kind::Atomic});
  }
  for (uint32_t i = 0; i < span_size_u32(fence_events.size()); ++i) {
    if (consan_moi_record_replay_fence_event_is_unpublished(fence_events[i]))
      continue;
    events.push_back({fence_events[i].event_index,
                      access_count + barrier_count + span_size_u32(atomic_events.size()) + i, i,
                      ReplayEvent::Kind::Fence});
  }
  std::stable_sort(events.begin(), events.end(),
                   [](const ReplayEvent &lhs, const ReplayEvent &rhs) {
                     if (lhs.event_index != rhs.event_index)
                       return lhs.event_index < rhs.event_index;
                     return lhs.input_order < rhs.input_order;
                   });

  for (const ReplayEvent &event : events) {
    if (event.kind == ReplayEvent::Kind::Barrier) {
      const ConSanMoiBarrierRecord &record = barrier_records[event.record_index];
      const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
      ReplayWorkgroupState &state = find_workgroup_state(generation, record.workgroup_x,
                                                         record.workgroup_y, record.workgroup_z);
      ++replay.processed_barrier_count;
      if (!state.in_barrier_run) {
        for (uint32_t &epoch : state.owner_epochs) {
          if (epoch < consan_moi_exact_shadow::max_epoch)
            ++epoch;
        }
        state.in_barrier_run = true;
      }
      continue;
    }

    if (event.kind == ReplayEvent::Kind::Fence) {
      const ConSanMoiRecordReplayFenceEvent &record = fence_events[event.record_index];
      ++replay.processed_fence_count;
      const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
      ReplayWorkgroupState &state = find_workgroup_state(generation, record.workgroup_x,
                                                         record.workgroup_y, record.workgroup_z);
      state.in_barrier_run = false;
      if (record.owner_id > consan_moi_exact_shadow::max_owner || record.scope == 0 ||
          record.scope > 3 || record.communication_token == 0) {
        ++replay.unsupported_fence_count;
        continue;
      }
      const bool acquire = record.kind == ConSanMoiFenceEventKind::Acquire ||
                           record.kind == ConSanMoiFenceEventKind::AcquireRelease;
      const bool release = record.kind == ConSanMoiFenceEventKind::Release ||
                           record.kind == ConSanMoiFenceEventKind::AcquireRelease;
      if (!acquire && !release) {
        ++replay.unsupported_fence_count;
        continue;
      }
      const uint32_t epoch = reconcile_owner_epoch(state, record.owner_id, record.epoch);
      bool imported_predecessor = false;
      if (acquire) {
        const ConSanMoiAtomicSyncResult token_result =
            replay_atomic_acquire(state, generation, record.communication_token, record.owner_id,
                                  record.instruction_offset);
        replay.metadata_full |= token_result.metadata_full;
        imported_predecessor = token_result.updated_record_count != 0u;
      }
      if (release) {
        const ConSanMoiAtomicSyncResult release_result =
            replay_atomic_release(state, generation, record.communication_token, record.owner_id,
                                  epoch, record.instruction_offset);
        replay.metadata_full |= release_result.metadata_full;
      }
      if (imported_predecessor)
        advance_owner_epoch(state, record.owner_id);
      continue;
    }

    if (event.kind == ReplayEvent::Kind::Atomic) {
      const ConSanMoiRecordReplayAtomicEvent &record = atomic_events[event.record_index];
      ++replay.processed_atomic_count;
      const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
      ReplayWorkgroupState &state = find_workgroup_state(generation, record.workgroup_x,
                                                         record.workgroup_y, record.workgroup_z);
      state.in_barrier_run = false;
      if (record.owner_id > consan_moi_exact_shadow::max_owner) {
        ++replay.unsupported_atomic_count;
        replay.metadata_full = true;
        continue;
      }
      const std::optional<ConSanMoiAtomicOutcome> outcome =
          consan_moi_record_replay_resolved_atomic_outcome(record);
      if (!outcome) {
        ++replay.unsupported_atomic_count;
        continue;
      }

      const uint32_t epoch = reconcile_owner_epoch(state, record.owner_id, record.epoch);
      ConSanMoiAtomicSyncResult atomic_result;
      switch (record.kind) {
      case ConSanMoiAtomicEventKind::Release:
        if (record.operation != ConSanMoiAtomicOperation::CompareExchange ||
            *outcome == ConSanMoiAtomicOutcome::Success)
          atomic_result = replay_atomic_release(state, generation, record.atomic_address,
                                                record.owner_id, epoch, record.instruction_offset);
        break;
      case ConSanMoiAtomicEventKind::Acquire:
        atomic_result = replay_atomic_acquire(state, generation, record.atomic_address,
                                              record.owner_id, record.instruction_offset);
        break;
      case ConSanMoiAtomicEventKind::AcquireRelease: {
        // RMW and CAS both consume the prior value. Failed CAS does not
        // publish the release half of an acquire-release event.
        atomic_result = replay_atomic_acquire(state, generation, record.atomic_address,
                                              record.owner_id, record.instruction_offset);
        if (record.operation != ConSanMoiAtomicOperation::CompareExchange ||
            *outcome == ConSanMoiAtomicOutcome::Success) {
          const ConSanMoiAtomicSyncResult release_result =
              replay_atomic_release(state, generation, record.atomic_address, record.owner_id,
                                    epoch, record.instruction_offset);
          atomic_result.metadata_full |= release_result.metadata_full;
        }
        break;
      }
      }
      replay.metadata_full |= atomic_result.metadata_full;
      if ((record.kind == ConSanMoiAtomicEventKind::Acquire ||
           record.kind == ConSanMoiAtomicEventKind::AcquireRelease) &&
          atomic_result.updated_record_count != 0u) {
        // A successful acquire closes the consumer's current segment. Later
        // accesses must not be covered by a token imported for an earlier
        // generation of the same reusable partial barrier.
        advance_owner_epoch(state, record.owner_id);
      }
      continue;
    }

    const uint32_t i = event.record_index;
    const ConSanMoiAccessRecord &record = access_records[i];
    // Fixed per-site Record/Replay buffers publish their static capacity up
    // front. A never-executed site therefore remains an all-zero slot; it is
    // neither a malformed access nor evidence loss. Any partially initialized
    // Empty record still falls through to the fail-closed unsupported path.
    if (consan_moi_access_record_is_unpublished(record))
      continue;
    const std::optional<ConSanMoiShadowAccessKind> access_kind =
        decode_access_kind(record.access_kind);
    if (!access_kind) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      continue;
    }
    if ((record.flags & ~kConSanMoiAccessRecordKnownFlags) != 0u) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      replay.metadata_full = true;
      continue;
    }
    if (record.wave_id > consan_moi_exact_shadow::max_owner) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      replay.metadata_full = true;
      continue;
    }

    const uint64_t lds_byte_offset_u64 =
        record.lds_byte_count != 0
            ? record.lds_byte_offset
            : static_cast<uint64_t>(record.start_cell) * consan_moi_exact_shadow::granule_bytes;
    const uint64_t lds_byte_count_u64 =
        record.lds_byte_count != 0
            ? record.lds_byte_count
            : static_cast<uint64_t>(record.cell_count) * consan_moi_exact_shadow::granule_bytes;
    if (lds_byte_offset_u64 > std::numeric_limits<uint32_t>::max() ||
        lds_byte_count_u64 > std::numeric_limits<uint32_t>::max()) {
      ++replay.processed_access_count;
      ++replay.unsupported_access_count;
      replay.metadata_full = true;
      replay.conflict = true;
      continue;
    }
    const uint32_t lds_byte_offset = static_cast<uint32_t>(lds_byte_offset_u64);
    const uint32_t lds_byte_count = static_cast<uint32_t>(lds_byte_count_u64);

    const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
    ReplayWorkgroupState &state = find_workgroup_state(generation, record.workgroup_x,
                                                       record.workgroup_y, record.workgroup_z);
    state.in_barrier_run = false;
    const uint32_t effective_epoch = reconcile_owner_epoch(state, record.wave_id, record.epoch);
    const ConSanMoiExactByteAccess access{
        generation,
        /*owner_id=*/record.wave_id,
        effective_epoch,
        *access_kind,
        lds_byte_offset,
        lds_byte_count,
        record.instruction_offset,
        record.lane_mask,
        // Unflagged lane masks summarize a wave access. They do not identify
        // separable address groups that could race with one another.
        (record.flags & kConSanMoiAccessRecordFlagExactAddressGroupMask) != 0u,
    };
    const ConSanMoiExactByteAccessResult access_result =
        state.exact_byte_shadow.access(access, state.acquired_epoch_tokens);
    ++replay.processed_access_count;
    if (access_result.conflict) {
      if (!access_result.prior) {
        replay.metadata_full = true;
        replay.conflict = true;
        append_diagnostic(state, make_metadata_full_diagnostic(access), record.event_index);
        continue;
      }
      replay.conflict = true;
      append_diagnostic(state, make_access_conflict_diagnostic(*access_result.prior, access),
                        record.event_index);
      replay.metadata_full |= access_result.capacity_exhausted;
      continue;
    }
    if (access_result.capacity_exhausted) {
      replay.metadata_full = true;
      replay.conflict = true;
      append_diagnostic(state, make_metadata_full_diagnostic(access), record.event_index);
      continue;
    }
    const uint64_t packed = pack_consan_moi_exact_shadow_entry(
        access.kind, access.owner_id, access.epoch, static_cast<uint32_t>(access.generation),
        access.instruction_offset);
    const uint64_t byte_end = static_cast<uint64_t>(access.lds_byte_offset) + access.lds_byte_count;
    const uint64_t start_cell = access.lds_byte_offset >> consan_moi_exact_shadow::granule_shift;
    const uint64_t end_cell = (byte_end + consan_moi_exact_shadow::granule_bytes - 1u) >>
                              consan_moi_exact_shadow::granule_shift;
    for (uint64_t cell = start_cell;
         cell < end_cell && cell < state.exported_exact_shadow_entries.size(); ++cell) {
      state.exported_exact_shadow_entries[cell] = packed;
    }
    const std::optional<ConSanMoiDiagnosticRecord> intra_wave =
        (record.flags & kConSanMoiAccessRecordFlagExactAddressGroupMask) != 0u
            ? make_intra_wave_write_diagnostic(access)
            : std::nullopt;
    if (!intra_wave)
      continue;
    replay.conflict = true;
    append_diagnostic(state, *intra_wave, record.event_index);
  }
  if (first_workgroup_index)
    std::copy(workgroups[*first_workgroup_index].exported_exact_shadow_entries.begin(),
              workgroups[*first_workgroup_index].exported_exact_shadow_entries.end(),
              exact_shadow_entries.begin());
  return replay;
}

} // namespace rocjitsu
