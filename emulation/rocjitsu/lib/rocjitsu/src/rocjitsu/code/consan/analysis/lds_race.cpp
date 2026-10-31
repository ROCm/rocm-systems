// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/lds_race.h"
#include "rocjitsu/code/consan/encoding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <map>
#include <optional>
#include <set>

namespace rocjitsu::consan {
namespace {

static_assert(kSyncCapacity <= 32, "Barrier witness masks must fit in uint32_t");

bool all_zero(std::span<const uint8_t> bytes) {
  return std::ranges::all_of(bytes, [](uint8_t value) { return value == 0; });
}

void issue(LdsRaceExecutionResult &result, uint32_t wave, uint32_t site, const char *reason) {
  result.coverage_complete = false;
  result.ordering_uncertain = true;
  ++result.integrity_events;
  if (result.issues.size() < kMaxIntegrityIssueExamples)
    result.issues.push_back({wave, site, reason});
}

struct Observation {
  LdsRaceParticipant participant;
  bool phase_valid = true;
  uint32_t lanes() const { return std::popcount(participant.mask); }
};

struct SiteGroup {
  std::vector<Observation> records;
  uint32_t sequence = 0;
  uint64_t mask = 0;
  std::set<uint32_t> offsets;
  bool ambiguous = false;
};

// A decoded barrier completion from one wave. read_wave validates its fields
// before retaining it. Ordering across waves additionally requires every
// participant to supply the same (group, occurrence) completion.
struct Completion {
  uint32_t group = 0;      // Index of the qualified static barrier group.
  uint32_t occurrence = 0; // One-based execution; must be 1 in this acyclic profile.
  // Wave-local access-sequence boundary: accesses at or below it precede the
  // completing wait, and accesses above it follow the wait.
  uint32_t sequence = 0;
  uint32_t phase = 0; // Number of completed barriers, including this one.
};

// Retained synchronization evidence for one wave. A missing or invalid journal
// entry does not discard other entries that can still prove ordering.
struct WaveEvidence {
  // Slot i belongs to static barrier group i; empty means no validated entry.
  std::array<std::optional<Completion>, kSyncCapacity> journal;
  // The row and journal satisfy the final-phase, sync-loss, and completion checks.
  // This does not imply full access coverage or consistent access phases;
  // read_wave checks those separately for each retained observation.
  bool complete = false;
};

void read_wave(const Manifest &facts, const CapturedExecution &execution,
               const LaunchLayout &layout, uint32_t wave, LdsRaceExecutionResult &result,
               std::vector<Observation> &observations, std::vector<WaveEvidence> &completions) {
  const size_t first_observation = observations.size();
  const std::span<const uint8_t> row =
      std::span(execution.report)
          .subspan(kReportHeaderBytes + wave * layout.row_stride, layout.row_stride);
  const uint32_t status = *wire::read_u32(row, wire::kRowStatus);
  const uint32_t loss = *wire::read_u32(row, wire::kRowLoss);
  const uint32_t final = *wire::read_u32(row, wire::kRowFinalSequence);
  constexpr uint32_t kCompletedRowStatus = wire::kRowEntered | wire::kRowExited;
  if (wire::read_u64(row, wire::kRowSerial) != execution.serial ||
      wire::read_u32(row, wire::kRowEntry) != facts.entry.id ||
      wire::read_u32(row, wire::kRowWave) != wave ||
      (status != kCompletedRowStatus && status != (kCompletedRowStatus | wire::kRowExhausted)) ||
      !all_zero(row.subspan(wire::kRowReservedBegin, kRowHeaderBytes - wire::kRowReservedBegin)) ||
      final > facts.sites.size() || loss > final ||
      (((status & wire::kRowExhausted) != 0) != (loss > 0))) {
    issue(result, wave, UINT32_MAX, "invalid_row_identity_status_or_sequence");
    return;
  }
  const uint32_t final_phase = *wire::read_u32(row, wire::kRowFinalPhase);
  WaveEvidence &evidence = completions[wave];
  evidence.complete =
      final_phase == facts.barriers.size() && wire::read_u32(row, wire::kRowSyncLoss) == 0;
  if (!evidence.complete)
    issue(result, wave, UINT32_MAX, "invalid_final_barrier_phase_or_loss");
  if (!facts.barriers.empty()) {
    uint32_t previous_sequence = 0;
    for (uint32_t slot = 0; slot < kSyncCapacity; ++slot) {
      const std::span<const uint8_t> record =
          row.subspan(sync_offset(facts.sites.size()) + slot * kSyncRecordBytes, kSyncRecordBytes);
      if (slot >= facts.barriers.size()) {
        if (!all_zero(record)) {
          evidence.complete = false;
          issue(result, wave, UINT32_MAX, "unexpected_barrier_completion");
        }
        continue;
      }
      const Completion completion = {.group = *wire::read_u32(record, wire::kSyncGroup),
                                     .occurrence = *wire::read_u32(record, wire::kSyncOccurrence),
                                     .sequence = *wire::read_u32(record, wire::kSyncSequence),
                                     .phase = *wire::read_u32(record, wire::kSyncPhase)};
      if (wire::read_u32(record, wire::kSyncCommit) != wire::kBarrierCompleted ||
          completion.group != slot || completion.occurrence != 1 || completion.phase != slot + 1 ||
          completion.phase > final_phase || completion.sequence > final ||
          completion.sequence < previous_sequence ||
          wire::read_u32(record, wire::kSyncParticipants) != layout.wave_count ||
          !all_zero(record.subspan(wire::kSyncReservedBegin))) {
        evidence.complete = false;
        issue(result, wave, UINT32_MAX, "missing_or_invalid_barrier_completion");
        continue;
      }
      previous_sequence = completion.sequence;
      evidence.journal[slot] = completion;
    }
  }
  result.lost_occurrences += loss;
  if (loss != 0)
    issue(result, wave, UINT32_MAX, "range_reservation_exhausted");
  std::map<uint32_t, SiteGroup> groups;
  for (const SiteFacts &site : facts.sites) {
    SiteGroup &group = groups[site.id];
    bool vacant = false;
    for (uint32_t address_group = 0; address_group < kAddressGroupsPerSite; ++address_group) {
      const std::span<const uint8_t> record =
          row.subspan(kRowHeaderBytes + (site.first_slot + address_group) * kAccessRecordBytes,
                      kAccessRecordBytes);
      if (all_zero(record)) {
        vacant = true;
        continue;
      }
      const uint32_t commit = *wire::read_u32(record, wire::kRecordCommit);
      const uint32_t sequence = *wire::read_u32(record, wire::kRecordSequence);
      const uint32_t offset = *wire::read_u32(record, wire::kRecordLdsOffset);
      const uint32_t width = *wire::read_u32(record, wire::kRecordWidth);
      const uint64_t mask = *wire::read_u64(record, wire::kRecordGuestMask);
      const uint32_t lanes =
          std::min(facts.wave_size, execution.dimensions.workgroup[0] - wave * facts.wave_size);
      const uint64_t active_mask = lanes == 64 ? UINT64_MAX : (uint64_t{1} << lanes) - 1;
      const uint32_t flags = *wire::read_u32(record, wire::kRecordFlags);
      if (commit != wire::kAccessCommitted ||
          wire::read_u32(record, wire::kRecordAnalysis) != uint32_t(site.analysis) ||
          wire::read_u32(record, wire::kRecordKind) != uint32_t(site.record) ||
          wire::read_u32(record, wire::kRecordSite) != site.id ||
          wire::read_u32(record, wire::kRecordRange) != site.range_id || sequence == 0 ||
          sequence > final || width != site.width || mask == 0 || (mask & ~active_mask) != 0 ||
          !all_zero(record.subspan(wire::kRecordReservedBegin))) {
        ++result.rejected_records;
        issue(result, wave, site.id, "invalid_record_commit_identity_mask_or_sequence");
        continue;
      }
      if (flags != wire::kExactGuestMask || offset > execution.group_bytes ||
          width > execution.group_bytes - offset) {
        ++result.rejected_records;
        issue(result, wave, site.id, "invalid_or_out_of_bounds_effective_range");
        continue;
      }
      if ((offset % width) != 0) {
        ++result.rejected_records;
        issue(result, wave, site.id, "unaligned_lds_alignment_mode_unavailable");
        continue;
      }
      if (vacant)
        issue(result, wave, site.id, "record_after_vacant_reservation");
      if ((group.sequence != 0 && group.sequence != sequence) || (group.mask & mask) != 0 ||
          !group.offsets.insert(offset).second)
        group.ambiguous = true;
      group.sequence = sequence;
      group.mask |= mask;
      const uint32_t phase = *wire::read_u32(record, wire::kRecordPhase);
      const uint32_t expected_phase = static_cast<uint32_t>(
          std::ranges::count_if(facts.barriers, [&](const BarrierFacts &barrier) {
            return barrier.original_wait < site.original_offset;
          }));
      const bool phase_valid = phase == expected_phase && phase <= final_phase;
      if (!phase_valid)
        issue(result, wave, site.id, "access_phase_contradicts_qualified_program");
      group.records.push_back(
          {{wave, site.id, sequence, mask, offset, width, site.original_offset, site.access, phase},
           phase_valid});
    }
  }
  std::map<uint32_t, std::vector<uint32_t>> occurrences;
  for (const auto &[site, group] : groups) {
    if (!group.records.empty())
      occurrences[group.sequence].push_back(site);
  }
  for (const auto &[sequence, sites] : occurrences) {
    (void)sequence;
    if (sites.size() > 1) {
      for (const uint32_t site : sites)
        groups[site].ambiguous = true;
    }
  }
  // The compiler admits only forward control flow. Distinct dynamically
  // executed sites therefore retain increasing original instruction order.
  std::map<uint32_t, SiteGroup>::iterator previous = groups.end();
  for (std::map<uint32_t, SiteGroup>::iterator it = groups.begin(); it != groups.end(); ++it) {
    if (it->second.records.empty())
      continue;
    if (previous != groups.end() && previous->second.sequence >= it->second.sequence) {
      previous->second.ambiguous = true;
      it->second.ambiguous = true;
    }
    previous = it;
  }
  if (occurrences.size() != final)
    issue(result, wave, UINT32_MAX, "missing_observed_occurrence");
  for (const auto &[site, group] : groups) {
    if (group.ambiguous) {
      result.rejected_records += group.records.size();
      issue(result, wave, site, "ambiguous_instruction_group_ownership");
    } else
      observations.insert(observations.end(), group.records.begin(), group.records.end());
  }
  // Validate every retained endpoint, including disjoint and Read/Read accesses.
  // A boundary contradiction is an integrity issue even without a candidate pair.
  bool consistent = true;
  for (Observation &observation : std::span(observations).subspan(first_observation)) {
    if (!observation.phase_valid) {
      consistent = false;
      continue;
    }
    const LdsRaceParticipant &access = observation.participant;
    for (const std::optional<Completion> &completion : evidence.journal) {
      // A completion's sequence counts accesses already observed by this wave.
      // A post-B0 access with phase=1, sequence=1 needs B0's boundary sequence=0;
      // boundary sequence=1 would instead place that same access before B0.
      if (completion &&
          (access.phase < completion->phase) != (access.sequence <= completion->sequence)) {
        // No boundary can certify this endpoint's position until its phase and
        // sequence agree. Its independently checked range/mask remain usable.
        observation.phase_valid = false;
        consistent = false;
        issue(result, wave, access.site, "access_sequence_contradicts_barrier_phase");
      }
    }
  }
  if (!facts.barriers.empty() && evidence.complete && consistent)
    ++result.completed_barrier_waves;
}

struct Bucket {
  // Reads and writes remain separate so counts and example generation never
  // enumerate the potentially quadratic Read/Read non-candidates.
  struct Wave {
    std::vector<const Observation *> reads, writes;
  };
  std::map<uint32_t, Wave> waves;
};

// Where one LDS access lies relative to validated workgroup barrier witnesses.
// Each mask bit identifies a journal slot. before and after describe the access
// relative to that slot's barrier, as in this two-barrier example:
//
//   B0 (slot 0) ---> access A ---> B1 (slot 1)
//   A.after  = 0b01: A follows B0.
//   A.before = 0b10: A precedes B1.
//
// A shared bit in A.before and B.after proves A -> barrier -> B. Each bit
// requires matching (group, occurrence) completions from every required wave;
// unequal access phases alone cannot supply it. The masks must remain wide
// enough for all kSyncCapacity journal slots (currently 16).
class Position {
public:
  uint32_t before = 0, after = 0;
  // Whether absence of an ordering witness is conclusive for this endpoint.
  // Without a shared ordering witness, both endpoints must be complete to call
  // the pair Unordered. Valid no-sync endpoints need no journal.
  bool complete = false;
  auto operator<=>(const Position &) const = default;
};

Position position(const Observation &observation, ScopeKind scope,
                  std::span<const WaveEvidence> evidence, uint32_t complete_witnesses,
                  bool complete) {
  Position result = {.complete = complete && observation.phase_valid};
  if (!observation.phase_valid)
    return result;
  // Qualification rules out synchronization throughout this scope. Once the
  // endpoint is valid, no other wave's exit or journal is needed to prove that
  // it is unordered with another valid endpoint in the same workgroup.
  if (scope == ScopeKind::NativeLdsNoSynchronization) {
    result.complete = true;
    return result;
  }
  const LdsRaceParticipant &access = observation.participant;
  for (uint32_t slot = 0; slot < kSyncCapacity; ++slot) {
    if ((complete_witnesses & (1u << slot)) == 0)
      continue;
    const Completion &completion = *evidence[access.wave].journal[slot];
    const bool before = access.phase < completion.phase;
    if (before != (access.sequence <= completion.sequence)) {
      result.complete = false;
      continue;
    }
    (before ? result.before : result.after) |= 1u << slot;
  }
  return result;
}

LdsRaceOrdering ordering(const Position &first, const Position &second) {
  // first and second identify candidate accesses, not their execution order.
  // A shared bit proves first -> barrier -> second; the other intersection
  // proves the reverse. With B0 in bit 0, first.before=0b1 and second.after=0b1
  // intersect at B0, establishing an ordering even if unrelated evidence is lost.
  if ((first.before & second.after) || (second.before & first.after))
    return LdsRaceOrdering::Ordered;
  // With B0 as the only witness, two accesses before it both have before=0b1,
  // after=0b0, so neither intersection orders them. Unordered requires complete
  // evidence; missing or contradictory required evidence keeps the candidate
  // Incomplete.
  return first.complete && second.complete ? LdsRaceOrdering::Unordered
                                           : LdsRaceOrdering::Incomplete;
}

std::vector<LdsRaceFinding> &examples_for(LdsRaceExecutionResult &result, LdsRaceOrdering order) {
  if (order == LdsRaceOrdering::Ordered)
    return result.ordered_examples;
  if (order == LdsRaceOrdering::Incomplete)
    return result.unresolved_examples;
  return result.cross_wave_examples;
}

void compare(const std::vector<const Observation *> &first,
             const std::vector<const Observation *> &second, uint32_t examples, ScopeKind scope,
             std::span<const WaveEvidence> evidence, uint32_t complete_witnesses, bool complete,
             LdsRaceExecutionResult &result) {
  if (first.empty() || second.empty())
    return;
  // The caller has selected conflicting ranges and distinct waves. Equal
  // Positions then have identical ordering against another group, so products
  // of group sizes and lane totals count pairs exactly (e.g. 3 * 2 = 6 pairs).
  // Only bounded examples below need individual identities.
  struct OrderingGroup {
    uint64_t count = 0, lanes = 0;
    std::vector<size_t> indices;
  };
  std::map<Position, OrderingGroup> first_groups, second_groups;
  for (const Observation *observation : first) {
    OrderingGroup &group =
        first_groups[position(*observation, scope, evidence, complete_witnesses, complete)];
    ++group.count;
    group.lanes += observation->lanes();
  }
  for (size_t index = 0; index < second.size(); ++index) {
    OrderingGroup &group =
        second_groups[position(*second[index], scope, evidence, complete_witnesses, complete)];
    ++group.count;
    group.lanes += second[index]->lanes();
    group.indices.push_back(index);
  }
  for (const auto &[left, left_group] : first_groups)
    for (const auto &[right, right_group] : second_groups) {
      const uint64_t pairs = left_group.count * right_group.count;
      switch (ordering(left, right)) {
      case LdsRaceOrdering::Ordered:
        result.ordered_pairs += pairs;
        break;
      case LdsRaceOrdering::Unordered:
        result.cross_wave_pairs += pairs;
        result.cross_wave_lane_pairs += left_group.lanes * right_group.lanes;
        break;
      case LdsRaceOrdering::Incomplete:
        result.ordering_unresolved_pairs += pairs;
        result.coverage_complete = false;
        result.ordering_uncertain = true;
        break;
      }
    }
  // Merge group indices so bounded examples preserve the canonical pair order.
  for (const Observation *left : first) {
    if (result.ordered_examples.size() >= examples &&
        result.cross_wave_examples.size() >= examples &&
        result.unresolved_examples.size() >= examples)
      break;
    const Position left_position = position(*left, scope, evidence, complete_witnesses, complete);
    std::map<Position, size_t> cursors;
    for (;;) {
      size_t next = second.size();
      Position selected;
      for (const auto &[right_position, group] : second_groups) {
        const size_t cursor = cursors[right_position];
        if (examples_for(result, ordering(left_position, right_position)).size() < examples &&
            cursor < group.indices.size() && group.indices[cursor] < next) {
          next = group.indices[cursor];
          selected = right_position;
        }
      }
      if (next == second.size())
        break;
      const LdsRaceOrdering order = ordering(left_position, selected);
      examples_for(result, order)
          .push_back({LdsRaceFindingKind::CrossWave, left->participant, second[next]->participant,
                      uint64_t{left->lanes()} * second[next]->lanes(), order});
      ++cursors[selected];
    }
  }
}

} // namespace

LdsRaceExecutionResult analyze_lds_race_execution(const Manifest &facts,
                                                  const CapturedExecution &execution,
                                                  uint32_t examples) {
  LdsRaceExecutionResult result = {.serial = execution.serial,
                                   .issues = {},
                                   .cross_wave_examples = {},
                                   .collision_examples = {},
                                   .ordered_examples = {},
                                   .unresolved_examples = {}};
  const FailureOr<LaunchLayout> layout =
      plan_launch(facts, execution.dimensions, execution.group_bytes, execution.private_bytes);
  const std::span<const uint8_t> report = std::span(execution.report);
  if (layout.failed() || report.size() != layout.value().report_bytes ||
      wire::read_u64(report, wire::kReportSerial) != execution.serial ||
      wire::read_u32(report, wire::kReportEntry) != facts.entry.id ||
      wire::read_u32(report, wire::kReportRows) != layout.value().wave_count ||
      wire::read_u32(report, wire::kReportRowStride) != layout.value().row_stride ||
      wire::read_u32(report, wire::kReportSites) != facts.sites.size() ||
      !all_zero(report.subspan(wire::kReportReservedBegin,
                               kReportHeaderBytes - wire::kReportReservedBegin))) {
    issue(result, UINT32_MAX, UINT32_MAX, "invalid_report_parent_or_extent");
    return result;
  }
  std::vector<Observation> observations;
  std::vector<WaveEvidence> completions(layout.value().wave_count);
  if (!facts.barriers.empty())
    result.required_barrier_waves = layout.value().wave_count;
  for (uint32_t wave = 0; wave < layout.value().wave_count; ++wave)
    read_wave(facts, execution, layout.value(), wave, result, observations, completions);
  const bool complete = std::ranges::all_of(completions, &WaveEvidence::complete);
  uint32_t complete_witnesses = 0;
  for (uint32_t slot = 0; slot < kSyncCapacity; ++slot) {
    const std::optional<Completion> &first = completions.front().journal[slot];
    if (first && std::ranges::all_of(completions, [&](const WaveEvidence &wave) {
          const std::optional<Completion> &other = wave.journal[slot];
          return other && other->group == first->group && other->occurrence == first->occurrence;
        }))
      complete_witnesses |= 1u << slot;
  }
  result.valid_observations = observations.size();
  std::map<uint32_t, Bucket> words;
  for (const Observation &observation : observations) {
    const LdsRaceParticipant &participant = observation.participant;
    Bucket &bucket = words[participant.offset];
    Bucket::Wave &wave = bucket.waves[participant.wave];
    (participant.access == AccessKind::Write ? wave.writes : wave.reads).push_back(&observation);
    if (participant.access == AccessKind::Write && observation.lanes() > 1) {
      const uint64_t pairs = uint64_t{observation.lanes()} * (observation.lanes() - 1) / 2;
      ++result.collision_groups;
      result.collision_lane_pairs += pairs;
      if (result.collision_examples.size() < examples)
        result.collision_examples.push_back(
            {LdsRaceFindingKind::InstructionCollision, participant, {}, pairs});
    }
  }
  // Admitted aligned b32 accesses overlap iff their byte offsets are equal.
  // Different dispatches never share this map, regardless of storage reuse.
  for (const auto &[offset, bucket] : words) {
    (void)offset;
    std::vector<const Observation *> prior_reads, prior_writes;
    for (const auto &[wave, accesses] : bucket.waves) {
      (void)wave;
      compare(prior_writes, accesses.writes, examples, facts.scope, completions, complete_witnesses,
              complete, result);
      compare(prior_writes, accesses.reads, examples, facts.scope, completions, complete_witnesses,
              complete, result);
      compare(prior_reads, accesses.writes, examples, facts.scope, completions, complete_witnesses,
              complete, result);
      prior_writes.insert(prior_writes.end(), accesses.writes.begin(), accesses.writes.end());
      prior_reads.insert(prior_reads.end(), accesses.reads.begin(), accesses.reads.end());
    }
  }
  return result;
}

} // namespace rocjitsu::consan
