// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/lds_race.h"
#include "rocjitsu/code/consan/encoding.h"

#include <algorithm>
#include <bit>
#include <map>
#include <set>

namespace rocjitsu::consan {
namespace {

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
  uint32_t lanes() const { return std::popcount(participant.mask); }
};

struct SiteGroup {
  std::vector<Observation> records;
  uint32_t sequence = 0;
  uint64_t mask = 0;
  std::set<uint32_t> offsets;
  bool ambiguous = false;
};

void read_wave(const Manifest &facts, const CapturedExecution &execution,
               const LaunchLayout &layout, uint32_t wave, LdsRaceExecutionResult &result,
               std::vector<Observation> &observations) {
  const auto row = std::span(execution.report)
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
  result.lost_occurrences += loss;
  if (loss != 0)
    issue(result, wave, UINT32_MAX, "range_reservation_exhausted");
  std::map<uint32_t, SiteGroup> groups;
  for (const SiteFacts &site : facts.sites) {
    auto &group = groups[site.id];
    bool vacant = false;
    for (uint32_t bank = 0; bank < kBanksPerSite; ++bank) {
      const auto record = row.subspan(
          kRowHeaderBytes + (site.first_slot + bank) * kAccessRecordBytes, kAccessRecordBytes);
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
      group.records.push_back(
          {{wave, site.id, sequence, mask, offset, width, site.original_offset, site.access}});
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
  auto previous = groups.end();
  for (auto it = groups.begin(); it != groups.end(); ++it) {
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
}

struct Bucket {
  // Reads and writes remain separate so counts and example generation never
  // enumerate the potentially quadratic Read/Read non-candidates.
  struct Wave {
    std::vector<const Observation *> reads, writes;
  };
  std::map<uint32_t, Wave> waves;
};

void compare(const std::vector<const Observation *> &first,
             const std::vector<const Observation *> &second, uint32_t examples,
             LdsRaceExecutionResult &result) {
  if (first.empty() || second.empty())
    return;
  result.cross_wave_pairs += uint64_t{first.size()} * second.size();
  uint64_t first_lanes = 0, second_lanes = 0;
  for (const auto *observation : first)
    first_lanes += observation->lanes();
  for (const auto *observation : second)
    second_lanes += observation->lanes();
  result.cross_wave_lane_pairs += first_lanes * second_lanes;
  if (second.empty())
    return;
  for (const auto *a : first) {
    if (result.cross_wave_examples.size() >= examples)
      break;
    for (const auto *b : second) {
      if (result.cross_wave_examples.size() >= examples)
        break;
      result.cross_wave_examples.push_back({LdsRaceFindingKind::CrossWave, a->participant,
                                            b->participant, uint64_t{a->lanes()} * b->lanes()});
    }
  }
}

} // namespace

LdsRaceExecutionResult analyze_lds_race_execution(const Manifest &facts,
                                                  const CapturedExecution &execution,
                                                  uint32_t examples) {
  LdsRaceExecutionResult result;
  result.serial = execution.serial;
  const auto layout =
      plan_launch(facts, execution.dimensions, execution.group_bytes, execution.private_bytes);
  const auto report = std::span(execution.report);
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
  for (uint32_t wave = 0; wave < layout.value().wave_count; ++wave)
    read_wave(facts, execution, layout.value(), wave, result, observations);
  result.valid_observations = observations.size();
  std::map<uint32_t, Bucket> words;
  for (const auto &observation : observations) {
    const auto &participant = observation.participant;
    auto &bucket = words[participant.offset];
    auto &wave = bucket.waves[participant.wave];
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
      compare(prior_writes, accesses.writes, examples, result);
      compare(prior_writes, accesses.reads, examples, result);
      compare(prior_reads, accesses.writes, examples, result);
      prior_writes.insert(prior_writes.end(), accesses.writes.begin(), accesses.writes.end());
      prior_reads.insert(prior_reads.end(), accesses.reads.begin(), accesses.reads.end());
    }
  }
  return result;
}

} // namespace rocjitsu::consan
