// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_diagnostic_report.h"

#include "util/log.h"

#include <algorithm>
#include <format>
#include <fstream>

namespace rocjitsu::amdgpu {
namespace {

using Finding = MemoryRaceChecker::Finding;
using Kind = MemoryRaceChecker::Kind;
using Space = MemoryRaceChecker::Space;
using Statistics = MemoryRaceChecker::Statistics;

const char *kind_name(Kind kind) {
  switch (kind) {
  case Kind::RAW:
    return "RAW";
  case Kind::WAR:
    return "WAR";
  case Kind::WAW:
    return "WAW";
  }
  return "RAW";
}

const char *action_name(const Finding &finding) {
  if (finding.space == Space::Global)
    return "cross-workgroup synchronization";
  return finding.requires_barrier ? "barrier" : "wait";
}

void write_statistics(std::ostream &out, const Statistics &statistics) {
  out << "{";
  bool first = true;
  const auto field = [&](std::string_view name, uint64_t value) {
    out << (first ? "" : ",") << "\"" << name << "\":" << value;
    first = false;
  };
  field("lds_accesses", statistics.lds_accesses);
  field("global_accesses", statistics.global_accesses);
  field("lane_ranges", statistics.lane_ranges);
  field("dword_candidates", statistics.dword_candidates);
  field("dword_visits", statistics.dword_visits);
  field("dword_duplicates_skipped", statistics.dword_duplicates_skipped);
  field("workgroup_lookups", statistics.workgroup_lookups);
  field("workgroup_creations", statistics.workgroup_creations);
  field("wave_search_steps", statistics.wave_search_steps);
  field("workgroup_map_lock_acquisitions", statistics.workgroup_map_lock_acquisitions);
  field("workgroup_map_lock_wait_ns", statistics.workgroup_map_lock_wait_ns);
  field("workgroup_state_lock_acquisitions", statistics.workgroup_state_lock_acquisitions);
  field("workgroup_state_lock_wait_ns", statistics.workgroup_state_lock_wait_ns);
  field("global_shard_lock_acquisitions", statistics.global_shard_lock_acquisitions);
  field("global_shard_lock_wait_ns", statistics.global_shard_lock_wait_ns);
  field("shadow_page_lookups", statistics.shadow_page_lookups);
  field("shadow_pages_created", statistics.shadow_pages_created);
  field("shadow_pages_rejected", statistics.shadow_pages_rejected);
  field("empty_record_heads", statistics.empty_record_heads);
  field("nonempty_record_heads", statistics.nonempty_record_heads);
  field("lds_records_visited", statistics.lds_records_visited);
  field("global_records_visited", statistics.global_records_visited);
  field("records_inserted", statistics.records_inserted);
  field("records_replaced", statistics.records_replaced);
  field("records_rejected", statistics.records_rejected);
  field("record_vector_growths", statistics.record_vector_growths);
  field("clock_matrix_growths", statistics.clock_matrix_growths);
  field("pending_vector_growths", statistics.pending_vector_growths);
  field("range_vector_growths", statistics.range_vector_growths);
  field("pending_candidates", statistics.pending_candidates);
  field("barrier_calls", statistics.barrier_calls);
  field("barrier_members", statistics.barrier_members);
  field("barrier_clock_cells", statistics.barrier_clock_cells);
  field("barrier_carried_ops", statistics.barrier_carried_ops);
  field("barrier_scratch_vectors", statistics.barrier_scratch_vectors);
  field("cleanup_workgroups", statistics.cleanup_workgroups);
  field("cleanup_dwords", statistics.cleanup_dwords);
  field("cleanup_pages", statistics.cleanup_pages);
  field("cleanup_records", statistics.cleanup_records);
  field("retained_bytes", statistics.retained_bytes);
  field("rejected_bytes", statistics.rejected_bytes);
  field("budget_reservation_attempts", statistics.budget_reservation_attempts);
  field("budget_reservation_failures", statistics.budget_reservation_failures);
  field("budget_bytes_released", statistics.budget_bytes_released);
  field("page_directory_growths", statistics.page_directory_growths);
  field("page_slab_allocations", statistics.page_slab_allocations);
  field("record_slab_allocations", statistics.record_slab_allocations);
  field("site_lookups", statistics.site_lookups);
  field("sites_interned", statistics.sites_interned);
  field("sites_reused", statistics.sites_reused);
  field("site_directory_growths", statistics.site_directory_growths);
  field("site_slab_allocations", statistics.site_slab_allocations);
  field("saturation_cache_hits", statistics.saturation_cache_hits);
  field("saturation_cache_inserts", statistics.saturation_cache_inserts);
  field("saturation_cache_invalidations", statistics.saturation_cache_invalidations);
  field("affine_accesses", statistics.affine_accesses);
  field("affine_lane_ranges", statistics.affine_lane_ranges);
  field("affine_dword_runs", statistics.affine_dword_runs);
  field("affine_dword_visits", statistics.affine_dword_visits);
  field("affine_pending_ops", statistics.affine_pending_ops);
  field("explicit_accesses", statistics.explicit_accesses);
  out << "}";
}

void write_dispatch_statistics(std::ostream &out,
                               const MemoryRaceChecker::DispatchStatistics &statistics) {
  out << "{\"dispatch\":" << statistics.dispatch
      << ",\"global_shadow_bytes_released\":" << statistics.global_shadow_bytes_released
      << ",\"global_shadow_peak_bytes\":" << statistics.global_shadow_peak_bytes
      << ",\"counters\":";
  write_statistics(out, statistics.counters);
  out << ",\"global_shards\":[";
  bool first = true;
  for (const auto &shard : statistics.global_shards) {
    out << (first ? "" : ",") << "{\"shard\":" << shard.shard << ",\"counters\":";
    first = false;
    write_statistics(out, shard.counters);
    out << "}";
  }
  out << "]}";
}

} // namespace

MemoryDiagnosticReport::MemoryDiagnosticReport(std::string path) : path_(std::move(path)) {
  // Findings are bounded, so allocate their complete shared storage before
  // producers can contend on the report.
  wait_findings_.reserve(kMaxFindings);
  race_findings_.reserve(kMaxFindings);
  recent_race_statistics_.reserve(kMaxRaceDispatchStatistics);
}

MemoryDiagnosticReport::~MemoryDiagnosticReport() { flush_if_needed(false); }

void MemoryDiagnosticReport::add_wait(const MemoryRaceChecker::WaveKey &wave,
                                      const MemoryWaitScoreboard::Hazard &hazard,
                                      std::string_view counter, uint32_t required_wait) {
  WaitFinding finding{wave, hazard, std::string(counter), required_wait};
  std::lock_guard lock(mutex_);
  ++observed_;
  ++revision_;
  if (wait_findings_.size() < kMaxFindings)
    wait_findings_.push_back(std::move(finding));
  else
    ++suppressed_;
}

void MemoryDiagnosticReport::add_race(const Finding &finding) { add_races(std::span(&finding, 1)); }

void MemoryDiagnosticReport::add_races(std::span<const Finding> findings) {
  if (findings.empty())
    return;
  std::lock_guard lock(mutex_);
  observed_ += findings.size();
  ++revision_;
  const size_t retained = std::min(findings.size(), kMaxFindings - race_findings_.size());
  race_findings_.insert(race_findings_.end(), findings.begin(), findings.begin() + retained);
  suppressed_ += findings.size() - retained;
}

void MemoryDiagnosticReport::add_suppressed_race() { add_suppressed_races(1); }

void MemoryDiagnosticReport::add_suppressed_races(uint64_t count) {
  if (!count)
    return;
  std::lock_guard lock(mutex_);
  observed_ += count;
  suppressed_ += count;
  ++revision_;
}

void MemoryDiagnosticReport::set_coverage_degraded() {
  std::lock_guard lock(mutex_);
  if (coverage_degraded_)
    return;
  coverage_degraded_ = true;
  ++revision_;
}

void MemoryDiagnosticReport::add_race_statistics(MemoryRaceChecker::DispatchStatistics statistics) {
  MemoryRaceChecker::DispatchStatistics retired;
  std::lock_guard lock(mutex_);
  ++completed_race_dispatches_;
  race_statistics_.add(statistics.counters);
  if (recent_race_statistics_.size() == kMaxRaceDispatchStatistics) {
    // Move ownership of the oldest shard detail out first so its allocation is
    // released only after the report mutex is released.
    retired = std::move(recent_race_statistics_.front());
    std::move(recent_race_statistics_.begin() + 1, recent_race_statistics_.end(),
              recent_race_statistics_.begin());
    recent_race_statistics_.back() = std::move(statistics);
  } else {
    recent_race_statistics_.push_back(std::move(statistics));
  }
  ++revision_;
}

void MemoryDiagnosticReport::flush() { flush_if_needed(true); }

void MemoryDiagnosticReport::flush_if_needed(bool write_empty) {
  // Only a report writer waits for another report writer. Producers take only
  // mutex_ and can continue while the snapshot is formatted and written.
  std::lock_guard writer_lock(writer_mutex_);
  Snapshot snapshot;
  {
    std::lock_guard state_lock(mutex_);
    if (revision_ == written_revision_ && (written_ || !write_empty))
      return;
    snapshot.wait_findings = wait_findings_;
    snapshot.race_findings = race_findings_;
    snapshot.race_statistics = race_statistics_;
    snapshot.recent_race_statistics = recent_race_statistics_;
    snapshot.completed_race_dispatches = completed_race_dispatches_;
    snapshot.observed = observed_;
    snapshot.suppressed = suppressed_;
    snapshot.revision = revision_;
    snapshot.coverage_degraded = coverage_degraded_;
  }

  const bool written = write_snapshot(snapshot);
  bool warn = false;
  {
    std::lock_guard state_lock(mutex_);
    if (written) {
      written_ = true;
      written_revision_ = snapshot.revision;
      write_error_reported_ = false;
    } else if (!write_error_reported_) {
      write_error_reported_ = true;
      warn = true;
    }
  }
  if (warn)
    util::Logger::warn(
        std::format("memory-diagnostics: cannot write memory_diagnostic_report {}", path_));
}

bool MemoryDiagnosticReport::write_snapshot(const Snapshot &snapshot) const {
  std::ofstream out(path_);
  if (!out)
    return false;
  out << "{\n  \"version\": 1,\n  \"findings\": [";
  bool first = true;
  for (const WaitFinding &finding : snapshot.wait_findings) {
    out << (first ? "" : ",");
    first = false;
    out << std::format("\n    {{\"checker\":\"memory-wait\",\"kind\":\"wait\",\"resource\":\"{}\","
                       "\"producer_pc\":{},\"consumer_pc\":{},\"process\":{},\"dispatch\":{},"
                       "\"workgroup\":{},\"wave\":{},\"required_wait\":{}}}",
                       finding.counter, finding.hazard.producer.pc, finding.hazard.consumer_pc,
                       finding.wave.process, finding.wave.dispatch, finding.wave.workgroup,
                       finding.wave.wave, finding.required_wait);
  }
  for (const Finding &finding : snapshot.race_findings) {
    out << (first ? "" : ",");
    first = false;
    out << std::format("\n    {{\"checker\":\"memory-race\",\"kind\":\"{}\",\"resource\":\"{}\","
                       "\"address\":{},\"size\":{},\"producer_pc\":{},\"consumer_pc\":{},"
                       "\"process\":{},\"dispatch\":{},\"producer_workgroup\":{},"
                       "\"producer_wave\":{},\"producer_lane\":{},\"consumer_workgroup\":{},"
                       "\"consumer_wave\":{},\"consumer_lane\":{},\"action\":\"{}\"}}",
                       kind_name(finding.kind), finding.space == Space::LDS ? "lds" : "global",
                       finding.address, finding.size, finding.producer.pc, finding.consumer.pc,
                       finding.consumer.wave.process, finding.consumer.wave.dispatch,
                       finding.producer.wave.workgroup, finding.producer.wave.wave,
                       finding.producer.lane, finding.consumer.wave.workgroup,
                       finding.consumer.wave.wave, finding.consumer.lane, action_name(finding));
  }
  out << "\n  ],\n  \"race_statistics\": {\n    \"completed_dispatches\": "
      << snapshot.completed_race_dispatches << ",\n    \"totals\": ";
  write_statistics(out, snapshot.race_statistics);
  out << ",\n    \"last_dispatch\": ";
  if (snapshot.recent_race_statistics.empty()) {
    out << "null";
  } else {
    write_dispatch_statistics(out, snapshot.recent_race_statistics.back());
  }
  out << ",\n    \"recent_dispatches\": [";
  bool first_dispatch = true;
  for (const auto &statistics : snapshot.recent_race_statistics) {
    out << (first_dispatch ? "" : ",");
    first_dispatch = false;
    write_dispatch_statistics(out, statistics);
  }
  out << "]";
  out << std::format("\n  }},\n  \"observed\": {},\n  \"suppressed\": {},\n  "
                     "\"coverage_degraded\": {}\n}}\n",
                     snapshot.observed, snapshot.suppressed,
                     snapshot.coverage_degraded ? "true" : "false");
  out.flush();
  return static_cast<bool>(out);
}

} // namespace rocjitsu::amdgpu
