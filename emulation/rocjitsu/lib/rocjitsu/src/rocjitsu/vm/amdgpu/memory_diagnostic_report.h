// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/memory_race_checker.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::amdgpu {

/// @brief Machine-readable record of one GPU's core memory diagnostics.
/// @details Collects `memory-wait` findings from every CU and `memory-race`
/// findings and bounded work statistics from the SoC-wide checker, then writes
/// them as JSON to this GPU's configured report path. Each kind keeps at most
/// kMaxFindings findings. JSON
/// `observed` counts delivered and explicitly suppressed events; the shared
/// `suppressed` count includes either vector overflowing and race events
/// rejected by the checker's console cap. The file is rewritten only when its
/// content changed, so a SoC that recorded nothing, such as one built only to
/// validate a config, never replaces a report written by another.
class MemoryDiagnosticReport {
public:
  static constexpr size_t kMaxFindings = 1024;

  explicit MemoryDiagnosticReport(std::string path);
  /// @brief Write findings recorded since the last flush().
  ~MemoryDiagnosticReport();

  MemoryDiagnosticReport(const MemoryDiagnosticReport &) = delete;
  MemoryDiagnosticReport &operator=(const MemoryDiagnosticReport &) = delete;

  const std::string &path() const { return path_; }

  /// @brief Record one register-readiness hazard from a wave's scoreboard.
  void add_wait(const MemoryRaceChecker::WaveKey &wave, const MemoryWaitScoreboard::Hazard &hazard,
                std::string_view counter, uint32_t required_wait);
  /// @brief Record one deduplicated address race.
  void add_race(const MemoryRaceChecker::Finding &finding);
  /// @brief Record a batch of deduplicated address races under one short lock.
  void add_races(std::span<const MemoryRaceChecker::Finding> findings);
  /// @brief Count an address race that the checker observed but did not report.
  void add_suppressed_race();
  /// @brief Count several address races rejected by the checker's console cap.
  void add_suppressed_races(uint64_t count);
  /// @brief Record that bounded race state stopped retaining accesses.
  void set_coverage_degraded();
  /// @brief Add fixed-work counters for one completed dispatch.
  /// @details The report retains cumulative totals and bounded recent
  /// per-dispatch/per-shard detail.
  void add_race_statistics(MemoryRaceChecker::DispatchStatistics statistics);
  /// @brief Rewrite the file if anything changed since it was last written.
  /// @details The first flush writes even an empty report, replacing one left
  /// at the same path by an earlier run.
  void flush();

private:
  static constexpr size_t kMaxRaceDispatchStatistics = 8;

  struct WaitFinding {
    MemoryRaceChecker::WaveKey wave;
    MemoryWaitScoreboard::Hazard hazard;
    std::string counter;
    uint32_t required_wait = 0;
  };

  struct Snapshot {
    std::vector<WaitFinding> wait_findings;
    std::vector<MemoryRaceChecker::Finding> race_findings;
    MemoryRaceChecker::Statistics race_statistics;
    std::vector<MemoryRaceChecker::DispatchStatistics> recent_race_statistics;
    uint64_t completed_race_dispatches = 0;
    uint64_t observed = 0;
    uint64_t suppressed = 0;
    uint64_t revision = 0;
    bool coverage_degraded = false;
  };

  void flush_if_needed(bool write_empty);
  bool write_snapshot(const Snapshot &snapshot) const;

  // Writers serialize with one another, but never hold the state mutex while
  // formatting JSON or waiting on the filesystem.
  std::mutex writer_mutex_;
  std::mutex mutex_;
  std::string path_;
  std::vector<WaitFinding> wait_findings_;
  std::vector<MemoryRaceChecker::Finding> race_findings_;
  MemoryRaceChecker::Statistics race_statistics_;
  std::vector<MemoryRaceChecker::DispatchStatistics> recent_race_statistics_;
  uint64_t completed_race_dispatches_ = 0;
  uint64_t observed_ = 0;
  uint64_t suppressed_ = 0;
  uint64_t revision_ = 0;
  uint64_t written_revision_ = 0;
  bool coverage_degraded_ = false;
  bool written_ = false;
  bool write_error_reported_ = false;
};

} // namespace rocjitsu::amdgpu
