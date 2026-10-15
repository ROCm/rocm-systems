// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/memory_race_checker.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::amdgpu {

/// @brief Machine-readable record of one GPU's core memory diagnostics.
/// @details Collects `memory-wait` findings from every CU and `memory-race`
/// findings from the SoC-wide checker, then writes them as JSON to this GPU's
/// configured report path. Each kind keeps at most kMaxFindings findings. JSON
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
  /// @brief Count an address race that the checker observed but did not report.
  void add_suppressed_race();
  /// @brief Record that bounded race state stopped retaining accesses.
  void set_coverage_degraded();
  /// @brief Rewrite the file if anything changed since it was last written.
  /// @details The first flush writes even an empty report, replacing one left
  /// at the same path by an earlier run.
  void flush();

private:
  struct WaitFinding {
    MemoryRaceChecker::WaveKey wave;
    MemoryWaitScoreboard::Hazard hazard;
    std::string counter;
    uint32_t required_wait = 0;
  };

  void write_locked();

  std::mutex mutex_;
  std::string path_;
  std::vector<WaitFinding> wait_findings_;
  std::vector<MemoryRaceChecker::Finding> race_findings_;
  uint64_t observed_ = 0;
  uint64_t suppressed_ = 0;
  bool coverage_degraded_ = false;
  bool dirty_ = false;
  bool written_ = false;
  bool write_error_reported_ = false;
};

} // namespace rocjitsu::amdgpu
