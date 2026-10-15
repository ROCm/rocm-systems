// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_set>
#include <vector>

namespace rocjitsu::amdgpu {

class MemoryDiagnosticReport;

/// @brief Address-aware companion to MemoryWaitScoreboard.
/// @details Register readiness remains per-wave and issuer-owned. This object
/// adds the state whose lifetime is wider than a wave: per-workgroup LDS
/// happens-before histories and a sharded dispatch-global shadow. One instance
/// is shared by every CP and CU of a SoC. A sharded map locates each
/// independently locked workgroup state. Global pages are assigned to sharded
/// locks, and an access holds one shard at a time. Unrelated workgroups or pages
/// can therefore proceed concurrently, while keys that hash to one shard can
/// serialize. Like the CU and CP, it identifies a dispatch by its SoC-wide
/// dispatch id and a workgroup by its grid-wide id within that dispatch.
class MemoryRaceChecker {
public:
  enum class Space : uint8_t { LDS, Global };
  enum class Kind : uint8_t { RAW, WAR, WAW };

  struct WaveKey {
    uint32_t process = 0;
    uint32_t dispatch = 0;
    uint32_t workgroup = 0;
    uint32_t wave = 0;
    uint64_t generation = 0;

    bool operator==(const WaveKey &) const = default;
  };

  /// Attributes shared by every lane of one instruction. `address`, `size`,
  /// and `lane` are used only by the single-range overloads.
  struct Access {
    WaveKey wave;
    uint64_t pc = 0;
    uint64_t address = 0;
    uint32_t size = 0;
    int32_t lane = -1;
    bool read = false;
    bool write = false;
    bool atomic = false;
    /// Same nonzero class means the ISA orders two same-wave accesses.
    uint32_t ordered_class = 0;
    std::array<MemoryWaitScoreboard::CompletionToken, 2> completions{};
    uint8_t completion_count = 0;
  };

  /// One contiguous byte range accessed by one lane (-1 for scalar or DMA).
  struct LaneRange {
    uint64_t address = 0;
    uint32_t size = 0;
    int32_t lane = -1;
  };

  struct Finding {
    Space space = Space::LDS;
    Kind kind = Kind::RAW;
    Access producer;
    Access consumer;
    uint64_t address = 0;
    uint32_t size = 0;
    bool requires_barrier = false;
  };

  using Reporter = std::function<void(const Finding &)>;

  static constexpr uint64_t kDefaultGlobalShadowBytes = uint64_t{512} << 20;

  /// @brief Create a checker that passes findings to @p reporter.
  /// @details An empty reporter logs each finding as a `memory-race:` warning.
  explicit MemoryRaceChecker(Reporter reporter = {});
  ~MemoryRaceChecker();

  void set_reporter(Reporter reporter);
  /// @brief Also record findings, suppressions and coverage loss in @p report.
  void set_report(std::shared_ptr<MemoryDiagnosticReport> report);
  /// @brief Bound the memory that remembers global accesses.
  /// @details Accesses beyond the budget are still checked against retained
  /// records but are not retained.
  void set_global_shadow_budget(uint64_t bytes);
  uint64_t global_shadow_budget() const { return global_shadow_budget_.load(); }

  /// @brief Register a wave of a workgroup that is starting.
  /// @details Every wave of a workgroup must be registered before any of them
  /// runs, so a barrier never treats an unregistered wave as having joined.
  void begin_wave(const WaveKey &key);
  /// @brief Mark a wave ended. Its retained accesses stay until a barrier orders them.
  void end_wave(const WaveKey &key);

  /// @brief Check one routed LDS instruction against the issuing wave's
  /// outstanding accesses and the workgroup's unsynchronized history, then
  /// retain it.
  void lds_access(const Access &instruction, std::span<const LaneRange> lanes,
                  const MemoryWaitScoreboard &scoreboard);
  void lds_access(const Access &access, const MemoryWaitScoreboard &scoreboard) {
    const LaneRange lane{access.address, access.size, access.lane};
    lds_access(access, std::span(&lane, 1), scoreboard);
  }
  /// @brief Check one global, scalar or tensor instruction against other
  /// workgroups of the same dispatch, then retain it.
  void global_access(const Access &instruction, std::span<const LaneRange> lanes);
  void global_access(const Access &access) {
    const LaneRange lane{access.address, access.size, access.lane};
    global_access(access, std::span(&lane, 1));
  }

  /// @brief Drop a wave's outstanding LDS operations that @p scoreboard proves complete.
  void retire_completed(const WaveKey &key, const MemoryWaitScoreboard &scoreboard);
  /// @brief Synchronize exactly the supplied waves.
  /// @details Supports workgroup, named and cluster barriers without treating
  /// nonparticipants as synchronized. Cluster members are synchronized per
  /// workgroup, because LDS history is per workgroup.
  void barrier(std::span<const WaveKey> members);

  /// @brief Release a retired workgroup's LDS history.
  void end_workgroup(uint32_t dispatch, uint32_t workgroup);
  /// @brief Release everything retained for a dispatch the CP no longer tracks.
  void end_dispatch(uint32_t dispatch);

  /// @brief Number of workgroups with live LDS state.
  size_t tracked_workgroups() const;
  /// @brief Storage currently charged against the global shadow budget.
  uint64_t global_shadow_bytes() const;
  /// @brief Distinct findings observed, including suppressed ones.
  uint64_t finding_count() const { return finding_count_.load(std::memory_order_relaxed); }
  uint64_t suppressed_count() const { return suppressed_count_.load(std::memory_order_relaxed); }
  bool coverage_degraded() const { return coverage_degraded_.load(std::memory_order_relaxed); }

private:
  struct WorkgroupState;
  struct WorkgroupShard;
  struct GlobalShard;

  struct FindingKey {
    Space space;
    Kind kind;
    uint32_t dispatch;
    uint64_t producer_pc;
    uint64_t consumer_pc;
    uint64_t address;
    bool operator==(const FindingKey &) const = default;
  };
  struct FindingKeyHash {
    size_t operator()(const FindingKey &key) const;
  };

  static constexpr size_t kWorkgroupShards = 64;
  static constexpr size_t kGlobalShards = 256;
  static constexpr size_t kMaxFindings = 64;

  std::shared_ptr<WorkgroupState> find_workgroup(const WaveKey &key, bool create);
  void barrier_workgroup(std::span<const WaveKey> members);
  void mark_coverage_degraded();
  void report(std::vector<Finding> &findings);
  void report_one(Finding finding);

  std::unique_ptr<WorkgroupShard[]> workgroups_;
  std::unique_ptr<GlobalShard[]> global_;
  std::atomic<uint64_t> global_shadow_budget_{kDefaultGlobalShadowBytes};
  // Set once any state is retained; until then dispatch cleanup has nothing to do.
  std::atomic<bool> tracking_{false};
  std::mutex report_mutex_;
  Reporter reporter_;
  std::shared_ptr<MemoryDiagnosticReport> report_;
  std::unordered_set<FindingKey, FindingKeyHash> reported_;
  std::atomic<uint64_t> finding_count_{0};
  std::atomic<uint64_t> suppressed_count_{0};
  std::atomic<bool> coverage_degraded_{false};
};

} // namespace rocjitsu::amdgpu
