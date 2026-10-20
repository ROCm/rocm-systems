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
  struct WorkgroupState;

public:
  enum class Space : uint8_t { LDS, Global };
  enum class Kind : uint8_t { RAW, WAR, WAW };

  struct LdsCheckMode {
    constexpr LdsCheckMode(bool same = true, bool cross = true)
        : same_wave(same), cross_wave(cross) {}
    bool same_wave;
    bool cross_wave;
  };

  struct WaveKey {
    uint32_t process = 0;
    uint32_t dispatch = 0;
    uint32_t workgroup = 0;
    uint32_t wave = 0;
    uint64_t generation = 0;

    bool operator==(const WaveKey &) const = default;
  };

  /// @brief Lifetime-safe cached location of one registered wave.
  /// @details The handle keeps its workgroup state alive and avoids repeated
  /// workgroup-map lookup and linear wave search. Checker methods still take
  /// the per-workgroup mutex before accessing the state.
  class WaveBinding {
  public:
    WaveBinding() = default;
    bool valid_for(const WaveKey &key) const;
    explicit operator bool() const { return state_ != nullptr; }

  private:
    friend class MemoryRaceChecker;
    std::shared_ptr<WorkgroupState> state_;
    WaveKey key_;
    uint32_t index_ = UINT32_MAX;
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

  /// A compact sequence of equal-size accesses issued by consecutive lanes.
  /// Lane i accesses [address + i * stride, address + i * stride + size).
  struct AffineLaneRange {
    uint64_t address = 0;
    uint64_t stride = 0;
    uint32_t size = 0;
    int32_t first_lane = -1;
    uint32_t count = 0;
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

  /// @brief Optional fixed-work counters for one or more completed dispatches.
  /// @details Collection is enabled when a MemoryDiagnosticReport is attached,
  /// keeping the ordinary console-only path free of timers and counter updates.
  struct Statistics {
    uint64_t lds_accesses = 0;
    uint64_t global_accesses = 0;
    uint64_t lane_ranges = 0;
    uint64_t dword_candidates = 0;
    uint64_t dword_visits = 0;
    uint64_t dword_duplicates_skipped = 0;
    uint64_t workgroup_lookups = 0;
    uint64_t workgroup_creations = 0;
    uint64_t wave_search_steps = 0;
    uint64_t workgroup_map_lock_acquisitions = 0;
    uint64_t workgroup_map_lock_wait_ns = 0;
    uint64_t workgroup_state_lock_acquisitions = 0;
    uint64_t workgroup_state_lock_wait_ns = 0;
    uint64_t global_shard_lock_acquisitions = 0;
    uint64_t global_shard_lock_wait_ns = 0;
    uint64_t shadow_page_lookups = 0;
    uint64_t shadow_pages_created = 0;
    uint64_t shadow_pages_rejected = 0;
    uint64_t empty_record_heads = 0;
    uint64_t nonempty_record_heads = 0;
    uint64_t lds_records_visited = 0;
    uint64_t global_records_visited = 0;
    uint64_t records_inserted = 0;
    uint64_t records_replaced = 0;
    uint64_t records_rejected = 0;
    uint64_t record_vector_growths = 0;
    uint64_t clock_matrix_growths = 0;
    uint64_t pending_vector_growths = 0;
    uint64_t range_vector_growths = 0;
    uint64_t pending_candidates = 0;
    uint64_t barrier_calls = 0;
    uint64_t barrier_members = 0;
    uint64_t barrier_clock_cells = 0;
    uint64_t barrier_carried_ops = 0;
    uint64_t barrier_scratch_vectors = 0;
    uint64_t cleanup_workgroups = 0;
    uint64_t cleanup_dwords = 0;
    uint64_t cleanup_pages = 0;
    uint64_t cleanup_records = 0;
    uint64_t retained_bytes = 0;
    uint64_t rejected_bytes = 0;
    uint64_t budget_reservation_attempts = 0;
    uint64_t budget_reservation_failures = 0;
    uint64_t budget_bytes_released = 0;
    uint64_t page_directory_growths = 0;
    uint64_t page_slab_allocations = 0;
    uint64_t record_slab_allocations = 0;
    uint64_t site_lookups = 0;
    uint64_t sites_interned = 0;
    uint64_t sites_reused = 0;
    uint64_t site_directory_growths = 0;
    uint64_t site_slab_allocations = 0;
    uint64_t saturation_cache_hits = 0;
    uint64_t saturation_cache_inserts = 0;
    uint64_t saturation_cache_invalidations = 0;
    uint64_t affine_accesses = 0;
    uint64_t affine_lane_ranges = 0;
    uint64_t affine_dword_runs = 0;
    uint64_t affine_dword_visits = 0;
    uint64_t affine_pending_ops = 0;
    uint64_t explicit_accesses = 0;

    void add(const Statistics &other);
    bool empty() const;
  };

  struct GlobalShardStatistics {
    uint32_t shard = 0;
    Statistics counters;
  };

  struct DispatchStatistics {
    uint32_t dispatch = 0;
    uint64_t global_shadow_bytes_released = 0;
    uint64_t global_shadow_peak_bytes = 0;
    Statistics counters;
    std::vector<GlobalShardStatistics> global_shards;
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
  /// records but are not retained. The limit is shared by every shard of this
  /// GPU checker; allocator metadata and fixed checker storage are excluded.
  void set_global_shadow_budget(uint64_t bytes);
  uint64_t global_shadow_budget() const;

  /// @brief Register every wave of a workgroup before any of them runs.
  /// @details Sizes the vector clock once and returns one cached binding per
  /// input key. All keys must identify the same workgroup.
  std::vector<WaveBinding> begin_workgroup(std::span<const WaveKey> members);
  /// @brief Register one wave for tests and fallback launch paths.
  WaveBinding begin_wave(const WaveKey &key);
  /// @brief Mark a wave ended. Its retained accesses stay until a barrier orders them.
  void end_wave(const WaveKey &key, const WaveBinding *binding = nullptr);

  /// @brief Check one routed LDS instruction against the issuing wave's
  /// outstanding accesses and the workgroup's unsynchronized history, then
  /// retain it.
  void lds_access(const Access &instruction, std::span<const LaneRange> lanes,
                  const MemoryWaitScoreboard &scoreboard, const WaveBinding *binding = nullptr,
                  LdsCheckMode mode = LdsCheckMode());
  void lds_access(const Access &instruction, const AffineLaneRange &lanes,
                  const MemoryWaitScoreboard &scoreboard, const WaveBinding *binding = nullptr,
                  LdsCheckMode mode = LdsCheckMode());
  void lds_access(const Access &access, const MemoryWaitScoreboard &scoreboard,
                  const WaveBinding *binding = nullptr, LdsCheckMode mode = LdsCheckMode()) {
    const LaneRange lane{access.address, access.size, access.lane};
    lds_access(access, std::span(&lane, 1), scoreboard, binding, mode);
  }
  /// @brief Check one global, scalar or tensor instruction against other
  /// workgroups of the same dispatch, then retain it.
  void global_access(const Access &instruction, std::span<const LaneRange> lanes);
  void global_access(const Access &instruction, const AffineLaneRange &lanes);
  void global_access(const Access &access) {
    const LaneRange lane{access.address, access.size, access.lane};
    global_access(access, std::span(&lane, 1));
  }

  /// @brief Drop a wave's outstanding LDS operations that @p scoreboard proves complete.
  void retire_completed(const WaveKey &key, const MemoryWaitScoreboard &scoreboard,
                        const WaveBinding *binding = nullptr);
  /// @brief Synchronize exactly the supplied waves.
  /// @details Supports workgroup, named and cluster barriers without treating
  /// nonparticipants as synchronized. Cluster members are synchronized per
  /// workgroup, because LDS history is per workgroup.
  void barrier(std::span<const WaveKey> members);

  /// @brief Release a retired workgroup's LDS history.
  void end_workgroup(uint32_t dispatch, uint32_t workgroup);
  /// @brief Release everything retained for a dispatch the CP no longer tracks.
  /// @param cleanup_workgroups Whether LDS-family state can exist for the dispatch.
  /// @param cleanup_global Whether global-family state can exist for the dispatch.
  void end_dispatch(uint32_t dispatch, bool cleanup_workgroups = true, bool cleanup_global = true);

  /// @brief Number of workgroups with live LDS state.
  size_t tracked_workgroups() const;
  /// @brief Storage currently charged against the global shadow budget.
  uint64_t global_shadow_bytes() const;
  /// @brief Highest checker-wide charged storage observed so far.
  uint64_t global_shadow_peak_bytes() const;
  /// @brief Distinct findings observed, including suppressed ones.
  uint64_t finding_count() const { return finding_count_.load(std::memory_order_relaxed); }
  uint64_t suppressed_count() const { return suppressed_count_.load(std::memory_order_relaxed); }
  bool coverage_degraded() const { return coverage_degraded_.load(std::memory_order_relaxed); }

private:
  struct WorkgroupShard;
  struct GlobalShard;
  struct GlobalBudget;
  struct AccessRanges {
    std::span<const LaneRange> explicit_lanes{};
    AffineLaneRange affine{};
    bool is_affine = false;
  };

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
  std::shared_ptr<WorkgroupState> resolve_workgroup(const WaveKey &key, const WaveBinding *binding,
                                                    bool create);
  uint32_t resolve_wave_index(WorkgroupState &state, const WaveKey &key, const WaveBinding *binding,
                              Statistics *statistics, bool create);
  std::unique_lock<std::mutex> lock_workgroup_state(WorkgroupState &state,
                                                    Statistics **statistics = nullptr);
  void merge_completed_statistics(uint32_t dispatch, const Statistics &statistics);
  void barrier_workgroup(std::span<const WaveKey> members);
  void mark_coverage_degraded();
  void lds_access_impl(const Access &instruction, const AccessRanges &lanes,
                       const MemoryWaitScoreboard &scoreboard, const WaveBinding *binding,
                       LdsCheckMode mode);
  void global_access_impl(const Access &instruction, const AccessRanges &lanes);
  void report(std::vector<Finding> &findings);

  std::unique_ptr<WorkgroupShard[]> workgroups_;
  std::unique_ptr<GlobalBudget> global_budget_;
  std::unique_ptr<GlobalShard[]> global_;
  struct StatisticsState;
  std::unique_ptr<StatisticsState> statistics_;
  std::atomic<bool> statistics_enabled_{false};
  // Set once any state is retained; until then dispatch cleanup has nothing to do.
  std::atomic<bool> tracking_{false};
  std::mutex report_mutex_;
  std::shared_ptr<const Reporter> reporter_;
  std::shared_ptr<MemoryDiagnosticReport> report_;
  std::unordered_set<FindingKey, FindingKeyHash> reported_;
  std::atomic<uint64_t> finding_count_{0};
  std::atomic<uint64_t> suppressed_count_{0};
  std::atomic<bool> coverage_degraded_{false};
};

} // namespace rocjitsu::amdgpu
