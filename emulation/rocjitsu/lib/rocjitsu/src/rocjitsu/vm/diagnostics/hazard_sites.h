// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hazard_sites.h
/// @brief Per-dispatch hazard sites shared by the compute units of one SoC.

#pragma once

#include "rocjitsu/vm/diagnostics/hazard_record.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocjitsu::diagnostics {

/// @brief One printed site. The waves, workgroups and compute units that repeat it share it.
struct HazardSiteKey {
  uint32_t dispatch = 0;
  uint64_t producer_pc = 0;
  uint64_t consumer_pc = 0;
  uint32_t reg = 0;
  uint8_t reg_class = 0;
  uint8_t checker = 0;
  bool operator==(const HazardSiteKey &) const = default;
};

enum class HazardDisposition { New, Repeat, Suppressed };

/// @brief Sites and kernel identities of the dispatches of one SoC.
/// @details Dispatch ids are unique within a SoC, including fanned-out dispatches,
/// so the command processors of all its XCDs share one table. Every occurrence is
/// counted; only the first occurrence of a site is printed. A compute unit counts its
/// repeats in its own cache without taking a lock; the shared table is locked once
/// per site and unit, and when the dispatch finishes.
class HazardSites {
public:
  /// @brief Default settings, for compute units without a command processor.
  HazardSites();
  /// @param gpu Ordinal of the simulated GPU, reported with every record.
  explicit HazardSites(Settings settings, uint32_t gpu = 0);
  ~HazardSites();
  HazardSites(const HazardSites &) = delete;
  HazardSites &operator=(const HazardSites &) = delete;

  const Settings &settings() const { return settings_; }
  uint32_t gpu() const { return gpu_; }
  /// @brief The process-wide writer, for checkers that do not link the simulator.
  ReportWriter &writer() const { return writer_; }

  void register_dispatch(DispatchIdentity identity);
  DispatchIdentity find_dispatch(uint32_t dispatch_id) const;

  /// @brief Count one occurrence reported by @p unit. Build and publish a record only for New.
  /// @details Calls for one unit must not run concurrently, as for a compute unit's
  /// issuing thread.
  HazardDisposition classify(const void *unit, const HazardSiteKey &key);
  /// @brief Print a site that classify() returned as New and append it to the report file.
  void publish(const HazardRecord &record) const;

  /// @brief Print the dispatch summary, then drop the dispatch's sites and identity.
  /// @details Call once every wave of the dispatch has retired.
  void finish_dispatch(uint32_t dispatch_id);

private:
  struct KeyHash {
    size_t operator()(const HazardSiteKey &key) const noexcept;
  };
  struct Site {
    std::atomic<uint64_t> occurrences{0};
  };
  struct Dispatch {
    std::unordered_map<HazardSiteKey, std::unique_ptr<Site>, KeyHash> sites;
    /// Occurrences of sites past max_sites_per_dispatch.
    Site suppressed;
    bool suppression_printed = false;
  };
  /// Repeats of one unit. Only that unit's thread writes a slot outside the mutex;
  /// installing a slot and merging a finished dispatch take the mutex.
  struct UnitCache {
    struct Slot {
      std::atomic<uint32_t> tag{0};
      std::atomic<uint32_t> dispatch{0};
      std::atomic<uint64_t> producer_pc{0};
      std::atomic<uint64_t> consumer_pc{0};
      std::atomic<uint64_t> pending{0};
      std::atomic<Site *> site{nullptr};
      std::atomic<bool> suppressed{false};
    };
    static constexpr size_t kWays = 4;
    static constexpr size_t kGroups = 16;
    std::mutex mutex;
    std::array<Slot, kWays * kGroups> slots{};
    std::array<uint8_t, kGroups> victim{};
  };
  struct UnitEntry {
    std::atomic<const void *> unit{nullptr};
    std::atomic<UnitCache *> cache{nullptr};
  };
  static constexpr unsigned kUnitBits = 11;
  static constexpr size_t kUnitEntries = size_t{1} << kUnitBits;

  UnitCache *cache_for(const void *unit);
  void install(UnitCache &cache, size_t group, uint32_t tag, const HazardSiteKey &key, Site *site,
               bool suppressed);
  static void flush(UnitCache::Slot &slot);
  void merge(UnitCache &cache, uint32_t dispatch_id);

  const Settings settings_;
  const uint32_t gpu_ = 0;
  ReportWriter &writer_;
  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, Dispatch> dispatches_;
  std::unordered_map<uint32_t, DispatchIdentity> identities_;
  std::unique_ptr<UnitEntry[]> unit_storage_;
  std::atomic<UnitEntry *> units_{nullptr};
  std::vector<std::unique_ptr<UnitCache>> caches_;
};

/// @brief The simulator's writer, shared by the hazard sites of every SoC.
ReportWriter &report_writer();

/// @brief Sites of a compute unit that has no command processor, as in unit tests.
HazardSites &unattached_sites(const void *unit);
/// @brief Drop the sites unattached_sites() created for @p unit.
void release_unattached_sites(const void *unit);

} // namespace rocjitsu::diagnostics
