// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legacy_page_table.h
/// @brief Frontend-neutral page-table data used by legacy host mappings.

#pragma once

#include "rocjitsu/vm/amdgpu/mtype.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rocjitsu::amdgpu {

inline constexpr uint64_t kLegacyPageShift = 12;
inline constexpr uint64_t kLegacyPageSize = uint64_t{1} << kLegacyPageShift;

/// @brief Who owns host storage named by a compatibility mapping.
enum class LegacyHostExtentOwner : uint8_t {
  Driver,
  /// Private read-write driver mapping backed by a file sealed against shrinking.
  DriverSealedRam,
  Application,
};

/// @brief Which device interface owns the GPU VA covered by an extent.
/// @details This is independent of who owns the host storage. In particular,
/// both a KFD allocation and a GEM BO may use application-owned host pages.
enum class LegacyGpuVaOwnerKind : uint8_t { Kfd, Gem };

struct LegacyGpuVaOwner {
  LegacyGpuVaOwnerKind kind = LegacyGpuVaOwnerKind::Kfd;
  /// Optional allocation or BO identity for ownership-preserving teardown.
  uint64_t identity = 0;

  bool operator==(const LegacyGpuVaOwner &) const = default;
};

/// @brief One host-backed interval within a GPU page.
class LegacyHostExtent {
public:
  LegacyHostExtent() = default;
  LegacyHostExtent(uint8_t *host, std::size_t backed_bytes, std::size_t page_offset,
                   LegacyHostExtentOwner storage_owner = LegacyHostExtentOwner::Application,
                   LegacyGpuVaOwner gpu_va_owner = {})
      : host_ptr(host), host_backed_bytes(backed_bytes), gpu_page_offset(page_offset),
        owner(storage_owner), va_owner(gpu_va_owner) {}

  uint8_t *host_ptr = nullptr;
  std::size_t host_backed_bytes = 0;
  std::size_t gpu_page_offset = 0;
  LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application;
  LegacyGpuVaOwner va_owner;

  bool operator==(const LegacyHostExtent &) const = default;
};

/// @brief Per-page compatibility translation entry.
class LegacyPageTableEntry {
public:
  LegacyPageTableEntry() = default;
  LegacyPageTableEntry(uint8_t *host_ptr, Mtype page_mtype,
                       LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application,
                       LegacyGpuVaOwner va_owner = {})
      : mtype(page_mtype), host_extents{{host_ptr, kLegacyPageSize, 0, owner, va_owner}} {}
  LegacyPageTableEntry(uint8_t *host_ptr, Mtype page_mtype, std::size_t host_backed_bytes,
                       std::size_t gpu_page_offset,
                       LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application,
                       LegacyGpuVaOwner va_owner = {})
      : mtype(page_mtype),
        host_extents{{host_ptr, host_backed_bytes, gpu_page_offset, owner, va_owner}} {}

  Mtype mtype = Mtype::RW;
  std::vector<LegacyHostExtent> host_extents;

  bool operator==(const LegacyPageTableEntry &) const = default;
};

using LegacyPageTable = std::pmr::unordered_map<uint64_t, LegacyPageTableEntry>;

/// @brief A run of whole, uniformly mapped GPU pages observed at one generation.
/// @details Every page in [begin, end) has exactly one full-page host extent at
/// page offset zero, the same MTYPE and the same storage-owner class, and the
/// host pages are consecutive, so one offset translates the whole run.
struct LegacyLocalExtent {
  uint64_t begin = 0;
  uint64_t end = 0;
  uint8_t *host_begin = nullptr;
  Mtype mtype = Mtype::RW;
  LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application;

  [[nodiscard]] bool contains(uint64_t address, uint64_t size) const {
    return address >= begin && address < end && size <= end - address;
  }
  [[nodiscard]] uint8_t *host(uint64_t address) const { return host_begin + (address - begin); }
};

/// @brief Local extents shared by every reader of one page-table generation.
/// @details Builders publish extents only while holding the page table shared
/// and after reading the generation under that lock. Mutators bump the
/// generation under the exclusive lock, so a published extent is exactly as
/// current as a copied PTE carrying the same generation.
class LegacyLocalExtentTable {
public:
  [[nodiscard]] std::optional<LegacyLocalExtent> find(uint64_t address, uint64_t generation) const {
    std::shared_lock lock(mutex_);
    if (generation != generation_)
      return std::nullopt;
    const auto it = upper_bound(address);
    if (it == extents_.begin())
      return std::nullopt;
    const auto &extent = *std::prev(it);
    if (address >= extent.end)
      return std::nullopt;
    return extent;
  }

  void publish(const LegacyLocalExtent &extent, uint64_t generation) {
    std::unique_lock lock(mutex_);
    if (generation < generation_)
      return;
    if (generation != generation_) {
      extents_.clear();
      generation_ = generation;
    }
    // Builders bound each walk, so one maximal run may be published in
    // pieces. Overlapping pieces of one generation belong to the same run and
    // share its host offset; merge them.
    LegacyLocalExtent merged = extent;
    auto first = upper_bound(extent.begin);
    if (first != extents_.begin() && std::prev(first)->end > extent.begin)
      --first;
    auto last = first;
    for (; last != extents_.end() && last->begin < extent.end; ++last) {
      if (last->begin < merged.begin) {
        merged.host_begin = last->host_begin;
        merged.begin = last->begin;
      }
      merged.end = std::max(merged.end, last->end);
    }
    extents_.insert(extents_.erase(first, last), merged);
  }

private:
  [[nodiscard]] std::vector<LegacyLocalExtent>::const_iterator upper_bound(uint64_t address) const {
    return std::upper_bound(extents_.begin(), extents_.end(), address,
                            [](uint64_t a, const LegacyLocalExtent &e) { return a < e.begin; });
  }
  [[nodiscard]] std::vector<LegacyLocalExtent>::iterator upper_bound(uint64_t address) {
    return std::upper_bound(extents_.begin(), extents_.end(), address,
                            [](uint64_t a, const LegacyLocalExtent &e) { return a < e.begin; });
  }

  mutable std::shared_mutex mutex_;
  uint64_t generation_ = 0;
  std::vector<LegacyLocalExtent> extents_;
};

/// @brief Lifetime-safe admission state for lock-free copied legacy PTEs.
///
/// @details A reader joins only while the generation it cached is current.
/// Mutations advance the generation first, then wait for already-admitted
/// readers before changing page-table storage. The shared state may safely
/// outlive the process and address-space objects retained by thread-local
/// caches. Admissions must be released by the acquiring thread.
class LegacyPageTableCacheState {
public:
  [[nodiscard]] uint64_t generation() const { return generation_.load(std::memory_order_seq_cst); }

  /// Extents built under this state's generations; see LegacyLocalExtentTable.
  [[nodiscard]] LegacyLocalExtentTable &local_extents() const { return local_extents_; }

  [[nodiscard]] bool try_acquire(uint64_t expected_generation) const {
    if (generation_.load(std::memory_order_seq_cst) != expected_generation)
      return false;
    reader_count().fetch_add(1, std::memory_order_seq_cst);
    if (generation_.load(std::memory_order_seq_cst) == expected_generation)
      return true;
    release();
    return false;
  }

  void release() const {
    auto &active = reader_count();
    if (active.fetch_sub(1, std::memory_order_seq_cst) == 1 &&
        draining_.load(std::memory_order_seq_cst))
      active.notify_all();
  }

  /// @brief Exclude new cached readers and drain the previous generation.
  /// @pre The caller owns the mutation-side serialization for the page table
  /// or address-space registration it is about to change.
  void invalidate_and_wait() {
    // Publish the wakeup requirement before invalidating admissions. A last
    // reader either sees this flag and wakes us, or leaves before the scan
    // observes its count. Ordinary releases never touch the library's shared
    // atomic-wait bookkeeping.
    draining_.store(true, std::memory_order_seq_cst);
    generation_.fetch_add(1, std::memory_order_seq_cst);
    for (auto &shard : readers_) {
      uint64_t active = shard.count.load(std::memory_order_seq_cst);
      while (active != 0) {
        shard.count.wait(active, std::memory_order_seq_cst);
        active = shard.count.load(std::memory_order_seq_cst);
      }
    }
    draining_.store(false, std::memory_order_seq_cst);
  }

private:
  // Reads dominate mutations. Keep the generation off the counters' cache
  // lines, and let unrelated simulation threads admit independently. Hash
  // collisions preserve exclusion, including nested reads on one thread.
  static constexpr std::size_t kReaderShards = 128;
  struct alignas(64) ReaderShard {
    std::atomic<uint64_t> count{0};
  };
  alignas(64) std::atomic<uint64_t> generation_{1};
  std::atomic<bool> draining_{false};
  mutable std::array<ReaderShard, kReaderShards> readers_;
  mutable LegacyLocalExtentTable local_extents_;

  std::atomic<uint64_t> &reader_count() const {
    // Thread identity gives inline calls in different shared libraries the
    // same slot, without retaining thread-owned state in the cache object.
    static thread_local const std::size_t shard =
        std::hash<std::thread::id>{}(std::this_thread::get_id()) % kReaderShards;
    return readers_[shard].count;
  }
};

} // namespace rocjitsu::amdgpu
