// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legacy_page_table.h
/// @brief Frontend-neutral page-table data used by legacy host mappings.

#pragma once

#include "rocjitsu/vm/amdgpu/mtype.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory_resource>
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

/// @brief One host-backed interval within a GPU page.
class LegacyHostExtent {
public:
  uint8_t *host_ptr = nullptr;
  std::size_t host_backed_bytes = 0;
  std::size_t gpu_page_offset = 0;
  LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application;

  bool operator==(const LegacyHostExtent &) const = default;
};

/// @brief Per-page compatibility translation entry.
class LegacyPageTableEntry {
public:
  LegacyPageTableEntry() = default;
  LegacyPageTableEntry(uint8_t *host_ptr, Mtype page_mtype,
                       LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application)
      : mtype(page_mtype), host_extents{{host_ptr, kLegacyPageSize, 0, owner}} {}
  LegacyPageTableEntry(uint8_t *host_ptr, Mtype page_mtype, std::size_t host_backed_bytes,
                       std::size_t gpu_page_offset,
                       LegacyHostExtentOwner owner = LegacyHostExtentOwner::Application)
      : mtype(page_mtype), host_extents{{host_ptr, host_backed_bytes, gpu_page_offset, owner}} {}

  Mtype mtype = Mtype::RW;
  std::vector<LegacyHostExtent> host_extents;

  bool operator==(const LegacyPageTableEntry &) const = default;
};

using LegacyPageTable = std::pmr::unordered_map<uint64_t, LegacyPageTableEntry>;

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

  std::atomic<uint64_t> &reader_count() const {
    // Thread identity gives inline calls in different shared libraries the
    // same slot, without retaining thread-owned state in the cache object.
    static thread_local const std::size_t shard =
        std::hash<std::thread::id>{}(std::this_thread::get_id()) % kReaderShards;
    return readers_[shard].count;
  }
};

} // namespace rocjitsu::amdgpu
