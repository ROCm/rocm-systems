// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legacy_page_table.h
/// @brief Frontend-neutral page-table data used by legacy host mappings.

#pragma once

#include "rocjitsu/vm/amdgpu/mtype.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <unordered_map>
#include <vector>

namespace rocjitsu::amdgpu {

inline constexpr uint64_t kLegacyPageShift = 12;
inline constexpr uint64_t kLegacyPageSize = uint64_t{1} << kLegacyPageShift;

/// @brief Who owns host storage named by a compatibility mapping.
enum class LegacyHostExtentOwner : uint8_t {
  Driver,
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
/// caches.
class LegacyPageTableCacheState {
public:
  [[nodiscard]] uint64_t generation() const { return generation_.load(std::memory_order_seq_cst); }

  [[nodiscard]] bool try_acquire(uint64_t expected_generation) const {
    if (generation_.load(std::memory_order_seq_cst) != expected_generation)
      return false;
    active_readers_.fetch_add(1, std::memory_order_seq_cst);
    if (generation_.load(std::memory_order_seq_cst) == expected_generation)
      return true;
    release();
    return false;
  }

  void release() const {
    if (active_readers_.fetch_sub(1, std::memory_order_seq_cst) == 1)
      active_readers_.notify_all();
  }

  /// @brief Exclude new cached readers and drain the previous generation.
  /// @pre The caller owns the mutation-side serialization for the page table
  /// or address-space registration it is about to change.
  void invalidate_and_wait() {
    generation_.fetch_add(1, std::memory_order_seq_cst);
    uint64_t active = active_readers_.load(std::memory_order_seq_cst);
    while (active != 0) {
      active_readers_.wait(active, std::memory_order_seq_cst);
      active = active_readers_.load(std::memory_order_seq_cst);
    }
  }

private:
  mutable std::atomic<uint64_t> generation_{1};
  mutable std::atomic<uint64_t> active_readers_{0};
};

} // namespace rocjitsu::amdgpu
