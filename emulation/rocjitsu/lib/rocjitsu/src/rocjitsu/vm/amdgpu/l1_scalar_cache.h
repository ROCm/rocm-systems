// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_L1_SCALAR_CACHE_H_
#define ROCJITSU_VM_AMDGPU_L1_SCALAR_CACHE_H_

#include "rocjitsu/vm/amdgpu/mtype.h"
#include "simdojo/components/cache.h"

#include <cstdint>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

class GpuMemory;
class L2Cache;

/// @brief L1 Scalar Cache (K$) controller for SMEM instructions.
///
/// 16KB, 64-byte lines, 4-way set-associative with LRU. Supports both
/// scalar loads and stores. Scalar stores allocate in K$ and mark lines
/// dirty; dirty lines are written back to L2 on eviction or when
/// `s_dcache_wb` is executed. `s_dcache_inv` invalidates all lines
/// without writeback.
///
/// CDNA3 K$ geometry: 64B lines, 64 sets, 4-way = 16KB.
class L1ScalarCache {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 6; // 64 bytes
  static constexpr uint32_t NUM_SETS = 64;
  static constexpr uint32_t ASSOCIATIVITY = 4;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;

  explicit L1ScalarCache(L2Cache *l2 = nullptr) : l2_(l2) {}

  /// @brief Set (or replace) the backing L2 cache.
  /// @param l2 New L2 cache (not owned).
  void set_l2(L2Cache *l2) { l2_ = l2; }

  /// @brief Set the memory subsystem for PTE MTYPE lookups.
  void set_memory(GpuMemory *mem) { memory_ = mem; }

  /// @brief Scalar load: read num_dwords contiguous dwords from addr.
  ///
  /// Fetches from K$ on hit, or fills from L2 on miss. Handles requests
  /// that span multiple cache lines.
  void load(uint64_t addr, uint32_t num_dwords, uint32_t *dst, uint32_t vmid = 0);

  /// @brief Scalar load: read num_bytes contiguous bytes from addr.
  void load_bytes(uint64_t addr, uint32_t num_bytes, uint8_t *dst, uint32_t vmid = 0);

  /// @brief Scalar store: write num_dwords contiguous dwords to addr.
  void store(uint64_t addr, uint32_t num_dwords, const uint32_t *src, uint32_t vmid = 0);

  /// @brief Write back all dirty K$ lines to L2 (s_dcache_wb).
  /// @param vmid Ignored. Each dirty line is written back under its own owning
  /// vmid (recorded in the line tag), so a caller-supplied vmid cannot be
  /// correct for a bulk writeback. Retained only for call-site signature symmetry.
  void writeback_all(uint32_t vmid = 0);

  /// @brief Invalidate all K$ lines without writeback (s_dcache_inv).
  void invalidate_all() { cache_.invalidate_all(); }

private:
  void ensure_line(uint64_t addr, uint32_t vmid = 0);
  void flush_line(uint64_t addr, uint32_t vmid = 0);

  CacheStore cache_;
  L2Cache *l2_;
  GpuMemory *memory_ = nullptr;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_L1_SCALAR_CACHE_H_
