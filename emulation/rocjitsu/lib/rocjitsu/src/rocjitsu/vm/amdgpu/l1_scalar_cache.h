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
/// scalar loads and stores. Cacheable scalar stores allocate in K$, update the
/// resident bytes, and write those bytes through to L2. K$ lines remain clean,
/// so eviction and `s_dcache_wb` never publish a cached full line.
/// `s_dcache_inv` invalidates all resident lines.
///
/// CDNA3 K$ geometry: 64B lines, 64 sets, 4-way = 16KB.
class L1ScalarCache {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 6; // 64 bytes
  static constexpr uint32_t NUM_SETS = 64;
  static constexpr uint32_t ASSOCIATIVITY = 4;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;

  explicit L1ScalarCache(L2Cache *l2 = nullptr);
  ~L1ScalarCache();

  L1ScalarCache(const L1ScalarCache &) = delete;
  L1ScalarCache &operator=(const L1ScalarCache &) = delete;
  L1ScalarCache(L1ScalarCache &&) = delete;
  L1ScalarCache &operator=(L1ScalarCache &&) = delete;

  /// @brief Set (or replace) the backing L2 cache.
  /// @param l2 New L2 cache (not owned).
  void set_l2(L2Cache *l2);

  /// @brief Set the memory subsystem for PTE MTYPE lookups.
  void set_memory(GpuMemory *mem);

  /// @brief Scalar load: read num_dwords contiguous dwords from addr.
  ///
  /// Fetches from K$ on hit, or fills from L2 on miss. Handles requests
  /// that span multiple cache lines.
  void load(uint64_t addr, uint32_t num_dwords, uint32_t *dst, uint32_t vmid = 0);

  /// @brief Scalar load: read num_bytes contiguous bytes from addr.
  void load_bytes(uint64_t addr, uint32_t num_bytes, uint8_t *dst, uint32_t vmid = 0);

  /// @brief Scalar store: write num_dwords contiguous dwords to addr.
  ///
  /// Cacheable stores read-allocate and write through each modified byte range
  /// to L2. UC and CC stores bypass K$.
  void store(uint64_t addr, uint32_t num_dwords, const uint32_t *src, uint32_t vmid = 0);

  /// @brief Handle s_dcache_wb; a no-op because K$ is write-through.
  /// @param vmid Ignored. Retained only for call-site signature symmetry.
  void writeback_all(uint32_t vmid = 0);

  /// @brief Invalidate all clean K$ lines (s_dcache_inv).
  void invalidate_all();

private:
  void ensure_line_locked(uint64_t addr, uint32_t vmid = 0);
  void flush_line_locked(uint64_t addr, uint32_t vmid = 0);
  void invalidate_all_locked();
  void synchronize_epoch_locked();

  CacheStore cache_;
  L2Cache *l2_;
  GpuMemory *memory_ = nullptr;
  uint64_t coherence_epoch_ = 0;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_L1_SCALAR_CACHE_H_
