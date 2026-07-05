// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_L1_VECTOR_CACHE_H_
#define ROCJITSU_VM_AMDGPU_L1_VECTOR_CACHE_H_

#include "rocjitsu/vm/amdgpu/mtype.h"
#include "simdojo/components/cache.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

class GpuMemory;
class L2Cache;

/// @brief L1 Vector Cache (V$) controller for FLAT/MUBUF/MTBUF instructions.
///
/// @details 32KB, 128-byte lines, 4-way set-associative with LRU. All cacheable
/// stores use write-through to L2 so that partial writes from different CUs
/// sharing the same L2 are properly merged at byte granularity.
///
/// CDNA3 V$ geometry: 128B lines, 64 sets, 4-way = 32KB.
class L1VectorCache {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 7; // 128 bytes
  static constexpr uint32_t NUM_SETS = 64;
  static constexpr uint32_t ASSOCIATIVITY = 4;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;
  static constexpr uint32_t LINE_SIZE = CacheStore::LINE_SIZE;

  explicit L1VectorCache(L2Cache *l2 = nullptr) : l2_(l2) {}

  void set_l2(L2Cache *l2) { l2_ = l2; }
  void set_memory(GpuMemory *mem) { memory_ = mem; }
  void load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size, uint32_t num_elems,
            uint8_t *dst, Mtype mtype, bool non_temporal, bool request_l1_bypass,
            uint32_t vmid = 0);

  void store(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size, uint32_t num_elems,
             const uint8_t *src, Mtype mtype, bool non_temporal, uint32_t vmid = 0);

  void invalidate(uint64_t addr) { cache_.invalidate(addr); }
  void invalidate_all() { cache_.invalidate_all(); }
  void flush_all();

  uint64_t store_count() const { return store_count_; }
  uint64_t store_active_count() const { return store_active_count_; }
  uint64_t store_l2_writes() const { return store_l2_writes_; }

private:
  void read_bytes(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype, bool non_temporal,
                  bool request_l1_bypass, uint32_t vmid);
  void write_bytes(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype, bool non_temporal,
                   uint32_t vmid);
  void ensure_line(uint64_t addr, uint32_t vmid);

  CacheStore cache_;
  L2Cache *l2_;
  GpuMemory *memory_ = nullptr;
  uint64_t store_count_ = 0;
  uint64_t store_active_count_ = 0;
  uint64_t store_l2_writes_ = 0;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_L1_VECTOR_CACHE_H_
