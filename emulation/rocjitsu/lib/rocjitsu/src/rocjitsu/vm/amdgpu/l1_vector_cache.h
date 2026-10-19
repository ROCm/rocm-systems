// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/memory_transaction.h"
#include "simdojo/components/cache.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

namespace rocjitsu {
namespace amdgpu {

class GpuMemory;
class L2Cache;
class DeviceCacheCoherence;
struct DeviceCacheAliasScope;
class L1VectorCacheTestAccess;
class RequestMtypeResolver;
struct CoalescedLaneRun;

/// @brief L1 Vector Cache (V$) controller for FLAT/MUBUF/MTBUF instructions.
///
/// @details 32KB, 128-byte lines, 4-way set-associative with LRU. All cacheable
/// stores use write-through to L2 so that partial writes from different CUs
/// sharing the same L2 are properly merged at byte granularity.
///
/// CDNA3 V$ geometry: 128B lines, 64 sets, 4-way = 32KB.
class L1VectorCache {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 7;
  static constexpr uint32_t NUM_SETS = 64;
  static constexpr uint32_t ASSOCIATIVITY = 4;

  using CacheStore =
      simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY,
                     simdojo::LRUPolicy<NUM_SETS, ASSOCIATIVITY>, /*LazyStorage=*/true>;
  static constexpr uint32_t LINE_SIZE = CacheStore::LINE_SIZE;

  explicit L1VectorCache(L2Cache *l2 = nullptr, MemoryTransactionPolicy transactions = {});

  MemoryTransactionPolicy transaction_policy() const { return transactions_; }
  ~L1VectorCache();

  L1VectorCache(const L1VectorCache &) = delete;
  L1VectorCache &operator=(const L1VectorCache &) = delete;
  L1VectorCache(L1VectorCache &&) = delete;
  L1VectorCache &operator=(L1VectorCache &&) = delete;

  /// @brief Rebind the backing L2 while the CU is quiesced.
  /// @details Registry transfer and backing pointer replacement are separate
  /// steps, so this must finish before ordinary accesses or maintenance.
  void set_l2(L2Cache *l2);
  void set_gpu_vm(GpuVm *gpu_vm);

  /// @param addr_base_offset Low bits of a uniform address contribution applied after
  /// swizzling. This does not change the first-byte addresses in @p addrs; it only keeps the
  /// contribution from moving logical swizzle-unit boundaries.
  /// @param swizzle_unit Bytes per swizzle unit (4 or 16).
  /// @param element_lane_masks Empty when every element uses @p lane_mask;
  /// otherwise contains exactly @p num_elems masks. In the latter form,
  /// @p lane_mask is the union of lanes valid for at least one element.
  VmAccessOutcome load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                       uint32_t num_elems, uint8_t *dst, Mtype mtype, bool non_temporal,
                       bool request_l1_bypass, uint32_t wf_size, uint32_t vmid = 0,
                       uint32_t addr_stride = 0, uint32_t addr_base_offset = 0,
                       std::span<const uint64_t> element_lane_masks = {},
                       uint32_t swizzle_unit = 4);

  /// @param addr_base_offset Low bits of a uniform address contribution applied after
  /// swizzling. This does not change the first-byte addresses in @p addrs; it only keeps the
  /// contribution from moving logical swizzle-unit boundaries.
  /// @param swizzle_unit Bytes per swizzle unit (4 or 16).
  /// @param element_lane_masks Empty when every element uses @p lane_mask;
  /// otherwise contains exactly @p num_elems masks. In the latter form,
  /// @p lane_mask is the union of lanes valid for at least one element.
  /// Private batching is opt-in for unobserved functional requests whose source
  /// data remain in instruction-owned storage until the operation completes.
  VmAccessOutcome store(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                        uint32_t num_elems, const uint8_t *src, Mtype mtype, bool non_temporal,
                        uint32_t wf_size, uint32_t vmid = 0, uint32_t addr_stride = 0,
                        uint32_t addr_base_offset = 0,
                        std::span<const uint64_t> element_lane_masks = {},
                        uint32_t swizzle_unit = 4, bool allow_private_batch = false);

  /// Read one translated local operand through V$ and L2 using the queue's
  /// exact address-space snapshot. Only complete cacheable local lines enter
  /// this path; the caller routes peer, system, and UC requests through VM.
  [[nodiscard]] VmAccessOutcome load_translated_local(const GpuVmAccess &access, uint64_t address,
                                                      std::span<uint8_t> bytes,
                                                      Mtype instruction_mtype, bool non_temporal,
                                                      bool request_l1_bypass);

  void invalidate(uint64_t addr, uint32_t vmid = 0);
  void invalidate_all();
  void flush_all();

  /// @brief Invalidate lines belonging to one ordinary VM root at completion.
  /// @details Matches resident VMID and cache identity across the whole root,
  /// including lines admitted under different page generations. Other roots
  /// and address spaces remain resident.
  void invalidate_for_completion(uint32_t vmid, uint64_t cache_identity);

  /// Probe only resident, byte-valid data under the supplied raw grant key.
  /// The span must be nonempty and contained in one V$ line. False is a miss;
  /// this method performs no VM admission or backing request.
  [[nodiscard]] bool try_read_peer_cached(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid,
                                          uint64_t admission_generation, uint64_t cache_identity);

  /// Optionally fill one V$ line fragment from a canonical peer read.
  /// The caller must hold exact ingress, current-VM, and canonical-line
  /// freshness authority through this call. The span must be nonempty and
  /// contained in one line. A dirty-line refusal leaves tags and replacement
  /// order unchanged; normal epoch reconciliation may discard stale lines.
  /// This method never invokes VM admission or backing traffic.
  [[nodiscard]] bool try_fill_peer_cached(uint64_t addr, std::span<const uint8_t> src,
                                          uint32_t vmid, uint64_t admission_generation,
                                          uint64_t cache_identity);

  uint64_t store_count() const {
    std::lock_guard lock(maintenance_mutex_);
    return store_count_;
  }
  uint64_t store_active_count() const {
    std::lock_guard lock(maintenance_mutex_);
    return store_active_count_;
  }
  uint64_t store_l2_writes() const {
    std::lock_guard lock(maintenance_mutex_);
    return store_l2_writes_;
  }

private:
  friend class DeviceCacheCoherence;
  friend class L1VectorCacheTestAccess;

  [[nodiscard]] uint64_t current_cache_identity(uint32_t vmid) const;
  [[nodiscard]] bool has_unidentified_overlap(std::span<const DeviceCacheAliasScope> aliases);
  void invalidate_selected_lines(std::span<const DeviceCacheAliasScope> aliases);
  void invalidate_all_lines();
  void synchronize_epoch();
  VmAccessOutcome read_bytes(uint64_t addr, uint8_t *dst, uint32_t size, bool non_temporal,
                             bool request_l1_bypass, uint32_t vmid, RequestMtypeResolver &mtypes);
  VmAccessOutcome write_bytes(uint64_t addr, const uint8_t *src, uint32_t size, bool non_temporal,
                              uint32_t vmid, RequestMtypeResolver &mtypes);
  VmAccessOutcome read_reordered(const CoalescedLaneRun &run, const uint64_t *addrs, uint8_t *dst,
                                 uint32_t stride, bool non_temporal, bool request_l1_bypass,
                                 uint32_t vmid, RequestMtypeResolver &mtypes);
  VmAccessOutcome write_reordered(const CoalescedLaneRun &run, const uint64_t *addrs,
                                  const uint8_t *src, uint32_t stride, bool non_temporal,
                                  uint32_t vmid, RequestMtypeResolver &mtypes);
  VmAccessOutcome store_swizzled(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                                 uint32_t num_elems, const uint8_t *src, Mtype mtype,
                                 bool non_temporal, uint32_t vmid, uint32_t addr_stride,
                                 uint32_t addr_base_offset,
                                 std::span<const uint64_t> element_lane_masks,
                                 uint32_t swizzle_unit, bool allow_private_batch,
                                 RequestMtypeResolver &mtypes);
  [[nodiscard]] VmAccessAdmission admit_access(uint64_t addr, uint32_t size, uint32_t vmid,
                                               VmAccessKind kind, bool report_fault = true) const;
  void cache_partial_bytes(uint64_t addr, const uint8_t *src, uint32_t size, uint32_t vmid,
                           uint64_t admission_generation, uint64_t cache_identity);
  // Store allocation leaves untouched bytes absent instead of reading backing.
  VmAccessOutcome ensure_line(uint64_t addr, uint32_t vmid, uint64_t admission_generation,
                              bool fetch_on_miss = true);

  CacheStore cache_;
  // Acquired before any L2 maintenance lock on ordinary accesses and scoped maintenance.
  mutable std::mutex maintenance_mutex_;
  L2Cache *l2_;
  const MemoryTransactionPolicy transactions_;
  std::shared_ptr<DeviceCacheCoherence> registered_domain_;
  GpuVm *gpu_vm_ = nullptr;
  VmMtypeCache mtype_cache_;
  uint64_t coherence_epoch_ = 0;
  uint64_t store_count_ = 0;
  uint64_t store_active_count_ = 0;
  uint64_t store_l2_writes_ = 0;
};

} // namespace amdgpu
} // namespace rocjitsu
