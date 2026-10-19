// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "simdojo/components/cache.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>

namespace rocjitsu {
namespace amdgpu {

class GpuMemory;
class L2Cache;
class DeviceCacheCoherence;
struct DeviceCacheAliasScope;
class L1ScalarCacheTestAccess;

/// @brief L1 Scalar Cache (K$) controller for SMEM instructions.
///
/// 16KB, 64-byte lines, 4-way set-associative with LRU. Supports both
/// scalar loads and stores. Cacheable scalar stores allocate in K$, update the
/// resident bytes, and write those bytes through to L2. K$ lines remain clean,
/// so eviction and `s_dcache_wb` never publish a cached full line.
/// `s_dcache_inv` invalidates all resident lines; `s_dcache_discard`
/// invalidates only the addressed line(s) for one VMID.
///
/// CDNA3 K$ geometry: 64B lines, 64 sets, 4-way = 16KB.
class L1ScalarCache {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 6; // 64 bytes
  static constexpr uint32_t NUM_SETS = 64;
  static constexpr uint32_t ASSOCIATIVITY = 4;

  using CacheStore =
      simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY,
                     simdojo::LRUPolicy<NUM_SETS, ASSOCIATIVITY>, /*LazyStorage=*/true>;

  explicit L1ScalarCache(L2Cache *l2 = nullptr);
  ~L1ScalarCache();

  L1ScalarCache(const L1ScalarCache &) = delete;
  L1ScalarCache &operator=(const L1ScalarCache &) = delete;
  L1ScalarCache(L1ScalarCache &&) = delete;
  L1ScalarCache &operator=(L1ScalarCache &&) = delete;

  /// @brief Set (or replace) the backing L2 cache while the CU is quiesced.
  /// @param l2 New L2 cache (not owned).
  /// @details Rebinding must finish before ordinary accesses or maintenance:
  /// registry transfer and backing pointer replacement are separate steps.
  void set_l2(L2Cache *l2);

  /// @brief Set the VM service used for process-scoped MTYPE lookups.
  void set_gpu_vm(GpuVm *gpu_vm);

  /// @brief Scalar load: read num_dwords contiguous dwords from addr.
  ///
  /// Fetches from K$ on hit, or fills from L2 on miss. Handles requests
  /// that span multiple cache lines. Instruction MTYPE is combined with page
  /// restrictions; CC/UC loads bypass resident K$ data and refetch through L2.
  /// Private RAM batching must be enabled only for unobserved functional requests.
  VmAccessOutcome load(uint64_t addr, uint32_t num_dwords, uint32_t *dst, uint32_t vmid = 0,
                       bool allow_private_batch = false, Mtype instruction_mtype = Mtype::RW);

  /// @brief Scalar load: read num_bytes contiguous bytes from addr.
  VmAccessOutcome load_bytes(uint64_t addr, uint32_t num_bytes, uint8_t *dst, uint32_t vmid = 0,
                             Mtype instruction_mtype = Mtype::RW);

  /// @brief Scalar store: write num_dwords contiguous dwords to addr.
  ///
  /// Cacheable stores allocate without reading untouched bytes and write through
  /// each modified byte range to L2. UC and CC stores bypass K$, whether the
  /// restriction comes from the instruction or the page.
  VmAccessOutcome store(uint64_t addr, uint32_t num_dwords, const uint32_t *src, uint32_t vmid = 0,
                        Mtype instruction_mtype = Mtype::RW);

  /// @brief Handle s_dcache_wb; a no-op because K$ is write-through.
  /// @param vmid Ignored. Retained only for call-site signature symmetry.
  void writeback_all(uint32_t vmid = 0);

  /// @brief Invalidate all clean K$ lines (s_dcache_inv).
  void invalidate_all();

  /// @brief Handle s_dcache_discard[_x2] without fetching or writing data.
  /// @param num_lines Number of consecutive 64-byte lines to invalidate (one or two).
  void discard(uint64_t addr, uint32_t num_lines, uint32_t vmid = 0);

  /// @brief Invalidate lines belonging to one ordinary VM root at completion.
  /// @details Matches resident VMID and cache identity across the whole root,
  /// including lines admitted under different page generations. Other roots
  /// and address spaces remain resident.
  void invalidate_for_completion(uint32_t vmid, uint64_t cache_identity);

  /// Probe only resident, byte-valid data under the supplied raw grant key.
  /// The span must be nonempty and contained in one K$ line. False is a miss;
  /// this method performs no VM admission or backing request.
  [[nodiscard]] bool try_read_peer_cached(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid,
                                          uint64_t admission_generation, uint64_t cache_identity);

  /// Optionally fill one K$ line fragment from a canonical peer read.
  /// The caller must hold exact ingress, current-VM, and canonical-line
  /// freshness authority through this call. The span must be nonempty and
  /// contained in one line. A dirty-line refusal leaves tags and replacement
  /// order unchanged; normal epoch reconciliation may discard stale lines.
  /// This method never invokes VM admission or backing traffic.
  [[nodiscard]] bool try_fill_peer_cached(uint64_t addr, std::span<const uint8_t> src,
                                          uint32_t vmid, uint64_t admission_generation,
                                          uint64_t cache_identity);

private:
  friend class DeviceCacheCoherence;
  friend class L1ScalarCacheTestAccess;

  [[nodiscard]] uint64_t current_cache_identity(uint32_t vmid) const;
  [[nodiscard]] bool has_unidentified_overlap(std::span<const DeviceCacheAliasScope> aliases);
  void invalidate_selected_lines(std::span<const DeviceCacheAliasScope> aliases);
  [[nodiscard]] VmAccessAdmission admit_access(uint64_t addr, uint32_t size, uint32_t vmid,
                                               VmAccessKind kind, bool report_fault = true) const;
  void cache_partial_bytes(uint64_t addr, const uint8_t *src, uint32_t size, uint32_t vmid,
                           uint64_t admission_generation, uint64_t cache_identity);
  VmAccessOutcome ensure_line(uint64_t addr, uint32_t vmid, uint64_t admission_generation,
                              bool fetch_on_miss = true);
  void flush_line(uint64_t addr, uint32_t vmid = 0);
  void invalidate_all_lines();
  void synchronize_epoch();

  CacheStore cache_;
  // Acquired before any L2 maintenance lock on ordinary accesses and scoped maintenance.
  std::mutex maintenance_mutex_;
  L2Cache *l2_;
  std::shared_ptr<DeviceCacheCoherence> registered_domain_;
  GpuVm *gpu_vm_ = nullptr;
  VmMtypeCache mtype_cache_;
  uint64_t coherence_epoch_ = 0;
};

} // namespace amdgpu
} // namespace rocjitsu
