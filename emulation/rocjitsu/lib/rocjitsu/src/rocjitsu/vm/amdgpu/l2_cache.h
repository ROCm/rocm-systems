// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file l2_cache.h
/// @brief L2 cache component shared per XCD.

#pragma once

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/l2_maintenance_mutex.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "simdojo/components/cache.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/message.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class PreparedCanonicalWriteback;

/// @brief L2 cache component shared per Accelerator Complex Die (XCD).
///
/// 128-byte lines, 2048 sets, 16-way set-associative = 4MB (default).
/// In functional mode, writes are kept visible through the backing store
/// (HBM controller or memory-side cache) via the requester port.
///
/// Mtype-aware behavior:
///   - UC: Flush/invalidate any resident L2 line, then bypass L2 and
///         forward directly to backing store.
///   - CC: Allocate in L2, MOESI coherence state tracking for CPU-GPU
///         shared memory. Write-through on CC stores.
///   - RW/WB: Allocate in L2, with functional-mode stores written through to
///            the backing store so completed dispatches are globally visible.
///   - NT: Allocate in L2 (L1 is bypassed, but L2 still caches).
///
/// Numeric-routed entries are keyed by (VMID, GPU virtual line). Handle-bound
/// local reads use their exact cache-generation identity as a separate key,
/// with the selected physical line and PTE policy retained in the tag.
/// Different aliases of the same backing line therefore keep
/// independent cached copies. Functional write() stores update backing memory
/// but do not invalidate other aliases; CC/UC access or explicit cache
/// maintenance is required before an alias refetches that data. Dirty lines are
/// written back through their owning VMID. Atomic RMW is intentionally stronger:
/// L2Cache coordinates all registered device caches before the backing RMW,
/// above GPU-VA-to-host mapping differences.
///
/// Serves as the backing store for both L1 Scalar (K$) and L1 Vector (V$).
///
/// Provides structural ports for the topology graph (IN for CU L1 miss
/// requests, OUT for HBM/fabric traffic).
///
/// @par Thread safety
/// Public cache operations are thread-safe.
/// Normal line accesses take shared maintenance admission before a per-set
/// mutex; whole-cache maintenance takes admission exclusively. Independent sets
/// proceed concurrently while admission is open. A writer that closes the gate
/// blocks new accesses to every set while existing accesses finish. Device
/// atomics and domain-wide maintenance exclude every L2 in the coherence domain.
/// Operations on a set are atomic, but multi-line requests are not snapshots.
///
/// Current-epoch accesses skip reconciliation. Stale accesses release shared
/// admission before taking the per-L2 reconciliation mutex, then reacquire
/// admission and recheck the epoch so peers reuse the first refresh. When both
/// locks are held, reconciliation precedes maintenance admission; admission
/// always precedes the per-set mutex. Direct maintenance never acquires the
/// reconciliation mutex. See L2MaintenanceMutex for admission and publication.
class L2Cache : public simdojo::Component {
public:
  static constexpr uint32_t LINE_SIZE_BITS = 7;
  static constexpr uint32_t NUM_SETS = 2048;
  static constexpr uint32_t ASSOCIATIVITY = 16;

  using CacheStore = simdojo::Cache<LINE_SIZE_BITS, NUM_SETS, ASSOCIATIVITY>;
  static constexpr uint32_t LINE_SIZE = CacheStore::LINE_SIZE;

  /// @brief Construct an L2Cache component.
  /// @param name Human-readable name (e.g., "xcd0.l2").
  explicit L2Cache(std::string name, std::shared_ptr<DeviceCacheCoherence> coherence =
                                         std::make_shared<DeviceCacheCoherence>());
  ~L2Cache() override;

  L2Cache(const L2Cache &) = delete;
  L2Cache &operator=(const L2Cache &) = delete;
  L2Cache(L2Cache &&) = delete;
  L2Cache &operator=(L2Cache &&) = delete;

  /// @brief Set the requester port used to reach the backing store.
  /// @param port The OUT port connected to the MSC or HBM controller.
  void set_req_port(simdojo::Port *port) { req_port_ = port; }

  /// @brief Set the backing memory for direct writeback in functional mode.
  /// @param mem GpuMemory instance (used when req_port_ has no link).
  void set_backing_memory(GpuMemory *mem) { backing_memory_ = mem; }

  /// @brief Set the legacy backing used only by out-of-band cache maintenance.
  /// @details This direct path avoids simulation-port traffic from SDMA's
  /// worker. Translated PCI/VFIO accesses do not use this compatibility path.
  void set_legacy_maintenance_memory(GpuMemory *memory) { legacy_maintenance_memory_ = memory; }
  /// @brief Set the VM service used only by out-of-band cache maintenance.
  /// @details The maintenance path snapshots nonzero VMIDs here instead of
  /// routing an SDMA worker through the simulation message topology.
  void set_legacy_maintenance_vm(GpuVm *gpu_vm) { legacy_maintenance_vm_ = gpu_vm; }
  /// @brief Set the VM service used for nonzero-VMID direct backing accesses.
  /// @details This is separate from the maintenance VM so direct functional
  /// accesses and out-of-band maintenance remain independently configurable.
  void set_gpu_vm(GpuVm *gpu_vm) { gpu_vm_ = gpu_vm; }

  /// @brief Rebind this cache to a device-local coherence domain before use.
  void set_coherence_domain(std::shared_ptr<DeviceCacheCoherence> coherence);
  const std::shared_ptr<DeviceCacheCoherence> &coherence_domain() const { return coherence_; }

  /// Diagnostic totals are exact at quiescence. Concurrent queries sample
  /// independent relaxed shards, not one instantaneous global snapshot.
  uint64_t backing_read_transactions() const {
    return diagnostic_total(&DiagnosticCounters::backing_reads);
  }
  uint64_t backing_write_transactions() const {
    return diagnostic_total(&DiagnosticCounters::backing_writes);
  }

  /// @brief Check the issuer's current mapping grant before cached bytes or
  /// backing are accessed. The generation labels resident cache lines.
  [[nodiscard]] VmAccessAdmission admit_access(uint64_t addr, uint32_t size, uint32_t vmid,
                                               VmAccessKind access, bool report_fault = true) const;

  /// @brief Return whether a range may be read speculatively for a cache fill.
  /// @details A failed query is not delivered as a GPU fault. Callers must
  /// bypass cache allocation and issue only the architectural operand so its
  /// ordinary access result determines whether execution succeeds or faults.
  [[nodiscard]] bool can_fetch_range(uint64_t addr, uint32_t size, uint32_t vmid = 0) const;

  /// @brief Read a cache line worth of data (or partial line).
  ///
  /// Used by L1 controllers to fetch on miss. In functional mode this refetches
  /// from backing memory to avoid stale clean lines across independent L2s.
  /// @param addr The starting address (line-aligned or not).
  /// @param dst Destination buffer.
  /// @param size Number of bytes to read.
  /// @param mtype Memory type for caching policy.
  VmAccessOutcome read(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype = Mtype::RW,
                       uint32_t vmid = 0);

  /// Read one cacheable local line through an exact handle-bound VM snapshot.
  /// The caller holds its device domain access lease across the L1/L2 request;
  /// this route never resolves the snapshot through a numeric VMID. Deferred
  /// system and peer translations are excluded by the line-policy probe.
  [[nodiscard]] VmAccessOutcome
  read_translated_local(const GpuVmAccess &access, uint64_t addr, std::span<uint8_t> dst,
                        Mtype instruction_mtype,
                        const DeviceCacheCoherence::LocalCacheAccess &domain_access);

  /// Probe only resident, byte-valid data under the supplied raw grant key.
  /// The span must be nonempty and contained in one L2 line. False is a miss;
  /// this method performs no VM admission or backing request.
  [[nodiscard]] bool try_read_peer_cached(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid,
                                          uint64_t admission_generation, uint64_t cache_identity);

  /// Optionally fill one L2 line fragment from a canonical peer read.
  /// The caller must hold exact ingress, current-VM, and canonical-line
  /// freshness authority through this call. The span must be nonempty and
  /// contained in one line. A dirty-line refusal leaves tags and replacement
  /// order unchanged; normal epoch reconciliation may discard stale lines.
  /// This method never publishes a dirty victim or sends backing traffic.
  [[nodiscard]] bool try_fill_peer_cached(uint64_t addr, std::span<const uint8_t> src,
                                          uint32_t vmid, uint64_t admission_generation,
                                          uint64_t cache_identity);

  /// Batch an unobserved scalar UC request only when this line is absent and
  /// the direct VM backing proves private, fault-free RAM. False has no guest
  /// effects; the caller must retain its original per-dword fallback.
  [[nodiscard]] bool try_read_scalar_ram(uint64_t addr, uint32_t *dst, uint32_t num_dwords,
                                         uint32_t vmid);

  /// @brief Write data to L2 (and possibly through to HBM).
  ///
  /// Used by L1 for write-through (CC) and write-back evictions.
  /// @param addr The starting address.
  /// @param src Source data.
  /// @param size Number of bytes to write.
  /// @param mtype Memory type for caching policy.
  VmAccessOutcome write(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype = Mtype::RW,
                        uint32_t vmid = 0);

  /// Optional same-line stores with a live private-RAM proof. Refusal changes
  /// neither memory nor cache replacement state. Logical counters stay per store.
  [[nodiscard]] bool try_write_private_dwords(std::span<const VmRamDwordStore> stores,
                                              Mtype instruction_mtype, Mtype effective_mtype,
                                              uint32_t vmid);

  /// Successful cached write chunks; UC bypass writes are not included.
  /// Like backing transaction totals, concurrent queries sample each shard.
  uint64_t write_count() const { return diagnostic_total(&DiagnosticCounters::writes); }

  /// @brief Fetch an entire cache line into the given buffer.
  ///
  /// Convenience method for L1 fills. Returns a full LINE_SIZE-byte line
  /// at the line-aligned address containing addr.
  /// @param addr Any address within the desired cache line.
  /// @param[out] line_buf Buffer of at least LINE_SIZE bytes.
  VmAccessOutcome fetch_line(uint64_t addr, uint8_t *line_buf, uint32_t vmid = 0);

  /// @brief Write back a full cache line from L1 eviction.
  /// @param line_addr Line-aligned address.
  /// @param[in] data Full cache line data (LINE_SIZE bytes).
  /// @param mtype Memory type for caching policy.
  VmAccessOutcome writeback_line(uint64_t line_addr, const uint8_t *data, Mtype mtype = Mtype::RW,
                                 uint32_t vmid = 0);

  /// @brief Write back only the modified bytes from a full L1 line snapshot.
  /// @param line_addr Line-aligned address.
  /// @param[in] data Full cache line data (LINE_SIZE bytes).
  /// @param dirty_offset First modified byte within the line.
  /// @param dirty_size Number of modified bytes.
  /// @param mtype Memory type for caching policy.
  /// @param vmid Owning process address space.
  VmAccessOutcome writeback_line(uint64_t line_addr, const uint8_t *data, uint32_t dirty_offset,
                                 uint32_t dirty_size, Mtype mtype = Mtype::RW, uint32_t vmid = 0);

  /// @brief Perform an atomic read-modify-write on a cache line.
  ///
  /// @details Establishes a device-local coherence boundary: dirty
  /// scalar and L2 state is published, the device coherence epoch is advanced,
  /// and every L2 remains quiescent through the backing RMW. Cached clean lines
  /// are discarded lazily on their next ordinary access.
  ///
  /// @param addr The memory address of the atomic target.
  /// @param size Access size in bytes (4 or 8).
  /// @param fn Callback: fn(line_data_ptr, line_offset). Must read the
  ///           old value, compute the new value, and write it in place.
  template <typename F>
  [[nodiscard]] VmAccessOutcome atomic_rmw(uint64_t addr, uint32_t size, F &&fn,
                                           uint32_t vmid = 0) {
    const auto vm_access = snapshot_atomic_access(vmid);
    const VmAccessOutcome admission = preflight_atomic_access(addr, size, vmid);
    if (admission != VmAccessOutcome::Complete)
      return admission;
    DeviceCacheCoherence::AtomicBoundary boundary = coherence_->acquire_atomic_boundary();
    return atomic_rmw(boundary, addr, size, std::forward<F>(fn), vmid, vm_access);
  }

  /// @brief Select local VM backing before acquiring the device coherence boundary.
  /// @details Take a fresh snapshot for each attempt, including retries. VMID-zero
  /// physical accesses and transport-backed atomics do not need a local snapshot.
  [[nodiscard]] std::optional<GpuVmAccess> snapshot_atomic_access(uint32_t vmid) const {
    if (!backing_memory_ || vmid == 0 || !gpu_vm_)
      return std::nullopt;
    return gpu_vm_->snapshot_vmid(vmid);
  }

  /// Check every architectural atomic lane before device-wide cache maintenance.
  [[nodiscard]] VmAccessOutcome preflight_atomic_access(uint64_t addr, uint32_t size,
                                                        uint32_t vmid) const;

  /// @brief Perform one backing atomic within an existing device boundary.
  /// @details A vector instruction can prepare the cache hierarchy once for all
  /// its lanes. Each backing RMW remains atomic with respect to host accesses.
  /// Legacy callers retain their per-lane preflight under this boundary.
  /// @param vm_access Result of snapshot_atomic_access(vmid) for this attempt.
  template <typename F>
  [[nodiscard]] VmAccessOutcome atomic_rmw(const DeviceCacheCoherence::AtomicBoundary &boundary,
                                           uint64_t addr, uint32_t size, F &&fn, uint32_t vmid,
                                           const std::optional<GpuVmAccess> &vm_access) {
    if (boundary.outcome() != VmAccessOutcome::Complete)
      return boundary.outcome();
    if (!boundary.belongs_to(coherence_.get()))
      return VmAccessOutcome::Malformed;

    const VmAccessOutcome admission = preflight_atomic_access(addr, size, vmid);
    if (admission != VmAccessOutcome::Complete)
      return admission;

    return atomic_rmw_preflighted(addr, size, std::forward<F>(fn), vmid, vm_access);
  }

  /// @brief Perform a prepared CU atomic after every lane was preflighted.
  /// @details The prepared lease already holds all cache locks and published
  /// the data epoch. Repeating VM registry admission here would invert the
  /// registry-to-cache lock order used by KFD alias maintenance.
  template <typename F>
  [[nodiscard]] VmAccessOutcome atomic_rmw(const DeviceCacheMaintenanceLease &boundary,
                                           uint64_t addr, uint32_t size, F &&fn, uint32_t vmid,
                                           const std::optional<GpuVmAccess> &vm_access) {
    if (!boundary.prepared_atomic_boundary_for(coherence_.get()))
      return VmAccessOutcome::Malformed;
    return atomic_rmw_preflighted(addr, size, std::forward<F>(fn), vmid, vm_access);
  }

  /// @brief Perform one backing atomic per active lane within an existing device boundary.
  /// @details Same contract as atomic_rmw() per lane, in lane order from @p next_lane.
  /// Locally backed VM accesses share one revocation lease and translation span across
  /// lanes, and the diagnostic counters are bumped once per call. @p next_lane is the first
  /// unapplied lane on return (the failing lane, or the lane count when complete).
  /// Accepts either an ordinary boundary or a preflighted maintenance lease.
  /// @param lane_addresses At most 64 per-lane addresses.
  /// @param fn Callback: fn(lane, data_ptr, offset).
  template <typename Boundary, typename F>
    requires(std::same_as<Boundary, DeviceCacheCoherence::AtomicBoundary> ||
             std::same_as<Boundary, DeviceCacheMaintenanceLease>)
  [[nodiscard]] VmAccessOutcome
  atomic_rmw_lanes(const Boundary &boundary, std::span<const uint64_t> lane_addresses,
                   uint64_t lane_mask, uint32_t size, F &&fn, uint32_t vmid,
                   const std::optional<GpuVmAccess> &vm_access, uint32_t &next_lane) {
    if (lane_addresses.size() > 64)
      return VmAccessOutcome::Malformed;
    if constexpr (std::same_as<Boundary, DeviceCacheCoherence::AtomicBoundary>) {
      if (boundary.outcome() != VmAccessOutcome::Complete)
        return boundary.outcome();
      if (!boundary.belongs_to(coherence_.get()))
        return VmAccessOutcome::Malformed;
    } else if (!boundary.prepared_atomic_boundary_for(coherence_.get())) {
      return VmAccessOutcome::Malformed;
    }
    const uint32_t lane_count = static_cast<uint32_t>(lane_addresses.size());
    if (backing_memory_ && vmid != 0 && gpu_vm_ != nullptr && vm_access) {
      uint32_t first = lane_count, active = 0;
      for (uint32_t lane = next_lane; lane < lane_count; ++lane) {
        if (lane_mask & (1ULL << lane)) {
          first = std::min(first, lane);
          ++active;
        }
      }
      if (active == 0) {
        next_lane = lane_count;
        return VmAccessOutcome::Complete;
      }
      next_lane = first;
      const VmAccessOutcome outcome = vm_access->atomic_modify_lanes(
          lane_addresses, lane_mask, size,
          [&](uint32_t lane, std::span<std::byte> target) {
            fn(lane, reinterpret_cast<uint8_t *>(target.data()), 0);
          },
          next_lane);
      // Count attempted lanes, including the failing one, just as atomic_rmw()
      // does. Later lanes were never attempted and must not count again on retry.
      if (outcome != VmAccessOutcome::Complete) {
        active = 0;
        for (uint32_t lane = first; lane < lane_count && lane <= next_lane; ++lane)
          active += (lane_mask & (1ULL << lane)) != 0;
      }
      // Only totals are reported; attribute the batch to its first lane's shard.
      auto &counters = diagnostics(lane_addresses[first]);
      counters.backing_reads.fetch_add(active, std::memory_order_relaxed);
      counters.backing_writes.fetch_add(active, std::memory_order_relaxed);
      return outcome;
    }
    for (uint32_t lane = next_lane; lane < lane_count; ++lane) {
      if (!(lane_mask & (1ULL << lane)))
        continue;
      const VmAccessOutcome outcome = atomic_rmw(
          boundary, lane_addresses[lane], size,
          [&](uint8_t *data, uint32_t offset) { fn(lane, data, offset); }, vmid, vm_access);
      if (outcome != VmAccessOutcome::Complete) {
        next_lane = lane;
        return outcome;
      }
    }
    next_lane = lane_count;
    return VmAccessOutcome::Complete;
  }

  /// @brief Flush a single L2 line (writeback if dirty, then invalidate).
  ///
  /// Used by CC reads to ensure cross-XCD coherence: dirty data is
  /// written back to the backing store before the line is invalidated,
  /// so a subsequent ensure_line() refetch gets the latest data.
  /// @param addr Any address within the target cache line.
  VmAccessOutcome flush_line(uint64_t addr, uint32_t vmid = 0);

  /// @brief Invalidate all L2 lines.
  void invalidate_all() {
    auto maintenance_lock = acquire_cache_maintenance();
    cache_.invalidate_all();
    clear_all_dirty_bytes();
  }

  /// @brief Invalidate L2 lines covering an address range.
  /// @details Used after a host/SDMA write has completed. Bytes inside the
  /// range are authoritative in backing memory. Dirty bytes outside a partial
  /// line update are preserved before the target VMID's line is invalidated.
  /// @param vmid Owning process address space to invalidate.
  VmAccessOutcome invalidate_range(uint64_t addr, uint32_t size, uint32_t vmid);

  /// @brief Publish and invalidate resident L2 lines before an external write.
  /// @details The entire dirty line is authoritative until the external write
  /// commits, including sibling bytes outside the requested range. A nonzero
  /// VMID is written through the exact retained VM binding that owns the line,
  /// directly to physical backing. A caller maintaining an L2/MSC hierarchy
  /// must flush the MSC first, then the L2, and must exclude alias refills from
  /// both caches until the write and both post-write invalidations complete.
  /// Release the MSC's access gate before calling this operation: VMID zero
  /// may publish through the ordinary L2-to-MSC port while holding L2 admission.
  /// Use an independent ingress barrier to exclude refills across both phases.
  /// An unsuccessful writeback retains its unpublished dirty bytes for retry.
  /// A nonlocal peer route returns failure with its dirty bytes retained;
  /// cacheable peer writeback needs a coordinated canonical-backing sink.
  /// No external effect may begin unless this operation completes.
  VmAccessOutcome flush_range_before_external_write(uint64_t addr, uint32_t size, uint32_t vmid);

  /// @brief Publish canonical alias lines through a prepared backing grant.
  /// @details The caller prepares the complete cache-line envelope and retains
  /// its counted ingress permit until this method releases every cache lock.
  /// Each dirty run must match the resident VM identity and nonzero grant
  /// generation. A failed or refused direct backing write leaves that run dirty
  /// for retry. No VM registry or KFD lookup occurs under the cache locks.
  VmAccessOutcome flush_range_before_external_write(uint64_t addr, uint32_t size, uint32_t vmid,
                                                    const PreparedCanonicalWriteback &prepared);

  /// @brief Drop resident L2 lines after an external write has committed.
  /// @details Never publishes old dirty bytes. Clears their byte masks before
  /// invalidating each overlapping line in the requested VMID. The caller
  /// must still exclude alias refills until this and MSC invalidation finish.
  void invalidate_range_after_external_write(uint64_t addr, uint32_t size, uint32_t vmid);

  /// @brief Flush all dirty L2 lines to HBM and invalidate.
  /// @param vmid Ignored. Each dirty line is written back under its own owning
  /// vmid (recorded in the line tag), so a caller-supplied vmid cannot be
  /// correct for a global flush. Retained only for call-site signature symmetry.
  VmAccessOutcome flush_all(uint32_t vmid = 0);

  /// @brief Flush resident lines owned by one ordinary VM binding at completion.
  /// @details Publish dirty bytes through the supplied exact access, then
  /// invalidate only lines with its VMID and cache identity. Refused or failed
  /// writes retain unpublished bytes for retry. A dirty line with stale
  /// admission returns Faulted without discarding it. The linked L2-to-MSC
  /// path cannot carry this binding and reports Malformed for dirty lines
  /// until it has an exact-access writeback path.
  [[nodiscard]] VmAccessOutcome flush_for_completion(const GpuVmAccess &access);

  /// @brief Create a completer port for a CU connection (one per CU).
  /// @param name Name suffix for the port (used for port naming).
  /// @returns Pointer to the newly created completer port.
  simdojo::Port *create_cpl_port(const std::string &name) {
    simdojo::PortID port_id = static_cast<simdojo::PortID>(cpl_ports_.size() + 1);
    auto port = std::make_unique<simdojo::Port>(
        "cpl_" + name, port_id, this, simdojo::PortDirection::IN, simdojo::PortProtocol::MEMORY);
    simdojo::Port *raw = add_port(std::move(port));
    cpl_ports_.push_back(raw);
    return raw;
  }

  /// @brief Return the requester port (for topology wiring to HBM/fabric).
  /// @returns Pointer to the requester port.
  simdojo::Port *req_port() { return req_port_; }

  /// @brief Return all completer ports (CU-facing).
  /// @returns Const reference to the vector of completer ports.
  const std::vector<simdojo::Port *> &cpl_ports() const { return cpl_ports_; }

private:
  friend class DeviceCacheCoherence;
  friend class L2CacheTestAccess;
  friend class KfdPreparedProviderTestAccess;

  [[nodiscard]] VmAccessAdmission query_cache_write_grant(uint64_t addr, uint32_t size,
                                                          uint32_t vmid) const;

  template <typename F>
  [[nodiscard]] VmAccessOutcome
  atomic_rmw_preflighted(uint64_t addr, uint32_t size, F &&fn, uint32_t vmid,
                         const std::optional<GpuVmAccess> &vm_access) {
    if (backing_memory_) {
      diagnostics(addr).backing_reads.fetch_add(1, std::memory_order_relaxed);
      diagnostics(addr).backing_writes.fetch_add(1, std::memory_order_relaxed);
      if (vmid == 0) {
        const bool modified =
            backing_memory_->atomic_modify(addr, size, [&](uint8_t *target) { fn(target, 0); });
        return modified ? VmAccessOutcome::Complete : VmAccessOutcome::Malformed;
      }
      if (gpu_vm_ == nullptr)
        return VmAccessOutcome::Unavailable;
      if (!vm_access)
        return VmAccessOutcome::Faulted;
      return vm_access->atomic_modify(addr, size, [&](std::span<std::byte> target) {
        fn(reinterpret_cast<uint8_t *>(target.data()), 0);
      });
    }
    assert((size == sizeof(uint32_t) || size == sizeof(uint64_t)) &&
           "L2 atomic size must be 4 or 8 bytes");
    simdojo::MemoryAtomicMutation mutation = [&](std::span<std::byte> target) {
      fn(reinterpret_cast<uint8_t *>(target.data()), 0);
    };
    return send_atomic_backing(addr, size, mutation, vmid);
  }

  static constexpr uint64_t MAX_INVALIDATE_RANGE_SET_LOCKS = 64;

  // Different sets can be accessed concurrently without sharing a host cache line.
  struct alignas(64) SetMutex {
    std::mutex mutex;
  };
  using SetMutexes = std::array<SetMutex, NUM_SETS>;

  std::mutex &set_mutex(uint64_t addr) const {
    return set_mutexes_[CacheStore::set_index(addr)].mutex;
  }

  class SetRangeLocks {
  public:
    SetRangeLocks(SetMutexes &mutexes, uint64_t line_start, uint64_t line_count)
        : mutexes_(mutexes) {
      std::array<bool, NUM_SETS> seen{};
      for (uint64_t i = 0; i < line_count; ++i) {
        const uint64_t line_addr = line_start + i * LINE_SIZE;
        uint32_t set = CacheStore::set_index(line_addr);
        if (!seen[set]) {
          seen[set] = true;
          sets_[count_++] = set;
        }
      }

      std::ranges::sort(sets_.begin(), sets_.begin() + count_);
      try {
        for (size_t i = 0; i < count_; ++i) {
          mutexes_[sets_[i]].mutex.lock();
          ++locked_;
        }
      } catch (...) {
        unlock_all();
        throw;
      }
    }

    SetRangeLocks(const SetRangeLocks &) = delete;
    SetRangeLocks &operator=(const SetRangeLocks &) = delete;

    ~SetRangeLocks() { unlock_all(); }

  private:
    void unlock_all() {
      while (locked_ > 0) {
        --locked_;
        mutexes_[sets_[locked_]].mutex.unlock();
      }
    }

    SetMutexes &mutexes_;
    std::array<uint32_t, NUM_SETS> sets_{};
    size_t count_ = 0;
    size_t locked_ = 0;
  };

  SetRangeLocks lock_sets_for_range(uint64_t line_start, uint64_t line_count) const {
    return SetRangeLocks(set_mutexes_, line_start, line_count);
  }

  using MaintenanceMutex = L2MaintenanceMutex;
  std::shared_lock<MaintenanceMutex> acquire_cache_access();
  std::shared_lock<MaintenanceMutex> try_acquire_peer_cache_access();
  std::unique_lock<MaintenanceMutex> acquire_cache_maintenance();
  void synchronize_epoch_locked();
  VmAccessOutcome cache_partial_bytes(uint64_t addr, const uint8_t *src, uint32_t size,
                                      uint32_t vmid, uint64_t admission_generation);
  // Store allocation leaves untouched bytes absent instead of reading backing.
  VmAccessOutcome ensure_line(uint64_t addr, uint32_t vmid, uint64_t admission_generation,
                              bool fetch_on_miss = true);
  VmAccessOutcome discard_stale_line_locked(uint64_t addr, uint32_t vmid,
                                            uint64_t admission_generation);
  [[nodiscard]] uint64_t current_cache_identity(uint32_t vmid) const;
  [[nodiscard]] bool direct_controlled_writeback(uint64_t address, uint32_t size,
                                                 uint32_t vmid) const noexcept;
  [[nodiscard]] bool retain_admitted_dirty_bytes(uint64_t line_addr, const simdojo::CacheTag &tag);
  VmAccessOutcome flush_line_locked(uint64_t addr, uint32_t vmid = 0);
  VmAccessOutcome flush_dirty_locked();
  [[nodiscard]] VmAccessOutcome flush_dirty_to_legacy_backing_locked();
  using DirtyMask = std::array<uint64_t, LINE_SIZE / 64>;
  [[nodiscard]] bool dirty_mask_overlaps_controlled(uint64_t line_addr, uint32_t vmid,
                                                    const DirtyMask &mask) const noexcept;
  [[nodiscard]] bool guarded_dirty_allocation_victim(uint64_t replacement_addr);
  void mark_dirty_bytes(uint64_t line_addr, uint32_t offset, uint32_t size, uint32_t vmid);
  bool clear_dirty_bytes(uint64_t line_addr, uint32_t offset, uint32_t size, uint32_t vmid);
  VmAccessOutcome publish_dirty_bytes(uint64_t line_addr, const uint8_t *data, uint32_t vmid,
                                      uint32_t discard_offset = 0, uint32_t discard_size = 0);
  VmAccessOutcome publish_dirty_bytes_before_external_write(uint64_t line_addr, const uint8_t *data,
                                                            const simdojo::CacheTag &tag,
                                                            const GpuVmAccess *access);
  [[nodiscard]] VmAccessOutcome publish_dirty_bytes_for_completion(uint64_t line_addr,
                                                                   const uint8_t *data,
                                                                   const simdojo::CacheTag &tag,
                                                                   const GpuVmAccess &access);
  [[nodiscard]] VmAccessOutcome
  publish_dirty_bytes_to_legacy_backing(uint64_t line_addr, const uint8_t *data, uint32_t vmid);
  bool has_legacy_maintenance_backing() const {
    return legacy_maintenance_memory_ != nullptr && legacy_maintenance_vm_ != nullptr;
  }
  void clear_all_dirty_bytes();
  VmAccessOutcome invalidate_range_locked(uint64_t addr, uint32_t size, uint32_t vmid,
                                          uint64_t line_start, uint64_t line_count);
  VmAccessOutcome flush_external_line_locked(uint64_t line_addr, uint32_t vmid,
                                             const GpuVmAccess *access);
  VmAccessOutcome flush_prepared_external_line_locked(uint64_t line_addr, uint64_t requested_begin,
                                                      uint64_t requested_last, uint32_t vmid,
                                                      const PreparedCanonicalWriteback &prepared);
  void invalidate_external_line_locked(uint64_t line_addr, uint32_t vmid);
  VmAccessOutcome send_backing(uint64_t addr, uint8_t *data, uint32_t size, simdojo::MessageOp op,
                               uint32_t vmid = 0);
  VmAccessOutcome send_atomic_backing(uint64_t addr, uint32_t size,
                                      const simdojo::MemoryAtomicMutation &mutation,
                                      uint32_t vmid = 0);
  static VmAccessOutcome access_outcome(simdojo::MessageStatus status);

  CacheStore cache_;
  mutable MaintenanceMutex maintenance_mutex_;
  std::mutex epoch_reconcile_mutex_;
  mutable SetMutexes set_mutexes_;
  simdojo::Port *req_port_ = nullptr;
  GpuMemory *backing_memory_ = nullptr; ///< Direct writeback path (functional mode).
  GpuMemory *legacy_maintenance_memory_ = nullptr;
  GpuVm *legacy_maintenance_vm_ = nullptr;
  GpuVm *gpu_vm_ = nullptr;
  std::shared_ptr<DeviceCacheCoherence> coherence_;
  uint64_t coherence_epoch_ = 0;
  std::mutex dirty_bytes_mutex_;
  std::map<std::pair<uint32_t, uint64_t>, DirtyMask> dirty_bytes_;
  std::atomic<bool> has_dirty_lines_{false};
  std::vector<simdojo::Port *> cpl_ports_;
  struct alignas(64) DiagnosticCounters {
    std::atomic<uint64_t> writes{0};
    std::atomic<uint64_t> backing_reads{0};
    std::atomic<uint64_t> backing_writes{0};
  };
  static constexpr uint32_t kDiagnosticShards = 64;

  DiagnosticCounters &diagnostics(uint64_t address) {
    const uint32_t set = CacheStore::set_index(address);
    // Fold the high set bits so page-aligned accesses use distinct shards.
    return diagnostics_[(set ^ (set >> 6)) & (kDiagnosticShards - 1)];
  }

  uint64_t diagnostic_total(std::atomic<uint64_t> DiagnosticCounters::*counter) const {
    uint64_t total = 0;
    for (const auto &shard : diagnostics_)
      total += (shard.*counter).load(std::memory_order_relaxed);
    return total;
  }

  // Accounting has no cache/coherence role and never shares a dirty-state line.
  std::array<DiagnosticCounters, kDiagnosticShards> diagnostics_{};
};

} // namespace amdgpu
} // namespace rocjitsu
