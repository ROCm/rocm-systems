// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kfd_process.h
/// @brief Per-process KFD state, analogous to the kernel's kfd_process.
///
/// @details Each process that opens /dev/kfd gets a KfdProcess instance holding
/// its allocations, queues, events, doorbells, and memory policies. The
/// SimulatedKfd owns a process table mapping fds to KfdProcess instances,
/// and delegates per-process ioctl operations through here.

#pragma once

#include "rocjitsu/kmd/linux/events.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/vm/amdgpu/gpu_handles.h"
#include "rocjitsu/vm/amdgpu/legacy_page_table.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "rocjitsu/vm/amdgpu/raw_backing_write_gate.h"
#include "util/distributed_shared_mutex.h"
#include "util/unique_handle.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <sys/types.h> // pid_t

namespace rocjitsu {

/// @brief DRM BO state (create info, placement, UMD metadata); defined by the interposer.
/// @details The KFD code only stores and copies pointers to it.
struct GemObject;
class PrivateGttCopySourceGate;

/// @brief Per-process KFD state.
///
/// @details Mirrors the kernel's kfd_process + kfd_process_device for a
/// single-GPU simulator. Each daemon client connection or local-mode session
/// gets one KfdProcess. The SimulatedKfd maintains a process table and
/// routes ioctls to the correct KfdProcess.
class KfdProcess {
public:
  class PageTableTransaction;

  /// @brief Per-GPU state within a process.
  struct PerGpuState {
    /// @brief One live client mapping of the canonical doorbell backing.
    struct DoorbellView {
      void *page = nullptr;
      uint64_t gpu_va = 0;
    };

    /// @brief Canonical shared backing retained for the process lifetime.
    int doorbell_memfd = -1;
    /// @brief Every client doorbell view still owned by the mapping layer.
    std::vector<DoorbellView> doorbell_views;
    /// @brief Stable driver-side alias used exclusively by the command processor.
    void *doorbell_monitor_page = nullptr;
    size_t doorbell_page_size = 0;
    uint64_t next_doorbell_offset = 0;
    std::vector<uint32_t> free_doorbell_offsets;
    uint64_t scratch_backing_va = 0;
    uint64_t trap_tba_addr = 0;
    uint64_t trap_tma_addr = 0;
    amdgpu::AddressSpaceHandle address_space;
  };

  /// @brief Construct a new KFD process with a unique process ID.
  /// @param process_id Unique identifier (analogous to PASID) for CP routing.
  /// @param num_gpus Number of GPU devices (sizes per-GPU state vector).
  explicit KfdProcess(uint32_t process_id, uint32_t num_gpus = 1);

  /// @brief Get the process ID (PASID analog).
  uint32_t process_id() const { return process_id_; }

  /// @brief Get the identity shared by this process lifetime's alias permits.
  uint64_t process_incarnation() const { return process_incarnation_; }

  pid_t client_pid() const { return client_pid_; }
  void set_client_pid(pid_t pid) { client_pid_ = pid; }

  uint32_t open_ref_count() const { return open_ref_count_.load(std::memory_order_relaxed); }
  void retain_open() { open_ref_count_.fetch_add(1, std::memory_order_relaxed); }

  /// @brief Drop one open reference; returns true when the last one is released.
  /// @details Asserts on underflow: every release must pair with a prior open or
  /// retain. An unbalanced release means an fd reference (primary or dup) was
  /// tracked without retaining, which would otherwise wrap the count and leak
  /// the process forever.
  bool release_open() {
    uint32_t prev = open_ref_count_.fetch_sub(1, std::memory_order_acq_rel);
    assert(prev > 0 && "KfdProcess open refcount underflow");
    return prev == 1;
  }

  /// @brief Stable driver-owned alias of one IPC-exported VRAM memfd.
  /// @details The alias starts at file offset zero and is independent of both
  /// processes' host pointers and GPU virtual addresses. Its last owner may
  /// outlive the source allocation, exported handle, and all imports.
  class GpuBacking {
  public:
    GpuBacking(size_t size, uint32_t gpu_id, bool controlled_peer_for_testing);
    ~GpuBacking();

    GpuBacking(const GpuBacking &) = delete;
    GpuBacking &operator=(const GpuBacking &) = delete;

    [[nodiscard]] static std::shared_ptr<const GpuBacking>
    create_memfd_alias(int fd, size_t size, uint32_t gpu_id,
                       bool controlled_peer_for_testing = false);
    [[nodiscard]] std::byte *data() const { return static_cast<std::byte *>(mapping_); }
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] uint32_t gpu_id() const { return gpu_id_; }
    [[nodiscard]] uint64_t backing_id() const { return backing_id_; }
    [[nodiscard]] bool controlled_peer_for_testing() const { return controlled_peer_for_testing_; }

  private:
    void *mapping_ = nullptr;
    size_t size_ = 0;
    uint32_t gpu_id_ = 0;
    const bool controlled_peer_for_testing_ = false;
    const uint64_t backing_id_;
  };

  /// @brief Request-bounded view of canonical BO bytes for a future peer route.
  /// @details Capturing a lease pins the alias after FREE, CLOSE, or VA reuse.
  /// Its offset is a memfd file offset, never a process virtual address.
  class GpuBackingLease {
  public:
    GpuBackingLease() = default;
    [[nodiscard]] explicit operator bool() const { return owner_ != nullptr; }
    [[nodiscard]] std::byte *data() const { return owner_ ? owner_->data() + offset_ : nullptr; }
    [[nodiscard]] std::span<std::byte> bytes() const { return {data(), size_}; }
    [[nodiscard]] uint64_t file_offset() const { return offset_; }
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] uint32_t owner_gpu_id() const { return owner_ ? owner_->gpu_id() : 0; }
    [[nodiscard]] uint64_t backing_id() const { return owner_ ? owner_->backing_id() : 0; }
    [[nodiscard]] bool controlled_peer_for_testing() const {
      return owner_ && owner_->controlled_peer_for_testing();
    }
    [[nodiscard]] bool same_backing(const GpuBackingLease &other) const {
      return owner_ != nullptr && owner_ == other.owner_;
    }

  private:
    friend class KfdProcess;
    GpuBackingLease(std::shared_ptr<const GpuBacking> owner, uint64_t offset, size_t size)
        : owner_(std::move(owner)), offset_(offset), size_(size) {}

    std::shared_ptr<const GpuBacking> owner_;
    uint64_t offset_ = 0;
    size_t size_ = 0;
  };

  enum class GpuGrantDomain : uint8_t { Unknown, System, Vram, Driver };

  /// @brief Read-only observation of one active retained backing MAP grant.
  /// @details This record is not an access permit or proof of PTE residency.
  /// Consumers recheck the live grant and VM binding before an effect or alias
  /// maintenance, including while MAP is installing or repairing its PTE.
  /// Canonical and local records live in separate snapshots; local records
  /// identify cacheable VRAM grants and use their own retained memfd owner.
  struct GpuAliasRecord {
    uint64_t backing_id = 0;
    uint32_t process_id = 0;
    uint64_t process_incarnation = 0;
    uint32_t gpu_id = 0;
    uint64_t gpu_va = 0;
    uint64_t backing_offset = 0;
    size_t extent = 0;
    uint64_t admission_generation = 0;
    // Local cacheable grants use a separate retained memfd owner. Canonical
    // peer grants keep their existing snapshot and uncached access policy.
    bool local = false;
  };
  using GpuAliasSnapshot = std::vector<GpuAliasRecord>;

  /// @brief GPU memory allocation descriptor.
  struct GpuAllocation {
    uint64_t gpu_va = 0;
    uint64_t size = 0;
    void *host_ptr = nullptr;
    uint32_t flags = 0;
    GpuGrantDomain domain = GpuGrantDomain::System;
    uint64_t handle = 0;
    int memfd = -1;
    // Public dma-buf access mode retained across IPC import and re-export.
    int export_access_mode = -1;
    uint32_t gpu_id = 0;
    bool user_va = false;
    bool imported = false;
    bool shared = false; ///< Simulator cache policy; does not alter requested ABI flags.
    int dmabuf_fd = -1;
    // Scratch backing is created by the driver for an executing GPU request,
    // rather than by an application KFD allocation/MAP sequence.
    bool driver_gpu_mapping = false;
    // A page-table allocation failure can leave a mapped prefix. A later
    // scratch request must finish the map before treating this VA as backed.
    bool scratch_mapping_incomplete = false;
    uint64_t driver_admission_generation = 0;
    // KFD MAP/UNMAP grants. The process page table is shared by all local
    // GPUs; each GPU's legacy VM binding consults the published grant index.
    std::unordered_set<uint32_t> mapped_device_ids;
    std::unordered_map<uint32_t, uint64_t> admission_generations;
    // A retired noncanonical grant can leave dirty cache lines that still
    // need the old CPU view, even after UNMAP removes its PTE and generation.
    bool ever_gpu_admitted = false;
    uint64_t page_table_mapped_bytes = 0;
    bool ipc_shared = false;
    // A controlled test may keep this canonical VRAM allocation cacheable
    // after IPC export. Public peer allocations never set this policy.
    bool cacheable_peer_for_testing = false;
    // A prior DMA-buf export may have created an untracked alias of this fd.
    bool dmabuf_exported = false;
    // A first ineligible export after local admission keeps the installed
    // cache policy while later MAP/mmap repairs extend that same local view.
    bool ipc_export_retains_local_policy = false;
    // The first export fixes peer-lease eligibility for this BO lifetime. A
    // later export must not promote an earlier handle's unleased imports into
    // a different canonical owner.
    bool backing_promotion_attempted = false;
    std::shared_ptr<const GpuBacking> backing_owner;
    // Stable local sink for cacheable VRAM lines. It never changes the
    // canonical backing_owner policy used by peer routes and PTE MTYPE.
    std::shared_ptr<const GpuBacking> local_backing_owner;
    // FREE keeps these generations after grant publication so failed PTE or
    // tombstone maintenance can retry the same closed batch.
    std::vector<GpuAliasRecord> retiring_aliases;
    bool retirement_in_progress = false;
    bool grants_retired = false;
    // A failed host-view or grant mutation may have changed only a PTE prefix.
    // Keep private GTT COPY admission closed until FREE removes this descriptor.
    bool private_gtt_copy_source_unstable = false;
    // Actual size of the CPU mmap owned by the driver. It may be shorter than
    // size when a caller maps only a prefix of a BO.
    size_t host_ptr_mapped_bytes = 0;
    // True when the driver created host_ptr (mmap it itself) and must munmap it
    // on teardown. False for caller-owned pages (e.g. reused MAP_FIXED pages
    // from the thunk) that the driver must never unmap, since unmapping them
    // races with the owning process still accessing the memory.
    bool host_ptr_owned = false;
    // MAP_FIXED installs a BO view in a caller-selected VMA. KFD owns the
    // backing descriptor, while the caller retains responsibility for munmap.
    bool host_ptr_caller_reserved = false;
    // DRM BO state (create info, placement, UMD metadata) for this backing. The
    // kernel BO keeps it after every GEM handle and dmabuf fd closes.
    std::shared_ptr<GemObject> bo_state;
    int host_view_prot = 0;
    // A fresh token is assigned whenever the tracked CPU memfd view changes.
    // Raw MAP generation and host-view generation must both match a COPY permit.
    uint64_t host_view_generation = 0;
    std::shared_ptr<PrivateGttCopySourceGate> private_gtt_copy_source_gate;
  };

  /// @brief Authorize one GPU request against the current KFD allocation grants.
  /// @details GPU access never takes alloc_mutex_. A pinned VM snapshot consults
  /// this live immutable index on every request, including after MAP/UNMAP.
  struct GpuGrantDecision {
    bool allowed = false;
    uint64_t generation = 0;
    bool cacheable = true;
    bool direct_backing_write_allowed = true;
    // Copied allocation metadata; a regular lookup never retains a BO owner.
    GpuGrantDomain domain = GpuGrantDomain::Unknown;
    uint32_t owner_gpu_id = 0;
    size_t first_extent_bytes = 0;
    bool peer_candidate = false;
    bool first_peer_candidate = false;
    // Only the first range may supply an owned local-write effect. This is
    // independent of the aggregate raw-backing write policy.
    bool first_controlled_local_write_candidate = false;
    uint32_t peer_owner_gpu_id = 0;
    bool mixed_peer_owners = false;
  };

  struct GpuPeerBackingDecision {
    GpuGrantDecision admission;
    GpuBackingLease backing;
  };

  [[nodiscard]] std::shared_ptr<const GpuAliasSnapshot> gpu_alias_snapshot() const {
    const auto grants = gpu_grants_.load(std::memory_order_acquire);
    return {grants, &grants->aliases};
  }

  [[nodiscard]] std::shared_ptr<const GpuAliasSnapshot> gpu_local_alias_snapshot() const {
    const auto grants = gpu_grants_.load(std::memory_order_acquire);
    return {grants, &grants->local_aliases};
  }

  struct GpuAliasSnapshots {
    std::shared_ptr<const GpuAliasSnapshot> canonical;
    std::shared_ptr<const GpuAliasSnapshot> local;
  };

  /// @brief Capture both alias kinds from one immutable grant publication.
  [[nodiscard]] GpuAliasSnapshots gpu_alias_snapshots() const {
    const auto grants = gpu_grants_.load(std::memory_order_acquire);
    return {{grants, &grants->aliases}, {grants, &grants->local_aliases}};
  }

  /// @brief Recheck the raw per-GPU MAP generation of a retained alias.
  /// @details UC access decisions intentionally expose generation zero, so an
  /// ingress consumer must compare against this live grant index instead.
  [[nodiscard]] bool gpu_alias_grant_current(const GpuAliasRecord &alias) const {
    if (alias.process_id != process_id_ || alias.process_incarnation != process_incarnation_)
      return false;
    const auto grants = gpu_grants_.load(std::memory_order_acquire);
    const GpuAliasSnapshot &aliases = alias.local ? grants->local_aliases : grants->aliases;
    return grants->valid && std::ranges::any_of(aliases, [&](const GpuAliasRecord &current) {
             return current.backing_id == alias.backing_id &&
                    current.process_incarnation == alias.process_incarnation &&
                    current.gpu_id == alias.gpu_id && current.gpu_va == alias.gpu_va &&
                    current.backing_offset == alias.backing_offset &&
                    current.extent == alias.extent &&
                    current.admission_generation == alias.admission_generation &&
                    current.local == alias.local;
           });
  }

  [[nodiscard]] GpuGrantDecision gpu_access_grant(uint32_t gpu_id, uint64_t address,
                                                  size_t size) const {
    return gpu_access_grant_impl(gpu_id, address, size, nullptr);
  }

  /// @brief Admit and retain a bounded peer backing for one GPU request.
  /// @details Ordinary CU/SDMA admission avoids owner reference traffic; only
  /// peer preparation asks the same live grant index for a retained backing.
  [[nodiscard]] GpuPeerBackingDecision gpu_peer_backing_grant(uint32_t gpu_id, uint64_t address,
                                                              size_t size) const {
    GpuPeerBackingDecision result;
    result.admission = gpu_access_grant_impl(gpu_id, address, size, &result.backing);
    return result;
  }

  /// @brief Retain a bounded local memfd sink for a prepared cache writeback.
  [[nodiscard]] GpuPeerBackingDecision gpu_local_backing_grant(uint32_t gpu_id, uint64_t address,
                                                               size_t size) const {
    GpuPeerBackingDecision result;
    result.admission = gpu_access_grant_impl(gpu_id, address, size, &result.backing, true);
    return result;
  }

private:
  [[nodiscard]] GpuGrantDecision gpu_access_grant_impl(uint32_t gpu_id, uint64_t address,
                                                       size_t size, GpuBackingLease *backing,
                                                       bool local_backing = false) const {
    if (size == 0 || size - 1 > UINT64_MAX - address)
      return {};
    const auto grants = gpu_grants_.load(std::memory_order_acquire);
    if (!grants->valid)
      return {};
    uint64_t generation = 0;
    bool matched = false;
    bool cacheable = true;
    std::shared_ptr<const GpuBacking> single_backing;
    uint64_t backing_offset = 0;
    uint64_t covered_end = address;
    uint64_t previous_end = 0;
    GpuGrantDecision decision;
    const uint64_t last_byte = address + size - 1;
    const auto first =
        std::upper_bound(grants->max_end_prefix.begin(), grants->max_end_prefix.end(), address);
    for (size_t index = static_cast<size_t>(first - grants->max_end_prefix.begin());
         index < grants->ranges.size(); ++index) {
      const auto &range = grants->ranges[index];
      if (range.gpu_va > last_byte)
        break;
      const uint64_t range_end = range.gpu_va + range.size;
      if (range_end <= address)
        continue;
      // Every touched range must join the preceding one without a hole or an
      // overlap. Keep scanning after covering the request to reject a second
      // allocation that overlaps an otherwise valid first one.
      if (range.gpu_va > covered_end || (matched && range.gpu_va < previous_end))
        return decision;
      if (!matched) {
        decision.domain = range.domain;
        decision.owner_gpu_id = range.owner_gpu_id;
        decision.first_extent_bytes =
            static_cast<size_t>(std::min<uint64_t>(size, range_end - address));
        decision.first_peer_candidate =
            range.domain == GpuGrantDomain::Vram && range.owner_gpu_id != gpu_id;
        decision.first_controlled_local_write_candidate =
            range.domain == GpuGrantDomain::Vram && range.owner_gpu_id == gpu_id &&
            range.backing_owner &&
            (range.cacheable_peer_for_testing ||
             range.backing_owner->controlled_peer_for_testing());
      }
      if (range.domain == GpuGrantDomain::Vram && range.owner_gpu_id != gpu_id) {
        if (decision.peer_candidate && decision.peer_owner_gpu_id != range.owner_gpu_id)
          decision.mixed_peer_owners = true;
        if (!decision.peer_candidate)
          decision.peer_owner_gpu_id = range.owner_gpu_id;
        decision.peer_candidate = true;
      }
      // The controlled peer fixture has no coherence path for owner raw
      // writes. Accumulate across the entire request, not just its first BO.
      if (range.domain == GpuGrantDomain::Vram && range.owner_gpu_id == gpu_id &&
          (range.cacheable_peer_for_testing ||
           (range.backing_owner && range.backing_owner->controlled_peer_for_testing())))
        decision.direct_backing_write_allowed = false;
      uint64_t range_generation = 0;
      if (range.driver_gpu_mapping) {
        range_generation = range.driver_admission_generation;
      } else {
        const auto grant = range.admission_generations.find(gpu_id);
        if (grant == range.admission_generations.end())
          return decision;
        range_generation = grant->second;
      }
      if (!matched) {
        generation = range_generation;
        // A canonical peer alias can change behind either GPU's caches.
        // Until the fabric provider supplies coherence, its local path must
        // bypass both L2 and the memory-side cache.
        cacheable = !range.backing_owner || range.cacheable_peer_for_testing;
        if (backing) {
          single_backing = local_backing ? range.local_backing_owner : range.backing_owner;
          backing_offset = address - range.gpu_va;
        }
      } else {
        cacheable = false;
        single_backing.reset();
      }
      matched = true;
      covered_end = range_end;
      previous_end = range_end;
    }
    if (matched && covered_end <= last_byte)
      return decision;
    // Queue rings and driver-owned mappings can live outside allocations_.
    // A span crossing grants is valid architecturally, but cannot use one
    // generation to identify every cache line it touches.
    if (backing && single_backing && backing_offset <= single_backing->size() &&
        size <= single_backing->size() - backing_offset)
      *backing = GpuBackingLease(std::move(single_backing), backing_offset, size);
    decision.allowed = true;
    decision.generation = cacheable ? generation : 0;
    decision.cacheable = cacheable;
    return decision;
  }

public:
  /// @brief Allocate an identity for the next GPU MAP or driver-created range.
  /// @pre The caller holds alloc_mutex_.
  uint64_t next_gpu_admission_generation_locked() {
    if (++gpu_admission_generation_ == 0)
      ++gpu_admission_generation_;
    return gpu_admission_generation_;
  }

  /// @pre The caller holds alloc_mutex_.
  uint64_t next_host_view_generation_locked() {
    if (++host_view_generation_ == 0)
      ++host_view_generation_;
    return host_view_generation_;
  }

  /// @brief Keep a freed GPU VA denied while the process VMID remains live.
  /// @pre The caller holds alloc_mutex_.
  void retire_gpu_range_locked(uint64_t gpu_va, uint64_t size) {
    if (size == 0 || size > UINT64_MAX - gpu_va)
      return;
    auto first = std::lower_bound(
        retired_gpu_ranges_.begin(), retired_gpu_ranges_.end(), gpu_va,
        [](const GpuGrantRange &range, uint64_t address) { return range.gpu_va < address; });
    if (first != retired_gpu_ranges_.begin() && (first - 1)->gpu_va + (first - 1)->size >= gpu_va)
      --first;

    uint64_t begin = gpu_va;
    uint64_t end = gpu_va + size;
    auto last = first;
    while (last != retired_gpu_ranges_.end() && last->gpu_va <= end) {
      begin = std::min(begin, last->gpu_va);
      end = std::max(end, last->gpu_va + last->size);
      ++last;
    }
    const size_t insert_index = static_cast<size_t>(first - retired_gpu_ranges_.begin());
    retired_gpu_ranges_.erase(first, last);
    retired_gpu_ranges_.insert(
        retired_gpu_ranges_.begin() + insert_index,
        {begin, end - begin, {}, false, 0, {}, GpuGrantDomain::Unknown, 0, {}});
  }

  /// @brief Replace a tombstone only for bytes claimed by a new allocation.
  /// @pre The caller holds alloc_mutex_.
  void claim_gpu_range_locked(uint64_t gpu_va, uint64_t size) {
    if (size == 0 || size > UINT64_MAX - gpu_va)
      return;
    const uint64_t end = gpu_va + size;
    std::vector<GpuGrantRange> remaining;
    remaining.reserve(retired_gpu_ranges_.size() + 1);
    for (const auto &range : retired_gpu_ranges_) {
      const uint64_t old_end = range.gpu_va + range.size;
      if (old_end <= gpu_va || end <= range.gpu_va) {
        remaining.push_back(range);
        continue;
      }
      if (range.gpu_va < gpu_va)
        remaining.push_back({range.gpu_va,
                             gpu_va - range.gpu_va,
                             {},
                             false,
                             0,
                             {},
                             GpuGrantDomain::Unknown,
                             0,
                             {}});
      if (end < old_end)
        remaining.push_back({end, old_end - end, {}, false, 0, {}, GpuGrantDomain::Unknown, 0, {}});
    }
    retired_gpu_ranges_ = std::move(remaining);
  }

  /// @brief Publish a copied grant view after changing allocations_ or MAP/UNMAP.
  /// @pre The caller holds alloc_mutex_.
  void publish_gpu_grants_locked();
  /// @brief Publish while retaining the request lock across grant and PTE work.
  /// @pre The caller holds alloc_mutex_ and owns a transaction for this process.
  void publish_gpu_grants_locked(PageTableTransaction &transaction);

private:
  void publish_gpu_grants_impl_locked() {
    auto next = std::make_shared<GpuGrantIndex>();
    next->ranges.reserve(allocations_.size() + retired_gpu_ranges_.size());
    for (const auto &[handle, allocation] : allocations_) {
      (void)handle;
      const auto domain =
          allocation.driver_gpu_mapping ? GpuGrantDomain::Driver : allocation.domain;
      next->ranges.push_back({allocation.gpu_va, allocation.size, allocation.admission_generations,
                              allocation.driver_gpu_mapping, allocation.driver_admission_generation,
                              allocation.backing_owner, domain, allocation.gpu_id,
                              allocation.local_backing_owner,
                              allocation.cacheable_peer_for_testing});
    }
    next->ranges.insert(next->ranges.end(), retired_gpu_ranges_.begin(), retired_gpu_ranges_.end());
    std::sort(next->ranges.begin(), next->ranges.end(),
              [](const GpuGrantRange &left, const GpuGrantRange &right) {
                return left.gpu_va < right.gpu_va;
              });
    next->max_end_prefix.reserve(next->ranges.size());
    uint64_t maximum_end = 0;
    for (const auto &range : next->ranges) {
      if (range.size == 0 || range.size > UINT64_MAX - range.gpu_va) {
        next->valid = false;
        break;
      }
      maximum_end = std::max(maximum_end, range.gpu_va + range.size);
      next->max_end_prefix.push_back(maximum_end);
    }
    if (next->valid) {
      for (const auto &range : next->ranges) {
        if (range.domain != GpuGrantDomain::Vram || range.driver_gpu_mapping)
          continue;
        const auto publish_aliases = [&](const std::shared_ptr<const GpuBacking> &owner,
                                         GpuAliasSnapshot &aliases, bool local) {
          if (!owner || owner->gpu_id() != range.owner_gpu_id || range.size > owner->size())
            return;
          for (const auto &[gpu_id, generation] : range.admission_generations) {
            if (generation == 0)
              continue;
            aliases.push_back({owner->backing_id(), process_id_, process_incarnation_, gpu_id,
                               range.gpu_va, 0, static_cast<size_t>(range.size), generation,
                               local});
          }
        };
        publish_aliases(range.backing_owner, next->aliases, false);
        // Once canonical promotion takes ownership, the local cacheable
        // generation cannot also appear as a second alias of those bytes.
        if (!range.backing_owner)
          publish_aliases(range.local_backing_owner, next->local_aliases, true);
      }
    }
    gpu_grants_.store(std::move(next), std::memory_order_release);
  }

public:
  /// @brief Memory policy descriptor.
  struct MemoryPolicy {
    uint64_t alternate_base = 0;
    uint64_t alternate_size = 0;
    uint32_t default_policy = 0;
    uint32_t alternate_policy = 0;
  };

  /// @brief Imported dmabuf descriptor.
  struct ImportedDmabuf {
    uint64_t handle = 0;
    int fd = -1;
    uint64_t size = 0;
    uint64_t va = 0;
    uint32_t gpu_id = 0;
  };

  /// @brief SVM range descriptor.
  struct SvmRange {
    uint64_t size = 0;
    std::unordered_map<uint32_t, uint32_t> attributes;
  };

  /// @brief Runtime enable state.
  struct RuntimeState {
    bool enabled = false;
    bool pending = false;
    uint32_t mode_mask = 0;
    uint32_t capabilities_mask = 0;
    uint64_t r_debug = 0;
  };

  /// @brief Debugger session state for one target process.
  ///
  /// @details Mirrors the debug-related fields the kernel maintains on
  /// @c struct @c kfd_process in
  /// @c drivers/gpu/drm/amd/amdkfd/kfd_priv.h.
  /// SimulatedKfd stores these sessions in a table keyed by the target's Linux
  /// pid, independently of KfdProcess, so a debugger can attach before the
  /// inferior opens /dev/kfd. This mirrors the real driver's DBG_TRAP_ENABLE
  /// path creating the target kfd_process.
  ///
  /// Field mapping to @c kfd_priv.h:
  /// | DebugSession field     | kfd_process field                               |
  /// |------------------------|-------------------------------------------------|
  /// | @ref enabled           | @c bool @c debug_trap_enabled                   |
  /// | @ref runtime_state     | @c kfd_runtime_info @c runtime_info.runtime_state (kfd_ioctl.h) |
  /// | @ref exception_enable_mask | @c uint64_t @c exception_enable_mask        |
  /// | @ref dbg_fd            | @c struct @c file* @c dbg_ev_file (flattened to fd) |
  /// | @ref debugger_pid      | @c struct @c kfd_process* @c debugger_process (stored as pid) |
  struct DebugSession {
    /// @brief Monotonic identity for this ENABLE lifetime.
    /// @details Async notification completion uses this to avoid mutating a
    /// replacement session after DISABLE followed by another ENABLE.
    uint64_t generation = 0;

    /// @brief Mirrors @c kfd_process::debug_trap_enabled.
    /// Set when the device process is debug-attached with a reserved VMID.
    bool enabled = false;

    /// @brief Mirrors @c kfd_process::runtime_info.runtime_state.
    /// Holds a @c kfd_dbg_runtime_state value (see @c kfd_ioctl.h).
    uint32_t runtime_state = 0;

    /// @brief Mirrors @c kfd_process::exception_enable_mask.
    /// Bitmask of exception classes that are forwarded to the debugger.
    uint64_t exception_enable_mask = 0;

    struct NotificationClaim {
      /// Complete exception event represented by an in-flight notifier write.
      uint64_t exception_mask = 0;
      /// False once a mask update leaves none of the event subscribed.
      bool continuously_subscribed = true;
      /// Bits consumed while this particular publication was in flight.
      uint64_t consumed_mask = 0;
    };

    /// Active whole-event ownership decisions, keyed by a local monotonic id.
    /// A SET_EXCEPTIONS_ENABLED transition invalidates a claim only when the
    /// complete event becomes unsubscribed, preserving ownership while one
    /// subscribed bit atomically replaces another.
    uint64_t next_notification_claim_id = 1;
    std::unordered_map<uint64_t, NotificationClaim> notification_claims;

    /// Process/device exception bits already used to wake this session.
    uint64_t notified_process_exception_mask = 0;
    /// Process/device exception bits reserved by a notifier write in flight.
    uint64_t pending_process_exception_mask = 0;
    /// In-flight process/device notifier writes per exception bit.
    std::array<uint32_t, 64> pending_process_exception_counts{};
    /// A failed or superseded publication still requires a background retry.
    bool notification_retry_needed = false;
    /// QUERY consumed a wake while queue publication still hid its status.
    bool notification_query_retry_needed = false;

    void consume_process_notification(uint64_t mask) {
      notified_process_exception_mask &= ~mask;
      for (auto &[_, claim] : notification_claims)
        claim.consumed_mask |= claim.exception_mask & mask;
    }

    void begin_process_notification(uint64_t mask) {
      for (uint32_t bit = 0; bit < 64; ++bit) {
        const uint64_t bit_mask = uint64_t{1} << bit;
        if ((mask & bit_mask) == 0)
          continue;
        ++pending_process_exception_counts[bit];
        pending_process_exception_mask |= bit_mask;
      }
    }

    void finish_process_notification(uint64_t mask) {
      for (uint32_t bit = 0; bit < 64; ++bit) {
        const uint64_t bit_mask = uint64_t{1} << bit;
        if ((mask & bit_mask) == 0 || pending_process_exception_counts[bit] == 0)
          continue;
        if (--pending_process_exception_counts[bit] == 0)
          pending_process_exception_mask &= ~bit_mask;
      }
    }

    /// @brief Previously configured process debug flags.
    uint32_t flags = 0;

    /// @brief Current wave launch mode.
    uint32_t launch_mode = 0;

    /// @brief Current wave-launch trap override mask.
    uint32_t launch_override_enable = 0;

    /// @brief Mirrors @c kfd_process::dbg_ev_file (flattened from @c struct @c file* to fd).
    /// File descriptor used as the debugger notification / poll target.
    /// -1 when no debugger is attached.
    int dbg_fd = -1;

    /// @brief Owns @ref dbg_fd when the daemon received it out-of-band.
    /// @details In daemon mode the notifier is a descriptor dup'd into the
    /// daemon's own fd table (SCM_RIGHTS), which the session must close when the
    /// debug session ends (DISABLE) or the process tears down. Engaged only in
    /// daemon mode; empty in local mode, where @ref dbg_fd is the debugger's own
    /// descriptor and is not owned here. RAII replaces an explicit close.
    UniqueDriverFd owned_dbg_fd;

    /// @brief Debugger-authorized access to the target's address space.
    /// @details The ptrace parent opens /proc/<target>/mem and transfers it to
    /// the daemon, which cannot use process_vm_readv/process_vm_writev itself.
    UniqueDriverFd target_mem_fd;

    /// @brief Pins the target process identity and reports target exit.
    /// @details Prevents a stale session from being mistaken for a later process
    /// that reuses the same numeric pid.
    UniqueDriverFd target_pidfd;

    /// @brief Pins the target's procfs directory used for ptrace authorization.
    /// @details Status is opened relative to this descriptor so authorization
    /// cannot silently switch to a process that reuses the numeric pid.
    UniqueDriverFd target_procfd;

    /// @brief Mirrors @c kfd_process::debugger_process (stored as pid instead of pointer).
    /// Linux PID of the attached debugger (ptrace parent). 0 when not attached.
    pid_t debugger_pid = 0;

    /// @brief Target exit was observed and owned resources were released.
    /// @details Keeps the pinned pidfd identity until a racing DISABLE consumes
    /// the session, so numeric PID reuse cannot turn process exit into EINVAL.
    bool target_exited = false;

    /// @brief The target's KfdProcess has been observed at least once.
    /// @details Distinguishes the two ways a target can have no KfdProcess.
    /// The real driver has only one: DBG_TRAP_ENABLE calls kfd_create_process,
    /// so from then on the kfd_process exists and a failed lookup means the
    /// process is gone (-ESRCH). Here the KfdProcess appears only when the
    /// inferior opens /dev/kfd, which can be after the debugger attaches, so
    /// "not up yet" and "torn down" are otherwise indistinguishable.
    bool saw_kfd_process = false;

    /// @brief Pins the debugger process identity and reports debugger exit.
    /// @details Mirrors the kernel's debugger-process notifier: the session is
    /// disabled when the debugger task exits, even if the target remains alive.
    UniqueDriverFd debugger_pidfd;

    /// @brief One programmed hardware address-watch register (TCP_WATCH0..3).
    struct AddressWatch {
      bool active = false;
      uint64_t address = 0;
      uint64_t mask = 0;
      uint32_t mode = 0;

      /// @brief Construct the full hardware compare state from the KFD UAPI.
      /// @details KFD transports only the programmable low 32 mask bits. The
      /// upper address bits are fixed compares in TCP_WATCH and must remain
      /// set, or unrelated addresses with the same low 32 bits alias.
      static constexpr AddressWatch from_kfd(uint64_t address, uint32_t mask, uint32_t mode) {
        return AddressWatch{true, address, 0xFFFFFFFF00000000ULL | mask, mode};
      }

      /// @brief Return whether an access overlaps the watched address block.
      [[nodiscard]] constexpr bool overlaps(uint64_t access_address, uint32_t bytes) const {
        if (!active || bytes == 0)
          return false;
        const uint64_t block_base = address & mask;
        const uint64_t block_size = ~mask + 1;
        const uint64_t access_end =
            access_address > UINT64_MAX - bytes ? UINT64_MAX : access_address + bytes;
        const uint64_t block_end = block_size == 0 || block_base > UINT64_MAX - block_size
                                       ? UINT64_MAX
                                       : block_base + block_size;
        return block_size == 0 || (access_address < block_end && block_base < access_end);
      }
    };
    /// The register file the topology advertises, so a debugger can never hold
    /// more watchpoints than the device snapshot told it exist.
    static constexpr uint32_t kMaxAddressWatches = kmd::kNumWatchPoints;
    std::array<AddressWatch, kMaxAddressWatches> address_watches;

    /// @brief Return a bit for every hardware watch slot matching an access.
    /// @details A single access may overlap multiple programmed watch ranges.
    /// Hardware reports all of them concurrently in TRAPSTS, allowing the
    /// debugger to associate every logical watchpoint with the same stop.
    /// @param matching_modes Bit set of watch modes the access satisfies,
    /// indexed by mode value. The modes overlap (a write is both NONREAD and
    /// ALL), so the caller -- which owns the KFD ABI -- resolves them into this
    /// set rather than passing a single mode to compare for equality.
    [[nodiscard]] constexpr uint32_t matching_address_watch_slots(uint64_t access_address,
                                                                  uint32_t bytes,
                                                                  uint32_t matching_modes) const {
      uint32_t slots = 0;
      for (uint32_t slot = 0; slot < kMaxAddressWatches; ++slot) {
        const auto &watch = address_watches[slot];
        if (watch.mode < 32 && ((matching_modes >> watch.mode) & 1u) != 0 &&
            watch.overlaps(access_address, bytes))
          slots |= uint32_t{1} << slot;
      }
      return slots;
    }
  };

  static constexpr uint64_t kPageShift = amdgpu::kLegacyPageShift;
  static constexpr uint64_t kPageSize = amdgpu::kLegacyPageSize;
  using HostExtentOwner = amdgpu::LegacyHostExtentOwner;
  using VaOwnerKind = amdgpu::LegacyGpuVaOwnerKind;
  using VaOwner = amdgpu::LegacyGpuVaOwner;
  using HostExtent = amdgpu::LegacyHostExtent;
  using PageTableEntry = amdgpu::LegacyPageTableEntry;
  using PageTable = amdgpu::LegacyPageTable;

  enum class PageTableMutationResult : uint8_t {
    Applied,
    ForeignOwnerCollision,
    InvalidRange,
    InvalidTransaction,
  };

  /// @brief Hold the process request lock across a GPU VA ownership change.
  /// @details GEM replacement must retain this token while its bookkeeping and
  /// PTE changes are committed, so a KFD writer cannot enter between them.
  class PageTableTransaction {
  public:
    PageTableTransaction(const PageTableTransaction &) = delete;
    PageTableTransaction &operator=(const PageTableTransaction &) = delete;
    PageTableTransaction(PageTableTransaction &&) = default;
    PageTableTransaction &operator=(PageTableTransaction &&) = default;

  private:
    friend class KfdProcess;
    explicit PageTableTransaction(KfdProcess &process)
        : process_(&process), request_lock_(*process.page_table_request_mutex_) {}

    KfdProcess *process_;
    std::unique_lock<util::DistributedSharedMutex> request_lock_;
  };

  [[nodiscard]] PageTableTransaction begin_page_table_transaction() {
    return PageTableTransaction(*this);
  }

  /// @brief Test published KFD grants while a VA ownership transaction is held.
  /// @details Missing or malformed state is treated as occupied. Unmapped
  /// allocations and retired tombstones have no active admission generation.
  bool active_gpu_grant_overlaps(PageTableTransaction &transaction, uint64_t gpu_va,
                                 size_t size) const {
    if (transaction.process_ != this || !transaction.request_lock_.owns_lock() || size == 0 ||
        size - 1 > UINT64_MAX - gpu_va)
      return true;
    const auto grants = gpu_grants_.load(std::memory_order_acquire);
    if (!grants || !grants->valid)
      return true;
    const uint64_t last_byte = gpu_va + size - 1;
    for (const auto &range : grants->ranges) {
      if (range.size == 0 || range.size > UINT64_MAX - range.gpu_va)
        return true;
      if (range.gpu_va <= last_byte && gpu_va <= range.gpu_va + range.size - 1 &&
          (!range.admission_generations.empty() || range.driver_gpu_mapping))
        return true;
    }
    return false;
  }

  /// @brief Preflight a proposed map while a transaction protects GEM state.
  /// @details Applied means no resident extent of the other interface overlaps
  /// the requested bytes. The later map must use this same transaction.
  PageTableMutationResult has_foreign_resident_overlap(PageTableTransaction &transaction,
                                                       uint64_t gpu_va, size_t size,
                                                       VaOwner va_owner) const {
    if (transaction.process_ != this || !transaction.request_lock_.owns_lock())
      return PageTableMutationResult::InvalidTransaction;
    if (size == 0)
      return PageTableMutationResult::Applied;
    if (size - 1 > UINT64_MAX - gpu_va)
      return PageTableMutationResult::InvalidRange;
    std::unique_lock lock(page_table_mutex_);
    return has_foreign_va_overlap_locked(gpu_va, size, va_owner, false)
               ? PageTableMutationResult::ForeignOwnerCollision
               : PageTableMutationResult::Applied;
  }

  /// @brief Map host pages into this process's GPU page table.
  /// @param mtype PTE MTYPE for these pages (derived from allocation flags).
  /// @param owner Who may revoke the backing; see HostExtentOwner. Defaults to
  ///        Application so an unannotated caller is validated rather than
  ///        trusted.
  PageTableMutationResult map_pages(uint64_t gpu_va, void *host_ptr, size_t size,
                                    amdgpu::Mtype mtype = amdgpu::Mtype::RW,
                                    HostExtentOwner owner = HostExtentOwner::Application,
                                    VaOwner va_owner = {}) {
    auto transaction = begin_page_table_transaction();
    return map_pages(transaction, gpu_va, host_ptr, size, mtype, owner, va_owner);
  }

  PageTableMutationResult map_pages(PageTableTransaction &transaction, uint64_t gpu_va,
                                    void *host_ptr, size_t size,
                                    amdgpu::Mtype mtype = amdgpu::Mtype::RW,
                                    HostExtentOwner owner = HostExtentOwner::Application,
                                    VaOwner va_owner = {}) {
    if (transaction.process_ != this || !transaction.request_lock_.owns_lock())
      return PageTableMutationResult::InvalidTransaction;
    if (size == 0)
      return PageTableMutationResult::Applied;
    if (size - 1 > UINT64_MAX - gpu_va)
      return PageTableMutationResult::InvalidRange;
    std::unique_lock lock(page_table_mutex_);
    if (has_foreign_va_overlap_locked(gpu_va, size, va_owner, false))
      return PageTableMutationResult::ForeignOwnerCollision;
    // MTYPE is a page field. A disjoint subpage map may coexist with the
    // other interface only when it preserves that page's existing policy.
    uint64_t inspected_va = gpu_va;
    size_t inspected_bytes = 0;
    while (inspected_bytes < size) {
      const auto page = page_table_.find(inspected_va >> kPageShift);
      if (page != page_table_.end() && page->second.mtype != mtype &&
          has_foreign_va_overlap_locked(inspected_va, 1, va_owner, false, true))
        return PageTableMutationResult::ForeignOwnerCollision;
      const size_t chunk =
          std::min<size_t>(kPageSize - (inspected_va & (kPageSize - 1)), size - inspected_bytes);
      inspected_va += chunk;
      inspected_bytes += chunk;
    }
    invalidate_page_policies_locked();
    auto *base = static_cast<uint8_t *>(host_ptr);
    uint64_t mapped_va = gpu_va;
    size_t host_offset = 0;
    try {
      while (host_offset < size) {
        const size_t gpu_page_offset = mapped_va & (kPageSize - 1);
        const size_t host_backed_bytes =
            std::min<size_t>(kPageSize - gpu_page_offset, size - host_offset);
        auto [page, inserted] =
            page_table_.try_emplace(mapped_va >> kPageShift, base + host_offset, mtype,
                                    host_backed_bytes, gpu_page_offset, owner, va_owner);
        if (!inserted) {
          page->second.mtype = mtype;
          replace_host_extent(page->second, {base + host_offset, host_backed_bytes, gpu_page_offset,
                                             owner, va_owner});
        }
        mapped_va += host_backed_bytes;
        host_offset += host_backed_bytes;
      }
    } catch (...) {
      // An allocation failure can leave a mapped prefix or an updated MTYPE.
      // Cached translations must still observe the changed page table.
      publish_page_table_mutation_locked();
      throw;
    }
    // Keep publication in the page-table critical section. Cached readers
    // validate this generation while holding the shared side of the same lock;
    // publishing after unlock would permit a stale-cache hit in between.
    publish_page_table_mutation_locked();
    return PageTableMutationResult::Applied;
  }

  /// @brief Unmap pages from this process's GPU page table.
  PageTableMutationResult unmap_pages(uint64_t gpu_va, size_t size, VaOwner va_owner = {}) {
    auto transaction = begin_page_table_transaction();
    return unmap_pages(transaction, gpu_va, size, va_owner);
  }

  PageTableMutationResult unmap_pages(PageTableTransaction &transaction, uint64_t gpu_va,
                                      size_t size, VaOwner va_owner = {}) {
    if (transaction.process_ != this || !transaction.request_lock_.owns_lock())
      return PageTableMutationResult::InvalidTransaction;
    if (size == 0)
      return PageTableMutationResult::Applied;
    if (size - 1 > UINT64_MAX - gpu_va)
      return PageTableMutationResult::InvalidRange;
    std::unique_lock lock(page_table_mutex_);
    if (has_foreign_va_overlap_locked(gpu_va, size, va_owner, true))
      return PageTableMutationResult::ForeignOwnerCollision;
    invalidate_page_policies_locked();
    uint64_t mapped_va = gpu_va;
    size_t unmapped_bytes = 0;
    try {
      while (unmapped_bytes < size) {
        const size_t chunk =
            std::min<size_t>(kPageSize - (mapped_va & (kPageSize - 1)), size - unmapped_bytes);
        auto page = page_table_.find(mapped_va >> kPageShift);
        if (page != page_table_.end()) {
          // Avoid building a temporary extent vector for a page being discarded.
          if (chunk != kPageSize)
            erase_host_extent(page->second, mapped_va & (kPageSize - 1), chunk);
          if (chunk == kPageSize || page->second.host_extents.empty())
            page_table_.erase(page);
        }
        mapped_va += chunk;
        unmapped_bytes += chunk;
      }
    } catch (...) {
      publish_page_table_mutation_locked();
      throw;
    }
    // See map_pages(): the mutation and generation publication are one
    // page-table critical section by design.
    publish_page_table_mutation_locked();
    return PageTableMutationResult::Applied;
  }

  /// @brief Replace mapped host pages while preserving their other PTE fields.
  /// @details The mutation and cache-generation publication occur under one
  /// page-table critical section. Only entries still pointing at the expected
  /// old page are changed.
  PageTableMutationResult remap_page_host_ptrs(uint64_t gpu_va, void *old_host_ptr,
                                               void *new_host_ptr, size_t size,
                                               VaOwner va_owner = {}) {
    auto transaction = begin_page_table_transaction();
    return remap_page_host_ptrs(transaction, gpu_va, old_host_ptr, new_host_ptr, size, va_owner);
  }

  PageTableMutationResult remap_page_host_ptrs(PageTableTransaction &transaction, uint64_t gpu_va,
                                               void *old_host_ptr, void *new_host_ptr, size_t size,
                                               VaOwner va_owner = {}) {
    if (transaction.process_ != this || !transaction.request_lock_.owns_lock())
      return PageTableMutationResult::InvalidTransaction;
    if (size == 0)
      return PageTableMutationResult::Applied;
    if (size - 1 > UINT64_MAX - gpu_va)
      return PageTableMutationResult::InvalidRange;
    std::unique_lock lock(page_table_mutex_);
    if (has_foreign_va_overlap_locked(gpu_va, size, va_owner, false))
      return PageTableMutationResult::ForeignOwnerCollision;
    invalidate_page_policies_locked();
    auto *old_base = static_cast<uint8_t *>(old_host_ptr);
    auto *new_base = static_cast<uint8_t *>(new_host_ptr);
    bool changed = false;
    uint64_t mapped_va = gpu_va;
    size_t host_offset = 0;
    while (host_offset < size) {
      auto page = page_table_.find(mapped_va >> kPageShift);
      if (page != page_table_.end()) {
        const uint64_t page_base = mapped_va & ~(kPageSize - 1);
        for (auto &extent : page->second.host_extents) {
          const uint64_t extent_va = page_base + extent.gpu_page_offset;
          if (extent_va < gpu_va || extent_va - gpu_va >= size)
            continue;
          const size_t extent_host_offset = extent_va - gpu_va;
          if (extent.host_ptr == old_base + extent_host_offset &&
              extent.host_ptr != new_base + extent_host_offset) {
            extent.host_ptr = new_base + extent_host_offset;
            changed = true;
          }
        }
      }
      const size_t chunk =
          std::min<size_t>(kPageSize - (mapped_va & (kPageSize - 1)), size - host_offset);
      mapped_va += chunk;
      host_offset += chunk;
    }
    if (changed)
      publish_page_table_mutation_locked();
    return PageTableMutationResult::Applied;
  }

  /// @brief Update the MTYPE of mapped pages and invalidate cached PTE copies.
  PageTableMutationResult set_page_mtype(uint64_t gpu_va, size_t size, amdgpu::Mtype mtype,
                                         VaOwner va_owner = {}) {
    auto transaction = begin_page_table_transaction();
    return set_page_mtype(transaction, gpu_va, size, mtype, va_owner);
  }

  PageTableMutationResult set_page_mtype(PageTableTransaction &transaction, uint64_t gpu_va,
                                         size_t size, amdgpu::Mtype mtype, VaOwner va_owner = {}) {
    if (transaction.process_ != this || !transaction.request_lock_.owns_lock())
      return PageTableMutationResult::InvalidTransaction;
    if (size == 0)
      return PageTableMutationResult::Applied;
    if (size - 1 > UINT64_MAX - gpu_va)
      return PageTableMutationResult::InvalidRange;
    std::unique_lock lock(page_table_mutex_);
    // MTYPE belongs to a whole PTE. Even a disjoint foreign extent within one
    // touched page would have its memory policy changed by this operation.
    if (has_foreign_va_overlap_locked(gpu_va, size, va_owner, false, true))
      return PageTableMutationResult::ForeignOwnerCollision;
    invalidate_page_policies_locked();
    bool changed = false;
    uint64_t mapped_va = gpu_va;
    size_t updated_bytes = 0;
    while (updated_bytes < size) {
      auto it = page_table_.find(mapped_va >> kPageShift);
      if (it != page_table_.end() && it->second.mtype != mtype) {
        it->second.mtype = mtype;
        changed = true;
      }
      const size_t chunk =
          std::min<size_t>(kPageSize - (mapped_va & (kPageSize - 1)), size - updated_bytes);
      mapped_va += chunk;
      updated_bytes += chunk;
    }
    if (changed)
      publish_page_table_mutation_locked();
    return PageTableMutationResult::Applied;
  }

  /// @brief Return the mutation counter used by GpuMemory translation caches.
  const uint64_t *page_table_generation() const { return &page_table_generation_; }

  /// @brief Return the retained token invalidated before each page-policy mutation.
  std::shared_ptr<const std::atomic<uint64_t>> page_table_mutation_epoch() const {
    return page_table_mutation_epoch_;
  }

  /// @brief Retained admission state for immutable copied PTE snapshots.
  std::shared_ptr<amdgpu::LegacyPageTableCacheState> page_table_cache_state() const {
    return page_table_cache_state_;
  }

  /// @brief Return the lease shared by page-table readers and mutations.
  std::shared_ptr<util::DistributedSharedMutex> page_table_request_mutex() const {
    return page_table_request_mutex_;
  }

  mutable util::DistributedSharedMutex page_table_mutex_;
  /// @brief Pool for page-table nodes and buckets.
  /// @details Writers serialize allocation under page_table_mutex_. Declaration
  /// order keeps the pool alive until the table is destroyed; capacity is
  /// retained for reuse until this process is destroyed.
  std::pmr::unsynchronized_pool_resource page_table_pool_;
  PageTable page_table_{&page_table_pool_};

  // -- Per-process state --

  uint32_t process_id_;
  // Address spaces are installed once for a KfdProcess and removed on close.
  // This distinguishes a later process/VM binding that reuses its numeric ID.
  const uint64_t process_incarnation_;
  pid_t client_pid_ = 0;
  std::atomic<uint32_t> open_ref_count_{1};

  /// @brief Serializes this process's ioctls, analogous to the real KFD's
  /// per-process lock. Taken as the outermost per-process lock in dispatch_ioctl.
  /// AMDKFD_IOC_WAIT_EVENTS is intentionally NOT taken under this lock: it blocks
  /// waiting for signals that SET_EVENT/RESET_EVENT (which DO run under this lock)
  /// must produce, so holding it would deadlock forward progress.
  std::mutex op_mutex_;
  /// @brief Canonical retirements queued for or holding the publication writer.
  /// @details Mmap and munmap can reject this process's own in-flight close
  /// without making unrelated processes fail on the global publication gate.
  std::atomic<uint32_t> canonical_retirements_pending_{0};
  mutable std::mutex alloc_mutex_;
  /// @brief Serializes scratch-pool backing allocation for this process.
  /// @details Every XCD of a fanned-out dispatch independently finds the same
  /// process-wide pool VA unbacked and races to map it; remapping a pool that
  /// already has live waves spilling into it would drop their data. Per-process
  /// rather than driver-wide so daemon clients do not serialize against each
  /// other. Lock order: hw_queue_mutex_ (held by the calling command processor)
  /// -> scratch_backing_mutex_ -> {alloc_mutex_, owned_fds_mutex_,
  /// page_table_mutex_}; nothing taken under it reaches a command processor.
  std::mutex scratch_backing_mutex_;
  std::unordered_map<uint64_t, GpuAllocation> allocations_;
  struct GpuGrantRange {
    uint64_t gpu_va = 0;
    uint64_t size = 0;
    std::unordered_map<uint32_t, uint64_t> admission_generations;
    bool driver_gpu_mapping = false;
    uint64_t driver_admission_generation = 0;
    std::shared_ptr<const GpuBacking> backing_owner;
    GpuGrantDomain domain = GpuGrantDomain::Unknown;
    uint32_t owner_gpu_id = 0;
    std::shared_ptr<const GpuBacking> local_backing_owner;
    bool cacheable_peer_for_testing = false;
  };
  struct GpuGrantIndex {
    std::vector<GpuGrantRange> ranges;
    GpuAliasSnapshot aliases;
    GpuAliasSnapshot local_aliases;
    // Monotonic end of all ranges through each sorted position. It locates an
    // earlier long overlap that a search on range starts alone would miss.
    std::vector<uint64_t> max_end_prefix;
    bool valid = true;
  };
  std::atomic<std::shared_ptr<const GpuGrantIndex>> gpu_grants_{
      std::make_shared<const GpuGrantIndex>()};
  std::shared_ptr<amdgpu::VmRawBackingWriteGate> raw_write_gate_ =
      std::make_shared<amdgpu::VmRawBackingWriteGate>();
  uint64_t gpu_admission_generation_ = 0;
  uint64_t host_view_generation_ = 0;
  std::vector<GpuGrantRange> retired_gpu_ranges_;
  uint64_t next_handle_ = 1;
  uint64_t next_gpu_va_;

  /// @brief Per-GPU state, indexed by gpu ordinal (0-based position in driver's gpus_ vector).
  std::vector<PerGpuState> gpu_state_;

  /// @brief Access per-GPU state by ordinal.
  PerGpuState &gpu(uint32_t ordinal) { return gpu_state_[ordinal]; }
  const PerGpuState &gpu(uint32_t ordinal) const { return gpu_state_[ordinal]; }

  uint32_t next_queue_id_ = 1;
  std::vector<uint32_t> active_queue_ids_;
  struct QueueDoorbellInfo {
    uint32_t gpu_ordinal;
    uint32_t doorbell_offset;
    amdgpu::QueueHandle queue_handle;
    std::optional<uint32_t> pm4_target_xcc;
  };
  std::unordered_map<uint32_t, QueueDoorbellInfo> queue_doorbell_map_;

  /// @brief Debug-relevant per-queue info reported by GET_QUEUE_SNAPSHOT.
  ///
  /// @details Captured when CREATE_QUEUE completes. rocm-dbgapi consumes the
  /// context-save-restore address/size to locate each queue's CWSR area (from
  /// which it walks the wave save state), plus the ring pointers to correlate
  /// dispatches. Mirrors the fields the kernel fills in
  /// @c kfd_queue_snapshot_entry (kfd_process_queue_manager.c:
  /// @c pqm_get_queue_snapshot).
  struct QueueSnapshotInfo {
    uint64_t ring_base_address = 0;
    uint64_t write_pointer_address = 0;
    uint64_t read_pointer_address = 0;
    uint64_t ctx_save_restore_address = 0;
    uint32_t ctx_save_restore_area_size = 0;
    uint32_t ring_size = 0;
    uint32_t queue_type = 0;
    uint32_t gpu_id = 0;
    uint32_t xcc_id = 0;
    /// CREATE_QUEUE supplied an exception-status address; the interrupt event is optional.
    bool has_runtime_exception_signal = false;
    uint64_t exception_status = 0; ///< Raised exceptions on this queue (KFD_EC_MASK bits).
    bool gws_allocated = false;    ///< Owns the process/GPU GWS grant until release or destruction.
    /// Exception bits whose ROCr ownership decision is still in flight.
    uint64_t runtime_exception_pending_status = 0;
    /// Runtime publications per bit; identical concurrent events may overlap.
    std::array<uint32_t, 64> runtime_exception_pending_counts{};
    /// Pending runtime bits for which at least one publication failed.
    uint64_t runtime_exception_failed_status = 0;
    /// Failed runtime-owned bits waiting for a debugger to resolve them.
    uint64_t runtime_exception_retained_status = 0;
    /// Runtime-owned bits already queried by the current debugger session.
    uint64_t runtime_exception_queried_status = 0;
    /// Session that owns the notification bookkeeping below.
    uint64_t debug_notification_session_generation = 0;
    /// Debugger notification writes currently in flight, counted per bit.
    std::array<uint32_t, 64> debug_notification_pending_counts{};
    /// Status hidden from QUERY until every corresponding write completes.
    uint64_t debug_notification_pending_status = 0;
    /// Status for which at least one notification write succeeded.
    uint64_t debug_notification_delivered_status = 0;
    /// Status retained for a later subscription even if notification fails.
    uint64_t debug_notification_retained_status = 0;
    /// Whole events represented by the flattened debugger status fields.
    /// Retaining their boundaries lets a retry preserve ownership when one
    /// subscribed bit in a combined event atomically replaces another.
    std::vector<uint64_t> debug_notification_events;

    void record_debug_notification_event(uint64_t mask) {
      if (mask != 0 && std::find(debug_notification_events.begin(), debug_notification_events.end(),
                                 mask) == debug_notification_events.end())
        debug_notification_events.push_back(mask);
    }

    void clear_debug_notification_events(uint64_t mask) {
      for (uint64_t &event : debug_notification_events)
        event &= ~mask;
      std::erase(debug_notification_events, uint64_t{0});
    }

    void prune_debug_notification_events() {
      for (uint64_t &event : debug_notification_events)
        event &= exception_status;
      std::erase(debug_notification_events, uint64_t{0});
    }

    /// @brief Return exception status that the debugger may currently consume.
    uint64_t debugger_visible_exception_status(uint64_t session_generation = 0) const {
      const uint64_t pending_status =
          session_generation == 0 || debug_notification_session_generation == session_generation
              ? debug_notification_pending_status
              : 0;
      return exception_status & ~runtime_exception_pending_status &
             ~runtime_exception_queried_status & ~pending_status;
    }

    /// @brief Reserve queue-exception bits for an in-flight ROCr decision.
    void begin_runtime_exception(uint64_t mask) {
      exception_status |= mask;
      record_debug_notification_event(mask);
      for (uint32_t bit = 0; bit < 64; ++bit) {
        const uint64_t bit_mask = uint64_t{1} << bit;
        if ((mask & bit_mask) == 0)
          continue;
        ++runtime_exception_pending_counts[bit];
        runtime_exception_pending_status |= bit_mask;
      }
    }

    /// @brief Resolve one in-flight ROCr decision without losing overlaps.
    uint64_t finish_runtime_exception(uint64_t mask, bool delivered, bool retain_failure) {
      uint64_t failed = 0;
      if (!delivered && retain_failure)
        runtime_exception_failed_status |= mask;
      for (uint32_t bit = 0; bit < 64; ++bit) {
        const uint64_t bit_mask = uint64_t{1} << bit;
        if ((mask & bit_mask) == 0)
          continue;
        if (runtime_exception_pending_counts[bit] == 0) {
          if (!delivered && retain_failure) {
            exception_status |= bit_mask;
            runtime_exception_retained_status |= bit_mask;
            runtime_exception_queried_status &= ~bit_mask;
            failed |= bit_mask;
          }
          continue;
        }
        if (--runtime_exception_pending_counts[bit] != 0)
          continue;
        runtime_exception_pending_status &= ~bit_mask;
        if ((runtime_exception_failed_status & bit_mask) != 0) {
          runtime_exception_retained_status |= bit_mask;
          runtime_exception_queried_status &= ~bit_mask;
          failed |= bit_mask;
        } else if (((debug_notification_pending_status | debug_notification_delivered_status |
                     debug_notification_retained_status | runtime_exception_retained_status) &
                    bit_mask) == 0) {
          exception_status &= ~bit_mask;
        }
        runtime_exception_failed_status &= ~bit_mask;
      }
      prune_debug_notification_events();
      return failed;
    }

    /// @brief Hide an event while its debugger-notifier write is in flight.
    void begin_debug_notification(uint64_t mask, uint64_t session_generation) {
      if (debug_notification_session_generation != session_generation) {
        debug_notification_session_generation = session_generation;
        debug_notification_pending_counts.fill(0);
        debug_notification_pending_status = 0;
        debug_notification_delivered_status = 0;
      }
      exception_status |= mask;
      for (uint32_t bit = 0; bit < 64; ++bit) {
        const uint64_t bit_mask = uint64_t{1} << bit;
        if ((mask & bit_mask) == 0)
          continue;
        ++debug_notification_pending_counts[bit];
        debug_notification_pending_status |= bit_mask;
      }
    }

    /// @brief Commit one debugger-notifier result without losing overlaps.
    void finish_debug_notification(uint64_t mask, uint64_t delivered_mask) {
      debug_notification_delivered_status |= mask & delivered_mask;
      for (uint32_t bit = 0; bit < 64; ++bit) {
        const uint64_t bit_mask = uint64_t{1} << bit;
        if ((mask & bit_mask) == 0 || debug_notification_pending_counts[bit] == 0)
          continue;
        if (--debug_notification_pending_counts[bit] != 0)
          continue;
        debug_notification_pending_status &= ~bit_mask;
        if ((debug_notification_delivered_status & bit_mask) == 0 &&
            (debug_notification_retained_status & bit_mask) == 0 &&
            (runtime_exception_pending_status & bit_mask) == 0 &&
            (runtime_exception_retained_status & bit_mask) == 0)
          exception_status &= ~bit_mask;
      }
      prune_debug_notification_events();
    }

    /// @brief Require another wake if this session later re-enables @p mask.
    void reset_debug_notification_delivery(uint64_t mask, uint64_t session_generation) {
      if (debug_notification_session_generation == session_generation)
        debug_notification_delivered_status &= ~mask;
    }

    /// @brief Consume only status currently visible to the debugger.
    void clear_debugger_exception_status(uint64_t mask) {
      const uint64_t consumed = mask & debugger_visible_exception_status();
      runtime_exception_queried_status |= consumed & runtime_exception_retained_status;
      exception_status &= ~(consumed & ~runtime_exception_retained_status);
      debug_notification_delivered_status &= ~consumed;
      debug_notification_retained_status &= ~consumed;
      clear_debug_notification_events(consumed & ~runtime_exception_retained_status);
    }

    /// @brief Resolve retained runtime ownership after a successful debugger resume.
    void resolve_runtime_exceptions() {
      const uint64_t resolved = runtime_exception_retained_status;
      runtime_exception_retained_status = 0;
      runtime_exception_queried_status &= ~resolved;
      runtime_exception_failed_status &= ~resolved;
      const uint64_t still_owned =
          runtime_exception_pending_status | debug_notification_pending_status |
          debug_notification_delivered_status | debug_notification_retained_status;
      exception_status &= ~(resolved & ~still_owned);
      prune_debug_notification_events();
    }

    /// @brief Clear every debugger-session-owned exception field.
    void clear_debugger_exception_state() {
      exception_status &= runtime_exception_pending_status | runtime_exception_retained_status;
      runtime_exception_queried_status = 0;
      debug_notification_session_generation = 0;
      debug_notification_pending_counts.fill(0);
      debug_notification_pending_status = 0;
      debug_notification_delivered_status = 0;
      debug_notification_retained_status = 0;
      prune_debug_notification_events();
    }

    /// @brief Area used by the XCC that owns this queue.
    uint64_t cwsr_xcc_address() const {
      return ctx_save_restore_address == 0
                 ? 0
                 : ctx_save_restore_address +
                       static_cast<uint64_t>(xcc_id) * ctx_save_restore_area_size;
    }
  };
  std::unordered_map<uint32_t, QueueSnapshotInfo> queue_snapshot_map_;

  EventState event_state_;

  std::unordered_map<uint32_t, MemoryPolicy> memory_policies_;
  std::unordered_map<uint64_t, ImportedDmabuf> imported_dmabufs_;
  std::unordered_map<int, uint64_t> fd_to_import_handle_;
  std::unordered_map<uint64_t, SvmRange> svm_ranges_;
  std::mutex runtime_mutex_;
  RuntimeState runtime_state_;

private:
  /// @pre page_table_mutex_ is held, and the range has been checked for wrap.
  bool has_foreign_va_overlap_locked(uint64_t gpu_va, size_t size, VaOwner va_owner, bool unmapping,
                                     bool whole_pages = false) const {
    uint64_t mapped_va = gpu_va;
    size_t examined_bytes = 0;
    while (examined_bytes < size) {
      const size_t page_offset = mapped_va & (kPageSize - 1);
      const size_t chunk = std::min<size_t>(kPageSize - page_offset, size - examined_bytes);
      const size_t range_begin = whole_pages ? 0 : page_offset;
      const size_t range_end = whole_pages ? kPageSize : page_offset + chunk;
      const auto page = page_table_.find(mapped_va >> kPageShift);
      if (page != page_table_.end()) {
        for (const auto &extent : page->second.host_extents) {
          const size_t extent_begin = extent.gpu_page_offset;
          const size_t extent_end = extent_begin + extent.host_backed_bytes;
          if (extent_end <= range_begin || range_end <= extent_begin)
            continue;
          if (extent.va_owner.kind != va_owner.kind ||
              (unmapping && va_owner.identity != 0 && extent.va_owner.identity != 0 &&
               extent.va_owner.identity != va_owner.identity))
            return true;
        }
      }
      mapped_va += chunk;
      examined_bytes += chunk;
    }
    return false;
  }

  static void normalize_host_extents(PageTableEntry &page) {
    auto &extents = page.host_extents;
    if (extents.size() > 1)
      std::ranges::sort(extents, {}, &HostExtent::gpu_page_offset);
    size_t out = 0;
    for (const auto &extent : extents) {
      if (extent.host_ptr == nullptr || extent.host_backed_bytes == 0)
        continue;
      if (out > 0) {
        auto &previous = extents[out - 1];
        if (previous.owner == extent.owner && previous.va_owner == extent.va_owner &&
            previous.gpu_page_offset + previous.host_backed_bytes == extent.gpu_page_offset &&
            previous.host_ptr + previous.host_backed_bytes == extent.host_ptr) {
          previous.host_backed_bytes += extent.host_backed_bytes;
          continue;
        }
      }
      extents[out++] = extent;
    }
    extents.resize(out);
  }

  static void replace_host_extent(PageTableEntry &page, HostExtent replacement) {
    const size_t replacement_begin = replacement.gpu_page_offset;
    const size_t replacement_end = replacement_begin + replacement.host_backed_bytes;
    if (replacement_begin == 0 && replacement_end == kPageSize) {
      page.host_extents = std::vector<HostExtent>{replacement};
      return;
    }
    std::vector<HostExtent> updated;
    updated.reserve(page.host_extents.size() + 1);
    for (const auto &extent : page.host_extents) {
      const size_t extent_begin = extent.gpu_page_offset;
      const size_t extent_end = extent_begin + extent.host_backed_bytes;
      if (extent_end <= replacement_begin || replacement_end <= extent_begin) {
        updated.push_back(extent);
        continue;
      }
      if (extent_begin < replacement_begin)
        updated.push_back({extent.host_ptr, replacement_begin - extent_begin, extent_begin,
                           extent.owner, extent.va_owner});
      if (replacement_end < extent_end)
        updated.push_back({extent.host_ptr + (replacement_end - extent_begin),
                           extent_end - replacement_end, replacement_end, extent.owner,
                           extent.va_owner});
    }
    updated.push_back(replacement);
    page.host_extents = std::move(updated);
    normalize_host_extents(page);
  }

  static void erase_host_extent(PageTableEntry &page, size_t erased_begin, size_t erased_bytes) {
    const size_t erased_end = erased_begin + erased_bytes;
    std::vector<HostExtent> updated;
    updated.reserve(page.host_extents.size() + 1);
    for (const auto &extent : page.host_extents) {
      const size_t extent_begin = extent.gpu_page_offset;
      const size_t extent_end = extent_begin + extent.host_backed_bytes;
      if (extent_end <= erased_begin || erased_end <= extent_begin) {
        updated.push_back(extent);
        continue;
      }
      if (extent_begin < erased_begin)
        updated.push_back({extent.host_ptr, erased_begin - extent_begin, extent_begin, extent.owner,
                           extent.va_owner});
      if (erased_end < extent_end)
        updated.push_back({extent.host_ptr + (erased_end - extent_begin), extent_end - erased_end,
                           erased_end, extent.owner, extent.va_owner});
    }
    page.host_extents = std::move(updated);
    normalize_host_extents(page);
  }

  // Invalidate copied policies before mutation, including a partial update
  // that throws before publishing the ordinary translation generation.
  void invalidate_page_policies_locked() {
    page_table_mutation_epoch_->fetch_add(1, std::memory_order_release);
    page_table_cache_state_->invalidate_and_wait();
  }

  void publish_page_table_mutation_locked() { ++page_table_generation_; }

  std::shared_ptr<util::DistributedSharedMutex> page_table_request_mutex_ =
      std::make_shared<util::DistributedSharedMutex>();

  /// @brief Page table version counter, bumped on every PTE mutation.
  /// @details GpuMemory keeps per-thread TLB-like translation caches keyed by
  ///          this generation. Mutations hold both page_table_request_mutex_
  ///          and page_table_mutex_; readers hold at least one of those locks,
  ///          so the counter itself does not need atomics.
  uint64_t page_table_generation_{1};
  /// @brief Retained atomic token for lockless copied-policy checks.
  /// @details Invalidate before mutation, including partially throwing updates.
  std::shared_ptr<std::atomic<uint64_t>> page_table_mutation_epoch_ =
      std::make_shared<std::atomic<uint64_t>>(1);
  std::shared_ptr<amdgpu::LegacyPageTableCacheState> page_table_cache_state_ =
      std::make_shared<amdgpu::LegacyPageTableCacheState>();
};

inline void KfdProcess::publish_gpu_grants_locked() {
  auto transaction = begin_page_table_transaction();
  publish_gpu_grants_locked(transaction);
}

inline void KfdProcess::publish_gpu_grants_locked(PageTableTransaction &transaction) {
  if (transaction.process_ != this || !transaction.request_lock_.owns_lock())
    throw std::logic_error("GPU grant publication requires a live page-table transaction");
  publish_gpu_grants_impl_locked();
}

} // namespace rocjitsu
