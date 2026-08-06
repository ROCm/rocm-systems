// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kfd_process.h
/// @brief Per-process KFD state, analogous to the kernel's kfd_process.
///
/// @details Each process that opens /dev/kfd gets a KfdProcess instance holding
/// its allocations, queues, events, doorbells, and memory policies. The
/// SimulatedKfd owns a process table mapping fds to KfdProcess instances,
/// and delegates per-process ioctl operations through here.

#ifndef ROCJITSU_KMD_LINUX_KFD_PROCESS_H_
#define ROCJITSU_KMD_LINUX_KFD_PROCESS_H_

#include "rocjitsu/kmd/linux/events.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "util/unique_handle.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/types.h> // pid_t

namespace rocjitsu {

/// @brief Per-process KFD state.
///
/// @details Mirrors the kernel's kfd_process + kfd_process_device for a
/// single-GPU simulator. Each daemon client connection or local-mode session
/// gets one KfdProcess. The SimulatedKfd maintains a process table and
/// routes ioctls to the correct KfdProcess.
class KfdProcess {
public:
  /// @brief Per-GPU state within a process.
  struct PerGpuState {
    void *doorbell_page = nullptr;
    size_t doorbell_page_size = 0;
    uint64_t doorbell_gpu_va = 0;
    uint64_t next_doorbell_offset = 0;
    std::vector<uint32_t> free_doorbell_offsets;
    uint64_t scratch_backing_va = 0;
    uint64_t trap_tba_addr = 0;
    uint64_t trap_tma_addr = 0;
  };

  /// @brief Construct a new KFD process with a unique process ID.
  /// @param process_id Unique identifier (analogous to PASID) for CP routing.
  /// @param num_gpus Number of GPU devices (sizes per-GPU state vector).
  explicit KfdProcess(uint32_t process_id, uint32_t num_gpus = 1)
      : process_id_(process_id), next_gpu_va_(0x1000000000ULL), gpu_state_(num_gpus) {}

  /// @brief Get the process ID (PASID analog).
  uint32_t process_id() const { return process_id_; }

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

  /// @brief GPU memory allocation descriptor.
  struct GpuAllocation {
    uint64_t gpu_va = 0;
    uint64_t size = 0;
    uint64_t backing_size = 0;
    void *host_ptr = nullptr;
    uint32_t flags = 0;
    amdgpu::Mtype backing_mtype = amdgpu::Mtype::RW;
    uint64_t handle = 0;
    int memfd = -1;
    uint32_t gpu_id = 0;
    bool user_va = false;
    bool imported = false;
    int dmabuf_fd = -1;
    bool permanent_backing = false;
    bool mapped_to_gpu = false;
    bool backing_authoritative = false;
    // Page-aligned allocation-relative intervals already consumed from an
    // existing MAP_FIXED client reservation. Keys are inclusive starts and
    // values are exclusive ends.
    std::map<uint64_t, uint64_t> imported_client_extents;
    // True when the driver created host_ptr (mmap it itself) and must munmap it
    // on teardown. False for caller-owned pages (e.g. reused MAP_FIXED pages
    // from the thunk) that the driver must never unmap, since unmapping them
    // races with the owning process still accessing the memory.
    bool host_ptr_owned = false;
  };

  /// @brief A stable lookup result for allocation-owned GPU backing.
  /// @details An inaccessible result identifies a managed address that must not
  /// fall through to identity-pointer translation. The guard keeps accessible
  /// backing alive until the caller finishes the memory operation.
  struct ResolvedBacking {
    uint8_t *address = nullptr;
    uint8_t *page_base = nullptr;
    uint8_t *range_base = nullptr;
    uint64_t range_size = 0;
    amdgpu::Mtype mtype = amdgpu::Mtype::RW;
    bool gpu_accessible = false;
    std::shared_lock<std::shared_mutex> guard;
  };

  [[nodiscard]] static bool is_valid_gpu_range(uint64_t gpu_va, uint64_t size) {
    return size != 0 && gpu_va <= UINT64_MAX - (size - 1);
  }

  /// @brief Whether an allocation already covers any page in a proposed range.
  /// @details The caller must hold alloc_mutex_. GPU mappings are page-granular,
  /// so byte-disjoint ranges sharing a page are treated as overlapping.
  [[nodiscard]] bool overlaps_allocation_locked(uint64_t gpu_va, uint64_t size) const {
    assert(is_valid_gpu_range(gpu_va, size));
    const uint64_t first_page = gpu_va >> kPageShift;
    const uint64_t end_page = ((gpu_va + size - 1) >> kPageShift) + 1;
    for (const auto &[handle, alloc] : allocations_) {
      (void)handle;
      const uint64_t alloc_first_page = alloc.gpu_va >> kPageShift;
      const uint64_t alloc_end_page = ((alloc.gpu_va + alloc.size - 1) >> kPageShift) + 1;
      if (first_page < alloc_end_page && alloc_first_page < end_page)
        return true;
    }
    return false;
  }

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

  /// @brief Per-process debugger session state.
  ///
  /// @details Mirrors the debug-related fields the kernel maintains on
  /// @c struct @c kfd_process in
  /// @c drivers/gpu/drm/amd/amdkfd/kfd_priv.h.
  /// Managed by the @c AMDKFD_IOC_DBG_TRAP ioctl handler
  /// (@c kfd_ioctl_set_debug_trap in the real driver).
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
    /// @brief Mirrors @c kfd_process::debug_trap_enabled.
    /// Set when the device process is debug-attached with a reserved VMID.
    bool enabled = false;

    /// @brief Mirrors @c kfd_process::runtime_info.runtime_state.
    /// Holds a @c kfd_dbg_runtime_state value (see @c kfd_ioctl.h).
    uint32_t runtime_state = 0;

    /// @brief Mirrors @c kfd_process::exception_enable_mask.
    /// Bitmask of exception classes that are forwarded to the debugger.
    uint64_t exception_enable_mask = 0;

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
    util::UniqueHandle owned_dbg_fd;

    /// @brief Mirrors @c kfd_process::debugger_process (stored as pid instead of pointer).
    /// Linux PID of the attached debugger (ptrace parent). 0 when not attached.
    pid_t debugger_pid = 0;
  };

  /// @brief Per-page translation entry, mirroring HW PTE fields.
  /// @details Stores the host pointer for VA→PA translation and the PTE MTYPE
  /// that the GPU MMU uses to override instruction-level caching. On real
  /// AMDGPU hardware, PTE bits 57:55 encode MTYPE per page.
  struct PageTableEntry {
    uint8_t *host_ptr = nullptr;
    amdgpu::Mtype mtype = amdgpu::Mtype::RW;
  };

  /// @brief Per-process GPU page table (GPU VA page number → PTE).
  /// @details Managed by the driver's mmap/munmap handlers. GpuMemory holds a
  ///          pointer to the active process's page table and resolves translations
  ///          on each memory access (TLB-like role).
  using PageTable = std::unordered_map<uint64_t, PageTableEntry>;

  static constexpr uint64_t kPageShift = 12;
  static constexpr uint64_t kPageSize = 1ULL << kPageShift;

  /// @brief Map host pages into this process's GPU page table.
  /// @param mtype PTE MTYPE for these pages (derived from allocation flags).
  void map_pages(uint64_t gpu_va, void *host_ptr, size_t size,
                 amdgpu::Mtype mtype = amdgpu::Mtype::RW) {
    std::unique_lock lock(page_table_mutex_);
    auto *base = static_cast<uint8_t *>(host_ptr);
    for (size_t off = 0; off < size; off += kPageSize)
      page_table_[(gpu_va + off) >> kPageShift] = {base + off, mtype};
    // Keep publication in the page-table critical section. Cached readers
    // validate this generation while holding the shared side of the same lock;
    // publishing after unlock would permit a stale-cache hit in between.
    publish_page_table_mutation_locked();
  }

  /// @brief Unmap pages from this process's GPU page table.
  void unmap_pages(uint64_t gpu_va, size_t size) {
    std::unique_lock lock(page_table_mutex_);
    for (size_t off = 0; off < size; off += kPageSize)
      page_table_.erase((gpu_va + off) >> kPageShift);
    // See map_pages(): the mutation and generation publication are one
    // page-table critical section by design.
    publish_page_table_mutation_locked();
  }

  /// @brief Replace mapped host pages while preserving their other PTE fields.
  /// @details The mutation and cache-generation publication occur under one
  /// page-table critical section. Only entries still pointing at the expected
  /// old page are changed.
  void remap_page_host_ptrs(uint64_t gpu_va, void *old_host_ptr, void *new_host_ptr, size_t size) {
    std::unique_lock lock(page_table_mutex_);
    auto *old_base = static_cast<uint8_t *>(old_host_ptr);
    auto *new_base = static_cast<uint8_t *>(new_host_ptr);
    bool changed = false;
    for (size_t off = 0; off < size; off += kPageSize) {
      auto it = page_table_.find((gpu_va + off) >> kPageShift);
      if (it != page_table_.end() && it->second.host_ptr == old_base + off &&
          it->second.host_ptr != new_base + off) {
        it->second.host_ptr = new_base + off;
        changed = true;
      }
    }
    if (changed)
      publish_page_table_mutation_locked();
  }

  /// @brief Update the MTYPE of mapped pages and invalidate cached PTE copies.
  void set_page_mtype(uint64_t gpu_va, size_t size, amdgpu::Mtype mtype) {
    std::unique_lock lock(page_table_mutex_);
    bool changed = false;
    for (size_t off = 0; off < size; off += kPageSize) {
      auto it = page_table_.find((gpu_va + off) >> kPageShift);
      if (it != page_table_.end() && it->second.mtype != mtype) {
        it->second.mtype = mtype;
        changed = true;
      }
    }
    if (changed)
      publish_page_table_mutation_locked();
  }

  /// @brief Return the mutation counter used by GpuMemory translation caches.
  const uint64_t *page_table_generation() const { return &page_table_generation_; }

  /// @brief Resolve a managed GPU range to allocation-owned host backing.
  /// @details Unmapped allocations still return a result with
  /// gpu_accessible=false, preventing unsafe identity-pointer fallback. The
  /// returned guard prevents teardown while accessible backing is in use.
  [[nodiscard]] std::optional<ResolvedBacking> resolve_backing(uint64_t gpu_va, size_t size) const {
    if (size == 0 || gpu_va > UINT64_MAX - (static_cast<uint64_t>(size) - 1))
      return std::nullopt;

    std::shared_lock lock(backing_ranges_mutex_);
    auto it = std::upper_bound(
        backing_ranges_.begin(), backing_ranges_.end(), gpu_va,
        [](uint64_t addr, const BackingRange &range) { return addr < range.first; });
    if (it == backing_ranges_.begin())
      return std::nullopt;
    --it;

    const uint64_t offset = gpu_va - it->first;
    if (offset >= it->size || static_cast<uint64_t>(size) > it->size - offset)
      return std::nullopt;

    const uintptr_t address_value = reinterpret_cast<uintptr_t>(it->host_base) + offset;
    return ResolvedBacking{reinterpret_cast<uint8_t *>(address_value),
                           reinterpret_cast<uint8_t *>(address_value - (gpu_va & (kPageSize - 1))),
                           it->host_base,
                           it->size,
                           it->mtype,
                           it->gpu_accessible,
                           std::move(lock)};
  }

  /// @brief Republish allocation-backed address ranges after a state change.
  /// @details The caller must hold alloc_mutex_. Acquiring the unique range lock
  /// waits for in-flight GPU accesses before backing can be unmapped.
  void refresh_backing_ranges_locked() { publish_backing_ranges_locked(); }

  mutable std::shared_mutex page_table_mutex_;
  PageTable page_table_;

  // -- Per-process state --

  uint32_t process_id_;
  pid_t client_pid_ = 0;
  std::atomic<uint32_t> open_ref_count_{1};

  /// @brief Serializes this process's ioctls, analogous to the real KFD's
  /// per-process lock. Taken as the outermost per-process lock in dispatch_ioctl.
  /// AMDKFD_IOC_WAIT_EVENTS is intentionally NOT taken under this lock: it blocks
  /// waiting for signals that SET_EVENT/RESET_EVENT (which DO run under this lock)
  /// must produce, so holding it would deadlock forward progress.
  std::mutex op_mutex_;
  mutable std::mutex alloc_mutex_;
  std::unordered_map<uint64_t, GpuAllocation> allocations_;
  // Page-aligned CPU alias intervals. Keys are inclusive host addresses and
  // values are exclusive ends.
  std::map<uintptr_t, uintptr_t> cpu_mappings_;
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
  };
  std::unordered_map<uint32_t, QueueDoorbellInfo> queue_doorbell_map_;

  EventState event_state_;

  std::unordered_map<uint32_t, MemoryPolicy> memory_policies_;
  std::unordered_map<uint64_t, ImportedDmabuf> imported_dmabufs_;
  std::unordered_map<int, uint64_t> fd_to_import_handle_;
  std::unordered_map<uint64_t, SvmRange> svm_ranges_;
  std::mutex runtime_mutex_;
  RuntimeState runtime_state_;

  mutable std::mutex debug_mutex_;
  DebugSession debug_session_;

private:
  void publish_page_table_mutation_locked() { ++page_table_generation_; }

  /// @brief Page table version counter, bumped on every PTE mutation.
  /// @details GpuMemory keeps per-thread TLB-like translation caches keyed by
  ///          this generation; all reads and writes occur while holding
  ///          page_table_mutex_, so the counter itself does not need atomics.
  uint64_t page_table_generation_{1};

  struct BackingRange {
    uint64_t first = 0;
    uint64_t size = 0;
    uint8_t *host_base = nullptr;
    amdgpu::Mtype mtype = amdgpu::Mtype::RW;
    bool gpu_accessible = false;
  };

  void publish_backing_ranges_locked() {
    std::vector<BackingRange> ranges;
    ranges.reserve(allocations_.size());
    for (const auto &[handle, alloc] : allocations_) {
      (void)handle;
      if (!alloc.permanent_backing || !alloc.host_ptr)
        continue;
      assert(is_valid_gpu_range(alloc.gpu_va, alloc.size));
      ranges.push_back({alloc.gpu_va, alloc.size, static_cast<uint8_t *>(alloc.host_ptr),
                        alloc.backing_mtype, alloc.mapped_to_gpu});
    }

    std::sort(ranges.begin(), ranges.end(), [](const BackingRange &lhs, const BackingRange &rhs) {
      return lhs.first < rhs.first;
    });
    for (size_t i = 1; i < ranges.size(); ++i) {
      assert(ranges[i - 1].first <= ranges[i].first);
      assert(ranges[i - 1].size <= ranges[i].first - ranges[i - 1].first &&
             "overlapping GPU allocation backings");
    }

    std::unique_lock lock(backing_ranges_mutex_);
    backing_ranges_ = std::move(ranges);
  }

  mutable std::shared_mutex backing_ranges_mutex_;
  std::vector<BackingRange> backing_ranges_;
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_KFD_PROCESS_H_
