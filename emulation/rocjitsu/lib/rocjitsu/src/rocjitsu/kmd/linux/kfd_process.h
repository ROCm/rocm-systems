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
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
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
    void *host_ptr = nullptr;
    uint32_t flags = 0;
    uint64_t handle = 0;
    int memfd = -1;
    uint32_t gpu_id = 0;
    bool user_va = false;
    bool imported = false;
    int dmabuf_fd = -1;
    // True when the driver created host_ptr (mmap it itself) and must munmap it
    // on teardown. False for caller-owned pages (e.g. reused MAP_FIXED pages
    // from the thunk) that the driver must never unmap, since unmapping them
    // races with the owning process still accessing the memory.
    bool host_ptr_owned = false;
  };

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

  static constexpr uint64_t kPageShift = 12;
  static constexpr uint64_t kPageSize = 1ULL << kPageShift;

  /// @brief One host-backed interval within a GPU page.
  struct HostExtent {
    uint8_t *host_ptr = nullptr;
    /// Number of host-allocation-backed bytes starting at host_ptr.
    size_t host_backed_bytes = 0;
    /// GPU-page offset that corresponds to host_ptr.
    size_t gpu_page_offset = 0;
  };

  /// @brief One inline host extent, spilling to dynamic storage only for split pages.
  class HostExtentList {
  public:
    HostExtentList() = default;
    HostExtentList(const HostExtentList &other) { copy_from(other); }
    HostExtentList(HostExtentList &&other) noexcept { move_from(std::move(other)); }
    HostExtentList(std::initializer_list<HostExtent> extents) {
      for (const auto &extent : extents)
        push_back(extent);
    }

    HostExtentList &operator=(const HostExtentList &other) {
      if (this != &other)
        copy_from(other);
      return *this;
    }
    HostExtentList &operator=(HostExtentList &&other) noexcept {
      if (this != &other)
        move_from(std::move(other));
      return *this;
    }

    [[nodiscard]] size_t size() const {
      if (std::holds_alternative<std::monostate>(storage_))
        return 0;
      if (std::holds_alternative<HostExtent>(storage_))
        return 1;
      return std::get<std::vector<HostExtent>>(storage_).size();
    }
    [[nodiscard]] bool empty() const { return size() == 0; }

    HostExtent *data() {
      if (auto *single = std::get_if<HostExtent>(&storage_))
        return single;
      if (auto *many = std::get_if<std::vector<HostExtent>>(&storage_))
        return many->data();
      return nullptr;
    }
    const HostExtent *data() const {
      if (const auto *single = std::get_if<HostExtent>(&storage_))
        return single;
      if (const auto *many = std::get_if<std::vector<HostExtent>>(&storage_))
        return many->data();
      return nullptr;
    }
    HostExtent *begin() { return data(); }
    const HostExtent *begin() const { return data(); }
    HostExtent *end() {
      auto *first = data();
      return first ? first + size() : nullptr;
    }
    const HostExtent *end() const {
      const auto *first = data();
      return first ? first + size() : nullptr;
    }
    HostExtent &front() { return (*this)[0]; }
    const HostExtent &front() const { return (*this)[0]; }
    HostExtent &back() { return (*this)[size() - 1]; }
    const HostExtent &back() const { return (*this)[size() - 1]; }
    HostExtent &operator[](size_t index) { return data()[index]; }
    const HostExtent &operator[](size_t index) const { return data()[index]; }

    void reserve(size_t capacity) {
      if (capacity <= 1)
        return;
      if (auto *many = std::get_if<std::vector<HostExtent>>(&storage_)) {
        many->reserve(capacity);
        return;
      }
      std::vector<HostExtent> many;
      many.reserve(capacity);
      if (auto *single = std::get_if<HostExtent>(&storage_))
        many.push_back(*single);
      storage_.emplace<std::vector<HostExtent>>(std::move(many));
    }

    void push_back(const HostExtent &extent) {
      if (std::holds_alternative<std::monostate>(storage_)) {
        storage_.emplace<HostExtent>(extent);
        return;
      }
      if (auto *single = std::get_if<HostExtent>(&storage_)) {
        std::vector<HostExtent> many;
        many.reserve(2);
        many.push_back(*single);
        many.push_back(extent);
        storage_.emplace<std::vector<HostExtent>>(std::move(many));
        return;
      }
      std::get<std::vector<HostExtent>>(storage_).push_back(extent);
    }

    void resize(size_t count) {
      if (count == 0) {
        storage_.emplace<std::monostate>();
        return;
      }
      if (count == 1) {
        if (auto *many = std::get_if<std::vector<HostExtent>>(&storage_)) {
          HostExtent single = many->front();
          storage_.emplace<HostExtent>(single);
        }
        return;
      }
      reserve(count);
      std::get<std::vector<HostExtent>>(storage_).resize(count);
    }

    HostExtentList &operator=(std::vector<HostExtent> extents) {
      if (extents.empty())
        storage_.emplace<std::monostate>();
      else if (extents.size() == 1)
        storage_.emplace<HostExtent>(extents.front());
      else
        storage_.emplace<std::vector<HostExtent>>(std::move(extents));
      return *this;
    }

  private:
    void copy_from(const HostExtentList &other) {
      if (const auto *single = std::get_if<HostExtent>(&other.storage_))
        storage_.emplace<HostExtent>(*single);
      else if (const auto *many = std::get_if<std::vector<HostExtent>>(&other.storage_))
        storage_.emplace<std::vector<HostExtent>>(*many);
      else
        storage_.emplace<std::monostate>();
    }

    void move_from(HostExtentList &&other) {
      if (auto *single = std::get_if<HostExtent>(&other.storage_))
        storage_.emplace<HostExtent>(*single);
      else if (auto *many = std::get_if<std::vector<HostExtent>>(&other.storage_))
        storage_.emplace<std::vector<HostExtent>>(std::move(*many));
      else
        storage_.emplace<std::monostate>();
    }

    std::variant<std::monostate, HostExtent, std::vector<HostExtent>> storage_;
  };

  /// @brief Per-page translation entry, mirroring HW PTE fields.
  /// @details A hardware PTE has one page-wide MTYPE, while local USERPTR
  /// allocations can contribute several disjoint host-backed intervals to the
  /// same GPU page. Keeping all intervals prevents a later sub-page mapping or
  /// unmapping from silently replacing an unrelated sibling.
  struct PageTableEntry {
    PageTableEntry() = default;
    PageTableEntry(uint8_t *host_ptr, amdgpu::Mtype page_mtype)
        : mtype(page_mtype), host_extents{{host_ptr, kPageSize, 0}} {}
    PageTableEntry(uint8_t *host_ptr, amdgpu::Mtype page_mtype, size_t host_backed_bytes,
                   size_t gpu_page_offset)
        : mtype(page_mtype), host_extents{{host_ptr, host_backed_bytes, gpu_page_offset}} {}

    amdgpu::Mtype mtype = amdgpu::Mtype::RW;
    HostExtentList host_extents;
  };

  /// @brief Per-process GPU page table (GPU VA page number → PTE).
  /// @details Managed by the driver's mmap/munmap handlers. GpuMemory holds a
  ///          pointer to the active process's page table and resolves translations
  ///          on each memory access (TLB-like role).
  using PageTable = std::unordered_map<uint64_t, PageTableEntry>;

  /// @brief Map host pages into this process's GPU page table.
  /// @param mtype PTE MTYPE for these pages (derived from allocation flags).
  void map_pages(uint64_t gpu_va, void *host_ptr, size_t size,
                 amdgpu::Mtype mtype = amdgpu::Mtype::RW) {
    std::unique_lock lock(page_table_mutex_);
    auto *base = static_cast<uint8_t *>(host_ptr);
    uint64_t mapped_va = gpu_va;
    size_t host_offset = 0;
    while (host_offset < size) {
      const size_t gpu_page_offset = mapped_va & (kPageSize - 1);
      const size_t host_backed_bytes =
          std::min<size_t>(kPageSize - gpu_page_offset, size - host_offset);
      auto [page, inserted] = page_table_.try_emplace(mapped_va >> kPageShift, base + host_offset,
                                                      mtype, host_backed_bytes, gpu_page_offset);
      if (!inserted) {
        page->second.mtype = mtype;
        replace_host_extent(page->second, {base + host_offset, host_backed_bytes, gpu_page_offset});
      }
      mapped_va += host_backed_bytes;
      host_offset += host_backed_bytes;
    }
    // Keep publication in the page-table critical section. Cached readers
    // validate this generation while holding the shared side of the same lock;
    // publishing after unlock would permit a stale-cache hit in between.
    publish_page_table_mutation_locked();
  }

  /// @brief Unmap pages from this process's GPU page table.
  void unmap_pages(uint64_t gpu_va, size_t size) {
    std::unique_lock lock(page_table_mutex_);
    uint64_t mapped_va = gpu_va;
    size_t unmapped_bytes = 0;
    while (unmapped_bytes < size) {
      const size_t chunk =
          std::min<size_t>(kPageSize - (mapped_va & (kPageSize - 1)), size - unmapped_bytes);
      auto page = page_table_.find(mapped_va >> kPageShift);
      if (page != page_table_.end()) {
        erase_host_extent(page->second, mapped_va & (kPageSize - 1), chunk);
        if (page->second.host_extents.empty())
          page_table_.erase(page);
      }
      mapped_va += chunk;
      unmapped_bytes += chunk;
    }
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
  }

  /// @brief Update the MTYPE of mapped pages and invalidate cached PTE copies.
  void set_page_mtype(uint64_t gpu_va, size_t size, amdgpu::Mtype mtype) {
    std::unique_lock lock(page_table_mutex_);
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
  }

  /// @brief Return the mutation counter used by GpuMemory translation caches.
  const uint64_t *page_table_generation() const { return &page_table_generation_; }

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
  static void normalize_host_extents(PageTableEntry &page) {
    auto &extents = page.host_extents;
    if (extents.size() > 1)
      std::sort(extents.begin(), extents.end(), [](const HostExtent &lhs, const HostExtent &rhs) {
        return lhs.gpu_page_offset < rhs.gpu_page_offset;
      });
    size_t out = 0;
    for (const auto &extent : extents) {
      if (extent.host_ptr == nullptr || extent.host_backed_bytes == 0)
        continue;
      if (out > 0) {
        auto &previous = extents[out - 1];
        if (previous.gpu_page_offset + previous.host_backed_bytes == extent.gpu_page_offset &&
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
        updated.push_back({extent.host_ptr, replacement_begin - extent_begin, extent_begin});
      if (replacement_end < extent_end)
        updated.push_back({extent.host_ptr + (replacement_end - extent_begin),
                           extent_end - replacement_end, replacement_end});
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
        updated.push_back({extent.host_ptr, erased_begin - extent_begin, extent_begin});
      if (erased_end < extent_end)
        updated.push_back(
            {extent.host_ptr + (erased_end - extent_begin), extent_end - erased_end, erased_end});
    }
    page.host_extents = std::move(updated);
    normalize_host_extents(page);
  }

  void publish_page_table_mutation_locked() { ++page_table_generation_; }

  /// @brief Page table version counter, bumped on every PTE mutation.
  /// @details GpuMemory keeps per-thread TLB-like translation caches keyed by
  ///          this generation; all reads and writes occur while holding
  ///          page_table_mutex_, so the counter itself does not need atomics.
  uint64_t page_table_generation_{1};
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_KFD_PROCESS_H_
