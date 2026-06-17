// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kfd_process.h
/// @brief Per-process KFD state, analogous to the kernel's kfd_process.
///
/// @details Each process that opens /dev/kfd gets a KfdProcess instance holding
/// its allocations, queues, events, doorbells, and memory policies. The
/// SimulatedDriver owns a process table mapping fds to KfdProcess instances,
/// and delegates per-process ioctl operations through here.

#ifndef ROCJITSU_KMD_LINUX_KFD_PROCESS_H_
#define ROCJITSU_KMD_LINUX_KFD_PROCESS_H_

#include "rocjitsu/kmd/linux/events.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu {

/// @brief Per-process KFD state.
///
/// @details Mirrors the kernel's kfd_process + kfd_process_device for a
/// single-GPU simulator. Each daemon client connection or local-mode session
/// gets one KfdProcess. The SimulatedDriver maintains a process table and
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
  bool release_open() { return open_ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1; }

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
  }

  /// @brief Unmap pages from this process's GPU page table.
  void unmap_pages(uint64_t gpu_va, size_t size) {
    std::unique_lock lock(page_table_mutex_);
    for (size_t off = 0; off < size; off += kPageSize)
      page_table_.erase((gpu_va + off) >> kPageShift);
  }

  mutable std::shared_mutex page_table_mutex_;
  PageTable page_table_;

  // -- Per-process state --

  uint32_t process_id_;
  pid_t client_pid_ = 0;
  std::atomic<uint32_t> open_ref_count_{1};

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

private:
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_KFD_PROCESS_H_
