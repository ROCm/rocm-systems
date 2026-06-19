// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_SIMULATED_KFD_H_
#define ROCJITSU_KMD_LINUX_SIMULATED_KFD_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace rocjitsu {

/// @brief 128-bit IPC share handle key, matching the kernel's random handle.
struct IpcHandleKey {
  uint32_t words[4];
  bool operator==(const IpcHandleKey &) const = default;
};

struct IpcHandleKeyHash {
  size_t operator()(const IpcHandleKey &key) const {
    size_t hash_value = std::hash<uint32_t>{}(key.words[0]);
    for (int idx = 1; idx < 4; idx++)
      hash_value ^= std::hash<uint32_t>{}(key.words[idx]) + 0x9e3779b9 + (hash_value << 6) +
                    (hash_value >> 2);
    return hash_value;
  }
};

/// @brief Exported IPC object stored in the driver's global IPC store.
struct IpcObject {
  uint32_t share_handle[4];
  int backing_memfd = -1;
  uint64_t allocation_size = 0;
  uint32_t allocation_flags = 0;
  uint32_t source_gpu_id = 0;
  uint32_t source_process_id = 0;
  uint64_t source_alloc_handle = 0;
};

/// @brief Simulated kernel-mode driver that routes KFD ioctls to the simulator.
///
/// @details Per-process state (allocations, queues, events, doorbells) is held
/// in KfdProcess instances. The driver maintains a process table and resolves
/// the target process from a process_id parameter — matching the real kernel's
/// kfd_chardev_ioctl which resolves kfd_process from filp->private_data.
///
/// The local-mode virtual interface (open/close/ioctl/mmap/munmap) operates on
/// the process created by open(). The daemon uses the process_id-aware
/// overloads so each client thread identifies itself by connection, not by
/// shared mutable state.
class SimulatedKfd : public LinuxKfd {
public:
  [[nodiscard]] bool daemon_mode() const { return daemon_mode_; }

  SimulatedKfd(SoC &soc, bool daemon_mode = false);
  SimulatedKfd(std::vector<SoC *> socs, std::vector<uint32_t> gpu_ids, bool daemon_mode = false);
  ~SimulatedKfd() override;

  /// @brief Local-mode interface (interposer). Operates on the local process.
  /// @{
  int open() override;
  int close() override;

  /// @brief Add one open reference to the local process without re-opening.
  /// @details Used by the interposer when an existing KFD fd is duplicated
  /// (dup/dup2/dup3/fcntl F_DUPFD). Each live fd holds one reference so the
  /// process is torn down only when the last fd is closed, not the first.
  /// @retval true A reference was added.
  /// @retval false No local process to retain (e.g. it was already torn down, or
  ///         daemon/remote mode); the caller must NOT treat the fd as retained.
  [[nodiscard]] bool retain_local_open() override;
  int ioctl(unsigned long request, void *arg) override;
  void *mmap(void *addr, size_t length, int prot, int flags, off_t offset) override;
  int munmap(void *addr, size_t length) override;
  /// @}

  /// @brief Daemon-mode process lifecycle. Thread-safe for concurrent clients.
  /// @{

  /// @brief Atomically create a new process and return its ID.
  /// @details Unlike open(), which sets local_process_id_ (not thread-safe for
  /// concurrent daemon clients), this method returns the ID directly so the
  /// caller can associate it with a specific client connection.
  uint32_t open_process(pid_t client_pid = 0);
  void set_process_client_pid(uint32_t process_id, pid_t client_pid);

  int ioctl(uint32_t process_id, unsigned long request, void *arg);
  void *mmap(uint32_t process_id, void *addr, size_t length, int prot, int flags, off_t offset);
  int munmap(uint32_t process_id, void *addr, size_t length);
  int close(uint32_t process_id);
  [[nodiscard]] int get_mmap_memfd(uint32_t process_id, off_t offset) const;
  /// @}

  /// @brief Local-mode get_mmap_memfd (uses local process).
  [[nodiscard]] int get_mmap_memfd(off_t offset) const;

  void setup_topology(const Sysfs::GpuInfo &gpu);
  void setup_topology(const config::KfdDeviceConfig &dev, uint32_t num_xcc);
  void setup_topology(const std::vector<config::KfdDeviceConfig> &devs, uint32_t num_xcc);
  bool is_doorbell_range(const void *addr, size_t length) const override;
  uint32_t gpu_id() const { return gpus_.empty() ? 0 : gpus_[0].gpu_id; }
  uint32_t num_gpus() const { return static_cast<uint32_t>(gpus_.size()); }
  const Sysfs &topology() const { return topology_; }
  std::string topology_path() const override { return topology_.path(); }
  std::string drm_path() const override { return topology_.drm_path(); }
  [[nodiscard]] int fd() const override { return fd_.load(std::memory_order_acquire); }
  [[nodiscard]] uint32_t local_process_id() const { return local_process_id_; }

  /// @brief Forget the primary KFD fd number without touching the process.
  /// @details Used by the interposer when dup2/dup3 atomically overwrites the
  /// primary KFD fd number: the number no longer refers to this driver, so stop
  /// classifying it as the local primary (fd() must no longer match it). Only
  /// clears if @p fd matches the current primary, so a stale call is a no-op.
  /// @returns kClearedDropRef if @p fd matched and was cleared (the local primary
  ///          holds one counted open reference, so the caller drops it via
  ///          close()); kNotPrimary if it did not match (a concurrent overwrite
  ///          won the race), so the caller must not release a reference.
  /// @details Runs under process_mutex_ so the CAS on fd_ is serialized with
  /// open()'s fd creation/selection/return: a racing dup2 can no longer clear
  /// fd_ in the window between open() publishing it and open() returning it, so
  /// open() never hands back -1 or an already-overwritten descriptor. Lock-free
  /// readers (driver_fd()/kfd_backend_of()/is_kfd_primary()) still observe fd_
  /// atomically.
  [[nodiscard]] PrimaryInvalidation invalidate_primary_fd(int fd) override;

  /// @brief Open-reference count of the local process, or 0 if none is alive.
  /// @details Introspection for tests/diagnostics. Each live KFD fd (the primary
  /// plus every dup) holds one reference; the process is destroyed at zero.
  [[nodiscard]] uint32_t local_open_ref_count() const;

  [[nodiscard]] bool owns_fd(int fd) const override;
  std::string redirect_sysfs_path(const char *path) const override;
  [[nodiscard]] bool handles_drm_render_minor(uint32_t minor) const override;
  [[nodiscard]] const Sysfs::GpuInfo *gpu_info_for_render_minor(uint32_t minor) const override;
  [[nodiscard]] int claim_fd(int real_fd);
  [[nodiscard]] bool owns_reserved_fd(int fd) const;

  /// @brief Derive the PTE MTYPE from KFD allocation flags (mirrors amdgpu).
  /// @details Public so the interposer's DRM GEM_VA path can install page-table
  /// entries with the same coherency type the KFD alloc requested.
  static amdgpu::Mtype pte_mtype_for_flags(uint32_t alloc_flags);

  /// @brief Install a host range into the local process's GPU page table.
  /// @details Drives DRM AMDGPU_GEM_VA MAP/REPLACE from the interposer: maps
  /// @p size bytes at @p gpu_va to @p host_ptr with the MTYPE derived from
  /// @p alloc_flags. No-op if the local process is gone.
  void gem_va_map(uint64_t gpu_va, void *host_ptr, size_t size, uint32_t alloc_flags);

  /// @brief Remove a GPU page-table range installed by gem_va_map (GEM_VA UNMAP).
  void gem_va_unmap(uint64_t gpu_va, size_t size);

  /// @brief Look up a KfdProcess by ID. Returns nullptr if not found.
  /// @details Public so the interposer can read a local allocation's flags when
  /// synthesizing GEM bookkeeping at EXPORT_DMABUF time.
  std::shared_ptr<KfdProcess> find_process(uint32_t process_id) const;

  /// @brief Per-GPU device state (mirrors kfd_dev in the kernel).
  struct GpuDevice {
    SoC *soc = nullptr;
    uint32_t gpu_id = 0;
    bool cps_initialized = false;
    kfd_process_device_apertures apertures{};
  };

private:
  /// @brief Look up the local-mode process.
  std::shared_ptr<KfdProcess> find_local_process() const;

  /// @brief Look up a KfdProcess by its client (Linux) pid. Used by
  /// AMDKFD_IOC_DBG_TRAP to resolve the debug target, mirroring the kernel's
  /// kfd_lookup_process_by_pid().
  /// @param pid Client (Linux) pid of the target process.
  /// @return The matching KfdProcess, or nullptr if none matches.
  std::shared_ptr<KfdProcess> find_process_by_client_pid(pid_t pid) const;

  /// @brief Look up a GpuDevice by gpu_id. Returns nullptr if not found.
  GpuDevice *find_gpu(uint32_t gpu_id);
  const GpuDevice *find_gpu(uint32_t gpu_id) const;

  /// @brief Get the ordinal (0-based index) for a gpu_id. Returns 0 if not found.
  uint32_t gpu_ordinal(uint32_t gpu_id) const {
    for (uint32_t i = 0; i < gpus_.size(); ++i)
      if (gpus_[i].gpu_id == gpu_id)
        return i;
    return 0;
  }

  void map_to_gpu(KfdProcess &proc, uint64_t gpu_va, void *host_ptr, size_t size,
                  amdgpu::Mtype mtype = amdgpu::Mtype::RW);
  void unmap_from_gpu(KfdProcess &proc, uint64_t gpu_va, size_t size);

  int dispatch_ioctl(KfdProcess &proc, unsigned long request, void *arg);
  void *dispatch_mmap(KfdProcess &proc, void *addr, size_t length, int prot, int flags,
                      off_t offset);
  int dispatch_munmap(KfdProcess &proc, void *addr, size_t length);
  int dispatch_get_mmap_memfd(KfdProcess &proc, off_t offset) const;

  int get_process_apertures_ioctl(void *arg) override;
  int acquire_vm_ioctl(void *arg) override;
  int get_available_memory_ioctl(void *arg) override;
  int set_memory_policy_ioctl(void *arg) override;
  int alloc_memory_ioctl(void *arg) override;
  int free_memory_ioctl(void *arg) override;
  int map_memory_ioctl(void *arg) override;
  int unmap_memory_ioctl(void *arg) override;
  int get_available_memory_ioctl(KfdProcess &proc, void *arg);
  int alloc_memory_ioctl(KfdProcess &proc, void *arg);
  int free_memory_ioctl(KfdProcess &proc, void *arg);
  int map_memory_ioctl(KfdProcess &proc, void *arg);
  int unmap_memory_ioctl(KfdProcess &proc, void *arg);
  int create_queue_ioctl(KfdProcess &proc, void *arg);
  int update_queue_ioctl(KfdProcess &proc, void *arg);
  int destroy_queue_ioctl(KfdProcess &proc, void *arg);
  int create_event_ioctl(KfdProcess &proc, void *arg);
  int set_memory_policy_ioctl(KfdProcess &proc, void *arg);
  int destroy_event_ioctl(KfdProcess &proc, void *arg);
  int set_event_ioctl(KfdProcess &proc, void *arg);
  int reset_event_ioctl(KfdProcess &proc, void *arg);
  int wait_events_ioctl(KfdProcess &proc, void *arg);
  int import_dmabuf_ioctl(KfdProcess &proc, void *arg);
  int export_dmabuf_ioctl(KfdProcess &proc, void *arg);
  int get_dmabuf_info_ioctl(KfdProcess &proc, void *arg);
  int ipc_export_handle_ioctl(KfdProcess &proc, void *arg);
  int ipc_import_handle_ioctl(KfdProcess &proc, void *arg);
  int svm_ioctl(KfdProcess &proc, void *arg);
  int runtime_enable_ioctl(KfdProcess &proc, void *arg);
  int debug_trap_ioctl(KfdProcess &caller, void *arg);
  int set_xnack_mode_ioctl(void *arg);
  int get_tile_config_ioctl(void *arg);
  bool allocate_scratch_backing(uint32_t process_id, uint64_t gpu_va, size_t size);

  /// @brief Lazily create the backing memfd exactly once across racing opens.
  /// @details CAS-publishes fd_ so concurrent open()/open_process() callers agree
  /// on a single memfd; losers close their own and adopt the winner's.
  /// @retval true fd_ holds a valid descriptor. @retval false memfd_create failed.
  [[nodiscard]] bool ensure_fd_created();

  /// @brief One-time per-GPU CP setup: apertures + interrupt/scratch callbacks.
  /// @details Idempotent per GpuDevice via the cps_initialized flag. The caller
  /// MUST hold process_mutex_ so the check-and-set of cps_initialized is atomic
  /// against concurrent daemon opens that would otherwise both register callbacks.
  void init_command_processors_locked();

  std::vector<GpuDevice> gpus_;
  bool daemon_mode_ = false;
  std::atomic<int> fd_{-1};

  /// @brief Process table mapping process_id to KfdProcess.
  /// @details Protected by process_mutex_ for concurrent daemon access.
  ///
  /// Global lock ordering (acquire in this order; never the reverse).
  /// op_mutex_ (KfdProcess) is the outermost per-process ioctl lock. Under it,
  /// process_mutex_ and alloc_mutex_ are independent siblings — they are NEVER
  /// held simultaneously (allocate_scratch_backing and close() both release
  /// process_mutex_ before taking alloc_mutex_):
  ///   op_mutex_ < process_mutex_
  ///   op_mutex_ < alloc_mutex_ < {ipc_mutex_, page_table_mutex_, owned_fds_mutex_}
  ///   op_mutex_ < runtime_mutex_ < alloc_mutex_        (runtime_enable_ioctl)
  ///   op_mutex_ < debug_mutex_ < runtime_mutex_        (debug_trap_ioctl)
  ///   process_mutex_ < interrupt_mutex_                (open()/open_process())
  /// The op_mutex_ in the debug rule is always the CALLER's, while debug_mutex_/
  /// runtime_mutex_ may belong to a DIFFERENT process (the debug target resolved
  /// by client pid). debug_trap_ioctl holds only the caller's op_mutex_ and never
  /// acquires the target's op_mutex_, so a cross-process attach cannot deadlock.
  /// interrupt_mutex_ is a leaf: the CP interrupt callback takes it and only
  /// descends into EventState::mutex_, and close() takes it only after releasing
  /// process_mutex_, so there is no cycle.
  /// The CP engine thread acquires hw_queue_mutex_ first, then reaches
  /// process_mutex_ (scratch resolver) or alloc_mutex_ (scratch allocator) through
  /// the callbacks — hw_queue_mutex_ -> process_mutex_ and, separately,
  /// hw_queue_mutex_ -> alloc_mutex_ (never both nested). To avoid an ABBA against
  /// that thread, an ioctl MUST NOT hold a per-process lock (alloc_mutex_) across a
  /// CommandProcessor call that takes hw_queue_mutex_ (e.g. register_queue) — build
  /// state under alloc_mutex_, release it, then call the CP.
  mutable std::mutex process_mutex_;
  std::unordered_map<uint32_t, std::shared_ptr<KfdProcess>> processes_;
  uint32_t next_process_id_ = 1;

  /// @brief Interrupt dispatch: process_id → EventState*.
  /// @details Protected by interrupt_mutex_. Decoupled from process_mutex_
  /// to avoid ABBA deadlocks with hw_queue_mutex_ in the CP doorbell thread.
  mutable std::mutex interrupt_mutex_;
  std::unordered_map<uint32_t, EventState *> event_dispatch_;

  /// @brief Process ID for local-mode (interposer). Set once in open().
  uint32_t local_process_id_ = 0;

  static constexpr kfd_process_device_apertures default_apertures_{
      .lds_base = 0x1000000000000ULL,
      .lds_limit = 0x10000FFFFFFFFULL,
      .scratch_base = 0x2000000000000ULL,
      .scratch_limit = 0x20000FFFFFFFFULL,
      .gpuvm_base = 0x1000000000ULL,
      .gpuvm_limit = 0x3FFFFFFFFFFFULL,
      .gpu_id = 0,
      .pad = 0,
  };

  /// @brief IPC handle store for cross-process memory sharing.
  /// @details Lock ordering: process_mutex_ < alloc_mutex_ < ipc_mutex_.
  mutable std::mutex ipc_mutex_;
  std::unordered_map<IpcHandleKey, IpcObject, IpcHandleKeyHash> ipc_store_;

  mutable std::mutex owned_fds_mutex_;
  std::unordered_set<int> owned_fds_;

  Sysfs topology_;

  static constexpr int kReservedFdCount = 256;
  int reserved_fd_base_ = 0;
  int next_reserved_fd_ = 0;

  void init_reserved_fd_range();
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_SIMULATED_KFD_H_
