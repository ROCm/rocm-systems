// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_SIMULATED_DRIVER_H_
#define ROCJITSU_KMD_LINUX_SIMULATED_DRIVER_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/driver.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace rocjitsu {

// KFD mmap offset encoding (mirrors kfd_priv.h).
inline constexpr uint64_t KFD_MMAP_TYPE_SHIFT = 62;
inline constexpr uint64_t KFD_MMAP_TYPE_MASK = 0x3ULL << KFD_MMAP_TYPE_SHIFT;
inline constexpr uint64_t KFD_MMAP_TYPE_DOORBELL = 0x3ULL << KFD_MMAP_TYPE_SHIFT;
inline constexpr uint64_t KFD_MMAP_TYPE_EVENTS = 0x2ULL << KFD_MMAP_TYPE_SHIFT;
inline constexpr uint64_t KFD_MMAP_GPU_ID_SHIFT = 46;

inline constexpr uint64_t kfd_mmap_gpu_id(uint32_t gpu_id) {
  return (static_cast<uint64_t>(gpu_id) << KFD_MMAP_GPU_ID_SHIFT) &
         ((1ULL << KFD_MMAP_TYPE_SHIFT) - (1ULL << KFD_MMAP_GPU_ID_SHIFT));
}

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
class SimulatedDriver : public Driver {
public:
  [[nodiscard]] bool daemon_mode() const { return daemon_mode_; }

  SimulatedDriver(SoC &soc, bool daemon_mode = false);
  SimulatedDriver(std::vector<SoC *> socs, std::vector<uint32_t> gpu_ids, bool daemon_mode = false);
  ~SimulatedDriver() override;

  /// @brief Local-mode interface (interposer). Operates on the local process.
  /// @{
  int open() override;
  int close() override;
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
  uint32_t open_process();

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
  bool is_doorbell_range(const void *addr, size_t length) const;
  uint32_t gpu_id() const { return gpus_.empty() ? 0 : gpus_[0].gpu_id; }
  uint32_t num_gpus() const { return static_cast<uint32_t>(gpus_.size()); }
  const Sysfs &topology() const { return topology_; }
  std::string topology_path() const { return topology_.path(); }
  [[nodiscard]] int fd() const { return fd_; }
  [[nodiscard]] uint32_t local_process_id() const { return local_process_id_; }
  [[nodiscard]] bool owns_fd(int fd) const;
  std::string redirect_sysfs_path(const char *path) const;
  [[nodiscard]] int claim_fd(int real_fd);
  [[nodiscard]] bool owns_reserved_fd(int fd) const;

  /// @brief Per-GPU device state (mirrors kfd_dev in the kernel).
  struct GpuDevice {
    SoC *soc = nullptr;
    uint32_t gpu_id = 0;
    bool cps_initialized = false;
    kfd_process_device_apertures apertures{};
  };

private:
  /// @brief Look up a KfdProcess by ID. Returns nullptr if not found.
  std::shared_ptr<KfdProcess> find_process(uint32_t process_id) const;

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

  int get_version_ioctl(void *arg);
  int get_clock_counters_ioctl(void *arg);
  int get_apertures_ioctl(void *arg);
  int acquire_vm_ioctl(void *arg);
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
  int set_xnack_mode_ioctl(void *arg);
  bool allocate_scratch_backing(uint32_t process_id, uint64_t gpu_va, size_t size);

  std::vector<GpuDevice> gpus_;
  bool daemon_mode_ = false;
  int fd_ = -1;

  /// @brief Process table mapping process_id to KfdProcess.
  /// @details Protected by process_mutex_ for concurrent daemon access.
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

#endif // ROCJITSU_KMD_LINUX_SIMULATED_DRIVER_H_
