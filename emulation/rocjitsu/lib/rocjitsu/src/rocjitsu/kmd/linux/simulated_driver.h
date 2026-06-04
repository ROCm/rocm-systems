// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_SIMULATED_DRIVER_H_
#define ROCJITSU_KMD_LINUX_SIMULATED_DRIVER_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/kmd/linux/events.h"
#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/driver.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

/// @brief Simulated kernel-mode driver that routes KFD ioctls to the simulator.
///
/// @details Also manages the global singleton instance used by the LD_PRELOAD
/// interposer. The interposer calls the static methods (get_or_create, lookup,
/// redirect_sysfs_path) which delegate to the singleton.
class SimulatedDriver : public Driver {
public:
  // -- Static singleton interface (used by the interposer) --

  /// @brief Get or lazily create the global driver singleton.
  /// @returns Pointer to the driver, or nullptr on failure.
  static SimulatedDriver *get_or_create();

  /// @brief Look up the driver by its KFD file descriptor.
  /// @param fd The file descriptor returned by open().
  /// @returns Pointer to the driver, or nullptr if fd doesn't match.
  static SimulatedDriver *lookup(int fd);

  /// @brief Get the KFD fd for the singleton (-1 if not open).
  static int kfd_fd();

  /// @brief Redirect a sysfs path through the generated topology.
  /// @param path The original sysfs path.
  /// @returns Redirected path, or empty string if no redirect needed.
  static std::string redirect_sysfs_path(const char *path);

  /// @brief Re-entry guard for driver construction (thread-local).
  static bool in_construction();

  // -- Instance interface --

  /// @brief Create a default driver from the RJ_CONFIG env var.
  static std::unique_ptr<SimulatedDriver> create_default();

  /// @brief Construct with a simulation engine and SoC.
  SimulatedDriver(simdojo::SimulationEngine &engine, SoC &soc);
  ~SimulatedDriver() override;

  int open() override;
  int close() override;
  int ioctl(unsigned long request, void *arg) override;
  void *mmap(void *addr, size_t length, int prot, int flags, off_t offset) override;
  int munmap(void *addr, size_t length) override;

  /// @brief Generate sysfs topology files for the interposer to redirect.
  /// @details Must be called before ROCR's hsa_init().
  void setup_topology(const Sysfs::GpuInfo &gpu);

  /// @brief Returns true if the address range overlaps the mapped doorbell aperture.
  bool is_doorbell_range(const void *addr, size_t length) const;

  /// Get the topology generator (for inspection/testing).
  const Sysfs &topology() const { return topology_; }

  /// Get the generated sysfs topology directory path.
  std::string topology_path() const { return topology_.path(); }

private:
  int get_version_ioctl(void *arg);
  int get_clock_counters_ioctl(void *arg);
  int get_apertures_ioctl(void *arg);
  int acquire_vm_ioctl(void *arg);
  int alloc_memory_ioctl(void *arg);
  int free_memory_ioctl(void *arg);
  int map_memory_ioctl(void *arg);
  int unmap_memory_ioctl(void *arg);
  int create_queue_ioctl(void *arg);
  int update_queue_ioctl(void *arg);
  int destroy_queue_ioctl(void *arg);
  int create_event_ioctl(void *arg);
  int set_memory_policy_ioctl(void *arg);
  int destroy_event_ioctl(void *arg);
  int set_event_ioctl(void *arg);
  int reset_event_ioctl(void *arg);
  int wait_events_ioctl(void *arg);
  int import_dmabuf_ioctl(void *arg);
  int export_dmabuf_ioctl(void *arg);
  int get_dmabuf_info_ioctl(void *arg);
  int svm_ioctl(void *arg);
  int runtime_enable_ioctl(void *arg);
  int set_xnack_mode_ioctl(void *arg);

  simdojo::SimulationEngine &engine_;
  SoC &soc_;
  int fd_ = -1; ///< Stable synthetic KFD fd (memfd); allocated once on first open(), reused across
                ///< close/reopen.

  uint32_t gpu_id_ = 0; ///< KFD gpu_id reported to userspace (set by setup_topology).
  /// CommandProcessor for this KFD device. Set once in open() from the SoC
  /// topology. All queue operations (create, flush, destroy) go through this
  /// pointer so no other code needs to know about the underlying XCD index.
  amdgpu::CommandProcessor *cp_ = nullptr;

  struct GpuAllocation {
    uint64_t gpu_va = 0;
    uint64_t size = 0;
    void *host_ptr = nullptr;
    uint32_t flags = 0;
    uint64_t handle = 0;
    bool user_va = false; ///< True if va_addr was provided by the caller (ROCR FMM path).
    bool imported = false;
    int dmabuf_fd = -1;
  };

  std::mutex alloc_mutex_;
  std::unordered_map<uint64_t, GpuAllocation> allocations_;
  uint64_t next_handle_ = 1;
  uint64_t next_gpu_va_ = 0x100000000ULL;

  /// @brief AMDGPU flat memory apertures (architecture-independent, 48-bit SVM).
  /// @details Mirrors kfd_flat_memory.c kfd_init_apertures_v9. The gpu_id field
  ///          is zero here; callers set it at copy time from gpu_id_.
  static constexpr kfd_process_device_apertures default_apertures_{
      .lds_base = 0x1000000000000ULL, // 1 << 48
      .lds_limit = 0x10000FFFFFFFFULL,
      .scratch_base = 0x2000000000000ULL, // 2 << 48
      .scratch_limit = 0x20000FFFFFFFFULL,
      .gpuvm_base = 0x1000000000ULL,    // 4 GB — above executable/stack/heap
      .gpuvm_limit = 0x3FFFFFFFFFFFULL, // 64 TB — safely below the Linux mmap
                                        // base (~124 TB on x86-64 with ASLR)
      .gpu_id = 0,
      .pad = 0,
  };

  uint32_t next_queue_id_ = 1;
  uint64_t next_doorbell_offset_ = 0;
  std::vector<uint32_t>
      active_queue_ids_; ///< Tracks queue IDs registered with the CP (for cleanup on close).
  void *doorbell_page_ =
      nullptr; ///< Mapped aperture page; base address given to the CP via set_doorbell_base().
  size_t doorbell_page_size_ = 0; ///< Size of the mapped doorbell aperture.
  uint64_t doorbell_gpu_va_ = 0;  ///< GPU VA associated with the doorbell aperture.

  EventState event_state_;

  struct MemoryPolicy {
    uint64_t alternate_base = 0;
    uint64_t alternate_size = 0;
    uint32_t default_policy = 0;
    uint32_t alternate_policy = 0;
  };

  struct ImportedDmabuf {
    uint64_t handle = 0;
    int fd = -1;
    uint64_t size = 0;
    uint64_t va = 0;
    uint32_t gpu_id = 0;
  };

  struct SvmRange {
    uint64_t size = 0;
    std::unordered_map<uint32_t, uint32_t> attributes;
  };

  struct RuntimeState {
    bool enabled = false;
    bool pending = false;
    uint32_t mode_mask = 0;
    uint32_t capabilities_mask = 0;
    uint64_t r_debug = 0;
  };

  std::unordered_map<uint32_t, MemoryPolicy> memory_policies_;
  std::unordered_map<uint64_t, ImportedDmabuf> imported_dmabufs_;
  std::unordered_map<int, uint64_t> fd_to_import_handle_;
  std::unordered_map<uint64_t, SvmRange> svm_ranges_;
  std::mutex runtime_mutex_;
  RuntimeState runtime_state_;

  uint64_t scratch_backing_va_ = 0; ///< Flat-scratch base from ROCR (SET_SCRATCH_BACKING_VA).
  uint64_t trap_tba_addr_ = 0;      ///< Trap Base Address from ROCR (SET_TRAP_HANDLER).
  uint64_t trap_tma_addr_ = 0;      ///< Trap Meta Address from ROCR (SET_TRAP_HANDLER).

  Sysfs topology_;
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_SIMULATED_DRIVER_H_
