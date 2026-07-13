// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file guest_kfd.h
/// @brief KFD discovery driver that appends one synthetic DBT guest GPU.

#ifndef ROCJITSU_KMD_LINUX_GUEST_KFD_H_
#define ROCJITSU_KMD_LINUX_GUEST_KFD_H_

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/kmd/linux/sysfs.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

namespace rocjitsu {

/// @brief KFD driver that exposes a guest GPU for DBT while forwarding host GPU work.
///
/// @details KFD emulation ends at discovery: this class appends one guest GPU
/// to KFD topology and process apertures, but does not execute guest queues.
/// Host-GPU ioctls are forwarded to the real /dev/kfd. If an execution ioctl
/// still targets the guest GPU, the driver returns an error so the missing HSA
/// forwarding path is visible.
class GuestKfd : public LinuxKfd {
public:
  /// @brief Construct a guest discovery driver from parsed DBT configuration.
  explicit GuestKfd(config::DbtGuestConfig config);

  /// @brief Close the real KFD fd and remove generated overlay state.
  ~GuestKfd() override;

  /// @brief Open the real /dev/kfd fd and lazily prepare guest discovery.
  int open() override;

  /// @brief Handle close for the /dev/kfd fd represented by this driver.
  int close() override;

  /// @brief Route guest discovery ioctls locally and host ioctls to real KFD.
  int ioctl(unsigned long request, void *arg) override;

  /// @brief Map host-backed KFD offsets and reject unsupported guest doorbells.
  void *mmap(void *addr, size_t length, int prot, int flags, off_t offset) override;

  /// @brief Forward unmaps for mappings created through this driver.
  int munmap(void *addr, size_t length) override;

  /// @brief Return the real /dev/kfd fd.
  [[nodiscard]] int fd() const override;

  /// @brief Return true when @p fd is an internal rocjitsu-owned fd.
  [[nodiscard]] bool owns_fd(int fd) const override;

  /// @brief Redirect KFD topology and guest DRM sysfs paths into the overlay.
  [[nodiscard]] std::string redirect_sysfs_path(const char *path) const override;

  /// @brief Return true if a mapping range overlaps a protected doorbell.
  [[nodiscard]] bool is_doorbell_range(const void *addr, size_t length) const override;

  /// @brief Return true when @p minor is the configured guest render node.
  [[nodiscard]] bool handles_drm_render_minor(uint32_t minor) const override;

  /// @brief Return synthetic AMDGPU metadata for the guest render node.
  [[nodiscard]] const Sysfs::GpuInfo *gpu_info_for_render_minor(uint32_t minor) const override;

  /// @brief Return the generated KFD topology root.
  [[nodiscard]] std::string topology_path() const override;

  /// @brief Return an empty DRM root because host DRM paths stay real.
  [[nodiscard]] std::string drm_path() const override { return {}; }

  /// @brief Detach inherited child-process state before destroying this copy.
  void reset_after_fork() override;

  /// @brief Prepare guest discovery without retaining an application open fd.
  bool prepare_for_discovery();

  /// @brief Add one open reference for a duplicated KFD fd.
  void retain_local_open() override;

private:
  class TopologyOverlay;

  /// @brief Open real KFD, generate topology, and select the host GPU.
  bool ensure_ready();

  /// @brief Open the process's real /dev/kfd fd while mutex_ is held.
  bool ensure_real_kfd_locked();

  /// @brief Forward one ioctl to the real /dev/kfd fd.
  int forward_ioctl(unsigned long request, void *arg);

  /// @brief Return real process apertures plus one synthetic guest aperture.
  int get_process_apertures_ioctl(void *arg) override;

  /// @brief Return guest clock-counter values or forward host requests.
  int get_clock_counters_ioctl(void *arg) override;

  /// @brief Succeed guest VM acquisition without creating a guest execution VM.
  int acquire_vm_ioctl(void *arg) override;

  /// @brief Report the configured guest-visible local memory size.
  int get_available_memory_ioctl(void *arg) override;

  /// @brief Accept guest startup memory policy setup and forward host policy.
  int set_memory_policy_ioctl(void *arg) override;

  /// @brief Allocate a synthetic KFD memory handle for guest startup bookkeeping.
  int alloc_memory_ioctl(void *arg) override;

  /// @brief Release a synthetic KFD memory handle or forward a real handle.
  int free_memory_ioctl(void *arg) override;

  /// @brief Rewrite guest gpu_id entries to the selected host before mapping.
  int map_memory_ioctl(void *arg) override;

  /// @brief Mirror map_memory rewrites for unmap requests.
  int unmap_memory_ioctl(void *arg) override;

  /// @brief Shared guest-to-host device-id rewrite for map/unmap memory ioctls.
  template <typename Args> int map_or_unmap_memory_ioctl(Args *args, unsigned long request);

  /// @brief Fail unsupported guest execution ioctls visibly.
  int reject_guest_execution_ioctl(unsigned long request, void *arg) const;

  /// @brief Return true when an ioctl argument names the synthetic guest GPU.
  bool request_targets_guest(unsigned long request, void *arg) const;

  /// @brief Build the synthetic aperture record appended after real apertures.
  kfd_process_device_apertures guest_apertures() const;

  config::DbtGuestConfig config_;
  Sysfs::GpuInfo guest_{};
  std::unique_ptr<TopologyOverlay> overlay_;
  mutable std::mutex mutex_;
  std::atomic<int> real_kfd_fd_{-1};
  uint32_t open_refs_ = 0;
  uint32_t host_gpu_id_ = 0;
  static constexpr uint64_t kSyntheticHandleBase = 1ULL << 63;
  uint64_t next_synthetic_handle_ = kSyntheticHandleBase;
  std::unordered_set<uint64_t> synthetic_handles_;
  std::unordered_set<uint64_t> synthetic_mmap_offsets_;
  std::atomic<bool> ready_{false};
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_GUEST_KFD_H_
