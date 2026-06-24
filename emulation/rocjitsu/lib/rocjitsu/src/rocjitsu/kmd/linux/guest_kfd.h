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
#include <mutex>
#include <string>
#include <unordered_set>

namespace rocjitsu {

/// @brief Generated topology overlay containing host KFD nodes plus one guest node.
///
/// @details The overlay copies the real host topology from sysfs, then appends a
/// guest GPU node generated from the configured KFD identity. Only KFD topology
/// paths and the guest DRM render node are redirected to this overlay; host DRM
/// paths remain real so ROCR can still create a host agent and execute on it.
class TopologyOverlay {
public:
  /// @brief Construct an empty overlay.
  TopologyOverlay() = default;

  /// @brief Remove any generated overlay directories owned by this instance.
  ~TopologyOverlay();

  /// @brief Overlays own temporary filesystem state and cannot be copied.
  TopologyOverlay(const TopologyOverlay &) = delete;

  /// @brief Overlays own temporary filesystem state and cannot be copied.
  TopologyOverlay &operator=(const TopologyOverlay &) = delete;

  /// @brief Build the overlay from the current host sysfs topology.
  /// @param guest Synthetic guest GPU properties to append.
  /// @returns true when topology and guest DRM paths are ready.
  bool generate(const Sysfs::GpuInfo &guest);

  /// @brief Remove generated overlay directories.
  void cleanup();

  /// @brief Drop inherited overlay ownership without removing parent-owned paths.
  void release_after_fork();

  /// @brief Generated KFD topology root.
  [[nodiscard]] const std::string &topology_path() const { return topology_dir_; }

  /// @brief Generated DRM root containing the guest render node.
  [[nodiscard]] const std::string &guest_drm_path() const { return guest_drm_dir_; }

  /// @brief Synthetic topology node ID assigned to the guest GPU.
  [[nodiscard]] uint32_t guest_node_id() const { return guest_node_id_; }

private:
  /// @brief Copy a host sysfs subtree into the generated overlay.
  bool copy_tree(const std::string &src, const std::string &dst);

  /// @brief Generate the appended guest KFD node and guest DRM metadata.
  bool copy_guest_node(const Sysfs::GpuInfo &guest);

  /// @brief Patch aggregate topology files after appending the guest node.
  bool patch_topology_files();

  std::string topology_dir_;
  std::string guest_drm_dir_;
  Sysfs guest_sysfs_;
  uint32_t guest_node_id_ = 0;
};

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

private:
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
  TopologyOverlay overlay_;
  mutable std::mutex mutex_;
  std::atomic<int> real_kfd_fd_{-1};
  uint32_t host_gpu_id_ = 0;
  static constexpr uint64_t kSyntheticHandleBase = 1ULL << 63;
  uint64_t next_synthetic_handle_ = kSyntheticHandleBase;
  std::unordered_set<uint64_t> synthetic_handles_;
  std::atomic<bool> ready_{false};
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_GUEST_KFD_H_
