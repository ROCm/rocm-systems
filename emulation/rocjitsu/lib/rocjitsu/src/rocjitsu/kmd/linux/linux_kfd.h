// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file linux_kfd.h
/// @brief Linux-specific KFD driver interface used by the syscall interposer.

#ifndef ROCJITSU_KMD_LINUX_LINUX_KFD_H_
#define ROCJITSU_KMD_LINUX_LINUX_KFD_H_

#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/driver.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace rocjitsu {

/// @brief KFD mmap offset encoding fields, matching kfd_priv.h.
inline constexpr uint64_t KFD_MMAP_TYPE_SHIFT = 62;
inline constexpr uint64_t KFD_MMAP_TYPE_MASK = 0x3ULL << KFD_MMAP_TYPE_SHIFT;
inline constexpr uint64_t KFD_MMAP_TYPE_DOORBELL = 0x3ULL << KFD_MMAP_TYPE_SHIFT;
inline constexpr uint64_t KFD_MMAP_TYPE_EVENTS = 0x2ULL << KFD_MMAP_TYPE_SHIFT;
inline constexpr uint64_t KFD_MMAP_GPU_ID_SHIFT = 46;

/// @brief Encode a KFD GPU id into the mmap offset GPU-id field.
inline constexpr uint64_t kfd_mmap_gpu_id(uint32_t gpu_id) {
  return (static_cast<uint64_t>(gpu_id) << KFD_MMAP_GPU_ID_SHIFT) &
         ((1ULL << KFD_MMAP_TYPE_SHIFT) - (1ULL << KFD_MMAP_GPU_ID_SHIFT));
}

/// @brief Linux KFD driver surface required by the LD_PRELOAD shim.
///
/// @details The base Driver interface handles /dev/kfd calls. The Linux shim
/// also needs sysfs redirection, DRM render-node ownership, and fd tracking.
/// SimulatedKfd and GuestKfd both implement this surface, but with
/// different ownership rules: simulation owns all visible GPU discovery, while
/// GuestKfd owns only the appended guest GPU.
class LinuxKfd : public Driver {
public:
  ~LinuxKfd() override = default;

  /// @brief Return the /dev/kfd fd represented by this driver.
  [[nodiscard]] virtual int fd() const = 0;

  /// @brief Reset inherited child-process state after fork().
  virtual void reset_after_fork() {}

  /// @brief Retain one duplicate open reference, if this driver tracks them.
  virtual void retain_local_open() {}

  /// @brief Return true when @p fd is owned internally by the driver.
  [[nodiscard]] virtual bool owns_fd(int fd) const = 0;

  /// @brief Redirect a sysfs path to a generated topology/DRM path.
  ///
  /// @details Returns an empty string when the driver does not own the path.
  [[nodiscard]] virtual std::string redirect_sysfs_path(const char *path) const = 0;

  /// @brief Return true if a memory range is a KFD doorbell mapping.
  [[nodiscard]] virtual bool is_doorbell_range(const void *addr, size_t length) const = 0;

  /// @brief Return true if this driver should synthesize /dev/dri/renderD@p minor.
  [[nodiscard]] virtual bool handles_drm_render_minor(uint32_t minor) const = 0;

  /// @brief Return synthetic GPU properties for a handled DRM render minor.
  [[nodiscard]] virtual const Sysfs::GpuInfo *gpu_info_for_render_minor(uint32_t minor) const = 0;

  /// @brief Return the generated KFD topology root, if any.
  [[nodiscard]] virtual std::string topology_path() const = 0;

  /// @brief Return a generated /sys/class/drm root for full DRM redirection.
  ///
  /// @details GuestKfd returns an empty string because host DRM entries
  /// must remain real; it redirects only the synthetic guest render node.
  [[nodiscard]] virtual std::string drm_path() const = 0;

  /// @brief Redirect standard KFD topology and DRM sysfs roots.
  ///
  /// @details Empty root paths are ignored. Returns empty when @p path does not
  /// name a handled sysfs root.
  static std::string redirect_sysfs_root_path(const char *path, const std::string &topology_path,
                                              const std::string &drm_path);

protected:
  /// @brief Parse /sys/class/drm/renderD<minor> paths.
  static bool parse_drm_render_path(std::string_view path, uint32_t *minor,
                                    std::string_view *suffix);

  /// @brief Parse /sys/dev/char/226:<minor> paths for DRM render nodes.
  static bool parse_sys_dev_drm_render_path(std::string_view path, uint32_t *minor,
                                            std::string_view *suffix);

  /// @brief Return a stable diagnostic name for a KFD ioctl request.
  static const char *ioctl_name(unsigned long request);

  /// @brief Handle AMDKFD_IOC_GET_VERSION.
  virtual int get_version_ioctl(void *arg);

  /// @brief Fill AMDKFD_IOC_GET_VERSION with the KFD ioctl ABI version.
  static int fill_get_version_ioctl(void *arg);

  /// @brief Handle AMDKFD_IOC_GET_CLOCK_COUNTERS.
  virtual int get_clock_counters_ioctl(void *arg);

  /// @brief Fill AMDKFD_IOC_GET_CLOCK_COUNTERS with synthetic monotonic counters.
  static int fill_get_clock_counters_ioctl(void *arg);

  /// @brief Handle AMDKFD_IOC_GET_PROCESS_APERTURES_NEW.
  virtual int get_process_apertures_ioctl(void *arg);

  /// @brief Handle AMDKFD_IOC_ACQUIRE_VM.
  virtual int acquire_vm_ioctl(void *arg);

  /// @brief Handle AMDKFD_IOC_AVAILABLE_MEMORY.
  virtual int get_available_memory_ioctl(void *arg);

  /// @brief Handle AMDKFD_IOC_SET_MEMORY_POLICY.
  virtual int set_memory_policy_ioctl(void *arg);

  /// @brief Handle AMDKFD_IOC_ALLOC_MEMORY_OF_GPU.
  virtual int alloc_memory_ioctl(void *arg);

  /// @brief Handle AMDKFD_IOC_FREE_MEMORY_OF_GPU.
  virtual int free_memory_ioctl(void *arg);

  /// @brief Handle AMDKFD_IOC_MAP_MEMORY_TO_GPU.
  virtual int map_memory_ioctl(void *arg);

  /// @brief Handle AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU.
  virtual int unmap_memory_ioctl(void *arg);
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_LINUX_KFD_H_
