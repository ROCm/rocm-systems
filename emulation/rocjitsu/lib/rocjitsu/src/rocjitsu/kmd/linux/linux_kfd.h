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

/// @brief The only WAIT_EVENTS timeout the KFD ABI treats as "wait forever".
///
/// @details Exactly this value, not a range. The next value down, 0xFFFFFFFE, is
/// finite -- and it is what ROCr clamps its longest wait to, so it is the large
/// timeout actually seen in practice. Classifying that as infinite would make it the
/// one timeout that never expires, which is not what the driver does with it. Treating
/// it as finite costs nothing: a deadline that far out still polls and still observes
/// shutdown cancellation, it merely also expires eventually. (Current kernels cap very
/// large finite waits well below the value asked for, so the real duration is shorter
/// again -- immaterial here, but it is not the literal millisecond count.)
inline constexpr uint32_t kWaitEventsInfiniteMs = 0xFFFFFFFFu;

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
/// SimulatedKfd and GuestKfd both implement this surface, but with different
/// ownership rules: simulation owns all visible GPU discovery, while GuestKfd
/// owns the appended guest and delegates host operations to either hardware or
/// a SimulatedKfd execution backend.
class LinuxKfd : public Driver {
public:
  ~LinuxKfd() override = default;

  /// @brief Return the /dev/kfd fd represented by this driver.
  [[nodiscard]] virtual int fd() const = 0;

  /// @brief Retain one duplicate open reference, if this driver tracks them.
  /// @retval true A reference was added.
  /// @retval false This driver does not track open references, or there is no
  ///         live local process to retain (e.g. it was already torn down); the
  ///         caller must NOT treat the fd as retained.
  [[nodiscard]] virtual bool retain_local_open() { return false; }

  /// @brief Number of open references this driver currently holds.
  /// @details The interposer decides when the local backend may be torn down by
  /// comparing this against the count it observed at publication (see
  /// InterposerContext::local_open_ref_baseline_), so a driver reports its RAW
  /// count here and does not need to know which references were app-facing.
  /// Default 0 for drivers that track no references — they are always idle.
  [[nodiscard]] virtual uint32_t local_open_ref_count() const { return 0; }

  /// @brief Release any thread blocked in a driver call so it returns and drops
  /// its driver snapshot, allowing teardown to complete.
  /// @details Called by the interposer once it has decided to tear the local
  /// backend down. A thread parked in an indefinite AMDKFD_IOC_WAIT_EVENTS holds a
  /// snapshot for the whole call, which would keep the driver alive indefinitely;
  /// only the driver can end that wait, since closing an fd does not cancel an
  /// in-flight one.
  ///
  /// Every blocking call the driver SERVES ITSELF has to be covered, not only the
  /// event wait: one that also parks a caller elsewhere -- on a debugger
  /// acknowledgement, say -- would hold teardown behind that call's own deadline
  /// instead. A driver that FORWARDS a blocking ioctl to a kernel it cannot interrupt
  /// is a known residual rather than a bug here: nothing in this process can end that
  /// wait, so teardown waits it out.
  ///
  /// Released waiters report a benign result their caller already handles
  /// (KFD_IOC_WAIT_RESULT_TIMEOUT for an event wait); this mutates no event, page,
  /// or process state. The destructive part of teardown belongs to close(). By the
  /// time this is called teardown is committed, so the release is a one-way latch
  /// for the driver's remaining life rather than something a later retain undoes.
  /// Default no-op for drivers with no blocking calls. Idempotent.
  virtual void begin_local_shutdown() {}

  /// @brief Outcome of invalidate_primary_fd(), telling the interposer how to
  /// follow up on a dup2/dup3 that overwrote a primary KFD fd number.
  enum class PrimaryInvalidation {
    /// @brief @p fd is not this driver's primary (or a concurrent overwrite
    /// already cleared it). The caller must NOT drop an open reference and should
    /// fall through to its dup-tracking path.
    kNotPrimary,
    /// @brief The primary was cleared AND it held one open reference the caller
    /// must now drop (via close()); used by drivers whose primary fd is counted
    /// in their open-reference bookkeeping (e.g. SimulatedKfd).
    kClearedDropRef,
    /// @brief The primary classification was cleared, but the caller must NOT
    /// drop an open reference: the driver's primary fd is internal and not part of
    /// the app-facing open-reference count (e.g. GuestKfd's hidden real /dev/kfd
    /// fd, which is kept alive by app dup references, not by the primary number).
    kClearedKeepRefs,
  };

  /// @brief Forget the primary KFD fd number without touching process state.
  /// @details Called by the interposer when dup2/dup3 atomically overwrites the
  /// primary KFD fd number: the number no longer refers to this driver, so it
  /// must stop reporting that number from fd()/owns_fd(). Default no-op for
  /// drivers that do not expose a primary fd number this way.
  /// @returns How the caller should follow up (see PrimaryInvalidation). Only
  ///          kClearedDropRef asks the caller to drop the primary's open reference
  ///          via close(), so a lost race or a driver whose primary carries no
  ///          counted reference never triggers a double/spurious release.
  [[nodiscard]] virtual PrimaryInvalidation invalidate_primary_fd(int fd) {
    (void)fd;
    return PrimaryInvalidation::kNotPrimary;
  }

  /// @brief Return true when @p fd is owned internally by the driver.
  [[nodiscard]] virtual bool owns_fd(int fd) const = 0;

  /// @brief Redirect a sysfs path to a generated topology/DRM path.
  ///
  /// @details Returns an empty string when the driver does not own the path.
  [[nodiscard]] virtual std::string redirect_sysfs_path(const char *path) const = 0;

  /// @brief Return true if a memory range is a KFD doorbell mapping.
  [[nodiscard]] virtual bool is_doorbell_range(const void *addr, size_t length) const = 0;

  /// @brief Perform an ordinary fixed mmap while retiring replaced client doorbell views.
  /// @details Simulator-backed drivers remove overlapping client views from KFD ownership before
  /// the mapping is replaced. Drivers without simulated execution pass the mmap through unchanged.
  virtual void *mmap_replacing_client_doorbell_views(void *addr, size_t length, int prot, int flags,
                                                     int fd, off_t offset) = 0;

  /// @brief Return true if this driver should synthesize /dev/dri/renderD@p minor.
  [[nodiscard]] virtual bool handles_drm_render_minor(uint32_t minor) const = 0;

  /// @brief Return synthetic GPU properties for a handled DRM render minor.
  [[nodiscard]] virtual const Sysfs::GpuInfo *gpu_info_for_render_minor(uint32_t minor) const = 0;

  /// @brief Return the generated KFD topology root, if any.
  [[nodiscard]] virtual std::string topology_path() const = 0;

  /// @brief Return a generated /sys/class/drm root for full DRM redirection.
  ///
  /// @details Hardware-backed GuestKfd returns an empty string because host DRM
  /// entries must remain real; simulator-backed GuestKfd delegates this to its
  /// SimulatedKfd backend.
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
