// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer.cpp
/// @brief LD_PRELOAD interposer that redirects KFD syscalls to the simulated driver.
///
/// @details Intercepts open, close, ioctl, mmap, munmap, and fopen to route
/// /dev/kfd operations and sysfs topology reads through SimulatedDriver.
/// All global state is managed by SimulatedDriver's static singleton interface.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <mutex>
#include <signal.h>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

using rocjitsu::SimulatedDriver;

namespace {

void rj_sigsegv_handler(int sig, siginfo_t *info, void *ctx) {
  auto *uc = static_cast<ucontext_t *>(ctx);
  void *fault_addr = info->si_addr;
  uint64_t pc = 0;
#ifdef __x86_64__
  pc = uc->uc_mcontext.gregs[REG_RIP];
#endif
  fprintf(stderr, "\n[rj CRASH] SIGSEGV at addr=%p pc=%#lx sig=%d code=%d\n", fault_addr, pc, sig,
          info->si_code);
  // Re-raise to get default behavior (core dump)
  signal(SIGSEGV, SIG_DFL);
  raise(SIGSEGV);
}

__attribute__((constructor)) void rj_install_signal_handler() {
  struct sigaction sa{};
  sa.sa_sigaction = rj_sigsegv_handler;
  sa.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, nullptr);
}

// Tracks fds opened by sysfs redirects so that relative openat() calls from the
// same dirfd can also be redirected. Maps fd → the redirected absolute path that
// was opened for it. Protected by g_sysfs_fd_mutex.
std::mutex g_sysfs_fd_mutex;
std::unordered_map<int, std::string> g_sysfs_fds;

// Tracks memfd stubs returned for /dev/dri/renderD* opens. ioctl() on these
// fds falls through to the kernel and returns ENOTTY unless we intercept them.
std::mutex g_drm_fd_mutex;
std::unordered_set<int> g_drm_fds;

std::mutex g_kfd_dup_mutex;
std::unordered_set<int> g_kfd_dup_fds;

bool is_kfd_primary_fd(int fd) { return fd == SimulatedDriver::kfd_fd(); }

bool is_kfd_duplicate_fd(int fd) {
  std::lock_guard<std::mutex> lock(g_kfd_dup_mutex);
  return g_kfd_dup_fds.count(fd) != 0;
}

bool is_kfd_tracked_fd(int fd) { return is_kfd_primary_fd(fd) || is_kfd_duplicate_fd(fd); }

void track_kfd_duplicate_fd(int fd) {
  if (fd < 0 || is_kfd_primary_fd(fd))
    return;
  std::lock_guard<std::mutex> lock(g_kfd_dup_mutex);
  g_kfd_dup_fds.insert(fd);
}

void untrack_kfd_duplicate_fd(int fd) {
  if (fd < 0)
    return;
  std::lock_guard<std::mutex> lock(g_kfd_dup_mutex);
  g_kfd_dup_fds.erase(fd);
}

void clear_kfd_duplicate_fds() {
  std::lock_guard<std::mutex> lock(g_kfd_dup_mutex);
  g_kfd_dup_fds.clear();
}

} // namespace

// Convert a standard fopen mode string to open(2) flags.
static int fopen_flags_from_mode(const char *mode) {
  bool plus = std::strchr(mode, '+') != nullptr;
  switch (mode[0]) {
  case 'w':
    return (plus ? O_RDWR : O_WRONLY) | O_CREAT | O_TRUNC;
  case 'a':
    return (plus ? O_RDWR : O_WRONLY) | O_CREAT | O_APPEND;
  default:
    return plus ? O_RDWR : O_RDONLY; // 'r' and fallback
  }
}

extern "C" {

int open(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }

  // Re-entry guard: pass through during driver construction.
  if (SimulatedDriver::in_construction())
    return static_cast<int>(syscall(SYS_openat, AT_FDCWD, path, flags, mode));

  if (std::string_view(path).starts_with("/dev/dri/renderD")) {
    if (!SimulatedDriver::lookup(SimulatedDriver::kfd_fd()))
      SimulatedDriver::get_or_create();
  }
  if (std::string_view(path).starts_with("/dev/dri/renderD") && SimulatedDriver::kfd_fd() >= 0) {
    int memfd = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_drm", 0));
    if (memfd >= 0) {
      std::lock_guard<std::mutex> lock(g_drm_fd_mutex);
      g_drm_fds.insert(memfd);
      return memfd;
    }
    return SimulatedDriver::kfd_fd();
  }

  // Intercept /dev/kfd — lazily create the simulated driver.
  if (std::strcmp(path, "/dev/kfd") == 0) {
    auto *drv = SimulatedDriver::get_or_create();
    if (!drv) {
      errno = ENODEV;
      return -1;
    }
    // If the driver was previously closed (e.g., Init() failed and scope
    // guard called Close()), re-open it to get a fresh fd.
    if (SimulatedDriver::kfd_fd() < 0) {
      drv->open();
      clear_kfd_duplicate_fds();
    }
    return SimulatedDriver::kfd_fd();
  }

  // Redirect sysfs topology and DRM reads to the generated directories.
  // Lazily create the driver if needed. amdsmi scans /sys/class/drm/
  // before opening /dev/kfd.
  if (std::string_view(path).starts_with("/sys/class/drm") ||
      std::string_view(path).starts_with("/sys/devices/virtual/kfd") ||
      std::string_view(path).starts_with("/sys/class/kfd")) {
    if (!SimulatedDriver::lookup(SimulatedDriver::kfd_fd()))
      SimulatedDriver::get_or_create();
  }
  std::string redirected = SimulatedDriver::redirect_sysfs_path(path);
  if (!redirected.empty()) {
    int fd = static_cast<int>(syscall(SYS_openat, AT_FDCWD, redirected.c_str(), flags, mode));
    if (fd >= 0) {
      std::lock_guard<std::mutex> lock(g_sysfs_fd_mutex);
      g_sysfs_fds[fd] = redirected;
    }
    return fd;
  }

  return static_cast<int>(syscall(SYS_openat, AT_FDCWD, path, flags, mode));
}

int open64(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  return open(path, flags, mode);
}

// glibc fortified open: _FORTIFY_SOURCE=2 rewrites open(path, flags) to
// __open_2(path, flags) when O_CREAT is not set. Passthrough only, DRM
// render node interception is disabled because the container's amdsmi
// binary has an inverted branch that calls drmFreeDevice on failure,
// causing heap corruption with uninitialized pointers.
int __open_2(const char *path, int oflag) {
  return static_cast<int>(syscall(SYS_openat, AT_FDCWD, path, oflag, 0));
}

int openat(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }

  if (path[0] == '/') {
    // Lazily create the driver for sysfs/DRM paths.
    if (std::string_view(path).starts_with("/sys/class/drm") ||
        std::string_view(path).starts_with("/sys/devices/virtual/kfd") ||
        std::string_view(path).starts_with("/sys/class/kfd")) {
      if (!SimulatedDriver::lookup(SimulatedDriver::kfd_fd()))
        SimulatedDriver::get_or_create();
    }
    std::string redirected = SimulatedDriver::redirect_sysfs_path(path);
    if (!redirected.empty()) {
      int fd = static_cast<int>(syscall(SYS_openat, AT_FDCWD, redirected.c_str(), flags, mode));
      if (fd >= 0) {
        std::lock_guard<std::mutex> lock(g_sysfs_fd_mutex);
        g_sysfs_fds[fd] = redirected;
      }
      return fd;
    }
  } else if (dirfd != AT_FDCWD) {
    // Relative path: check if dirfd was opened via a sysfs redirect.
    // If so, form the full redirected path and try to redirect the open.
    std::string dir_path;
    {
      std::lock_guard<std::mutex> lock(g_sysfs_fd_mutex);
      auto it = g_sysfs_fds.find(dirfd);
      if (it != g_sysfs_fds.end())
        dir_path = it->second;
    }
    if (!dir_path.empty()) {
      std::string full = dir_path + "/" + path;
      int fd = static_cast<int>(syscall(SYS_openat, AT_FDCWD, full.c_str(), flags, mode));
      if (fd >= 0) {
        std::lock_guard<std::mutex> lock(g_sysfs_fd_mutex);
        g_sysfs_fds[fd] = full;
      }
      return fd;
    }
  }

  return static_cast<int>(syscall(SYS_openat, dirfd, path, flags, mode));
}

int openat64(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  return openat(dirfd, path, flags, mode);
}

int close(int fd) {
  {
    std::lock_guard<std::mutex> lock(g_sysfs_fd_mutex);
    g_sysfs_fds.erase(fd);
  }
  {
    std::lock_guard<std::mutex> lock(g_drm_fd_mutex);
    if (g_drm_fds.erase(fd)) {
      syscall(SYS_close, fd);
      return 0;
    }
  }
  if (is_kfd_duplicate_fd(fd)) {
    untrack_kfd_duplicate_fd(fd);
    return static_cast<int>(syscall(SYS_close, fd));
  }
  auto *drv = SimulatedDriver::lookup(fd);
  if (drv) {
    int rc = drv->close();
    clear_kfd_duplicate_fds();
    return rc;
  }
  return static_cast<int>(syscall(SYS_close, fd));
}

int ioctl(int fd, unsigned long request, ...) {
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);

  // DRM ioctl request codes. Defined locally to avoid pulling in drm headers.
  // Computed from _IOWR('d', nr, size) / _IOW('d', nr, size).
  constexpr unsigned long kDrmIoctlVersion = 0xc0406400;    // _IOWR('d', 0x00, drm_version)
  constexpr unsigned long kDrmIoctlAmdgpuInfo = 0x40186445; // _IOW('d', 0x45, drm_amdgpu_info)

  // Handle ioctls on DRM render node stubs. The kernel would return ENOTTY
  // on a memfd, which ROCR/amdsmi treat as a fatal error.
  {
    std::lock_guard<std::mutex> lock(g_drm_fd_mutex);
    if (g_drm_fds.count(fd)) {
      if (request == kDrmIoctlVersion && arg) {
        struct drm_version {
          int version_major, version_minor, version_patchlevel;
          size_t name_len;
          char *name;
          size_t date_len;
          char *date;
          size_t desc_len;
          char *desc;
        };
        auto *ver = static_cast<drm_version *>(arg);
        ver->version_major = 3;
        ver->version_minor = 57;
        ver->version_patchlevel = 0;
        static constexpr const char drv_name[] = "amdgpu";
        if (ver->name && ver->name_len >= sizeof(drv_name) - 1)
          std::memcpy(ver->name, drv_name, sizeof(drv_name));
        ver->name_len = sizeof(drv_name) - 1;
        if (ver->date && ver->date_len > 0)
          ver->date[0] = '\0';
        ver->date_len = 1;
        if (ver->desc && ver->desc_len > 0)
          ver->desc[0] = '\0';
        ver->desc_len = 1;
        return 0;
      }
      if (request == kDrmIoctlAmdgpuInfo && arg) {
        struct drm_amdgpu_info {
          uint64_t return_pointer;
          uint32_t return_size;
          uint32_t query;
          uint64_t pad;
        };
        auto *info = static_cast<drm_amdgpu_info *>(arg);
        if (info->return_pointer && info->return_size > 0)
          std::memset(reinterpret_cast<void *>(info->return_pointer), 0, info->return_size);
        return 0;
      }
      // Unhandled DRM ioctl — return error instead of faking success.
      // Returning 0 for unknown ioctls causes drmGetDevice to operate on
      // uninitialized data, leading to invalid free() on garbage pointers.
      errno = EINVAL;
      return -1;
    }
  }

  auto *drv = SimulatedDriver::lookup(fd);
  if (!drv && is_kfd_duplicate_fd(fd))
    drv = SimulatedDriver::lookup(SimulatedDriver::kfd_fd());
  if (drv)
    return drv->ioctl(request, arg);

  return static_cast<int>(syscall(SYS_ioctl, fd, request, arg));
}

int dup(int oldfd) {
  int rc = static_cast<int>(syscall(SYS_dup, oldfd));
  if (rc >= 0) {
    if (is_kfd_tracked_fd(oldfd))
      track_kfd_duplicate_fd(rc);
    else
      untrack_kfd_duplicate_fd(rc);
  }
  return rc;
}

int dup2(int oldfd, int newfd) {
  int rc = static_cast<int>(syscall(SYS_dup2, oldfd, newfd));
  if (rc >= 0) {
    if (is_kfd_tracked_fd(oldfd))
      track_kfd_duplicate_fd(rc);
    else
      untrack_kfd_duplicate_fd(rc);
  }
  return rc;
}

#ifdef SYS_dup3
int dup3(int oldfd, int newfd, int flags) {
  int rc = static_cast<int>(syscall(SYS_dup3, oldfd, newfd, flags));
  if (rc >= 0) {
    if (is_kfd_tracked_fd(oldfd))
      track_kfd_duplicate_fd(rc);
    else
      untrack_kfd_duplicate_fd(rc);
  }
  return rc;
}
#endif

namespace {
enum class FcntlArgKind { None, Int, Ptr };

FcntlArgKind fcntl_arg_kind(int cmd) {
  switch (cmd) {
  case F_DUPFD:
  case F_DUPFD_CLOEXEC:
  case F_SETFD:
  case F_SETFL:
  case F_SETOWN:
  case F_SETSIG:
  case F_SETLEASE:
  case F_NOTIFY:
  case F_SETPIPE_SZ:
  case F_ADD_SEALS:
    return FcntlArgKind::Int;
#ifdef F_SETLK
  case F_SETLK:
  case F_SETLKW:
#endif
#if defined(F_SETLK64) && (!defined(F_SETLK) || F_SETLK64 != F_SETLK)
  case F_SETLK64:
  case F_SETLKW64:
#endif
  case F_GETLK:
#if defined(F_GETLK64) && (!defined(F_GETLK) || F_GETLK64 != F_GETLK)
  case F_GETLK64:
#endif
#ifdef F_GETOWNER_UIDS
  case F_GETOWNER_UIDS:
#endif
#ifdef F_GET_RW_HINT
  case F_GET_RW_HINT:
#endif
#ifdef F_SET_RW_HINT
  case F_SET_RW_HINT:
#endif
#ifdef F_GET_FILE_RW_HINT
  case F_GET_FILE_RW_HINT:
#endif
#ifdef F_SET_FILE_RW_HINT
  case F_SET_FILE_RW_HINT:
#endif
    return FcntlArgKind::Ptr;
#ifdef F_SETOWN_EX
  case F_SETOWN_EX:
    return FcntlArgKind::Ptr;
#endif
#ifdef F_GETOWN_EX
  case F_GETOWN_EX:
    return FcntlArgKind::Ptr;
#endif
  default:
    return FcntlArgKind::None;
  }
}
} // namespace

int fcntl(int fd, int cmd, ...) {
  va_list ap;
  va_start(ap, cmd);
  FcntlArgKind kind = fcntl_arg_kind(cmd);
  long rc = 0;
  switch (kind) {
  case FcntlArgKind::Int: {
    int arg = va_arg(ap, int);
    rc = syscall(SYS_fcntl, fd, cmd, arg);
    break;
  }
  case FcntlArgKind::Ptr: {
    void *arg = va_arg(ap, void *);
    rc = syscall(SYS_fcntl, fd, cmd, arg);
    break;
  }
  case FcntlArgKind::None:
  default:
    rc = syscall(SYS_fcntl, fd, cmd, 0L);
    break;
  }
  va_end(ap);

  if (rc >= 0 && (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC)) {
    if (is_kfd_tracked_fd(fd))
      track_kfd_duplicate_fd(static_cast<int>(rc));
    else
      untrack_kfd_duplicate_fd(static_cast<int>(rc));
  }
  return static_cast<int>(rc);
}

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  auto *drv = SimulatedDriver::lookup(fd);
  if (drv)
    return drv->mmap(addr, length, prot, flags, offset);

  return reinterpret_cast<void *>(syscall(SYS_mmap, addr, length, prot, flags, fd, offset));
}

int mprotect(void *addr, size_t length, int prot) {
  auto *drv = SimulatedDriver::lookup(SimulatedDriver::kfd_fd());
  if (drv && drv->is_doorbell_range(addr, length)) {
    errno = EPERM;
    return -1;
  }
  return static_cast<int>(syscall(SYS_mprotect, addr, length, prot));
}

int munmap(void *addr, size_t length) {
  auto *drv = SimulatedDriver::lookup(SimulatedDriver::kfd_fd());
  if (drv) {
    int ret = drv->munmap(addr, length);
    if (ret != -ENOENT)
      return ret;
  }
  return static_cast<int>(syscall(SYS_munmap, addr, length));
}

// -- libdrm interposition --

int amdgpu_device_initialize(int /*fd*/, uint32_t *major_version, uint32_t *minor_version,
                             void **device_handle) {
  if (SimulatedDriver::kfd_fd() < 0)
    return -1;
  *major_version = 3;
  *minor_version = 57;
  static int dummy_handle = 1;
  *device_handle = &dummy_handle;
  return 0;
}

int amdgpu_device_initialize2(int fd, bool /*deduplicate_device*/, uint32_t *major_version,
                              uint32_t *minor_version, void **device_handle) {
  return amdgpu_device_initialize(fd, major_version, minor_version, device_handle);
}

int amdgpu_device_deinitialize(void * /*device_handle*/) { return 0; }

int amdgpu_device_get_fd(void * /*device_handle*/) { return SimulatedDriver::kfd_fd(); }

// -- fopen / freopen interposition (sysfs redirect) --

FILE *fopen(const char *path, const char *mode) {
  if (!path || !mode)
    return nullptr;

  const char *actual = path;
  std::string redirected;
  if (!SimulatedDriver::in_construction()) {
    // Lazily create the driver for sysfs/DRM paths.
    if (std::string_view(path).starts_with("/sys/class/drm") ||
        std::string_view(path).starts_with("/sys/devices/virtual/kfd") ||
        std::string_view(path).starts_with("/sys/class/kfd")) {
      if (!SimulatedDriver::lookup(SimulatedDriver::kfd_fd()))
        SimulatedDriver::get_or_create();
    }
    redirected = SimulatedDriver::redirect_sysfs_path(path);
    if (!redirected.empty())
      actual = redirected.c_str();
  }

  int fd =
      static_cast<int>(syscall(SYS_openat, AT_FDCWD, actual, fopen_flags_from_mode(mode), 0644));
  if (fd < 0)
    return nullptr;
  return fdopen(fd, mode);
}

FILE *fopen64(const char *path, const char *mode) { return fopen(path, mode); }

/// @brief Redirect freopen through the sysfs interposer.
/// @details Closes the existing stream and opens a new one at the redirected
/// path. The returned FILE* may differ from stream — callers must use the
/// return value (standard freopen contract).
FILE *freopen(const char *path, const char *mode, FILE *stream) {
  if (!path || !mode)
    return nullptr;
  // Close the old stream if provided. The C standard allows stream to be
  // null, but GCC's nonnull attribute on fclose triggers a warning.
  RJ_DIAGNOSTIC_PUSH
  RJ_DIAGNOSTIC_IGNORE_NONNULL_COMPARE
  if (stream)
    ::fclose(stream);
  RJ_DIAGNOSTIC_POP
  return fopen(path, mode);
}

FILE *freopen64(const char *path, const char *mode, FILE *stream) {
  return freopen(path, mode, stream);
}

// -- opendir interposition --
// glibc's opendir uses internal __openat64 which bypasses LD_PRELOAD.
// We must interpose opendir directly to redirect /sys/class/drm/ and
// /sys/devices/virtual/kfd/ directory listings.

DIR *opendir(const char *name) {
  using real_opendir_t = DIR *(*)(const char *);
  static real_opendir_t real_opendir =
      reinterpret_cast<real_opendir_t>(dlsym(RTLD_NEXT, "opendir"));

  if (!SimulatedDriver::in_construction()) {
    if (std::string_view(name).starts_with("/sys/class/drm") ||
        std::string_view(name).starts_with("/sys/devices/virtual/kfd") ||
        std::string_view(name).starts_with("/sys/class/kfd")) {
      if (!SimulatedDriver::lookup(SimulatedDriver::kfd_fd()))
        SimulatedDriver::get_or_create();
    }
    std::string redirected = SimulatedDriver::redirect_sysfs_path(name);
    if (!redirected.empty())
      return real_opendir(redirected.c_str());
  }
  return real_opendir(name);
}

} // extern "C"
