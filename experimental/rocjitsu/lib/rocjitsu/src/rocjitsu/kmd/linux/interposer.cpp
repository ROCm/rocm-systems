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
#include <fcntl.h>
#include <linux/memfd.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

using rocjitsu::SimulatedDriver;

namespace {

// Tracks fds opened by sysfs redirects so that relative openat() calls from the
// same dirfd can also be redirected. Maps fd → the redirected absolute path that
// was opened for it. Protected by g_sysfs_fd_mutex.
std::mutex g_sysfs_fd_mutex;
std::unordered_map<int, std::string> g_sysfs_fds;

// Tracks memfd stubs returned for /dev/dri/renderD* opens. ioctl() on these
// fds falls through to the kernel and returns ENOTTY unless we intercept them.
std::mutex g_drm_fd_mutex;
std::unordered_set<int> g_drm_fds;

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

  // Intercept DRM render node opens — return a memfd stub that FMM can hold.
  // Track these fds so ioctl() can return 0 rather than ENOTTY.
  if (std::strncmp(path, "/dev/dri/renderD", 16) == 0 && SimulatedDriver::kfd_fd() >= 0) {
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
    if (SimulatedDriver::kfd_fd() < 0)
      drv->open();
    return SimulatedDriver::kfd_fd();
  }

  // Redirect sysfs topology reads to the generated directory.
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

int openat(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }

  if (path[0] == '/') {
    // Absolute path: redirect directly if it matches the sysfs prefix.
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
  auto *drv = SimulatedDriver::lookup(fd);
  if (drv)
    return drv->close();
  return static_cast<int>(syscall(SYS_close, fd));
}

int ioctl(int fd, unsigned long request, ...) {
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);

  // Return success for any ioctl on DRM render node stubs — the kernel would
  // return ENOTTY on a memfd, which ROCR may treat as a fatal error.
  {
    std::lock_guard<std::mutex> lock(g_drm_fd_mutex);
    if (g_drm_fds.count(fd))
      return 0;
  }

  auto *drv = SimulatedDriver::lookup(fd);
  if (drv)
    return drv->ioctl(request, arg);

  return static_cast<int>(syscall(SYS_ioctl, fd, request, arg));
}

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  auto *drv = SimulatedDriver::lookup(fd);
  if (drv)
    return drv->mmap(addr, length, prot, flags, offset);

  return reinterpret_cast<void *>(syscall(SYS_mmap, addr, length, prot, flags, fd, offset));
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

} // extern "C"
