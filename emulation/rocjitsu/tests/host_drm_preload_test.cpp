// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file host_drm_preload_test.cpp
/// @brief Supply a host DRM descriptor identity without requiring a physical GPU.
/// @details Loaded after rocjitsu so its RTLD_NEXT stat calls see this identity.

#include <dlfcn.h>
#include <sys/stat.h>

namespace {
thread_local int host_fd = -1;
thread_local dev_t host_device = 0;

template <typename Stat> int finish_stat(int fd, Stat *info, int result) {
  if (result == 0 && fd == host_fd) {
    info->st_mode = (info->st_mode & ~S_IFMT) | S_IFCHR;
    info->st_rdev = host_device;
  }
  return result;
}
} // namespace

extern "C" __attribute__((visibility("default"))) void rj_test_host_drm(int fd, dev_t device) {
  host_fd = fd;
  host_device = device;
}

extern "C" __attribute__((visibility("default"))) int fstat(int fd, struct stat *info) {
  static auto next = reinterpret_cast<decltype(&fstat)>(dlsym(RTLD_NEXT, "fstat"));
  return finish_stat(fd, info, next(fd, info));
}

extern "C" __attribute__((visibility("default"))) int fstat64(int fd, struct stat64 *info) {
  static auto next = reinterpret_cast<decltype(&fstat64)>(dlsym(RTLD_NEXT, "fstat64"));
  return finish_stat(fd, info, next(fd, info));
}

extern "C" __attribute__((visibility("default"))) int __fxstat(int version, int fd,
                                                               struct stat *info) {
  static auto next = reinterpret_cast<decltype(&__fxstat)>(dlsym(RTLD_NEXT, "__fxstat"));
  return finish_stat(fd, info, next(version, fd, info));
}

extern "C" __attribute__((visibility("default"))) int __fxstat64(int version, int fd,
                                                                 struct stat64 *info) {
  static auto next = reinterpret_cast<decltype(&__fxstat64)>(dlsym(RTLD_NEXT, "__fxstat64"));
  return finish_stat(fd, info, next(version, fd, info));
}
