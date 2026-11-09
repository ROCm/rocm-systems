// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file host_drm_preload_test.cpp
/// @brief Supply a host DRM descriptor identity without requiring a physical GPU.
/// @details Loaded after rocjitsu so its RTLD_NEXT stat calls see this identity.

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
thread_local int host_fd = -1;
thread_local dev_t host_device = 0;
thread_local int swap_original = -1;
thread_local int swap_replacement = -1;
thread_local bool swapped = false;

void *finish_mmap(void *mapped) {
  if (mapped != MAP_FAILED && swap_original >= 0) {
    swapped = syscall(SYS_dup3, swap_replacement, swap_original, O_CLOEXEC) == swap_original;
    swap_original = -1;
  }
  return mapped;
}

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

// Reuse the public fd after the kernel has mapped its file but before the
// interposer applies dump policy. This library sits beneath the interposer.
extern "C" __attribute__((visibility("default"))) void rj_test_swap_fd_after_mmap(int original,
                                                                                  int replacement) {
  swap_original = original;
  swap_replacement = replacement;
  swapped = false;
}
extern "C" __attribute__((visibility("default"))) bool rj_test_mmap_swapped_fd() { return swapped; }
extern "C" __attribute__((visibility("default"))) void *mmap(void *address, size_t size, int prot,
                                                             int flags, int fd, off_t offset) {
  static auto next = reinterpret_cast<decltype(&mmap)>(dlsym(RTLD_NEXT, "mmap"));
  return finish_mmap(next(address, size, prot, flags, fd, offset));
}
extern "C" __attribute__((visibility("default"))) void *mmap64(void *address, size_t size, int prot,
                                                               int flags, int fd, off64_t offset) {
  static auto next = reinterpret_cast<decltype(&mmap64)>(dlsym(RTLD_NEXT, "mmap64"));
  return finish_mmap(next(address, size, prot, flags, fd, offset));
}
