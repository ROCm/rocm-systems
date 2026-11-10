// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file topology_probe_preload_test.cpp
/// @brief Inject KFD topology probe errors into the CLI without a GPU.

#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>

namespace {
constexpr const char *kPrimary = "/sys/devices/virtual/kfd/kfd/topology/nodes";
constexpr const char *kFallback = "/sys/class/kfd/kfd/topology/nodes";

bool topology_path(const char *path) {
  return std::strcmp(path, kPrimary) == 0 || std::strcmp(path, kFallback) == 0;
}

template <typename Stat> bool probe_stat(const char *path, Stat *info, int &result) {
  const char *mode = std::getenv("RJ_TOPOLOGY_FAILURE");
  if (!mode || !topology_path(path))
    return false;
  if (std::strcmp(path, kPrimary) == 0) {
    errno = std::strcmp(mode, "probe") == 0 ? EACCES : ENOENT;
    result = -1;
  } else {
    *info = {};
    info->st_mode = S_IFDIR | 0555;
    result = 0;
  }
  return true;
}

bool deny_directory(const char *path) {
  if (!std::getenv("RJ_TOPOLOGY_FAILURE") || !topology_path(path))
    return false;
  errno = EACCES;
  return true;
}
} // namespace

extern "C" __attribute__((visibility("default"))) int stat(const char *path, struct stat *info) {
  int result;
  if (probe_stat(path, info, result))
    return result;
  static auto next = reinterpret_cast<decltype(&stat)>(dlsym(RTLD_NEXT, "stat"));
  return next(path, info);
}

extern "C" __attribute__((visibility("default"))) int stat64(const char *path,
                                                             struct stat64 *info) {
  int result;
  if (probe_stat(path, info, result))
    return result;
  static auto next = reinterpret_cast<decltype(&stat64)>(dlsym(RTLD_NEXT, "stat64"));
  return next(path, info);
}

extern "C" __attribute__((visibility("default"))) DIR *opendir(const char *path) {
  if (deny_directory(path))
    return nullptr;
  static auto next = reinterpret_cast<decltype(&opendir)>(dlsym(RTLD_NEXT, "opendir"));
  return next(path);
}

extern "C" __attribute__((visibility("default"))) int openat(int fd, const char *path, int flags,
                                                             ...) {
  if (deny_directory(path))
    return -1;
  mode_t mode = 0;
  if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, mode_t);
    va_end(args);
  }
  static auto next = reinterpret_cast<decltype(&openat)>(dlsym(RTLD_NEXT, "openat"));
  return next(fd, path, flags, mode);
}

extern "C" __attribute__((visibility("default"))) int __openat_2(int fd, const char *path,
                                                                 int flags) {
  return openat(fd, path, flags);
}
