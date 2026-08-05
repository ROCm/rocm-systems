// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_passthrough.h
/// @brief Resolved libc entry points used to bypass rocjitsu interposer wrappers.

#ifndef ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_
#define ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_

#include <cstddef>
#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace rocjitsu {

/// @brief Real libc function pointers resolved with dlsym(RTLD_NEXT).
///
/// @details The LD_PRELOAD interposer shadows these symbols. Code that needs
/// to intentionally pass through to the host kernel or filesystem should call
/// these pointers instead of plain libc symbols, which would recurse back into
/// the wrappers.
class LibcPassthrough {
public:
  int (*openat)(int, const char *, int, ...) = nullptr;
  int (*close)(int) = nullptr;
  ssize_t (*read)(int, void *, size_t) = nullptr;
  ssize_t (*write)(int, const void *, size_t) = nullptr;
  int (*ioctl)(int, unsigned long, ...) = nullptr;
  void *(*mmap)(void *, size_t, int, int, int, off_t) = nullptr;
  int (*munmap)(void *, size_t) = nullptr;
  int (*mprotect)(void *, size_t, int) = nullptr;
  int (*madvise)(void *, size_t, int) = nullptr;
  int (*memfd_create)(const char *, unsigned int) = nullptr;
  int (*dup)(int) = nullptr;
  int (*dup2)(int, int) = nullptr;
  int (*dup3)(int, int, int) = nullptr;
  int (*fcntl)(int, int, ...) = nullptr;
  FILE *(*fopen)(const char *, const char *) = nullptr;
  FILE *(*freopen)(const char *, const char *, FILE *) = nullptr;
  DIR *(*opendir)(const char *) = nullptr;
  struct dirent *(*readdir)(DIR *) = nullptr;
  int (*closedir)(DIR *) = nullptr;
  int (*stat)(const char *, struct stat *) = nullptr;
  int (*lstat)(const char *, struct stat *) = nullptr;
  int (*access)(const char *, int) = nullptr;
  int (*fstat_fn)(int, struct stat *) = nullptr;
  ssize_t (*readlink_fn)(const char *, char *, size_t) = nullptr;
  pid_t (*fork)() = nullptr;

  /// @brief Return true after all required symbols have been resolved.
  [[nodiscard]] bool ready() const { return initialized_; }

  /// @brief Resolve the real libc functions from the next dynamic object.
  void resolve();

private:
  bool initialized_ = false;
};

/// @brief Return the process-wide libc pass-through table.
LibcPassthrough &libc_passthrough();

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_
