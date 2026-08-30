// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_passthrough.h
/// @brief Resolved libc entry points used to bypass rocjitsu interposer wrappers.

#ifndef ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_
#define ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_

#include "util/unique_handle.h"

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
  /// @brief The nine legacy stat aliases rocJITsu also exports.
  /// @details Resolved EAGERLY here rather than through a function-local static in
  /// each wrapper. A lazy static's C++ initialization guard can be inherited
  /// mid-initialization by a forked child whose initializing thread no longer
  /// exists, deadlocking the child before it can even reach the owner-PID gate. The
  /// interposer constructor is single-threaded, so resolving here has no such
  /// window. struct stat64 is spelled void* to keep this header free of
  /// _LARGEFILE64_SOURCE ordering constraints; the wrappers cast back.
  int (*fstat64_fn)(int, void *) = nullptr;
  int (*fxstat_fn)(int, int, struct stat *) = nullptr;
  int (*fxstat64_fn)(int, int, void *) = nullptr;
  int (*stat64_fn)(const char *, void *) = nullptr;
  int (*lstat64_fn)(const char *, void *) = nullptr;
  int (*xstat_fn)(int, const char *, struct stat *) = nullptr;
  int (*xstat64_fn)(int, const char *, void *) = nullptr;
  int (*lxstat_fn)(int, const char *, struct stat *) = nullptr;
  int (*lxstat64_fn)(int, const char *, void *) = nullptr;
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

/// @brief UniqueHandle traits that close through the libc pass-through table.
/// @details The interposer and the drivers link into ONE DSO, and the interposer
/// exports close() with default visibility, so a bare ::close() from driver code
/// binds to the interposer's own hook, which classifies the fd against the active
/// driver and can reach init_mutex_ via release_local_open(). Driver code reached
/// through an interposed ioctl/mmap would then re-enter the interposer while it is
/// mid-dispatch, on an fd the interposer knows nothing about. Every fd OWNED BY A
/// DRIVER therefore uses UniqueDriverFd, never util::UniqueHandle, keeping "the
/// driver never re-enters the interposer" total. Same rationale as
/// safe_fstat()/safe_fcntl().
struct PassthroughFdTraits {
  using handle_type = int;

  static handle_type invalid() noexcept { return -1; }
  static bool is_valid(handle_type fd) noexcept { return fd >= 0; }
  static void close(handle_type fd) noexcept {
    // The table is unresolved until resolve() runs. These traits are reachable from
    // any driver TU, so fall back to ::close rather than calling a null pointer;
    // that path can only be taken before the interposer is live, so it cannot
    // re-enter the hook this type exists to avoid.
    auto *real_close = libc_passthrough().close;
    static_cast<void>(real_close ? real_close(fd) : ::close(fd));
  }
};

using UniqueDriverFd = util::BasicUniqueHandle<PassthroughFdTraits>;

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_
