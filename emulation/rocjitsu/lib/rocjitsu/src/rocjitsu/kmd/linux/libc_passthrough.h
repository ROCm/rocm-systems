// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_passthrough.h
/// @brief Resolved libc entry points used to bypass rocjitsu interposer wrappers.

#ifndef ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_
#define ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_

#include "rocjitsu/kmd/linux/scoped_cancellation_disable.h"
#include "util/unique_handle.h"

#include <cstddef>
#include <cstdio>
#include <dirent.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Forward declaration keeps legacy aliases independent of feature-macro order.
struct stat64;

namespace rocjitsu {

/// @brief Named libc wrappers for interposer bypass and unwindable calls.
/// @details Interposed operations use RTLD_NEXT to preserve the next preload's
/// implementation without re-entering rocjitsu. Non-interposed operations may
/// also need non-noexcept pointers to preserve forced unwinding. Mutex acquisition
/// uses RTLD_DEFAULT to match normal release and preserve sanitizer interception.
/// Optional wrappers remain null when unavailable; callers implement the stated
/// fallback or return an unsupported-feature error. read/write/close retain libc
/// cancellation points: internal cleanup must protect the entire ownership or
/// publication transaction, while application forwarding stays cancellable.
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
  void *(*mremap)(void *, size_t, size_t, int, ...) = nullptr;
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
  char *(*realpath_fn)(const char *, char *) = nullptr;
  char *(*realpath_chk_fn)(const char *, char *, size_t) = nullptr;
  /// @brief The nine legacy stat aliases rocJITsu also exports.
  /// @details Resolved EAGERLY here rather than through a function-local static in
  /// each wrapper. A lazy static's C++ initialization guard can be inherited
  /// mid-initialization by a forked child whose initializing thread no longer
  /// exists, deadlocking the child before it can even reach the owner-PID gate. The
  /// interposer constructor is single-threaded, so resolving here has no such
  /// window. Keep exact function types when calling another instrumented preload.
  int (*fstat64_fn)(int, struct stat64 *) = nullptr;
  int (*fxstat_fn)(int, int, struct stat *) = nullptr;
  int (*fxstat64_fn)(int, int, struct stat64 *) = nullptr;
  int (*stat64_fn)(const char *, struct stat64 *) = nullptr;
  int (*lstat64_fn)(const char *, struct stat64 *) = nullptr;
  int (*xstat_fn)(int, const char *, struct stat *) = nullptr;
  int (*xstat64_fn)(int, const char *, struct stat64 *) = nullptr;
  int (*lxstat_fn)(int, const char *, struct stat *) = nullptr;
  int (*lxstat64_fn)(int, const char *, struct stat64 *) = nullptr;
  pid_t (*fork)() = nullptr;
  // Non-noexcept signatures preserve forced unwinding from signal handlers.
  int (*clock_gettime)(clockid_t, timespec *) = nullptr;
  int (*pthread_mutex_lock)(pthread_mutex_t *) = nullptr;
  int (*sem_init)(sem_t *, int, unsigned int) = nullptr;
  int (*sem_destroy)(sem_t *) = nullptr;
  int (*sem_post)(sem_t *) = nullptr;
  // Optional on older libc versions; callers must handle a missing wrapper.
  int (*sem_clockwait)(sem_t *, clockid_t, const timespec *) = nullptr;
  int (*pidfd_open)(pid_t, unsigned int) = nullptr;
  ssize_t (*getdents64)(int, void *, size_t) = nullptr;

  /// @brief Return true after all required symbols have been resolved.
  [[nodiscard]] bool ready() const { return initialized_; }

  /// @brief Resolve required and optional wrappers using their lookup contracts.
  /// @details Use RTLD_NEXT for interposer bypass. For non-interposed operations
  /// whose interception must match direct calls, use RTLD_DEFAULT. ready() requires
  /// the mandatory entries, not optional feature wrappers or legacy aliases.
  void resolve();

private:
  bool initialized_ = false;
};

/// @brief Find a libc mapping wrapper without allocating through dlsym.
/// @details Only for mmap/munmap/mremap before resolve() completes. Inspect the
/// already-loaded libc's ELF tables under dl_iterate_phdr's loader lock; do not
/// call an IFUNC resolver or another preload during allocator initialization.
void *bootstrap_mapping_symbol(const char *name);

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
    ScopedCancellationDisable cancellation;
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
