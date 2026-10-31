// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_passthrough.h
/// @brief Resolved libc entry points used to bypass rocjitsu interposer wrappers.

#pragma once

#include "util/unique_handle.h"

#include <cstddef>
#include <cstdio>
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/prctl.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// Forward declaration keeps legacy aliases independent of feature-macro order.
struct stat64;
struct statx;

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
  long (*syscall)(long, ...) = nullptr;
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
  off_t (*lseek_fn)(int, off_t, int) = nullptr;
  off64_t (*lseek64_fn)(int, off64_t, int) = nullptr;
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
  /// @brief Directory-relative metadata hooks, resolved eagerly for fork safety.
  int (*fstatat_fn)(int, const char *, struct stat *, int) = nullptr;
  int (*fstatat64_fn)(int, const char *, struct stat64 *, int) = nullptr;
  int (*fxstatat_fn)(int, int, const char *, struct stat *, int) = nullptr;
  int (*fxstatat64_fn)(int, int, const char *, struct stat64 *, int) = nullptr;
  int (*faccessat_fn)(int, const char *, int, int) = nullptr;
  ssize_t (*readlinkat_fn)(int, const char *, char *, size_t) = nullptr;
  int (*statx_fn)(int, const char *, int, unsigned int, struct statx *) = nullptr;
  pid_t (*fork)() = nullptr;
  decltype(&::getsockname) getsockname = nullptr;
  decltype(&::sendmsg) sendmsg = nullptr;
  decltype(&::recvmsg) recvmsg = nullptr;
  decltype(&::shmget) shmget = nullptr;
  decltype(&::shmat) shmat = nullptr;
  decltype(&::shmdt) shmdt = nullptr;
  decltype(&::shmctl) shmctl = nullptr;
  decltype(&::getrandom) getrandom = nullptr;
  decltype(&::prlimit) prlimit = nullptr;
  decltype(&::ftruncate) ftruncate = nullptr;
  decltype(&::fallocate) fallocate = nullptr;
  decltype(&::socketpair) socketpair = nullptr;
  decltype(&::bind) bind = nullptr;
  decltype(&::epoll_ctl) epoll_ctl = nullptr;
  decltype(&::prctl) prctl = nullptr;
  decltype(&::epoll_create1) epoll_create1 = nullptr;
  decltype(&::eventfd) eventfd = nullptr;
  decltype(&::epoll_wait) epoll_wait = nullptr;
  decltype(&::accept4) accept4 = nullptr;
  decltype(&::getsockopt) getsockopt = nullptr;
  decltype(&::socket) socket = nullptr;
  decltype(&::connect) connect = nullptr;
  decltype(&::listen) listen = nullptr;
  decltype(&::execve) execve = nullptr;
  decltype(&::setsid) setsid = nullptr;
  decltype(&::waitpid) waitpid = nullptr;
  decltype(&::poll) poll = nullptr;
  decltype(&::clone) clone = nullptr;
  decltype(&::_exit) exit_immediately = nullptr;
  int (*pidfd_open)(pid_t, unsigned int) = nullptr;
  int (*close_range)(unsigned int, unsigned int, int) = nullptr;
  ssize_t (*getdents64)(int, void *, size_t) = nullptr;

  /// @brief Return true after all required symbols have been resolved.
  [[nodiscard]] bool ready() const { return initialized_; }

  /// @brief Resolve the real libc functions from the next dynamic object.
  void resolve();

private:
  bool initialized_ = false;
};

/// @brief Return the process-wide libc pass-through table.
LibcPassthrough &libc_passthrough();

/// @brief Compatibility entry points for operations absent from older libc releases.
/// @details Prefer the resolved libc wrapper. If unavailable, use the syscall
/// entry point; callers retain typed arguments and errno. Standalone callers
/// also use this fallback before the pass-through table is resolved.
int host_pidfd_open(pid_t pid, unsigned int flags);
int host_close_range(unsigned int first, unsigned int last, int flags);
ssize_t host_getdents64(int fd, void *buffer, size_t size);

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
