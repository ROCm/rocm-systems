// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_passthrough.h
/// @brief Resolved libc entry points used to bypass rocjitsu interposer wrappers.

#ifndef ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_
#define ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_

#include <atomic>
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
  /// @brief Return true after all required symbols have been resolved.
  [[nodiscard]] bool ready() const { return initialized_.load(std::memory_order_acquire); }

  /// @brief Resolve the real libc functions from the next dynamic object.
  void resolve();

  int openat(int dirfd, const char *path, int flags, mode_t mode = 0);
  int close(int fd);
  ssize_t read(int fd, void *buf, size_t count);
  ssize_t write(int fd, const void *buf, size_t count);
  int ioctl(int fd, unsigned long request, void *arg);
  void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
  int munmap(void *addr, size_t length);
  int mprotect(void *addr, size_t length, int prot);
  int madvise(void *addr, size_t length, int advice);
  int memfd_create(const char *name, unsigned int flags);
  int dup(int oldfd);
  int dup2(int oldfd, int newfd);
  int dup3(int oldfd, int newfd, int flags);
  int fcntl(int fd, int cmd, int arg);
  int fcntl(int fd, int cmd, long arg);
  int fcntl(int fd, int cmd, void *arg);
  FILE *fopen(const char *path, const char *mode);
  FILE *freopen(const char *path, const char *mode, FILE *stream);
  DIR *opendir(const char *name);
  struct dirent *readdir(DIR *dirp);
  int closedir(DIR *dirp);
  int stat(const char *path, struct stat *buf);
  int lstat(const char *path, struct stat *buf);
  int access(const char *path, int mode);
  int fstat_fn(int fd, struct stat *buf);
  ssize_t readlink_fn(const char *path, char *buf, size_t bufsiz);
  pid_t fork();

private:
  using OpenatFn = int (*)(int, const char *, int, ...);
  using CloseFn = int (*)(int);
  using ReadFn = ssize_t (*)(int, void *, size_t);
  using WriteFn = ssize_t (*)(int, const void *, size_t);
  using IoctlFn = int (*)(int, unsigned long, ...);
  using MmapFn = void *(*)(void *, size_t, int, int, int, off_t);
  using MunmapFn = int (*)(void *, size_t);
  using MprotectFn = int (*)(void *, size_t, int);
  using MadviseFn = int (*)(void *, size_t, int);
  using MemfdCreateFn = int (*)(const char *, unsigned int);
  using DupFn = int (*)(int);
  using Dup2Fn = int (*)(int, int);
  using Dup3Fn = int (*)(int, int, int);
  using FcntlFn = int (*)(int, int, ...);
  using FopenFn = FILE *(*)(const char *, const char *);
  using FreopenFn = FILE *(*)(const char *, const char *, FILE *);
  using OpendirFn = DIR *(*)(const char *);
  using ReaddirFn = struct dirent *(*)(DIR *);
  using ClosedirFn = int (*)(DIR *);
  using StatFn = int (*)(const char *, struct stat *);
  using LstatFn = int (*)(const char *, struct stat *);
  using AccessFn = int (*)(const char *, int);
  using FstatFn = int (*)(int, struct stat *);
  using ReadlinkFn = ssize_t (*)(const char *, char *, size_t);
  using ForkFn = pid_t (*)();

  template <typename Fn> static Fn resolve(std::atomic<Fn> &slot, const char *name);

  std::atomic<OpenatFn> openat_{nullptr};
  std::atomic<CloseFn> close_{nullptr};
  std::atomic<ReadFn> read_{nullptr};
  std::atomic<WriteFn> write_{nullptr};
  std::atomic<IoctlFn> ioctl_{nullptr};
  std::atomic<MmapFn> mmap_{nullptr};
  std::atomic<MunmapFn> munmap_{nullptr};
  std::atomic<MprotectFn> mprotect_{nullptr};
  std::atomic<MadviseFn> madvise_{nullptr};
  std::atomic<MemfdCreateFn> memfd_create_{nullptr};
  std::atomic<DupFn> dup_{nullptr};
  std::atomic<Dup2Fn> dup2_{nullptr};
  std::atomic<Dup3Fn> dup3_{nullptr};
  std::atomic<FcntlFn> fcntl_{nullptr};
  std::atomic<FopenFn> fopen_{nullptr};
  std::atomic<FreopenFn> freopen_{nullptr};
  std::atomic<OpendirFn> opendir_{nullptr};
  std::atomic<ReaddirFn> readdir_{nullptr};
  std::atomic<ClosedirFn> closedir_{nullptr};
  std::atomic<StatFn> stat_{nullptr};
  std::atomic<LstatFn> lstat_{nullptr};
  std::atomic<AccessFn> access_{nullptr};
  std::atomic<FstatFn> fstat_{nullptr};
  std::atomic<ReadlinkFn> readlink_{nullptr};
  std::atomic<ForkFn> fork_{nullptr};
  std::atomic<bool> initialized_{false};
};

/// @brief Return the process-wide libc pass-through table.
LibcPassthrough &libc_passthrough();

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_LIBC_PASSTHROUGH_H_
