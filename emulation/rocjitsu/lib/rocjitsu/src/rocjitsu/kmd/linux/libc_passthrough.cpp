// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_passthrough.cpp
/// @brief dlsym(RTLD_NEXT) libc pass-through table.

#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/syscall_compat.h"

#include "util/dynamic_loader.h"

#include <cassert>
#include <dlfcn.h>

namespace rocjitsu {

void LibcPassthrough::resolve() {
  if (initialized_)
    return;

  auto *handle = RTLD_NEXT;
  openat = util::lookup_symbol<decltype(openat)>(handle, "openat");
  close = util::lookup_symbol<decltype(close)>(handle, "close");
  read = util::lookup_symbol<decltype(read)>(handle, "read");
  write = util::lookup_symbol<decltype(write)>(handle, "write");
  ioctl = util::lookup_symbol<decltype(ioctl)>(handle, "ioctl");
  syscall = util::lookup_symbol<decltype(syscall)>(handle, "syscall");
  mmap = util::lookup_symbol<decltype(mmap)>(handle, "mmap");
  munmap = util::lookup_symbol<decltype(munmap)>(handle, "munmap");
  mremap = util::lookup_symbol<decltype(mremap)>(handle, "mremap");
  mprotect = util::lookup_symbol<decltype(mprotect)>(handle, "mprotect");
  madvise = util::lookup_symbol<decltype(madvise)>(handle, "madvise");
  memfd_create = util::lookup_symbol<decltype(memfd_create)>(handle, "memfd_create");
  dup = util::lookup_symbol<decltype(dup)>(handle, "dup");
  dup2 = util::lookup_symbol<decltype(dup2)>(handle, "dup2");
  dup3 = util::lookup_symbol<decltype(dup3)>(handle, "dup3");
  fcntl = util::lookup_symbol<decltype(fcntl)>(handle, "fcntl");
  fopen = util::lookup_symbol<decltype(fopen)>(handle, "fopen");
  freopen = util::lookup_symbol<decltype(freopen)>(handle, "freopen");
  opendir = util::lookup_symbol<decltype(opendir)>(handle, "opendir");
  readdir = util::lookup_symbol<decltype(readdir)>(handle, "readdir");
  closedir = util::lookup_symbol<decltype(closedir)>(handle, "closedir");
  stat = util::lookup_symbol<decltype(stat)>(handle, "stat");
  lstat = util::lookup_symbol<decltype(lstat)>(handle, "lstat");
  access = util::lookup_symbol<decltype(access)>(handle, "access");
  lseek_fn = util::lookup_symbol<decltype(lseek_fn)>(handle, "lseek");
  lseek64_fn = util::lookup_symbol<decltype(lseek64_fn)>(handle, "lseek64");
  fstat_fn = util::lookup_symbol<decltype(fstat_fn)>(handle, "fstat");
  // Legacy aliases, resolved eagerly so no wrapper needs a lazy static (see the
  // declarations for why that matters across fork).
  fstat64_fn = util::lookup_symbol<decltype(fstat64_fn)>(handle, "fstat64");
  fxstat_fn = util::lookup_symbol<decltype(fxstat_fn)>(handle, "__fxstat");
  fxstat64_fn = util::lookup_symbol<decltype(fxstat64_fn)>(handle, "__fxstat64");
  stat64_fn = util::lookup_symbol<decltype(stat64_fn)>(handle, "stat64");
  lstat64_fn = util::lookup_symbol<decltype(lstat64_fn)>(handle, "lstat64");
  xstat_fn = util::lookup_symbol<decltype(xstat_fn)>(handle, "__xstat");
  xstat64_fn = util::lookup_symbol<decltype(xstat64_fn)>(handle, "__xstat64");
  lxstat_fn = util::lookup_symbol<decltype(lxstat_fn)>(handle, "__lxstat");
  lxstat64_fn = util::lookup_symbol<decltype(lxstat64_fn)>(handle, "__lxstat64");
  readlink_fn = util::lookup_symbol<decltype(readlink_fn)>(handle, "readlink");
  realpath_fn = util::lookup_symbol<decltype(realpath_fn)>(handle, "realpath");
  realpath_chk_fn = util::lookup_symbol<decltype(realpath_chk_fn)>(handle, "__realpath_chk");
  fstatat_fn = util::lookup_symbol<decltype(fstatat_fn)>(handle, "fstatat");
  fstatat64_fn = util::lookup_symbol<decltype(fstatat64_fn)>(handle, "fstatat64");
  fxstatat_fn = util::lookup_symbol<decltype(fxstatat_fn)>(handle, "__fxstatat");
  fxstatat64_fn = util::lookup_symbol<decltype(fxstatat64_fn)>(handle, "__fxstatat64");
  faccessat_fn = util::lookup_symbol<decltype(faccessat_fn)>(handle, "faccessat");
  readlinkat_fn = util::lookup_symbol<decltype(readlinkat_fn)>(handle, "readlinkat");
  statx_fn = util::lookup_symbol<decltype(statx_fn)>(handle, "statx");
  fork = util::lookup_symbol<decltype(fork)>(handle, "fork");
  getsockname = util::lookup_symbol<decltype(getsockname)>(handle, "getsockname");
  sendmsg = util::lookup_symbol<decltype(sendmsg)>(handle, "sendmsg");
  recvmsg = util::lookup_symbol<decltype(recvmsg)>(handle, "recvmsg");
  shmget = util::lookup_symbol<decltype(shmget)>(handle, "shmget");
  shmat = util::lookup_symbol<decltype(shmat)>(handle, "shmat");
  shmdt = util::lookup_symbol<decltype(shmdt)>(handle, "shmdt");
  shmctl = util::lookup_symbol<decltype(shmctl)>(handle, "shmctl");
  getrandom = util::lookup_symbol<decltype(getrandom)>(handle, "getrandom");
  prlimit = util::lookup_symbol<decltype(prlimit)>(handle, "prlimit");
  ftruncate = util::lookup_symbol<decltype(ftruncate)>(handle, "ftruncate");
  fallocate = util::lookup_symbol<decltype(fallocate)>(handle, "fallocate");
  socketpair = util::lookup_symbol<decltype(socketpair)>(handle, "socketpair");
  bind = util::lookup_symbol<decltype(bind)>(handle, "bind");
  epoll_ctl = util::lookup_symbol<decltype(epoll_ctl)>(handle, "epoll_ctl");
  prctl = util::lookup_symbol<decltype(prctl)>(handle, "prctl");
  epoll_create1 = util::lookup_symbol<decltype(epoll_create1)>(handle, "epoll_create1");
  eventfd = util::lookup_symbol<decltype(eventfd)>(handle, "eventfd");
  epoll_wait = util::lookup_symbol<decltype(epoll_wait)>(handle, "epoll_wait");
  accept4 = util::lookup_symbol<decltype(accept4)>(handle, "accept4");
  getsockopt = util::lookup_symbol<decltype(getsockopt)>(handle, "getsockopt");
  socket = util::lookup_symbol<decltype(socket)>(handle, "socket");
  connect = util::lookup_symbol<decltype(connect)>(handle, "connect");
  listen = util::lookup_symbol<decltype(listen)>(handle, "listen");
  execve = util::lookup_symbol<decltype(execve)>(handle, "execve");
  setsid = util::lookup_symbol<decltype(setsid)>(handle, "setsid");
  waitpid = util::lookup_symbol<decltype(waitpid)>(handle, "waitpid");
  poll = util::lookup_symbol<decltype(poll)>(handle, "poll");
  clone = util::lookup_symbol<decltype(clone)>(handle, "clone");
  exit_immediately = util::lookup_symbol<decltype(exit_immediately)>(handle, "_exit");
  // These interfaces postdate libc versions used by supported build hosts.
  pidfd_open = util::lookup_symbol<decltype(pidfd_open)>(handle, "pidfd_open");
  close_range = util::lookup_symbol<decltype(close_range)>(handle, "close_range");
  getdents64 = util::lookup_symbol<decltype(getdents64)>(handle, "getdents64");
  // Keep ready() false unless every required interposed libc entry point was
  // resolved. In release builds the asserts disappear, so the boolean must not
  // claim readiness while any later call would dereference a null function
  // pointer.
  initialized_ = openat && close && read && write && ioctl && syscall && mmap && munmap &&
                 mprotect && madvise && memfd_create && dup && dup2 && dup3 && fcntl && fopen &&
                 freopen && opendir && readdir && closedir && stat && lstat && access && fstat_fn &&
                 readlink_fn && realpath_fn && fork && getsockname && sendmsg && recvmsg &&
                 shmget && shmat && shmdt && shmctl && getrandom && prlimit && ftruncate &&
                 fallocate && socketpair && bind && epoll_ctl && prctl && epoll_create1 &&
                 eventfd && epoll_wait && accept4 && getsockopt && socket && connect && listen &&
                 execve && setsid && waitpid && poll && clone && exit_immediately;
  assert(initialized_);
}

LibcPassthrough &libc_passthrough() {
  static LibcPassthrough real;
  return real;
}

int host_pidfd_open(pid_t pid, unsigned int flags) {
  const auto &host = libc_passthrough();
  if (host.pidfd_open)
    return host.pidfd_open(pid, flags);
  auto call = host.syscall ? host.syscall : ::syscall;
  return call(SYS_pidfd_open, static_cast<long>(pid), static_cast<unsigned long>(flags));
}

int host_close_range(unsigned int first, unsigned int last, int flags) {
  const auto &host = libc_passthrough();
  if (host.close_range)
    return host.close_range(first, last, flags);
  auto call = host.syscall ? host.syscall : ::syscall;
  return call(SYS_close_range, static_cast<unsigned long>(first), static_cast<unsigned long>(last),
              static_cast<long>(flags));
}

ssize_t host_getdents64(int fd, void *buffer, size_t size) {
  const auto &host = libc_passthrough();
  if (host.getdents64)
    return host.getdents64(fd, buffer, size);
  auto call = host.syscall ? host.syscall : ::syscall;
  return call(SYS_getdents64, static_cast<long>(fd), buffer, size);
}

} // namespace rocjitsu
