// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_passthrough.cpp
/// @brief dlsym(RTLD_NEXT) libc pass-through table.

#include "rocjitsu/kmd/linux/libc_passthrough.h"

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
  fork = util::lookup_symbol<decltype(fork)>(handle, "fork");
  // Keep ready() false unless every required interposed libc entry point was
  // resolved. In release builds the asserts disappear, so the boolean must not
  // claim readiness while any later call would dereference a null function
  // pointer.
  initialized_ = openat && close && read && write && ioctl && mmap && munmap && mprotect &&
                 madvise && memfd_create && dup && dup2 && dup3 && fcntl && fopen && freopen &&
                 opendir && readdir && closedir && stat && lstat && access && fstat_fn &&
                 readlink_fn && fork;
  assert(initialized_);
}

LibcPassthrough &libc_passthrough() {
  static LibcPassthrough real;
  return real;
}

} // namespace rocjitsu
