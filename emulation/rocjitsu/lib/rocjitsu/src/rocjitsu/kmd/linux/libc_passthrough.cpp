// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_passthrough.cpp
/// @brief dlsym(RTLD_NEXT) libc pass-through table.

#include "rocjitsu/kmd/linux/libc_passthrough.h"

#include "util/dynamic_loader.h"

#include <cassert>
#include <cstdint>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rocjitsu {

namespace {

void *raw_mmap_syscall(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  // syscall(2) is the libc wrapper: on kernel errors it returns -1 and sets errno.
  long rc = syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
  if (rc == -1)
    return MAP_FAILED;
  return reinterpret_cast<void *>(static_cast<uintptr_t>(rc));
}

int raw_munmap_syscall(void *addr, size_t length) {
  long rc = syscall(SYS_munmap, addr, length);
  assert(rc == 0 || rc == -1);
  return static_cast<int>(rc);
}

} // namespace

template <typename Fn> Fn LibcPassthrough::resolve(std::atomic<Fn> &slot, const char *name) {
  if (auto fn = slot.load(std::memory_order_acquire))
    return fn;
  auto fn = util::lookup_symbol<Fn>(RTLD_NEXT, name);
  assert(fn);
  slot.store(fn, std::memory_order_release);
  return fn;
}

void LibcPassthrough::resolve() {
  if (ready())
    return;

  static_cast<void>(resolve(openat_, "openat"));
  static_cast<void>(resolve(close_, "close"));
  static_cast<void>(resolve(read_, "read"));
  static_cast<void>(resolve(write_, "write"));
  static_cast<void>(resolve(ioctl_, "ioctl"));
  static_cast<void>(resolve(mmap_, "mmap"));
  static_cast<void>(resolve(munmap_, "munmap"));
  static_cast<void>(resolve(mprotect_, "mprotect"));
  static_cast<void>(resolve(madvise_, "madvise"));
  static_cast<void>(resolve(memfd_create_, "memfd_create"));
  static_cast<void>(resolve(dup_, "dup"));
  static_cast<void>(resolve(dup2_, "dup2"));
  static_cast<void>(resolve(dup3_, "dup3"));
  static_cast<void>(resolve(fcntl_, "fcntl"));
  static_cast<void>(resolve(fopen_, "fopen"));
  static_cast<void>(resolve(freopen_, "freopen"));
  static_cast<void>(resolve(opendir_, "opendir"));
  static_cast<void>(resolve(readdir_, "readdir"));
  static_cast<void>(resolve(closedir_, "closedir"));
  static_cast<void>(resolve(stat_, "stat"));
  static_cast<void>(resolve(lstat_, "lstat"));
  static_cast<void>(resolve(access_, "access"));
  static_cast<void>(resolve(fstat_, "fstat"));
  static_cast<void>(resolve(readlink_, "readlink"));
  static_cast<void>(resolve(fork_, "fork"));
  initialized_.store(true, std::memory_order_release);
}

int LibcPassthrough::openat(int dirfd, const char *path, int flags, mode_t mode) {
  return resolve(openat_, "openat")(dirfd, path, flags, mode);
}

int LibcPassthrough::close(int fd) { return resolve(close_, "close")(fd); }

ssize_t LibcPassthrough::read(int fd, void *buf, size_t count) {
  return resolve(read_, "read")(fd, buf, count);
}

ssize_t LibcPassthrough::write(int fd, const void *buf, size_t count) {
  return resolve(write_, "write")(fd, buf, count);
}

int LibcPassthrough::ioctl(int fd, unsigned long request, void *arg) {
  return resolve(ioctl_, "ioctl")(fd, request, arg);
}

void *LibcPassthrough::mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  if (auto fn = mmap_.load(std::memory_order_acquire))
    return fn(addr, length, prot, flags, fd, offset);
  return raw_mmap_syscall(addr, length, prot, flags, fd, offset);
}

int LibcPassthrough::munmap(void *addr, size_t length) {
  if (auto fn = munmap_.load(std::memory_order_acquire))
    return fn(addr, length);
  return raw_munmap_syscall(addr, length);
}

int LibcPassthrough::mprotect(void *addr, size_t length, int prot) {
  return resolve(mprotect_, "mprotect")(addr, length, prot);
}

int LibcPassthrough::madvise(void *addr, size_t length, int advice) {
  return resolve(madvise_, "madvise")(addr, length, advice);
}

int LibcPassthrough::memfd_create(const char *name, unsigned int flags) {
  return resolve(memfd_create_, "memfd_create")(name, flags);
}

int LibcPassthrough::dup(int oldfd) { return resolve(dup_, "dup")(oldfd); }

int LibcPassthrough::dup2(int oldfd, int newfd) {
  return resolve(dup2_, "dup2")(oldfd, newfd);
}

int LibcPassthrough::dup3(int oldfd, int newfd, int flags) {
  return resolve(dup3_, "dup3")(oldfd, newfd, flags);
}

int LibcPassthrough::fcntl(int fd, int cmd, int arg) {
  return resolve(fcntl_, "fcntl")(fd, cmd, arg);
}

int LibcPassthrough::fcntl(int fd, int cmd, long arg) {
  return resolve(fcntl_, "fcntl")(fd, cmd, arg);
}

int LibcPassthrough::fcntl(int fd, int cmd, void *arg) {
  return resolve(fcntl_, "fcntl")(fd, cmd, arg);
}

FILE *LibcPassthrough::fopen(const char *path, const char *mode) {
  return resolve(fopen_, "fopen")(path, mode);
}

FILE *LibcPassthrough::freopen(const char *path, const char *mode, FILE *stream) {
  return resolve(freopen_, "freopen")(path, mode, stream);
}

DIR *LibcPassthrough::opendir(const char *name) {
  return resolve(opendir_, "opendir")(name);
}

struct dirent *LibcPassthrough::readdir(DIR *dirp) {
  return resolve(readdir_, "readdir")(dirp);
}

int LibcPassthrough::closedir(DIR *dirp) { return resolve(closedir_, "closedir")(dirp); }

int LibcPassthrough::stat(const char *path, struct stat *buf) {
  return resolve(stat_, "stat")(path, buf);
}

int LibcPassthrough::lstat(const char *path, struct stat *buf) {
  return resolve(lstat_, "lstat")(path, buf);
}

int LibcPassthrough::access(const char *path, int mode) {
  return resolve(access_, "access")(path, mode);
}

int LibcPassthrough::fstat_fn(int fd, struct stat *buf) {
  return resolve(fstat_, "fstat")(fd, buf);
}

ssize_t LibcPassthrough::readlink_fn(const char *path, char *buf, size_t bufsiz) {
  return resolve(readlink_, "readlink")(path, buf, bufsiz);
}

pid_t LibcPassthrough::fork() { return resolve(fork_, "fork")(); }

LibcPassthrough &libc_passthrough() {
  static LibcPassthrough real;
  return real;
}

} // namespace rocjitsu
