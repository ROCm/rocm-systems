// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file early_mmap_preload_test.cpp
/// @brief Test helper whose constructor calls mmap before rocjitsu initializes.

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

template <size_t N> [[noreturn]] void fail(const char (&message)[N]) {
  ssize_t ignored = write(STDERR_FILENO, message, N - 1);
  static_cast<void>(ignored);
  _exit(EXIT_FAILURE);
}

} // namespace

__attribute__((constructor)) static void early_mmap_constructor() {
  struct stat st {};
  if (stat("/dev/null", &st) != 0)
    fail("early mmap preload: stat failed\n");
  if (access("/dev/null", R_OK) != 0)
    fail("early mmap preload: access failed\n");

  char link_target[256]{};
  if (readlink("/proc/self/exe", link_target, sizeof(link_target)) <= 0)
    fail("early mmap preload: readlink failed\n");

  int fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    fail("early mmap preload: open failed\n");
  if (fstat(fd, &st) != 0)
    fail("early mmap preload: fstat failed\n");
  int dup_fd = dup(fd);
  if (dup_fd < 0)
    fail("early mmap preload: dup failed\n");
  if (fcntl(dup_fd, F_GETFD, 0) < 0)
    fail("early mmap preload: fcntl failed\n");
  if (close(dup_fd) != 0)
    fail("early mmap preload: close dup failed\n");
  if (close(fd) != 0)
    fail("early mmap preload: close fd failed\n");

  errno = 0;
  if (mmap(nullptr, 0, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) != MAP_FAILED ||
      errno != EINVAL)
    fail("early mmap preload: zero-length mmap did not fail with EINVAL\n");

  void *ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    fail("early mmap preload: mmap failed\n");

  static_cast<char *>(ptr)[0] = 7;

  errno = 0;
  if (munmap(ptr, 0) == 0 || errno != EINVAL)
    fail("early mmap preload: zero-length munmap did not fail with EINVAL\n");

  if (munmap(ptr, 4096) != 0)
    fail("early mmap preload: munmap failed\n");
}
