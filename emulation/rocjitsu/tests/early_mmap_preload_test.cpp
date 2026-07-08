// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file early_mmap_preload_test.cpp
/// @brief Test helper whose constructor calls mmap before rocjitsu initializes.

#include <cerrno>
#include <cstdlib>
#include <sys/mman.h>
#include <unistd.h>

namespace {

template <size_t N> [[noreturn]] void fail(const char (&message)[N]) {
  static_cast<void>(write(STDERR_FILENO, message, N - 1));
  _exit(EXIT_FAILURE);
}

} // namespace

__attribute__((constructor)) static void early_mmap_constructor() {
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
