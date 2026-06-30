// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file early_mmap_preload_test.cpp
/// @brief Test helper whose constructor calls mmap before rocjitsu initializes.

#include <sys/mman.h>
#include <unistd.h>

__attribute__((constructor)) static void early_mmap_constructor() {
  void *ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    _exit(101);

  static_cast<char *>(ptr)[0] = 7;

  if (munmap(ptr, 4096) != 0)
    _exit(102);
}
