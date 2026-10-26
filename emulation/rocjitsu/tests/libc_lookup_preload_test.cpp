// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_lookup_preload_test.cpp
/// @brief Inject allocator mapping re-entry and unavailable optional libc wrappers.

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <sys/mman.h>
#include <unistd.h>

static bool exercise_allocator = false;
static bool inside_allocator = false;
static const char *missing_wrapper = nullptr;

extern "C" __attribute__((visibility("default"))) void *dlsym(void *handle, const char *name) {
  // The versions cover glibc's combined libdl and its older x86-64/AArch64 ABIs.
  // dlvsym is not interposed by this helper.
  void *(*next)(void *, const char *) = nullptr;
  for (const char *version : {"GLIBC_2.34", "GLIBC_2.2.5", "GLIBC_2.17"}) {
    if (auto *symbol = dlvsym(RTLD_NEXT, "dlsym", version)) {
      next = reinterpret_cast<void *(*)(void *, const char *)>(symbol);
      break;
    }
  }
  if (!next)
    _exit(90);
  if (inside_allocator)
    _exit(91); // A mapping hook recursed into allocating lookup.
  if (exercise_allocator) {
    inside_allocator = true;
    void *storage = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (storage == MAP_FAILED)
      _exit(92);
    static_cast<char *>(storage)[0] = 42;
    storage = mremap(storage, 4096, 8192, MREMAP_MAYMOVE);
    if (storage == MAP_FAILED || static_cast<char *>(storage)[0] != 42)
      _exit(93);
    if (munmap(storage, 8192) != 0)
      _exit(94);
    inside_allocator = false;
    exercise_allocator = false;
  }
  if (missing_wrapper && std::strcmp(missing_wrapper, name) == 0)
    return nullptr;
  return next(handle, name);
}

__attribute__((constructor)) static void exercise_lookup_reentry() {
  missing_wrapper = std::getenv("ROCJITSU_TEST_MISSING_LIBC_WRAPPER");
  if (!std::getenv("ROCJITSU_TEST_LOOKUP_REENTRY"))
    return;
  exercise_allocator = true;
  if (!dlsym(RTLD_DEFAULT, "getpid") || exercise_allocator)
    _exit(95);
}
