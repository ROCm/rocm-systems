// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <dlfcn.h>
#include <new>
#include <unistd.h>

// Preloaded only by the analyzer subprocess test. This size selects the CLI's
// input vector, before decoding; ordinary allocations use libstdc++ unchanged.
void *operator new(std::size_t bytes) {
  if (bytes == 123457) {
    constexpr char message[] = "ConSan test: throwing input allocation\n";
    if (::write(STDERR_FILENO, message, sizeof(message) - 1) != sizeof(message) - 1)
      std::_Exit(88);
    throw std::bad_alloc();
  }
  static const auto allocate =
      reinterpret_cast<void *(*)(std::size_t)>(::dlsym(RTLD_NEXT, "_Znwm"));
  if (allocate == nullptr)
    std::abort();
  return allocate(bytes);
}
