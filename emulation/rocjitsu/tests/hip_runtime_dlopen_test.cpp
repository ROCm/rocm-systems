// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_runtime_dlopen_test.cpp
/// @brief Exercises HIP initialization when the runtime has no link-time owner.

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

namespace {

using HipInit = int (*)(unsigned int);

template <typename Function> Function resolve(void *library, const char *name) {
  static_assert(sizeof(Function) == sizeof(void *));
  void *symbol = dlsym(library, name);
  Function function = nullptr;
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s /path/to/libamdhip64.so /path/to/hsa-tool.so\n", argv[0]);
    return 2;
  }

  void *hip = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (hip == nullptr) {
    std::fprintf(stderr, "dlopen(%s) failed: %s\n", argv[1], dlerror());
    return 3;
  }

  const HipInit hip_init = resolve<HipInit>(hip, "hipInit");
  if (hip_init == nullptr) {
    std::fprintf(stderr, "dlsym(hipInit) failed: %s\n", dlerror());
    return 4;
  }
  const int status = hip_init(0);
  if (status != 0) {
    std::fprintf(stderr, "hipInit failed: %d\n", status);
    return 5;
  }

  if (dlclose(hip) != 0) {
    std::fprintf(stderr, "dlclose(libamdhip64) failed: %s\n", dlerror());
    return 6;
  }

  void *tool = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (tool == nullptr) {
    std::fprintf(stderr, "HSA tool was unloaded with the dynamic HIP runtime: %s\n", dlerror());
    return 7;
  }
  if (dlclose(tool) != 0) {
    std::fprintf(stderr, "dlclose(HSA tool probe) failed: %s\n", dlerror());
    return 8;
  }
  return 0;
}
