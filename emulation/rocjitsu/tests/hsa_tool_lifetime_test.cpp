// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

namespace {

using Retain = bool (*)();

Retain resolve_retain(void *module) {
  static_assert(sizeof(Retain) == sizeof(void *));
  void *symbol = dlsym(module, "rj_test_retain_hsa_tool_dso");
  Retain retain = nullptr;
  std::memcpy(&retain, &symbol, sizeof(retain));
  return retain;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s /path/to/test-module.so\n", argv[0]);
    return 2;
  }

  void *module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (module == nullptr) {
    std::fprintf(stderr, "dlopen(%s) failed: %s\n", argv[1], dlerror());
    return 3;
  }
  const Retain retain = resolve_retain(module);
  if (retain == nullptr) {
    std::fprintf(stderr, "dlsym(retain) failed: %s\n", dlerror());
    return 4;
  }
  if (!retain()) {
    std::fputs("the module could not retain its own DSO\n", stderr);
    return 5;
  }
  if (dlclose(module) != 0) {
    std::fprintf(stderr, "first dlclose failed: %s\n", dlerror());
    return 6;
  }

  void *probe = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (probe == nullptr) {
    std::fprintf(stderr, "the retained DSO was unloaded: %s\n", dlerror());
    return 7;
  }
  if (dlclose(probe) != 0) {
    std::fprintf(stderr, "probe dlclose failed: %s\n", dlerror());
    return 8;
  }
  return 0;
}
