// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstdio>
#include <dlfcn.h>

namespace rocjitsu::amdgpu {
class Wavefront;
}

int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  // The interposer owns process-lifetime state. Keep it mapped through exit,
  // as it is during normal preloaded execution.
  void *core = ::dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
  if (!core) {
    std::fprintf(stderr, "core load failed: %s\n", ::dlerror());
    return 1;
  }
  void *observer = ::dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!observer) {
    std::fprintf(stderr, "register observer load failed: %s\n", ::dlerror());
    return 1;
  }
  using Probe = void (*)(rocjitsu::amdgpu::Wavefront *, uint32_t, uint32_t);
  auto probe = reinterpret_cast<Probe>(::dlsym(observer, "rj_probe_register_observer"));
  if (!probe) {
    std::fprintf(stderr, "register observer entry point missing: %s\n", ::dlerror());
    return 1;
  }
  probe(nullptr, 0, 0);
  ::dlclose(observer);
  ::dlclose(core);
  return 0;
}
