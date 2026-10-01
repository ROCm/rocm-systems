// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <bit>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>

#ifndef PROVIDER_FLAG_VALUE
#define PROVIDER_FLAG_VALUE 0
#endif

extern "C" const char *provider_config();
extern "C" const char *provider_digest();
extern "C" int provider_flag();
extern "C" const char *provider_generated_config();

int main(int argc, char **argv) {
  if (argc == 2) {
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      std::fprintf(stderr, "%s\n", dlerror());
      return 1;
    }
    void *symbol = dlsym(handle, "provider_digest");
    if (!symbol)
      return 1;
    const auto digest = std::bit_cast<const char *(*)()>(symbol);
    // Match the production resolver's build-digest rejection criterion.
    const bool rejected = std::strcmp(PROVIDER_DIGEST, digest()) != 0;
    dlclose(handle);
    std::puts(rejected ? "mixed_config=rejected" : "mixed_config=accepted");
    return rejected ? 0 : 1;
  }
  std::printf("config=%s digest=%s flag=%d\n", provider_config(), provider_digest(),
              provider_flag());
  return std::strcmp(PROVIDER_CONFIG, provider_config()) != 0 ||
                 std::strcmp(PROVIDER_CONFIG, provider_generated_config()) != 0 ||
                 std::strcmp(PROVIDER_DIGEST, provider_digest()) != 0 ||
                 PROVIDER_FLAG_VALUE != provider_flag()
             ? 1
             : 0;
}
