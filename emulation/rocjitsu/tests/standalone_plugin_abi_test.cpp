// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_exports.h"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>

int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  // Load without core objects or librocjitsu.so so unresolved simulator
  // helpers cannot be supplied accidentally by the host process.
  void *module = ::dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!module) {
    std::fprintf(stderr, "standalone plugin load failed: %s\n", ::dlerror());
    return 1;
  }
  const auto metadata = reinterpret_cast<rocjitsu::PluginMetadataFn>(
      ::dlsym(module, rocjitsu::kPluginMetadataSymbol));
  const auto create =
      reinterpret_cast<rocjitsu::PluginCreateFn>(::dlsym(module, rocjitsu::kPluginCreateSymbol));
  const auto destroy =
      reinterpret_cast<rocjitsu::PluginDestroyFn>(::dlsym(module, rocjitsu::kPluginDestroySymbol));
  if (!metadata || !create || !destroy) {
    std::fprintf(stderr, "standalone plugin loader exports missing\n");
    ::dlclose(module);
    return 1;
  }
  const rocjitsu::PluginMetadata *description = metadata();
  if (!description || !description->name || std::strcmp(description->name, argv[2]) != 0) {
    std::fprintf(stderr, "standalone plugin metadata does not match %s\n", argv[2]);
    ::dlclose(module);
    return 1;
  }
  rocjitsu::PluginHandle plugin = create("{}");
  if (!plugin) {
    std::fprintf(stderr, "standalone plugin creation failed\n");
    ::dlclose(module);
    return 1;
  }
  destroy(plugin);
  ::dlclose(module);
  return 0;
}
