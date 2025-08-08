/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "libpci_helper.h"

#include <cstdio>
#include <dlfcn.h>
#include "exception.h"


namespace rocprofiler {
PcieAccessApi* api = nullptr;
void* libpciaccess_handle = nullptr;

// A function just to load all the symbols you need
PcieAccessApi* LoadPcieAccessLibAPI() {
  char* libpciaccess_tool = getenv("LIBPCI_TOOL_LIB");
  if (libpciaccess_tool == nullptr) {
    libpciaccess_tool = "libpciaccess.so";
  }
  libpciaccess_handle = dlopen(libpciaccess_tool, RTLD_NOW);
  if (libpciaccess_handle == nullptr) {
    fprintf(stderr, "ROCProfiler: can't load libpciaccess library \"%s\"\n", libpciaccess_tool);
    fprintf(stderr, "%s\n", dlerror());
    abort();
  }
  dlerror(); /* Clear any existing error */

  api = new PcieAccessApi();
  api->pci_device_find_by_slot =
      reinterpret_cast<pci_device_find_by_slot_handler_t>(dlsym(
          libpciaccess_handle, "pci_device_find_by_slot"));
  if (api->pci_device_find_by_slot == nullptr) {
    fprintf(stderr,
        "ROCProfiler: libpciaccess library corrupted, pci_device_find_by_slot() method is "
        "expected\n");
    fprintf(stderr, "%s\n", dlerror());
    abort();
  }
  api->pci_device_probe =
      reinterpret_cast<pci_device_probe_handler_t>(dlsym(
          libpciaccess_handle, "pci_device_probe"));
  if (api->pci_device_probe == nullptr) {
    fprintf(stderr,
        "ROCProfiler: libpciaccess library corrupted, pci_device_probe() method is "
        "expected\n");
    fprintf(stderr, "%s\n", dlerror());
    abort();
  }
  api->pci_device_map_range =
      reinterpret_cast<pci_device_map_range_handler_t>(dlsym(
          libpciaccess_handle, "pci_device_map_range"));
  if (api->pci_device_map_range == nullptr) {
    fprintf(stderr,
        "ROCProfiler: libpciaccess library corrupted, pci_device_map_range() method is "
        "expected\n");
    fprintf(stderr, "%s\n", dlerror());
    abort();
  }
  api->pci_device_unmap_range =
      reinterpret_cast<pci_device_unmap_range_handler_t>(dlsym(
          libpciaccess_handle, "pci_device_unmap_range"));

  if (api->pci_device_unmap_range == nullptr) {
    fprintf(stderr,
        "ROCProfiler: libpciaccess library corrupted, pci_device_unmap_range() method is "
        "expected\n");
    fprintf(stderr, "%s\n", dlerror());
    abort();
  }
  api->pci_system_init =
      reinterpret_cast<pci_system_init_handler_t>(dlsym(
          libpciaccess_handle, "pci_system_init"));
  if (api->pci_system_init == nullptr) {
    fprintf(stderr,
        "ROCProfiler: libpciaccess library corrupted, pci_system_init() method is "
        "expected\n");
    fprintf(stderr, "%s\n", dlerror());
    abort();
  }
  api->pci_system_cleanup =
      reinterpret_cast<pci_system_cleanup_handler_t>(dlsym(
          libpciaccess_handle, "pci_system_cleanup"));
  if (api->pci_system_cleanup == nullptr) {
    fprintf(stderr,
        "ROCProfiler: libpciaccess library corrupted, pci_system_cleanup() method is "
        "expected\n");
    fprintf(stderr, "%s\n", dlerror());
    abort();
  }
  return api;
}

const PcieAccessApi* GetPciAccessLibApi() {
  if (api == nullptr) {
    api = LoadPcieAccessLibAPI();
  }
  return api;
}

void UnLoadPcieAccessLibAPI() {
  if (api == nullptr) {
    return;
  }
  api->pci_device_find_by_slot = nullptr;
  api->pci_device_probe = nullptr;
  api->pci_device_map_range = nullptr;
  api->pci_device_unmap_range = nullptr;
  api->pci_system_init = nullptr;
  api->pci_system_cleanup = nullptr;
  delete api;
  api = nullptr;
  if (dlclose(libpciaccess_handle) != 0) {
    warning("Warning: Failed to close libpciaccess handle: %s\n", dlerror());
  }
}
}  // namespace rocprofiler
