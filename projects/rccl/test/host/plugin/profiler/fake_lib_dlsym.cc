/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The profiler_vN adapter tests hand getNcclProfiler_vN a FakeLib (exported symbol -> address) as the library
// handle. This resolves it, as profiler-test.cc's ncclOsDlsym does for that binary.

#include <map>
#include <string>

#include "os.h"

void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) {
  const auto* lib = static_cast<const std::map<std::string, void*>*>(handle);
  auto it = lib->find(symbol);
  return it == lib->end() ? nullptr : it->second;
}
