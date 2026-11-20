// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// A shared library that links libamdcuid_static.a and exports one function of
// its own. Its dynamic symbol table must hold nothing of the archive's.

#include "amd_cuid.h"

extern "C" __attribute__((visibility("default"))) const char* cuid_exports_probe_version() {
  return amdcuid_library_version_to_string();
}
