// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* A legacy HSA tool wrapper that a fork child must never enter. */
#include <hsa/hsa.h>

#include <cstdint>

// Only the stable prefix needed for this fixture is copied. The vendored
// hsa_api_trace.h expects extra ROCr-private inc/ headers in its include path.
struct ApiTableVersion {
  uint32_t major_id;
  uint32_t minor_id;
  uint32_t step_id;
  uint32_t reserved;
};

struct CoreApiTablePrefix {
  ApiTableVersion version;
  decltype(&hsa_init) init;
  decltype(&hsa_shut_down) shut_down;
  decltype(&hsa_system_get_info) system_get_info;
};

struct RootTablePrefix {
  ApiTableVersion version;
  CoreApiTablePrefix* core;
};

static hsa_status_t intercepted_system_get_info(hsa_system_info_t, void*) {
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

extern "C" bool OnLoad(RootTablePrefix* tables, uint64_t, uint64_t,
                       const char* const*) {
  tables->core->system_get_info = intercepted_system_get_info;
  return true;
}

extern "C" void OnUnload() {}
