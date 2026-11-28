/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_trace.hpp"

#include "hip_trace_probes.h"

#include <rocm/trace/trace.h>

#include <cstdlib>
#include <cstring>

// The rocm_hip provider, with the name and GUID of the ROCm trace contract (rocm/trace/values.h).
ROCM_TRACE_DEFINE_PROVIDER(rocm_hip, HIP);

// What the probes record (ROCM_TRACE_CAPTURE_*), read by hip_trace_probes.cpp. Set once, before
// the first table is installed.
int rocm_hip_trace_capture = ROCM_TRACE_CAPTURE_FULL;

namespace hip {
namespace trace {
namespace {

// ROCM_TRACE_DISABLE=1: install no probes, so API calls go straight to the implementation.
bool DisabledByEnvironment() {
  const char* value = std::getenv("ROCM_TRACE_DISABLE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

// ROCM_TRACE_CAPTURE=none|scalars|full; anything else, or unset, is full.
int CaptureLevelFromEnvironment() {
  const char* value = std::getenv("ROCM_TRACE_CAPTURE");
  if (value != nullptr && std::strcmp(value, "none") == 0) return ROCM_TRACE_CAPTURE_NONE;
  if (value != nullptr && std::strcmp(value, "scalars") == 0) return ROCM_TRACE_CAPTURE_SCALARS;
  return ROCM_TRACE_CAPTURE_FULL;
}

// Registered when the first table is installed, so a session can enable it from then on, and
// unregistered when libamdhip64 is unloaded, before the probes that write to it go away.
class Provider {
 public:
  Provider() {
    rocm_hip_trace_capture = CaptureLevelFromEnvironment();
    TraceLoggingRegister(rocm_hip);
  }
  ~Provider() { TraceLoggingUnregister(rocm_hip); }

  Provider(const Provider&) = delete;
  Provider& operator=(const Provider&) = delete;
};

bool Enabled() {
  static const bool enabled = !DisabledByEnvironment();
  if (enabled) {
    static Provider provider;
  }
  return enabled;
}

}  // namespace

void InstallProbes(HipDispatchTable* table) {
  if (Enabled()) rocm_trace_install_HipDispatchTable(table);
}

void InstallProbes(HipCompilerDispatchTable* table) {
  if (Enabled()) rocm_trace_install_HipCompilerDispatchTable(table);
}

}  // namespace trace
}  // namespace hip
