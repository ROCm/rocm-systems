/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// rocm_hip API events: every HIP API call through a dispatch table writes <api>_enter and
// <api>_exit through TraceLogging (LTTng-UST on Linux, ETW on Windows), for a profiler outside the
// process to record. The probes are generated (hip_trace_probes.cpp); see README.md here.

#pragma once

#include <hip/amd_detail/hip_api_trace.hpp>

namespace hip {
namespace trace {

// Install the probes in a dispatch table: each non-null entry is replaced with a probe that calls
// it. Call when the table is filled, before it is handed to tools, so the probe is innermost.
// Not thread-safe. With ROCM_TRACE_DISABLE=1 this does nothing.
void InstallProbes(HipDispatchTable* table);
void InstallProbes(HipCompilerDispatchTable* table);

// The tools table holds runtime callbacks for profilers, not API functions: nothing to install.
inline void InstallProbes(HipToolsDispatchTable*) {}

}  // namespace trace
}  // namespace hip
