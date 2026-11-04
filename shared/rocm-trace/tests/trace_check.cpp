// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compile-only: trace.h with a real TraceLogging implementation, as C++.

#include <rocm/trace/trace.h>

ROCM_TRACE_DEFINE_PROVIDER(rocm_trace_check_cpp, HSA);

#if defined(_WIN32)
static_assert(ROCM_TRACE_LEVEL_INFORMATION == ROCM_TRACE_ETW_LEVEL_INFORMATION);
static_assert(ROCM_TRACE_LEVEL_VERBOSE == ROCM_TRACE_ETW_LEVEL_VERBOSE);
#else
static_assert(ROCM_TRACE_LEVEL_INFORMATION ==
              ROCM_TRACE_LTTNG_LEVEL_INFORMATION);
static_assert(ROCM_TRACE_LEVEL_VERBOSE == ROCM_TRACE_LTTNG_LEVEL_VERBOSE);
#endif

bool rocm_trace_check_cpp_enabled();

bool rocm_trace_check_cpp_enabled() {
  return ROCM_TRACE_ENABLED(rocm_trace_check_cpp, "check",
                            ROCM_TRACE_LEVEL_VERBOSE, ROCM_TRACE_KW_DISPATCH);
}
