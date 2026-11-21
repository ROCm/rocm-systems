/* Copyright (c) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT */

/* Compile-only: trace.h with a real TraceLogging implementation, as C. */

#include <rocm/trace/trace.h>

ROCM_TRACE_DEFINE_PROVIDER(rocm_trace_check_c, HIP);

#if defined(_WIN32)
_Static_assert(ROCM_TRACE_LEVEL_INFORMATION == ROCM_TRACE_ETW_LEVEL_INFORMATION,
               "information level");
_Static_assert(ROCM_TRACE_LEVEL_VERBOSE == ROCM_TRACE_ETW_LEVEL_VERBOSE,
               "verbose level");
#else
_Static_assert(ROCM_TRACE_LEVEL_INFORMATION ==
                   ROCM_TRACE_LTTNG_LEVEL_INFORMATION,
               "information level");
_Static_assert(ROCM_TRACE_LEVEL_VERBOSE == ROCM_TRACE_LTTNG_LEVEL_VERBOSE,
               "verbose level");
#endif

int rocm_trace_check_c_enabled(void);

int rocm_trace_check_c_enabled(void) {
  return ROCM_TRACE_ENABLED(rocm_trace_check_c, "check",
                            ROCM_TRACE_LEVEL_INFORMATION, ROCM_TRACE_KW_API)
             ? 1
             : 0;
}

/* TraceLoggingOpcode's values come from <winmeta.h> on Windows and from the
 * LTTng flavour's own header on Linux; trace.h provides them on both. */
static_assert(WINEVENT_OPCODE_START == 1 && WINEVENT_OPCODE_STOP == 2,
              "opcodes");
