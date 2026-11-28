/* Copyright (c) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT */

/* ROCm trace contract for runtimes: providers with the contract names and
 * GUIDs, and an enable check that works on both TraceLogging transports.
 *
 * Requires a TraceLogging implementation: the Windows SDK on Windows, the
 * LTTng flavour of microsoft/tracelogging (with liblttng-ust) on Linux. This
 * header names no library; the runtime links the transport it uses. C11 or
 * later (TraceLogging's LTTng macros use static_assert), and C++. */

#ifndef ROCM_TRACE_TRACE_H
#define ROCM_TRACE_TRACE_H

#include "rocm/trace/values.h"

#if defined(_WIN32)
/* First, in its own block: the other Windows headers need its types. */
#include <windows.h>

#include <TraceLoggingProvider.h>
#include <evntrace.h>
#include <winmeta.h> /* WINEVENT_OPCODE_*, which TraceLoggingOpcode takes */
#else
#if !defined(__cplusplus)
#include <assert.h> /* static_assert in C11 */
#endif
#include <tracelogging/TraceLoggingProvider.h> /* also WINEVENT_OPCODE_* */
#endif

/* Define a provider with its contract name and GUID, for example:
 *
 *   ROCM_TRACE_DEFINE_PROVIDER(rocm_hip, HIP);
 *
 * ID is the suffix of a ROCM_TRACE_PROVIDER_<ID>_NAME in values.h. */
#define ROCM_TRACE_DEFINE_PROVIDER(provider, ID)                               \
  TRACELOGGING_DEFINE_PROVIDER(provider, ROCM_TRACE_PROVIDER_##ID##_NAME,      \
                               ROCM_TRACE_PROVIDER_##ID##_GUID_INIT)

/* Levels, in each transport's own numbering (values.h lists the numbers). */
#define ROCM_TRACE_LEVEL_INFORMATION TRACE_LEVEL_INFORMATION
#define ROCM_TRACE_LEVEL_VERBOSE TRACE_LEVEL_VERBOSE

/* Whether a session enabled an event: per event on LTTng, per provider level
 * and keyword on ETW. Use it before work done only for tracing, and for
 * control events, where the runtime acts on the answer. Plain data events need
 * no explicit check: TraceLoggingWrite checks its own event.
 *
 *   provider  the symbol given to ROCM_TRACE_DEFINE_PROVIDER
 *   event     the event name, a string literal (LTTng)
 *   level     ROCM_TRACE_LEVEL_INFORMATION or ROCM_TRACE_LEVEL_VERBOSE (ETW)
 *   keyword   the event's ROCM_TRACE_KW_* bits (ETW)
 *
 * The LTTng flavour has only TraceLoggingEventEnabled and the Windows SDK only
 * TraceLoggingProviderEnabled, so each branch ignores the arguments the other
 * uses. */
#if defined(_WIN32)
#define ROCM_TRACE_ENABLED(provider, event, level, keyword)                    \
  TraceLoggingProviderEnabled(provider, level, keyword)
#else
#define ROCM_TRACE_ENABLED(provider, event, level, keyword)                    \
  TraceLoggingEventEnabled(provider, event)
#endif

#endif /* ROCM_TRACE_TRACE_H */
