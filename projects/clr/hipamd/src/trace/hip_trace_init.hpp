/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Lifecycle entry points for the HIP trace backend, kept in a header of their own so
// that hip_context.cpp and hip_runtime.cpp do not have to pull in <windows.h>,
// <TraceLoggingProvider.h> and hip_prof_str.h.

namespace hip {
namespace trace {

#if defined(HIP_TRACE_BACKEND_TRACELOGGING)

// Registers the TraceLogging provider. Must not run under loader lock, so it is driven
// from hip::init() rather than from DllMain attach: when a session is already active,
// ETW delivers the enablement callback inline on the registering thread.
//
// Consequence: the first HIP API call in a process is never traced. It evaluates the
// enablement gate before the dispatch that triggers hip::init(), so the provider is
// still unregistered at that point. In practice that first call is a device probe
// rather than a launch or a transfer.
void initialize();

// Unregisters the provider. Unlike EventRegister, EventUnregister is explicitly
// permitted during DLL_PROCESS_DETACH.
void finalize();

#else

inline void initialize() {}
inline void finalize() {}

#endif

}  // namespace trace
}  // namespace hip
