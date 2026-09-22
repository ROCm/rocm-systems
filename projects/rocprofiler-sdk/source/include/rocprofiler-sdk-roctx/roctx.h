// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

/**
 * @file roctx.h
 * @brief ROCTx API interface for AMD code annotation and profiling
 *
 * @mainpage ROCTx API Specification
 *
 * @section introduction Introduction
 * ROCTx is a comprehensive library that implements the AMD code annotation API. It provides
 * essential functionality for:
 * - Event annotation and marking
 * - Code range tracking and management
 * - Profiler control and customization
 * - Thread and device naming capabilities
 *
 * Key features:
 * - Nested range tracking with push/pop functionality
 * - Process-wide range management
 * - Thread-specific and global profiler control
 * - Device and stream naming support
 * - HSA agent and HIP device management
 *
 * The API is divided into several main components:
 * 1. Markers - For single event annotations
 * 2. Ranges - For tracking code execution spans
 * 3. Profiler Control - For managing profiling tool behavior
 * 4. Naming Utilities - For labeling threads, devices, and streams
 *
 * Thread Safety:
 * - Range operations are thread-local and thread-safe
 * - Marking operations are thread-safe
 * - Profiler control operations are process-wide
 *
 * Integration:
 * - Compatible with HIP runtime
 * - Supports HSA (Heterogeneous System Architecture)
 * - Provides both C and C++ interfaces
 *
 * Performance Considerations:
 * - Minimal overhead for marking and range operations
 * - Thread-local storage for efficient range stacking
 * - Lightweight profiler control mechanisms
 *
 * @note All string parameters must be null-terminated
 * @warning Proper nesting of range push/pop operations is user's responsibility
 */

#include <stddef.h>
#include <stdint.h>

#include "rocprofiler-sdk-roctx/defines.h"
#include "rocprofiler-sdk-roctx/types.h"
#include "rocprofiler-sdk-roctx/version.h"

ROCTX_EXTERN_C_INIT

/** @defgroup marker_group ROCTx Markers
 *
 * @brief Markers are used to annotate specific events in the code execution.
 *
 * @{
 */

/**
 * Mark an event.
 *
 * @param[in] message The message associated with the event.
 */
void
roctxMarkA(const char* message) ROCTX_API ROCTX_NONNULL(1);

/** @} */

/** @defgroup range_group ROCTx Ranges
 *
 * @brief Ranges are used to describe a span of code execution in a ROCm application.
 *
 * Ranges can be nested, and the API provides functions to start and stop ranges.
 * Ranges are thread-local, meaning that each thread can have its own stack of
 * ranges. The API provides functions to push and pop ranges from the stack.
 * The API also provides functions to start and stop ranges, which are
 * process-wide. Each range is assigned a unique ID, which can be used to
 * identify the range when stopping it.
 * @{
 */

/**
 * Start a new nested range.
 *
 * Nested ranges are stacked and local to the current CPU thread.
 *
 * @param[in] message The message associated with this range.
 *
 * @return Returns the level this nested range is started at. Nested range
 * levels are 0 based.
 */
int
roctxRangePushA(const char* message) ROCTX_API ROCTX_NONNULL(1);

/**
 * Stop the current nested range.
 *
 * Stop the current nested range, and pop it from the stack. If a nested range
 * was active before the last one was started, it becomes again the current
 * nested range.
 *
 * @return Returns the level the stopped nested range was started at, or a
 * negative value if there was no nested range active.
 */
int
roctxRangePop() ROCTX_API;

/**
 * @brief Starts a process range.
 *
 * Start/stop ranges can be started and stopped in different threads. Each
 * timespan is assigned a unique range ID.
 *
 * @param [in] message The message associated with this range.
 *
 * @return Returns the ID of the new range.
 */
roctx_range_id_t
roctxRangeStartA(const char* message) ROCTX_API ROCTX_NONNULL(1);

/**
 * Stop a process range.
 *
 * @param [in] id ::roctx_range_id_t returned from ::roctxRangeStartA to stop
 */
void
roctxRangeStop(roctx_range_id_t id) ROCTX_API;

/** @} */

/** @defgroup PROFILER_COMM ROCTx Application control/customization of profiling tools
 *
 * @brief Applications can invoke these functions to control/customize profiling tool behavior.
 *
 * @{
 */

/**
 * @brief Request any currently running profiling tool that is should stop collecting data.
 *
 * Within a profiling tool, it is recommended that the tool cache all active contexts at the time of
 * the request and then stop them. By convention, the application should pass zero to indicate a
 * global pause of the profiler in the current process. If the application wishes to pause only the
 * current thread, the application should obtain the thread ID via @ref roctxGetThreadId.
 *
 * @param [in] tid Zero for all threads in current process or non-zero for a specific thread
 *
 * @return int A profiling tool may choose to set this value to a non-zero value to indicate a
 * failure while executing the request or lack of support. If the profiling tool supports pausing
 * but is already paused, the tool should ignore the request and return zero.
 */
int
roctxProfilerPause(roctx_thread_id_t tid) ROCTX_API;

/**
 * @brief Request any currently running profiling tool that is should resume collecting data.
 *
 * Within a profiling tool, it is recommended that the tool re-activated the active contexts which
 * were cached when the pause request was issued. By convention, the application should pass zero to
 * indicate a global pause of the profiler in the current process. If the application wishes to
 * pause only the current thread, the application should obtain the thread ID via @ref
 * roctxGetThreadId.
 *
 * @param [in] tid Zero for all threads in current process or non-zero for a specific thread
 *
 * @return int A profiling tool may choose to set this value to a non-zero value to indicate a
 * failure while executing the request or lack of support. If the profiling tool is supports
 * resuming but is already active, the tool should ignore the request and return zero.
 */
int
roctxProfilerResume(roctx_thread_id_t tid) ROCTX_API;

/** @} */

/** \defgroup UTILITIES ROCTx Utility functions
 *
 * @brief Utility functions for profiling tools to customize their behavior.
 *
 * @{
 */

/**
 * @brief Indicate to a profiling tool that, where possible, you would like the current CPU OS
 * thread to be labeled by the provided name in the output of the profiling tool.
 *
 * Rocprofiler does not provide explicit support for how profiling tools handle this request:
 * support for this capability is tool specific. ROCTx does NOT rename the thread via
 * `pthread_setname_np`.
 *
 * @param [in] name Name for the current OS thread
 *
 * @return int A profiling tool may choose to set this value to a non-zero value to indicate a
 * failure while executing the request or lack of support
 */
int
roctxNameOsThread(const char* name) ROCTX_API ROCTX_NONNULL(1);

/**
 * @brief Indicate to a profiling tool that, where possible, you would like the given HSA agent
 * to be labeled by the provided name in the output of the profiling tool.
 *
 * Rocprofiler does not provide any explicit support for how profiling tools handle this request:
 * support for this capability is tool specific.
 *
 * @param [in] name Name for the specified agent
 * @param [in] agent Pointer to a HSA agent identifier
 *
 * @return int A profiling tool may choose to set this value to a non-zero value to indicate a
 * failure while executing the request or lack of support
 */
int
roctxNameHsaAgent(const char* name, const struct hsa_agent_s* agent) ROCTX_API ROCTX_NONNULL(1, 2);

/**
 * @brief Indicate to a profiling tool that, where possible, you would like the given HIP device id
 * to be labeled by the provided name in the output of the profiling tool.
 *
 * Rocprofiler does not provide any explicit support for how profiling tools handle this request:
 * support for this capability is tool specific.
 *
 * @param [in] name Name for the specified device
 * @param [in] device_id HIP device ordinal
 *
 * @return int A profiling tool may choose to set this value to a non-zero value to indicate a
 * failure while executing the request or lack of support
 */
int
roctxNameHipDevice(const char* name, int device_id) ROCTX_API ROCTX_NONNULL(1);

/**
 * @brief Indicate to a profiling tool that, where possible, you would like the given HIP stream
 * to be labeled by the provided name in the output of the profiling tool.
 *
 * Rocprofiler does not provide any explicit support for how profiling tools handle this request:
 * support for this capability is tool specific.
 *
 * @param [in] name Name for the specified stream
 * @param [in] stream A `hipStream_t` value (hipStream_t == ihipStream_t*)
 *
 * @return int A profiling tool may choose to set this value to a non-zero value to indicate a
 * failure while executing the request or lack of support
 */
int
roctxNameHipStream(const char* name, const struct ihipStream_t* stream) ROCTX_API ROCTX_NONNULL(1);

/** @} */

/** @defgroup UTILITIES ROCTx Utility functions
 *
 * @{
 */

/**
 * @brief Retrieve a id value for the current thread which will be identical to the id value a
 * profiling tool gets via `rocprofiler_get_thread_id(rocprofiler_thread_id_t*)`
 *
 * @param tid [out] Pointer to where the value should be placed
 *
 * @return int A profiling tool may choose to set this value to a non-zero value to indicate a
 * failure while executing the request or lack of support
 */
int
roctxGetThreadId(roctx_thread_id_t* tid) ROCTX_API ROCTX_NONNULL(1);

/** @} */

ROCTX_EXTERN_C_FINI

#if !defined(roctxRangeStart)
/** @brief For future compatibility */
#    define roctxRangeStart(message) roctxRangeStartA(message)
#endif

#if !defined(roctxMark)
/** @brief For future compatibility */
#    define roctxMark(message) roctxMarkA(message)
#endif

#if !defined(roctxRangePush)
/** @brief For future compatibility */
#    define roctxRangePush(message) roctxRangePushA(message)
#endif
