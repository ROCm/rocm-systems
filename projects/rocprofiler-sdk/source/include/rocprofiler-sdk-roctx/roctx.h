// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

/** \mainpage ROCTX API Specification
 *
 * \section introduction Introduction
 * ROCTX is a library that implements the AMD code annotation API.  It provides
 * the support necessary to annotate events and code ranges in applications.
 */

/**
 * \file
 * ROCTX API interface.
 */

#include <stddef.h>
#include <stdint.h>

#include "rocprofiler-sdk-roctx/defines.h"
#include "rocprofiler-sdk-roctx/types.h"
#include "rocprofiler-sdk-roctx/version.h"

ROCTX_EXTERN_C_INIT

/** \defgroup marker_group ROCTX Markers
 *
 * Marker annotations are used to describe events in a ROCm application.
 *
 * @{
 */

/**
 * Mark an event.
 *
 * \param[in] message The message associated with the event.
 */
void
roctxMarkA(const char* message) ROCTX_API ROCTX_NONNULL(1);

/** @} */

/** \defgroup range_group ROCTX Ranges
 *
 * Range annotations are used to describe events in a ROCm application.
 *
 * @{
 */

/**
 * Start a new nested range.
 *
 * Nested ranges are stacked and local to the current CPU thread.
 *
 * \param[in] message The message associated with this range.
 *
 * \return Returns the level this nested range is started at. Nested range
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
 * \return Returns the level the stopped nested range was started at, or a
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
 */
void
roctxRangeStop(roctx_range_id_t id) ROCTX_API;

/** @} */

/** \defgroup PROFILER_COMM ROCTX Application control/customization of profiling tools
 *
 * Applications can invoke these functions to control/customize profiling tool behavior.
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
roctxNameOsThread(const char*) ROCTX_API ROCTX_NONNULL(1);

/**
 * @brief Indicate to a profiling tool that, where possible, you would like the given HSA agent
 * to be labeled by the provided name in the output of the profiling tool.
 *
 * Rocprofiler does not provide any explicit support for how profiling tools handle this request:
 * support for this capability is tool specific.
 *
 * @param [in] name Name for the specified agent
 * @param [in] stream Pointer to a HSA agent identifier
 *
 * @return int A profiling tool may choose to set this value to a non-zero value to indicate a
 * failure while executing the request or lack of support
 */
int
roctxNameHsaAgent(const char* name, const struct hsa_agent_s*) ROCTX_API ROCTX_NONNULL(1, 2);

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

/** \defgroup UTILITIES ROCTx Utility functions
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
#    define roctxRangeStart(message) roctxRangeStartA(message)
#endif

#if !defined(roctxMark)
#    define roctxMark(message) roctxMarkA(message)
#endif

#if !defined(roctxRangePush)
#    define roctxRangePush(message) roctxRangePushA(message)
#endif
