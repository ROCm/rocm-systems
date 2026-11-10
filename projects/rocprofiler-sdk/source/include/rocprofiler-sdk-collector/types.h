// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocprofiler-sdk-collector/defines.h"

#include <stddef.h>
#include <stdint.h>

/**
 * @defgroup COLLECTOR_DATA_TYPES Data types
 * @brief Data types of the out-of-process collector
 *
 * @{
 */

/**
 * @brief Status codes.
 */
typedef enum rocprofiler_collector_status_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_COLLECTOR_STATUS_SUCCESS = 0,             ///< No error occurred
    ROCPROFILER_COLLECTOR_STATUS_ERROR,                   ///< Generalized error
    ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT,  ///< Invalid function argument
    ROCPROFILER_COLLECTOR_STATUS_ERROR_INCOMPATIBLE_ABI,  ///< Struct size not supported
    ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE,     ///< Call not valid in this state
    ROCPROFILER_COLLECTOR_STATUS_ERROR_LTTNG_NOT_FOUND,   ///< lttng-sessiond not found
    ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION_DAEMON,    ///< Session daemon did not start
    ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION,           ///< Trace session control failed
    ROCPROFILER_COLLECTOR_STATUS_ERROR_OUTPUT_DIRECTORY,  ///< Output directory not usable
    ROCPROFILER_COLLECTOR_STATUS_ERROR_LAUNCH,            ///< The target could not be started
    ROCPROFILER_COLLECTOR_STATUS_LAST,
} rocprofiler_collector_status_t;

/**
 * @brief What a traced process does when its trace buffer is full.
 */
typedef enum rocprofiler_collector_buffer_policy_t  // NOLINT(performance-enum-size)
{
    /// The writing thread waits for space, up to the blocking timeout, as rocprofiler-sdk's
    /// ROCPROFILER_BUFFER_POLICY_LOSSLESS does. An event still without space at the timeout
    /// is discarded and counted.
    ROCPROFILER_COLLECTOR_BUFFER_POLICY_LOSSLESS = 0,
    /// The event is discarded and counted; the thread never waits.
    ROCPROFILER_COLLECTOR_BUFFER_POLICY_DISCARD,
    ROCPROFILER_COLLECTOR_BUFFER_POLICY_LAST,
} rocprofiler_collector_buffer_policy_t;

/**
 * @brief A collector: one trace session, with its private session daemon.
 */
typedef struct rocprofiler_collector_t
{
    uint64_t handle;
} rocprofiler_collector_t;

/**
 * @brief Configuration of a collector. Zero-initialize it, set @p size to
 * @p sizeof(rocprofiler_collector_config_t), then set the fields needed.
 */
typedef struct rocprofiler_collector_config_t
{
    uint64_t size;  ///< sizeof(rocprofiler_collector_config_t), for ABI checks

    /// Directory for the capture; created if it does not exist and must be empty if it
    /// does. The trace is written to "<output_dir>/trace" and the session daemon's log to
    /// "<output_dir>/lttng-sessiond.log".
    const char* output_dir;

    /// Installation prefix of lttng-tools (bin/lttng-sessiond and
    /// lib/lttng/libexec/lttng-consumerd below it). NULL: the
    /// ROCPROFILER_COLLECTOR_LTTNG_PREFIX environment variable, else lttng-sessiond is
    /// looked up in PATH.
    const char* lttng_prefix;

    /// NULL-terminated array of trace providers whose events are recorded, e.g.
    /// "rocm_hip". NULL: { "rocm_hip", "rocm_hsa", NULL }.
    const char* const* providers;

    /// Bytes per sub-buffer, a power of 2; 0: 512 KiB. Each traced process has a buffer
    /// per CPU; buffer_policy says what happens when it is full.
    uint64_t subbuffer_size;
    uint64_t num_subbuffers;  ///< sub-buffers per buffer, a power of 2; 0: 4

    /// With ROCPROFILER_COLLECTOR_BUFFER_POLICY_LOSSLESS, the longest a thread waits for
    /// space for one event, in microseconds. 0: 1 second; UINT64_MAX: no limit.
    uint64_t blocking_timeout_us;

    /// 0 (ROCPROFILER_COLLECTOR_BUFFER_POLICY_LOSSLESS) by default. Lossless applies to
    /// processes started with rocprofiler_collector_launch, and their children: the
    /// collector sets LTTNG_UST_ALLOW_BLOCKING=1 in their environment.
    rocprofiler_collector_buffer_policy_t buffer_policy;

    uint8_t reserved_padding[20];
} rocprofiler_collector_config_t;

/** @} */

#if defined(__cplusplus)
static_assert(sizeof(rocprofiler_collector_config_t) == 80,
              "Increment the size of reserved_padding after adding a member");
#endif
