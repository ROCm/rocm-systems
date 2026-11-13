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

/**
 * @file collector.h
 * @brief Out-of-process collector: records the runtimes' trace events (TraceLogging over
 * LTTng) for a process started from the tool's own process, with no profiler code in it.
 *
 * A collector runs a private LTTng session daemon (its own ``LTTNG_HOME``) and one trace
 * session in capture mode: the events are written to disk during the run and read after
 * it stops.
 *
 * @code{.c}
 * rocprofiler_collector_config_t cfg = {0};
 * cfg.size       = sizeof(cfg);
 * cfg.output_dir = "/tmp/capture";
 *
 * rocprofiler_collector_t col;
 * rocprofiler_collector_create(&cfg, &col);  // daemon running, session created
 * rocprofiler_collector_start(col);          // events enabled: start before the target
 * int pid = 0;
 * rocprofiler_collector_launch(col, argv, &pid);
 * int status = 0;
 * rocprofiler_collector_wait(col, pid, &status);
 * rocprofiler_collector_stop(col);           // buffers flushed to <output_dir>/trace
 * rocprofiler_collector_destroy(col);        // session destroyed, daemon stopped
 * @endcode
 *
 * Only one collector can exist in a process at a time: the session daemon is selected
 * through the process's ``LTTNG_HOME`` environment variable, which the collector sets
 * from create until destroy. Calls on one collector must not run concurrently.
 */

#include "rocprofiler-sdk-collector/defines.h"
#include "rocprofiler-sdk-collector/types.h"

/**
 * @defgroup VERSIONING_GROUP Library Versioning
 * @brief Version information about the interface and the installed library.
 *
 * @{
 */

#include "rocprofiler-sdk-collector/version.h"

ROCPROFILER_COLLECTOR_EXTERN_C_INIT

/**
 * @brief Query the version of the installed library.
 *
 * @param [out] major The major version number
 * @param [out] minor The minor version number
 * @param [out] patch The patch version number
 * @return ::ROCPROFILER_COLLECTOR_STATUS_SUCCESS
 */
rocprofiler_collector_status_t
rocprofiler_collector_get_version(uint32_t* major,
                                  uint32_t* minor,
                                  uint32_t* patch) ROCPROFILER_COLLECTOR_API;

/** @} */

/**
 * @defgroup COLLECTOR_GROUP Collector
 * @brief Session control for one traced process tree
 *
 * @{
 */

/**
 * @brief Create a collector: prepare the output directory, start a private session daemon
 * and create a capture session with one user-space channel (per-process buffers, discard
 * mode) recording the configured providers' events with the vpid and vtid contexts.
 * Tracing is not started.
 *
 * @param [in] config The configuration; see ::rocprofiler_collector_config_t
 * @param [out] collector The new collector
 * @return ::ROCPROFILER_COLLECTOR_STATUS_SUCCESS, or
 *   - ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT: a NULL argument, no output_dir, or
 *     an empty providers list
 *   - ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INCOMPATIBLE_ABI: config->size is too small
 *   - ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE: another collector exists
 *   - ::ROCPROFILER_COLLECTOR_STATUS_ERROR_OUTPUT_DIRECTORY: not creatable, or not empty
 *   - ::ROCPROFILER_COLLECTOR_STATUS_ERROR_LTTNG_NOT_FOUND: no lttng-sessiond found
 *   - ::ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION_DAEMON: the daemon did not start
 *   - ::ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION: creating the session failed
 */
rocprofiler_collector_status_t
rocprofiler_collector_create(const rocprofiler_collector_config_t* config,
                             rocprofiler_collector_t* collector) ROCPROFILER_COLLECTOR_API
    ROCPROFILER_COLLECTOR_NONNULL(1, 2);

/**
 * @brief Start tracing. Start before launching the target: a process records its events
 * from the moment it registers with the session daemon.
 *
 * @param [in] collector The collector
 * @return ::ROCPROFILER_COLLECTOR_STATUS_SUCCESS,
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT (unknown collector),
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE (already started, or stopped) or
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION
 */
rocprofiler_collector_status_t
rocprofiler_collector_start(rocprofiler_collector_t collector) ROCPROFILER_COLLECTOR_API;

/**
 * @brief Start a process traced by this collector: @p argv[0] is looked up in PATH, and
 * the process inherits the caller's environment, with ``LTTNG_HOME`` naming the
 * collector's session daemon. The child processes it starts with fork and exec are traced
 * too; a child that only forks (no exec) records events only if ``liblttng-ust-fork.so``
 * is preloaded in the target, which this function does not do.
 *
 * @param [in] collector The collector, started
 * @param [in] argv NULL-terminated argument vector; argv[0] is the program
 * @param [out] pid The process id
 * @return ::ROCPROFILER_COLLECTOR_STATUS_SUCCESS,
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT,
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE (not started) or
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_LAUNCH (fork failed, or the program could not be
 *   executed)
 */
rocprofiler_collector_status_t
rocprofiler_collector_launch(rocprofiler_collector_t collector,
                             const char* const*      argv,
                             int*                    pid) ROCPROFILER_COLLECTOR_API
    ROCPROFILER_COLLECTOR_NONNULL(2, 3);

/**
 * @brief Wait for a process started with ::rocprofiler_collector_launch to exit.
 *
 * @param [in] collector The collector
 * @param [in] pid The process id from ::rocprofiler_collector_launch
 * @param [out] exit_status The exit status, or 128 + the signal number if a signal ended it
 * @return ::ROCPROFILER_COLLECTOR_STATUS_SUCCESS,
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT (unknown collector or pid) or
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR
 */
rocprofiler_collector_status_t
rocprofiler_collector_wait(rocprofiler_collector_t collector,
                           int                     pid,
                           int*                    exit_status) ROCPROFILER_COLLECTOR_API
    ROCPROFILER_COLLECTOR_NONNULL(3);

/**
 * @brief Stop tracing and wait until every buffer is written to the trace directory.
 *
 * @param [in] collector The collector, started
 * @return ::ROCPROFILER_COLLECTOR_STATUS_SUCCESS,
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT,
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE (not started) or
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION
 */
rocprofiler_collector_status_t
rocprofiler_collector_stop(rocprofiler_collector_t collector) ROCPROFILER_COLLECTOR_API;

/**
 * @brief The trace directory (CTF), "<output_dir>/trace". Complete after
 * ::rocprofiler_collector_stop.
 *
 * @param [in] collector The collector
 * @param [out] path The path, valid until ::rocprofiler_collector_destroy
 * @return ::ROCPROFILER_COLLECTOR_STATUS_SUCCESS or
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT
 */
rocprofiler_collector_status_t
rocprofiler_collector_get_trace_path(rocprofiler_collector_t collector,
                                     const char**            path) ROCPROFILER_COLLECTOR_API
    ROCPROFILER_COLLECTOR_NONNULL(2);

/**
 * @brief Destroy the collector: stop tracing if it is started, destroy the session, stop
 * the session daemon and restore ``LTTNG_HOME``. The output directory is kept.
 *
 * @param [in] collector The collector
 * @return ::ROCPROFILER_COLLECTOR_STATUS_SUCCESS,
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT or
 *   ::ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION (the collector is destroyed regardless)
 */
rocprofiler_collector_status_t
rocprofiler_collector_destroy(rocprofiler_collector_t collector) ROCPROFILER_COLLECTOR_API;

/**
 * @brief The name of a status code, e.g. "ROCPROFILER_COLLECTOR_STATUS_SUCCESS".
 *
 * @param [in] status The status code
 * @return The name, or NULL for a value that is not a status code
 */
const char*
rocprofiler_collector_get_status_name(rocprofiler_collector_status_t status)
    ROCPROFILER_COLLECTOR_API;

/**
 * @brief A description of a status code, e.g. "Session daemon did not start".
 *
 * @param [in] status The status code
 * @return The description, or NULL for a value that is not a status code
 */
const char*
rocprofiler_collector_get_status_string(rocprofiler_collector_status_t status)
    ROCPROFILER_COLLECTOR_API;

/** @} */

ROCPROFILER_COLLECTOR_EXTERN_C_FINI
