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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @addtogroup CONTEXT_OPERATIONS
 * @{
 */

/**
 * @brief Flags controlling the behavior of context synchronization.
 */
typedef enum rocprofiler_synchronize_context_flags_t
{
    ROCPROFILER_SYNCHRONIZE_CONTEXT_NONE   = 0,
    ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG = (1 << 0),  ///< populate result and return
    ROCPROFILER_SYNCHRONIZE_CONTEXT_NOFLUSH =
        (1 << 1),  ///< do not flush buffer(s) associated with context
    ROCPROFILER_SYNCHRONIZE_CONTEXT_LAST = ROCPROFILER_SYNCHRONIZE_CONTEXT_NOFLUSH,
} rocprofiler_synchronize_context_flags_t;

/**
 * @brief Options set by the user
 */
typedef struct rocprofiler_synchronize_context_options_t
{
    uint64_t size;   ///< must be set by user
    uint64_t flags;  ///< bitset of rocprofiler_synchronize_context_flags_t
    /// Bounds each wait the call makes, not the call as a whole, so the total may exceed it.
    /// Zero means wait indefinitely.
    uint64_t timeout_ns;
} rocprofiler_synchronize_context_options_t;

/**
 * @brief Result returned by the runtime after attempting to synchronize contexts.
 */
typedef struct rocprofiler_synchronize_context_result_t
{
    uint64_t size;                 ///< must be set by user
    uint64_t num_records;          ///< estimated number of records outstanding before call
    uint64_t num_timeout_records;  ///< estimated number of records remaining after timeout
} rocprofiler_synchronize_context_result_t;

/**
 * @brief (experimental) Wait until records have been written for already-submitted GPU operations.
 *
 * Records for asynchronous operations are written on runtime-owned threads after the operation
 * retires, so a flush issued straight after a device synchronize can return nothing for work that
 * has finished. This waits for them to be written and subsequently performs a flush to make all
 * records visible.
 *
 * @warning Expensive, and it slows tracing while it runs: it blocks until record production
 * catches up and stalls queue creation and destruction. Call it once before reading a buffer,
 * never per dispatch or on a hot path. Calling it from a buffer or record callback deadlocks:
 * those run on the threads it waits for.
 *
 * Covers kernel dispatch, memory copy, HIP event tracing, dispatch and device counter collection,
 * signal-less dispatch, and services extracted inline at dispatch completion; PC sampling agent
 * buffers are flushed for the clients owning @p context. Records written by the calling thread,
 * such as API tracing, never need it. Free-running sources are not covered.
 *
 * The wait spans record production process-wide, since the threads producing records are not
 * partitioned by context; only the flush is scoped to @p context, which is flushed buffer by
 * buffer. The flush runs even when the wait timed out, so the records that did land are still
 * delivered and the status reports the shortfall.
 *
 * @param [in] context Context whose records the caller needs written; must be owned by the caller
 * @param [in] options Options controlling the behavior of synchronization. May be NULL, which
 * selects no flags and an unbounded wait.
 * @param [out] result Filled with the outcome of the synchronization attempt. May be NULL unless
 * ::ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG is set. Its counts are zeroed before anything else
 * runs, so they are readable on every status this returns.
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS Record writing is caught up for the covered kinds
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND @p context does not exist
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT @p result is NULL while
 * ::ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG is set, which would leave nothing to return
 * @retval ::ROCPROFILER_STATUS_ERROR_TIMEOUT The operation timed out before all records were
 * written. Nothing was discarded, @p result reports the estimated shortfall, and the call may be
 * retried.
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_synchronize_context(rocprofiler_context_id_t                         context,
                                const rocprofiler_synchronize_context_options_t* options,
                                rocprofiler_synchronize_context_result_t* result) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
