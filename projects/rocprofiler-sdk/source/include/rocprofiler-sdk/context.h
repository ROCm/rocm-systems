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

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup CONTEXT_OPERATIONS Context Handling
 * @brief Associate services with a handle. This handle is used to activate/deactivate the services
 * during the application runtime.
 *
 * @{
 */

/**
 * The NULL Context handle.
 */
#define ROCPROFILER_CONTEXT_NONE ROCPROFILER_HANDLE_LITERAL(rocprofiler_context_id_t, 0UL)

/**
 * @brief Create context.
 *
 * @param [out] context_id Context identifier
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_create_context(rocprofiler_context_id_t* context_id) ROCPROFILER_API
    ROCPROFILER_NONNULL(1);

/**
 * @brief Start context.
 *
 * @param [in] context_id Identifier for context to be activated
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_start_context(rocprofiler_context_id_t context_id) ROCPROFILER_API;

/**
 * @brief Stop context.
 *
 * @param [in] context_id Identifier for context to be deactivated
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_stop_context(rocprofiler_context_id_t context_id) ROCPROFILER_API;

/**
 * @brief Query whether context is currently active.
 *
 * @param [in] context_id Context identifier for the query
 * @param [out] status If context is active, this will be a nonzero value
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS The input context id identified a registered context
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND The input context id did not identify a
 * registered context
 */
rocprofiler_status_t
rocprofiler_context_is_active(rocprofiler_context_id_t context_id, int* status) ROCPROFILER_API
    ROCPROFILER_NONNULL(2);

/**
 * @brief Query whether the context is valid
 *
 * @param [in] context_id Context identifier for the query
 * @param [out] status If context is invalid, this will be a nonzero value
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_context_is_valid(rocprofiler_context_id_t context_id, int* status) ROCPROFILER_API
    ROCPROFILER_NONNULL(2);

/**
 * @brief Wait until records have been written for already-submitted GPU operations.
 *
 * Records for asynchronous operations are written on runtime-owned threads after the operation
 * retires, so a flush issued straight after a device synchronize can return nothing for work that
 * has finished. This waits for them to be written; flush afterwards to receive them.
 *
 * @warning Expensive, and it slows tracing while it runs: it blocks until record production
 * catches up and stalls queue creation and destruction. Call it once before reading a buffer,
 * never per dispatch or on a hot path. Calling it from a buffer or record callback deadlocks:
 * those run on the threads it waits for.
 *
 * Covers kernel dispatch, memory copy, HIP event tracing, dispatch and device counter collection,
 * signal-less dispatch, and services extracted inline at dispatch completion; PC sampling agent
 * buffers are flushed for the clients owning @p contexts. Records written by the calling thread,
 * such as API tracing, never need it. Free-running sources are not covered.
 *
 * @param [in] contexts Contexts whose records the caller needs written; must be owned by the caller
 * @param [in] num_contexts Number of entries in @p contexts
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS Record writing is caught up for the covered kinds
 * @retval ::ROCPROFILER_STATUS_ERROR_RECORDS_OUTSTANDING Records remain outstanding; nothing was
 * discarded and the call may be retried
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND A context in @p contexts does not exist
 */
rocprofiler_status_t
rocprofiler_synchronize_contexts(const rocprofiler_context_id_t* contexts,
                                 size_t num_contexts) ROCPROFILER_API ROCPROFILER_NONNULL(1);

/** @} */

ROCPROFILER_EXTERN_C_FINI
