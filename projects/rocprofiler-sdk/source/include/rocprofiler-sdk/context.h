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
 * Kernel dispatch and memory copy records are built on threads owned by the HSA runtime, after the
 * operation retires. Synchronizing on the operation itself -- with hipDeviceSynchronize or any
 * equivalent -- does not wait for that, so a @ref ::rocprofiler_flush_buffer issued immediately
 * afterwards can deliver nothing for work that has demonstrably finished. This function closes
 * that gap. It is the analogue of a device synchronize for record production.
 *
 * Records are written, not delivered. Call @ref ::rocprofiler_flush_buffer afterwards to receive
 * them. This function does not flush, does not stop any context, and does not prevent other
 * threads from submitting work.
 *
 * @par What it costs
 * This blocks, and the wait is open-ended by design: like a device synchronize, work submitted
 * concurrently by other threads extends it. Callers that cannot tolerate an unbounded stall should
 * ensure their own work has completed first, which is the intended usage. The wait is process-wide
 * even when a subset of contexts is named, and it stalls queue creation and destruction for its
 * duration.
 *
 * @par What it covers
 * The kernel dispatch and memory copy paths, and everything emitted from the dispatch completion
 * handler alongside them: dispatch counter collection, dispatch SPM, dispatch thread trace, the
 * dispatch-scoped PC sampling hook, scratch memory reported with a dispatch, and correlation id
 * retirement. API tracing records are written synchronously by the calling thread and never need
 * this.
 *
 * @par What it does NOT cover
 * Agent-scoped services -- device counter collection, device SPM and the PC sampling service
 * threads -- scratch memory events raised independently of a dispatch, and the KFD event kinds.
 * PC sampling internal agent buffers are drained by @ref ::rocprofiler_flush_buffer rather than
 * here.
 *
 * @par Signal-less dispatch
 * Signal-less dispatch records are covered. If the dispatch log cannot be quiesced within its
 * internal budget, signal-less dispatch is disabled for the remainder of the process and later
 * work takes the ordinary completion-signal path, which still produces records. Throughput of the
 * dispatch path may change; records are not dropped.
 *
 * @param [in] contexts Contexts the caller is synchronizing. Validated, and reserved for scoping
 * the wait in future revisions; the wait performed today is process-wide.
 * @param [in] num_contexts Number of entries in @p contexts.
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS Record writing is caught up for the covered kinds
 * @retval ::ROCPROFILER_STATUS_ERROR_TIMEOUT An internal drain gave up with work still in flight.
 * Records remain outstanding; the caller may retry.
 * @retval ::ROCPROFILER_STATUS_ERROR_REENTRANT_CALL Invoked from a buffer or tracing callback.
 * Those run on the threads this function waits for, so the wait could never be satisfied.
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND A context in @p contexts does not exist
 */
rocprofiler_status_t
rocprofiler_synchronize_contexts(const rocprofiler_context_id_t* contexts,
                                 size_t num_contexts) ROCPROFILER_API ROCPROFILER_NONNULL(1);

/** @} */

ROCPROFILER_EXTERN_C_FINI
