// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup kernel_replay_service Kernel replay (dispatch counting)
 * @brief Optional multi-pass hardware counter collection for a single kernel dispatch.
 *
 * When enabled, the SDK re-executes a targeted dispatch once per counter batch while the
 * application observes a single completion. Directly-allocated device memory is saved before the
 * first pass and restored between passes, so every pass runs against identical inputs, and
 * @p record_callback is invoked once per pass with the same @c dispatch_id.
 *
 * This API is mutually exclusive with ::rocprofiler_configure_callback_dispatch_counting_service
 * and ::rocprofiler_configure_buffer_dispatch_counting_service on the same context: configure
 * kernel replay first, or use only the regular dispatch counting entry points without replay.
 *
 * @{
 */

/**
 * @brief Returns the number of replay passes (counter batches) to run for a replayed dispatch.
 *
 * Invoked by the SDK on the replay path to learn how many times to re-execute the dispatch. The
 * tool returns the number of counter batches it wants collected (each pass collects one batch via
 * @c dispatch_callback). A return value of 0 or 1 disables replay for the dispatch.
 *
 * @param [in] user_data User data supplied at configuration time.
 */
typedef uint64_t (*rocprofiler_kernel_replay_pass_count_cb_t)(void* user_data);

/**
 * @brief Configure dispatch counting with kernel replay enabled on a context.
 *
 * @param [in] context_id Context identifier.
 * @param [in] dispatch_callback Invoked once per replay pass before enqueue (counter batch).
 * @param [in] dispatch_callback_args User data for @p dispatch_callback.
 * @param [in] record_callback Invoked once per pass with counter records for that pass.
 * @param [in] record_callback_args User data for @p record_callback.
 * @param [in] pass_count_callback Invoked by the SDK to learn how many replay passes to run.
 * @param [in] pass_count_callback_args User data for @p pass_count_callback.
 *
 * @retval ::ROCPROFILER_STATUS_SUCCESS On success.
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND Invalid context id.
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID Context already has dispatch counting or
 *         kernel replay configured.
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT Null callback.
 * @retval Other errors from the underlying dispatch counting configuration path.
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_configure_kernel_replay_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void*                                      dispatch_callback_args,
    rocprofiler_dispatch_counting_record_cb_t  record_callback,
    void*                                      record_callback_args,
    rocprofiler_kernel_replay_pass_count_cb_t  pass_count_callback,
    void*                                      pass_count_callback_args) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
