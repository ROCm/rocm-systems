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

#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup EXTERNAL_CORRELATION External Correlation IDs
 * @brief User-defined correlation identifiers to supplement rocprofiler generated correlation ids
 *
 * @{
 */

/**
 * @brief Push default value for `external` field in @ref rocprofiler_correlation_id_t onto stack.
 *
 * External correlation ids are thread-local values. However, if rocprofiler internally requests an
 * external correlation id on a non-main thread and an external correlation id has not been pushed
 * for this thread, the external correlation ID will default to the latest external correlation id
 * on the main thread -- this allows tools to push an external correlation id once on the main
 * thread for, say, the MPI rank or process-wide UUID and this value will be used by all subsequent
 * child threads.
 *
 * @param [in] context Associated context
 * @param [in] tid thread identifier. @see rocprofiler_get_thread_id
 * @param [in] external_correlation_id User data to place in external field in @ref
 * rocprofiler_correlation_id_t
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND Context does not exist
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT Thread id is not valid
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_push_external_correlation_id(rocprofiler_context_id_t context,
                                         rocprofiler_thread_id_t  tid,
                                         rocprofiler_user_data_t  external_correlation_id);

/**
 * @brief Pop default value for `external` field in @ref rocprofiler_correlation_id_t off of stack
 *
 * @param [in] context Associated context
 * @param [in] tid thread identifier. @see rocprofiler_get_thread_id
 * @param [out] external_correlation_id Correlation id data popped off the stack
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND Context does not exist
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT Thread id is not valid
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_pop_external_correlation_id(rocprofiler_context_id_t context,
                                        rocprofiler_thread_id_t  tid,
                                        rocprofiler_user_data_t* external_correlation_id);

/** @} */

ROCPROFILER_EXTERN_C_FINI
