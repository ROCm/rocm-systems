// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

/** @} */

/**
 * @brief Push default value for `external` field in @ref rocprofiler_correlation_id_t onto stack.
 *
 * @param context [in] Associated context
 * @param tid [in] thread identifier. @see rocprofiler_get_thread_id
 * @param external_correlation_id [in] User data to place in external field in @ref
 * rocprofiler_correlation_id_t
 * @return rocprofiler_status_t Returns ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND if the context
 * does not exist. Returns ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT if thread id is not valid.
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_push_external_correlation_id(rocprofiler_context_id_t context,
                                         rocprofiler_thread_id_t  tid,
                                         rocprofiler_user_data_t  external_correlation_id);

/**
 * @brief Pop default value for `external` field in @ref rocprofiler_correlation_id_t off of stack
 *
 * @param context [in] Associated context
 * @param tid [in] thread identifier. @see rocprofiler_get_thread_id
 * @param external_correlation_id [out] Correlation id data popped off the stack
 * @return rocprofiler_status_t Returns ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND if the context
 * does not exist. Returns ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT if thread id is not valid.
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_pop_external_correlation_id(rocprofiler_context_id_t context,
                                        rocprofiler_thread_id_t  tid,
                                        rocprofiler_user_data_t* external_correlation_id);

ROCPROFILER_EXTERN_C_FINI
