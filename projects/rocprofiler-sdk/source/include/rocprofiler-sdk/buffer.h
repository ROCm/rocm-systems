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
 * @defgroup BUFFER_HANDLING Buffer Handling
 * @brief Creation, destruction, and flushing of buffers populated with data from rocprofiler
 *
 * @{
 *
 * Every Buffer is associated with a specific service kind.
 * OR
 * Every Buffer is associated with a specific service ID.
 *
 */

/**
 * @brief  Async callback function.
 *
 * The SDK guarantees that only records whose kind was explicitly configured for
 * the associated context are forwarded to this callback.  Records arriving
 * through the internal ring buffer with a kind that is not present in the
 * context's configured set (e.g. because hipInit raced with
 * rocprofiler_configure and caused the SDK to register callbacks for all 33
 * buffer tracing categories before the tool context was fully established) are
 * filtered out by the dispatch path and never delivered here.  Callers
 * therefore do not need to defend against blind struct reinterpretation of
 * records with uninitialized operation fields.
 *
 * @code{.cpp}
 *  for(size_t i = 0; i < num_headers; ++i)
 *  {
 *      rocprofiler_record_header_t* hdr = headers[i];
 *      if(hdr->kind == ROCPROFILER_RECORD_KIND_PC_SAMPLE)
 *      {
 *          auto* data = static_cast<rocprofiler_pc_sample_t*>(&hdr->payload);
 *          ...
 *      }
 *  }
 * @endcode
 */
typedef void (*rocprofiler_buffer_tracing_cb_t)(rocprofiler_context_id_t      context,
                                                rocprofiler_buffer_id_t       buffer_id,
                                                rocprofiler_record_header_t** headers,
                                                size_t                        num_headers,
                                                void*                         data,
                                                uint64_t                      drop_count);

/**
 * @brief Query whether a buffer tracing kind is among the categories configured
 *        for the given context.
 *
 * This is the predicate used by the internal buffer-flush dispatch path to
 * implement the kind-filter guard (ROCM-8888).  It is exposed here so that
 * higher-level wrappers and test harnesses can replicate the same check without
 * duplicating the context-lookup logic.
 *
 * @param [in]  context  Context whose configured-kind set is queried.
 * @param [in]  kind     Buffer tracing kind to test.
 * @param [out] result   Set to non-zero when @p kind is configured, zero otherwise.
 * @return ::rocprofiler_status_t
 *   - ::ROCPROFILER_STATUS_SUCCESS        — query completed (check @p result).
 *   - ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND — @p context is not valid.
 */
rocprofiler_status_t
rocprofiler_query_buffer_tracing_kind_configured(rocprofiler_context_id_t        context,
                                                 rocprofiler_buffer_tracing_kind_t kind,
                                                 int*                            result) ROCPROFILER_API
    ROCPROFILER_NONNULL(3);

/**
 * @brief Create buffer.
 *
 * @param [in] context Context identifier associated with buffer
 * @param [in] size Size of the buffer in bytes
 * @param [in] watermark - watermark size, where the callback is called, if set
 * to 0 then the callback will be called on every record
 * @param [in] policy Behavior policy when buffer is full
 * @param [in] callback Callback to invoke when buffer is flushed/full
 * @param [in] callback_data Data to provide in callback function
 * @param [out] buffer_id Identification handle for buffer
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_create_buffer(rocprofiler_context_id_t        context,
                          size_t                          size,
                          size_t                          watermark,
                          rocprofiler_buffer_policy_t     policy,
                          rocprofiler_buffer_tracing_cb_t callback,
                          void*                           callback_data,
                          rocprofiler_buffer_id_t*        buffer_id) ROCPROFILER_API
    ROCPROFILER_NONNULL(5, 7);

/**
 * @brief Destroy buffer.
 *
 * @param [in] buffer_id
 * @return ::rocprofiler_status_t
 *
 * Note: This will destroy the buffer even if it is not empty. The user can
 * call @ref ::rocprofiler_flush_buffer before it to make sure the buffer is empty.
 */
rocprofiler_status_t
rocprofiler_destroy_buffer(rocprofiler_buffer_id_t buffer_id) ROCPROFILER_API;

/**
 * @brief Flush buffer.
 *
 * @param [in] buffer_id
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_flush_buffer(rocprofiler_buffer_id_t buffer_id) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI