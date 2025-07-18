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

/** @} */

ROCPROFILER_EXTERN_C_FINI
