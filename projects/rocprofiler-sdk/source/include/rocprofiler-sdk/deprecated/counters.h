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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @brief (deprecated) Callback that gives a list of available dimensions for a counter
 *
 * @param [in] id Counter id the dimension data is for
 * @param [in] dim_info An array of dimensions for the counter
 *      ::rocprofiler_iterate_counter_dimensions was called on.
 * @param [in] num_dims Number of dimensions
 * @param [in] user_data User data supplied by
 *      ::rocprofiler_iterate_agent_supported_counters
 */
ROCPROFILER_SDK_DEPRECATED(
    "Function using this alias has been deprecated. See rocprofiler_iterate_counter_dimensions")
typedef rocprofiler_status_t (*rocprofiler_available_dimensions_cb_t)(
    rocprofiler_counter_id_t                           id,
    const rocprofiler_counter_record_dimension_info_t* dim_info,
    size_t                                             num_dims,
    void*                                              user_data);

/**
 * @brief (deprecated) Return information about the dimensions that exists for a specific counter
 * and the extent of each dimension.
 *
 * @param [in] id counter id to query dimension info for.
 * @param [in] info_cb Callback to return dimension information for counter
 * @param [in] user_data data to pass into the callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if dimension exists
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if counter is not found
 * @retval ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND if counter does not have this dimension
 */
ROCPROFILER_SDK_DEPRECATED("Information now available in rocprofiler_counter_info_v1_t. "
                           "This function will be removed in the future.")
rocprofiler_status_t
rocprofiler_iterate_counter_dimensions(rocprofiler_counter_id_t              id,
                                       rocprofiler_available_dimensions_cb_t info_cb,
                                       void* user_data) ROCPROFILER_API;

/**
 * @brief (deprecated) This call returns the number of instances specific counter contains.
 *
 * @param [in] agent_id rocprofiler agent identifier
 * @param [in] counter_id counter id (obtained from iterate_agent_supported_counters)
 * @param [out] instance_count number of instances the counter has
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counter found
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if counter not found
 */
ROCPROFILER_SDK_DEPRECATED("Information now available in rocprofiler_counter_info_v1_t. "
                           "This function will be removed in the future.")
rocprofiler_status_t
rocprofiler_query_counter_instance_count(rocprofiler_agent_id_t   agent_id,
                                         rocprofiler_counter_id_t counter_id,
                                         size_t*                  instance_count) ROCPROFILER_API
    ROCPROFILER_NONNULL(3);

ROCPROFILER_EXTERN_C_FINI
