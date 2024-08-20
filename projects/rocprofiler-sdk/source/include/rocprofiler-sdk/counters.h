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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup COUNTERS Hardware counters Information
 * @brief Query functions related to hardware counters
 * @{
 */

/**
 * @brief Query counter id information from record_id.
 *
 * @param [in] id record id from rocprofiler_record_counter_t
 * @param [out] counter_id counter id associated with the record
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if id decoded
 */
rocprofiler_status_t
rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                    rocprofiler_counter_id_t*         counter_id) ROCPROFILER_API
    ROCPROFILER_NONNULL(2);

/**
 * @brief Query dimension position from record_id. If the dimension does not exist
 *        in the counter, the return will be 0.
 *
 * @param [in] id record id from @ref rocprofiler_record_counter_t
 * @param [in]  dim dimension for which positional info is requested (currently only
 *              0 is allowed, i.e. flat array without dimension).
 * @param [out] pos value of the dimension in id
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if dimension decoded
 */
rocprofiler_status_t
rocprofiler_query_record_dimension_position(rocprofiler_counter_instance_id_t  id,
                                            rocprofiler_counter_dimension_id_t dim,
                                            size_t* pos) ROCPROFILER_API ROCPROFILER_NONNULL(3);

/**
 * @brief Callback that gives a list of available dimensions for a counter
 *
 * @param [in] id Counter id the dimension data is for
 * @param [in] dim_info An array of dimensions for the counter
 *      @ref rocprofiler_iterate_counter_dimensions was called on.
 * @param [in] num_dims Number of dimensions
 * @param [in] user_data User data supplied by
 *      @ref rocprofiler_iterate_agent_supported_counters
 */
typedef rocprofiler_status_t (*rocprofiler_available_dimensions_cb_t)(
    rocprofiler_counter_id_t                   id,
    const rocprofiler_record_dimension_info_t* dim_info,
    size_t                                     num_dims,
    void*                                      user_data);

/**
 * @brief Return information about the dimensions that exists for a specific counter
 *        and the extent of each dimension.
 *
 * @param [in] id counter id to query dimension info for.
 * @param [in] info_cb Callback to return dimension information for counter
 * @param [in] user_data data to pass into the callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if dimension exists
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if counter is not found
 * @retval ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND if counter does not have this dimension
 */
rocprofiler_status_t
rocprofiler_iterate_counter_dimensions(rocprofiler_counter_id_t              id,
                                       rocprofiler_available_dimensions_cb_t info_cb,
                                       void* user_data) ROCPROFILER_API;

/**
 * @brief Query Counter info such as name or description.
 *
 * @param [in] counter_id counter to get info for
 * @param [in] version Version of struct in info, see @ref rocprofiler_counter_info_version_id_t for
 * available types
 * @param [out] info rocprofiler_counter_info_{version}_t struct to write info to.
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counter found
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if counter not found
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI Version is not supported
 */
rocprofiler_status_t
rocprofiler_query_counter_info(rocprofiler_counter_id_t              counter_id,
                               rocprofiler_counter_info_version_id_t version,
                               void* info) ROCPROFILER_API ROCPROFILER_NONNULL(3);

/**
 * @brief This call returns the number of instances specific counter contains.
 *
 * @param [in] agent_id rocprofiler agent identifier
 * @param [in] counter_id counter id (obtained from iterate_agent_supported_counters)
 * @param [out] instance_count number of instances the counter has
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counter found
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if counter not found
 */
rocprofiler_status_t
rocprofiler_query_counter_instance_count(rocprofiler_agent_id_t   agent_id,
                                         rocprofiler_counter_id_t counter_id,
                                         size_t*                  instance_count) ROCPROFILER_API
    ROCPROFILER_NONNULL(3);

/**
 * @brief Callback that gives a list of counters available on an agent. The
 *        counters variable is owned by rocprofiler and should not be free'd.
 *
 * @param [in] agent_id Agent ID of the current callback
 * @param [in] counters An array of counters that are avialable on the agent
 *      @ref rocprofiler_iterate_agent_supported_counters was called on.
 * @param [in] num_counters Number of counters contained in counters
 * @param [in] user_data User data supplied by
 *      @ref rocprofiler_iterate_agent_supported_counters
 */
typedef rocprofiler_status_t (*rocprofiler_available_counters_cb_t)(
    rocprofiler_agent_id_t    agent_id,
    rocprofiler_counter_id_t* counters,
    size_t                    num_counters,
    void*                     user_data);

/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in] agent_id GPU agent identifier
 * @param [in] cb callback to caller to get counters
 * @param [in] user_data data to pass into the callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR if no counters found for agent
 */
rocprofiler_status_t
rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                             rocprofiler_available_counters_cb_t cb,
                                             void* user_data) ROCPROFILER_API
    ROCPROFILER_NONNULL(2);

/** @} */

ROCPROFILER_EXTERN_C_FINI
