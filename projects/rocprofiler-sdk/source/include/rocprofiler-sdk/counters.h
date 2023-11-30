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
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                    rocprofiler_counter_id_t* counter_id) ROCPROFILER_NONNULL(2);

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
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_record_dimension_position(rocprofiler_counter_instance_id_t  id,
                                            rocprofiler_counter_dimension_id_t dim,
                                            size_t* pos) ROCPROFILER_NONNULL(3);

/**
 * @brief Return information about the dimension for a specified counter. This call
 *        is primary for future use not related to this alpha since the only dimension
 *        supported is 0 (flat array without dimension).
 *
 * @param [in] id counter id to query dimension info for.
 * @param [in]  dim dimension (currently only 0 is allowed)
 * @param [out] info info on the dimension (name, instance_size)
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if dimension exists
 * @retval ROCPROFILER_STATUS_ERROR if the dimension does not
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_record_dimension_info(rocprofiler_counter_id_t             id,
                                        rocprofiler_counter_dimension_id_t   dim,
                                        rocprofiler_record_dimension_info_t* info)
    ROCPROFILER_NONNULL(3);

/**
 * @brief Query Counter name. Name is a pointer controlled by rocprofiler and
 *        should not be free'd or modified.
 *
 * @param [in] counter_id counter for which to get its name.
 * @param [out] name returns a pointer to the name of the counter
 * @param [out] size returns the size of the name returned
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counter found
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if counter not found
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_counter_name(rocprofiler_counter_id_t counter_id, const char** name, size_t* size)
    ROCPROFILER_NONNULL(2, 3);

/**
 * @brief This call returns the number of instances specific counter contains.
 *        WARNING: There is a restriction on this call in the alpha/beta release
 *        of rocprof. This call will not return correct instance information in
 *        tool_init and must be called as part of the dispatch callback for accurate
 *        instance counting information. The reason for this restriction is that HSA
 *        is not yet loaded on tool_init.
 *
 * @param [in] agent rocprofiler agent
 * @param [in] counter_id counter id (obtained from iterate_agent_supported_counters)
 * @param [out] instance_count number of instances the counter has
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counter found
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if counter not found
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_counter_instance_count(rocprofiler_agent_t      agent,
                                         rocprofiler_counter_id_t counter_id,
                                         size_t* instance_count) ROCPROFILER_NONNULL(3);

/**
 * @brief Callback that gives a list of counters available on an agent. The
 *        counters variable is owned by rocprofiler and should not be free'd.
 *
 * @param [in] counters An array of counters that are avialable on the agent
 *      @ref rocprofiler_iterate_agent_supported_counters was called on.
 * @param [in] num_counters Number of counters contained in counters
 * @param [in] user_data User data supplied by
 *      @ref rocprofiler_iterate_agent_supported_counters
 */
typedef rocprofiler_status_t (*rocprofiler_available_counters_cb_t)(
    rocprofiler_counter_id_t* counters,
    size_t                    num_counters,
    void*                     user_data);

/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in] agent GPU agent
 * @param [in] cb callback to caller to get counters
 * @param [in] user_data data to pass into the callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR if no counters found for agent
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_t                 agent,
                                             rocprofiler_available_counters_cb_t cb,
                                             void* user_data) ROCPROFILER_NONNULL(2);

/** @} */

ROCPROFILER_EXTERN_C_FINI
