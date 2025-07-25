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

#include <rocprofiler-sdk/deprecated/counters.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup COUNTERS Hardware counters Information
 * @brief Query functions related to hardware counters
 * @{
 */

/**
 * @brief (experimental) Counter info struct version 0
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_counter_info_v0_t
{
    rocprofiler_counter_id_t id;               ///< Id of this counter
    const char*              name;             ///< Name of the counter
    const char*              description;      ///< Description of the counter
    const char*              block;            ///< Block of the counter (non-derived only)
    const char*              expression;       ///< Counter expression (derived counters only)
    uint8_t                  is_constant : 1;  ///< If this counter is HW constant
    uint8_t                  is_derived  : 1;  ///< If this counter is a derived counter
} rocprofiler_counter_info_v0_t;

/**
 * @brief (experimental) Represents metadata about a single dimension of a counter instance.
 *
 * This structure provides the name of the dimension (e.g., "XCC", "SE", etc.)
 * and the index indicating the position of a specific counter instance within that dimension.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_counter_dimension_info_t
{
    uint64_t    size;  ///< Size of this structure. Used for versioning and validation.
    const char* dimension_name;
    size_t      index;

    /**
     * @var dimension_name
     * @brief Name of the dimension this instance belongs to.

     * @var index
     * @brief Position (zero-based) of the instance within the specified dimension.
     */

} rocprofiler_counter_dimension_info_t;

/**
 * @brief (experimental) Describes a specific counter instance and its position across multiple
 * dimensions.
 *
 * This structure provides the unique instance ID, associated counter ID, number of dimensions for
 * the instance, and a pointer to an array of metadata describing each dimension's name and index.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_counter_record_dimension_instance_info_t
{
    uint64_t size;  ///< Size of this structure. Used for versioning and validation.
    rocprofiler_counter_instance_id_t            instance_id;
    uint64_t                                     counter_id;
    uint64_t                                     dimensions_count;
    const rocprofiler_counter_dimension_info_t** dimensions;

    /**
     * @var instance_id
     * @brief Encoded identifier for the instance, which includes the counter ID and all dimension
     positions.

     * @var counter_id
     * @brief Identifier of the counter associated with this instance.

     * @var dimensions_count
     * @brief Number of dimensions associated with this instance.

     * @var dimensions
     * @brief Array of pointers to dimension info structures, each representing one dimension
     *        and the position of this instance within that dimension.
     *
     * The array has `dimensions_count` elements, and each element is a pointer to a
     * `rocprofiler_counter_dimension_info_t` structure.
     */

} rocprofiler_counter_record_dimension_instance_info_t;

/**
 * @brief (experimental) Counter info struct version 1. Combines information from
 * ::rocprofiler_counter_info_v0_t with the dimension information.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_counter_info_v1_t
{
    uint64_t                 size;  ///< Size of this structure. Used for versioning and validation.
    rocprofiler_counter_id_t id;    ///< Id of this counter
    const char*              name;  ///< Name of the counter
    const char*              description;      ///< Description of the counter
    const char*              block;            ///< Block of the counter (non-derived only)
    const char*              expression;       ///< Counter expression (derived counters only)
    uint8_t                  is_constant : 1;  ///< If this counter is HW constant
    uint8_t                  is_derived  : 1;  ///< If this counter is a derived counter

    uint64_t                                                     dimensions_count;
    const rocprofiler_counter_record_dimension_info_t**          dimensions;
    uint64_t                                                     dimensions_instances_count;
    const rocprofiler_counter_record_dimension_instance_info_t** dimensions_instances;

    /// @var dimensions_count
    /// @brief Number of dimensions for the counter
    ///
    /// @var dimensions
    /// @brief Dimension information of the counter
    ///
    /// @var dimensions_instances_count
    /// @brief Number of unique instances for this counter, across all dimension combinations.
    ///
    /// @var dimensions_instances
    /// @brief Array of pointers to instance info structs, each describing a unique instance
    ///        and its specific dimension mapping.
} rocprofiler_counter_info_v1_t;

/**
 * @brief (experimental) Query counter id information from record_id.
 *
 * @param [in] id record id from rocprofiler_counter_record_t
 * @param [out] counter_id counter id associated with the record
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if id decoded
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                    rocprofiler_counter_id_t*         counter_id) ROCPROFILER_API
    ROCPROFILER_NONNULL(2);

/**
 * @brief (experimental) Query dimension position from record_id. If the dimension does not exist
 *        in the counter, the return will be 0.
 *
 * @param [in] id record id from ::rocprofiler_counter_record_t
 * @param [in]  dim dimension for which positional info is requested (currently only
 *              0 is allowed, i.e. flat array without dimension).
 * @param [out] pos value of the dimension in id
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if dimension decoded
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_query_record_dimension_position(rocprofiler_counter_instance_id_t  id,
                                            rocprofiler_counter_dimension_id_t dim,
                                            size_t* pos) ROCPROFILER_API ROCPROFILER_NONNULL(3);

/**
 * @brief (experimental) Query Counter info such as name or description.
 *
 * @param [in] counter_id counter to get info for
 * @param [in] version Version of struct in info, see ::rocprofiler_counter_info_version_id_t for
 * available types
 * @param [out] info rocprofiler_counter_info_{version}_t struct to write info to.
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counter found
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if counter not found
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI Version is not supported
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_query_counter_info(rocprofiler_counter_id_t              counter_id,
                               rocprofiler_counter_info_version_id_t version,
                               void* info) ROCPROFILER_API ROCPROFILER_NONNULL(3);

/**
 * @brief (experimental) Callback that gives a list of counters available on an agent. The
 *        counters variable is owned by rocprofiler and should not be free'd.
 *
 * @param [in] agent_id Agent ID of the current callback
 * @param [in] counters An array of counters that are avialable on the agent
 *      ::rocprofiler_iterate_agent_supported_counters was called on.
 * @param [in] num_counters Number of counters contained in counters
 * @param [in] user_data User data supplied by
 *      ::rocprofiler_iterate_agent_supported_counters
 */
ROCPROFILER_SDK_EXPERIMENTAL typedef rocprofiler_status_t (*rocprofiler_available_counters_cb_t)(
    rocprofiler_agent_id_t    agent_id,
    rocprofiler_counter_id_t* counters,
    size_t                    num_counters,
    void*                     user_data);

/**
 * @brief (experimental) Query Agent Counters Availability.
 *
 * @param [in] agent_id GPU agent identifier
 * @param [in] cb callback to caller to get counters
 * @param [in] user_data data to pass into the callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR if no counters found for agent
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                             rocprofiler_available_counters_cb_t cb,
                                             void* user_data) ROCPROFILER_API
    ROCPROFILER_NONNULL(2);

/**
 * @brief (experimental) Creates a new counter based on a derived metric provided. The counter will
 * only be available for counter collection profiles created after the addition of this counter. Due
 * to the regeneration of internal ASTs and dimension cache, this call may be slow and should
 * generally be avoided in performance sensitive code blocks (i.e. dispatch callbacks).
 *
 * @param [in] name The name of the new counter.
 * @param [in] name_len The length of the counter name.
 * @param [in] expr The counter expression, formatted identically to YAML counter definitions.
 * @param [in] expr_len The length of the expression.
 * @param [in] agent The rocprofiler_agent_id_t specifying the agent for which to create the
 * counter.
 * @param [in] description The description of the new counter (optional).
 * @param [in] description_len The length of the description.
 * @param [out] counter_id The rocprofiler_counter_id_t of the created counter.
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if the counter was successfully created.
 * @retval ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED if the counter could not be created.
 * @retval ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT if a counter argument is incorrect
 * @retval ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND if the agent is not found
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_create_counter(const char*               name,
                           size_t                    name_len,
                           const char*               expr,
                           size_t                    expr_len,
                           const char*               description,
                           size_t                    description_len,
                           rocprofiler_agent_id_t    agent,
                           rocprofiler_counter_id_t* counter_id) ROCPROFILER_NONNULL(1, 3, 8);

/** @} */

ROCPROFILER_EXTERN_C_FINI
