// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @brief (experimental) SPM parameter types for configuring sampling behavior.
 **/
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_parameter_type_t
{
    ROCPROFILER_SPM_PARAMETER_TYPE_NONE = 0,
    ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
    ROCPROFILER_SPM_PARAMETER_TYPE_LAST,
} rocprofiler_spm_parameter_type_t;

/**
 * @brief (experimental) Describes an available SPM configuration for an agent.
 *
 * Returned via ::rocprofiler_spm_query_agent_configurations to describe the
 * supported SPM parameter types and their valid value ranges for a given agent.
 **/
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_available_configuration_t
{
    uint64_t                         size;  ///< Size of this struct
    rocprofiler_spm_parameter_type_t type;  ///< The SPM parameter type; used to tag the union below
    union
    {
        struct
        {
            uint64_t min_interval;  ///< Minimum supported value for this parameter
            uint64_t max_interval;  ///< Maximum supported value for this parameter
        } interval;                 ///< Interval configuration for sample interval parameter types
    };
} rocprofiler_spm_available_configuration_t;

/**
 * @brief (experimental) Callback that provides the available SPM configurations for an agent.
 *        The config array is owned by rocprofiler and should not be freed.
 *
 * @param [in] config Array of pointers to available SPM configurations
 * @param [in] num_config Number of configurations in the array
 * @param [in] user_data User data supplied by ::rocprofiler_spm_query_agent_configurations
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef rocprofiler_status_t (*rocprofiler_spm_available_configurations_cb_t)(
    const rocprofiler_spm_available_configuration_t** config,
    size_t                                            num_config,
    void*                                             user_data);

/**
 *  @brief (experimental) Query supported agent configurations for SPM.
 *       Supported configurations are returned in a callback.
 *
 * @param [in] agent_id agent for which the configurations are queried for
 * @param [in] cb  callback in which configurations are returned
 * @param [in] user_data user data to be passed to the callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if configurations were successfully queried
 * @retval ROCPROFILER_STATUS_ERROR if configurations could not be queried
 * @retval ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND if agent not found
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 **/
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_spm_query_agent_configurations(rocprofiler_agent_id_t                        agent_id,
                                           rocprofiler_spm_available_configurations_cb_t cb,
                                           void* user_data) ROCPROFILER_API ROCPROFILER_NONNULL(2);

/**
 * @brief (experimental) SPM configuration parameter used to configure sampling behavior.
 *
 * Passed to ::rocprofiler_spm_create_counter_config to specify SPM sampling parameters.
 * Supported configurations can be queried via ::rocprofiler_spm_query_agent_configurations.
 * @var size
 * @brief Size of this struct.
 *
 * @var type
 * @brief The SPM parameter type.
 *
 * @var value
 * @brief The value for this parameter.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_parameters_t
{
    uint64_t                         size;
    rocprofiler_spm_parameter_type_t type;
    uint64_t                         value;
} rocprofiler_spm_parameters_t;

/**
 * @brief (experimental) Create SPM Counter Configuration. A config is bound to an agent but can
 *        be used across many contexts. The config has a fixed set of counters
 *        that are collected (and specified by counter_list) and parameters. The available
 *        counters for an agent can be queried using
 *        ::rocprofiler_spm_iterate_agent_supported_counters. An existing config
 *        may be supplied via config_id to use as a base for the new config.
 *        All counters in the existing config will be copied over to the new
 *        config. The existing config will remain unmodified and usable with
 *        the new config id being returned in config_id.
 *
 * @param [in] agent_id Agent identifier
 * @param [in] counters_list List of GPU counters
 * @param [in] counters_count Size of counters list
 * @param [in] parameters SPM parameter configuration
 * @param [in] parameters_count Number of parameters
 * @param [in,out] config_id Identifier for GPU SPM counters group. If an existing
                   config is supplied, that counters will be copied
                   over to a new config (returned via this id)
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if config created
 * @retval ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT if agent does not support an input
 counter for sampling
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 * @retval ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED if the ROCPROFILER_SPM_BETA_ENABLED is not set
 * @retval ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT if input counters exceed the hardware limit
 * @retval ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND if agent not found
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if an input counter is not found in metrics
 file
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_spm_create_counter_config(rocprofiler_agent_id_t           agent_id,
                                      rocprofiler_counter_id_t*        counters_list,
                                      size_t                           counters_count,
                                      rocprofiler_spm_parameters_t**   parameters,
                                      size_t                           parameters_count,
                                      rocprofiler_counter_config_id_t* config_id) ROCPROFILER_API
    ROCPROFILER_NONNULL(2, 4, 6);

/**
 * @brief (experimental) Destroy SPM Profile Configuration.
 *
 * @param [in] config_id
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if config destroyed
 * @retval ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND if the profile is not found
 * @retval ROCPROFILER_STATUS_ERROR if config could not be destroyed
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_spm_destroy_counter_config(rocprofiler_counter_config_id_t config_id) ROCPROFILER_API;

/**
 * @brief (experimental) SPM record flags.
 *
 **/
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_record_flag_t
{
    ROCPROFILER_SPM_RECORD_FLAG_NONE         = 0,       ///< flag value none
    ROCPROFILER_SPM_RECORD_FLAG_DATA         = 1 << 0,  ///< records with data
    ROCPROFILER_SPM_RECORD_FLAG_DATA_LOSS    = 1 << 1,  ///< records with data loss
    ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END = 1 << 2,  ///< dispatch complete, no records
    ROCPROFILER_SPM_RECORD_FLAG_LAST         = 1 << 3,
} rocprofiler_spm_record_flag_t;

/**
 * @brief (experimental) Kernel dispatch data for profile counting callbacks.
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_dispatch_counting_service_data_t
{
    uint64_t                           size;            ///< Size of this struct
    rocprofiler_async_correlation_id_t correlation_id;  ///< Correlation ID for this dispatch
    rocprofiler_kernel_dispatch_info_t dispatch_info;   ///< Dispatch info
} rocprofiler_spm_dispatch_counting_service_data_t;

typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_counter_record_t
{
    uint64_t size;  ///< Size of this structure. Used for versioning and validation.
    rocprofiler_dispatch_id_t
        dispatch_id;  ///< dispatch id used to determine the dispatch this record belongs to.
    rocprofiler_counter_instance_id_t id;         ///< Counter instance id
    rocprofiler_agent_id_t            agent_id;   ///< Agent on which the record is collected
    rocprofiler_timestamp_t           timestamp;  ///< timestamp of the sample
    double value;  ///< SPM sample for the counter with counter instance id: id
} rocprofiler_spm_counter_record_t;

/**
 * @brief (experimental) Counting record callback. This is a callback is invoked when the kernel
 *        execution is complete and contains the counter profile data requested in
 *        ::rocprofiler_spm_dispatch_counting_service_cb_t. Only used with
 *        ::rocprofiler_spm_configure_callback_dispatch_service
 *
 * @param [in] dispatch_data kernel dispatch data
 * @param [in] records array of pointers to the rocprofiler_spm_counter_record_t.
          Memory of records is managed by the SDK. It is valid only within this callback
 * @param [in] record_count  size of the record array
 * @param [in] flags  rocprofiler_spm_record_flag_t
 * @param [in] userdata user data supplied by dispatch callback
 * @param [in] record Callback data supplied via dispatch configure service
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef void (*rocprofiler_spm_dispatch_counting_record_cb_t)(
    const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
    const rocprofiler_spm_counter_record_t**                records,
    size_t                                                  record_count,
    rocprofiler_spm_record_flag_t                           flags,
    rocprofiler_user_data_t                                 userdata,
    void*                                                   record_callback_args);

/**
 * @brief (experimental) Kernel Dispatch Callback. This is a callback that is invoked before the
 * kernel is enqueued into the HSA queue. What counters to collect for a kernel are set via passing
 * back a profile config (config) in this callback. These counters will be collected and emplaced in
 * the buffer with ::rocprofiler_buffer_id_t used when setting up this callback or will be returned
 * via a callback used when setting up this callback
 *
 * @param [in] dispatch_data kernel dispatch data
 * @param [out] config  spm counter config
 * @param [out] user_data User data unique to this dispatch. Returned in record callback
 * @param [in] callback_data_args Callback supplied via dispatch configure service
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef void (*rocprofiler_spm_dispatch_counting_service_cb_t)(
    const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
    rocprofiler_counter_config_id_t*                        config,
    rocprofiler_user_data_t*                                user_data,
    void*                                                   callback_data_args);

/**
 * @brief (experimental) Query Agent SPM Counters Availability.
 *
 * @param [in] agent_id GPU agent identifier
 * @param [in] cb callback to caller to get counters
 * @param [in] user_data data to pass into the callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if all counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND invalid agent
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 * @retval ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED agent has no supported SPM counter
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_spm_iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                                 rocprofiler_available_counters_cb_t cb,
                                                 void* user_data) ROCPROFILER_API
    ROCPROFILER_NONNULL(2);

/**
 * @brief (experimental) Configure callback dispatch profile Counting Service.
 *        Collects the counters in dispatch packets and calls a callback
 *        with the counters collected during that dispatch.
 *
 * @param [in] context_id context id
 * @param [in] dispatch_callback callback to perform when dispatch is enqueued
 * @param [in] dispatch_callback_args callback data for dispatch callback
 * @param [in] record_callback  Record callback for completed profile data
 * @param [in] record_callback_args Callback args for record callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if the context can be configured for SPM dispatch service
 * @retval ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED if the ROCPROFILER_SPM_BETA_ENABLED is not set
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID Invalid context
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT conflicting services being enabled in the
 * context
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_spm_configure_callback_dispatch_service(
    rocprofiler_context_id_t                       context_id,
    rocprofiler_spm_dispatch_counting_service_cb_t dispatch_callback,
    void*                                          dispatch_callback_args,
    rocprofiler_spm_dispatch_counting_record_cb_t  record_callback,
    void* record_callback_args) ROCPROFILER_API ROCPROFILER_NONNULL(2, 4);

/**
 * @brief (experimental) Configure buffered dispatch spm service.
 *        Collects the counters in dispatch packets and stores them
 *        in a buffer with buffer_id. The buffer may contain packets from more than
 *        one dispatch (denoted by dispatch id). Will trigger the
 *        callback based on the parameters setup in buffer_id_t.
 *
 * @param [in] context_id context id
 * @param [in] buffer_id id of the buffer to use for the counting service
 * @param [in] callback callback to perform when dispatch is enqueued
 * @param [in] callback_data_args callback data
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if the context can be configured for SPM buffer dispatch
 * service
 * @retval ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND if the buffer is not found
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID invalid input context
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT conflicting services being enabled in the
 * context
 * @retval ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED if the ROCPROFILER_SPM_BETA_ENABLED is not set
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_spm_configure_buffer_dispatch_service(
    rocprofiler_context_id_t                       context_id,
    rocprofiler_buffer_id_t                        buffer_id,
    rocprofiler_spm_dispatch_counting_service_cb_t callback,
    void* callback_data_args) ROCPROFILER_API ROCPROFILER_NONNULL(3);

ROCPROFILER_EXTERN_C_FINI
