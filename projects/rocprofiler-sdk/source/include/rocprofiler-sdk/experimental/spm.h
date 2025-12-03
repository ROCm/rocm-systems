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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @brief (experimental) SPM parameter type used to configure SPM service.
 *
 **/
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_parameter_type_t
{
    ROCPROFILER_SPM_PARAMETER_TYPE_SCLK_COUNT = 0,  ///< No. of sc clock counts to do sampling
    ROCPROFILER_SPM_PARAMETER_TYPE_BUFFER_SIZE,     ///< SPM Buffer size, in bytes.
    ROCPROFILER_SPM_PARAMETER_TYPE_TIMEOUT_MS,
    ROCPROFILER_SPM_PARAMETER_TYPE_LAST
    ///< SPM timeout in ms. Time to wait to read the SPM buffer
} rocprofiler_spm_parameter_type_t;

/**
 * @brief SPM Profile Configurations
 * @see rocprofiler_spm_create_counter_config for how to create.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_counter_config_id_t
{
    uint64_t handle;  ///< Opaque handle
} rocprofiler_spm_counter_config_id_t;

/**
 * @brief (experimental) SPM parameter type and value.
 *
 **/
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_parameter_t
{
    rocprofiler_spm_parameter_type_t type;   ///< SPM Parameter type
    uint64_t                         value;  ///< SPM Parameter value
} rocprofiler_spm_parameter_t;

/**
 * @brief (experimental) Create SPM Counter Configuration. A config is bound to an agent but can
 *        be used across many contexts. The config has a fixed set of counters
 *        that are collected (and specified by counter_list) and parameters. The available
 *        counters for an agent can be queried using
 *        ::rocprofiler_iterate_spm_supported_counters. An existing config
 *        may be supplied via config_id to use as a base for the new config.
 *        All counters and parameters in the existing config will be copied over to the new
 *        config. The existing config will remain unmodified and usable with
 *        the new config id being returned in config_id.
 *
 * @param [in] agent_id Agent identifier
 * @param [in] counters_list List of GPU counters
 * @param [in] counters_count Size of counters list
 * @param [in] parameters Parameters list of rocprofiler_spm_parameter_type_t
 * @param [in] parameter_count Number of parameters
 * @param [in,out] config_id Identifier for GPU SPM counters group. If an existing
                   config is supplied, that profiles counters and parameters will be copied
                   over to a new config (returned via this id)
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if config created
 * @retval ROCPROFILER_STATUS_ERROR if config could not be created
 * @retval ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT if agent does not support an input
 counter
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 * @retval ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED if the ROCPROFILER_SPM_BETA_ENABLED is not set
 * @retval ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT if input counters exceed the hardware limit
 * @retval ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND if agent not found
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if an input counter is not found in metrics
 file
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_spm_create_counter_config(rocprofiler_agent_id_t               agent_id,
                                      rocprofiler_counter_id_t*            counters_list,
                                      size_t                               counters_count,
                                      rocprofiler_spm_parameter_t*         parameters,
                                      size_t                               parameter_count,
                                      rocprofiler_spm_counter_config_id_t* config_id)
    ROCPROFILER_API ROCPROFILER_NONNULL(2, 6);

/**
 * @brief (experimental) Destroy SPM Profile Configuration.
 *
 * @param [in] config_id
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if config destroyed
 * @retval ROCPROFILER_STATUS_ERROR if config could not be destroyed
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_spm_destroy_counter_config(rocprofiler_spm_counter_config_id_t config_id)
    ROCPROFILER_API;

/**
 * @brief (experimental) SPM record flags.
 *
 **/
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_record_flag_t
{
    ROCPROFILER_SPM_RECORD_FLAG_DATA_LOST = 0,  ///< records with data loss
    ROCPROFILER_SPM_RECORD_FLAG_DATA,           ///< records with data
    ROCPROFILER_SPM_RECORD_FLAG_END,            ///< End of agent service
    ROCPROFILER_SPM_RECORD_FLAG_LAST,
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

/**
 * @brief (experimental) SPM record counter record.
 *
 **/
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_spm_counter_record_t
{
    uint64_t size;  ///< Size of this structure. Used for versioning and validation.
    rocprofiler_counter_instance_id_t id;         ///< Counter instance id
    rocprofiler_agent_id_t            agent_id;   ///< Agent on which the record is collected
    rocprofiler_timestamp_t           timestamp;  ///< timestamp of the sample
    uint64_t value;  ///< SPM sample for the counter with counter instance id: id
} rocprofiler_spm_counter_record_t;

/**
 * @brief (experimental) Callback to receive SPM data
 *
 * @param [in] dispatch_data kernel dispatch data
 * @param [in] records pointer to the array of SPM records
 * @param [in] record_count  size of the record array
 * @param [in] flags  rocprofiler_spm_record_flag_t
 * @param [in] userdata user data supplied by dispatch callback
 * @param [in] record Callback data supplied via dispatch configure service
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef void (*rocprofiler_spm_dispatch_counting_record_cb_t)(
    rocprofiler_spm_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_spm_counter_record_t*                records,
    size_t                                           record_count,
    uint8_t                                          flags,
    rocprofiler_user_data_t*                         userdata,
    void*                                            record_callback_args);
/**
 * @brief (experimental) Callback query if dispatch should be profiled
 *
 * @param [in] dispatch_data kernel dispatch data
 * @param [in] config  spm counter config
 * @param [out] user_data User data unique to this dispatch. Returned in record callback
 * @param [in] callback_data_args Callback supplied via dispatch configure service
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef void (*rocprofiler_spm_dispatch_counting_service_cb_t)(
    rocprofiler_spm_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_spm_counter_config_id_t*             config,
    rocprofiler_user_data_t*                         user_data,
    void*                                            callback_data_args);

/**
 * @defgroup SPM counter
 * @brief Query and config functions related to SPM hardware counters
 * @{
 */
/**
 * @brief (experimental) Callback that gives a list of counters available on an agent. The
 *        counters variable is owned by rocprofiler and should not be free'd.
 *
 * @param [in] agent_id Agent ID of the current callback
 * @param [in] counters An array of counters that are available on the agent
 *      ::rocprofiler_iterate_spm_supported_counters was called on.
 * @param [in] num_counters Number of counters contained in counters
 * @param [in] user_data User data supplied by
 *      ::rocprofiler_iterate_spm_supported_counters
 */
ROCPROFILER_SDK_EXPERIMENTAL typedef rocprofiler_status_t (
    *rocprofiler_available_spm_counters_cb_t)(rocprofiler_agent_id_t    agent_id,
                                              rocprofiler_counter_id_t* counters,
                                              size_t                    num_counters,
                                              void*                     user_data);

/**
 * @brief (experimental) Query Agent SPM Counters Availability.
 *
 * @param [in] agent_id GPU agent identifier
 * @param [in] cb callback to caller to get counters
 * @param [in] user_data data to pass into the callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if all counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND invalid agent
 * @retval ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED agent has no supported SPM counter
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_iterate_spm_supported_counters(rocprofiler_agent_id_t                  agent_id,
                                           rocprofiler_available_spm_counters_cb_t cb,
                                           void* user_data) ROCPROFILER_API ROCPROFILER_NONNULL(2);

/**
 * @brief (experimental) Configure SPM in dispatch monitoring mode
 *
 * @param [in] context_id context id
 * @param [in] dispatch_callback callback to perform when dispatch is enqueued
 * @param [in] dispatch_callback_args callback data for dispatch callback
 * @param [in] record_callback  Record callback for completed profile data
 * @param [in] record_callback_args Callback args for record callback
 * @return ::rocprofiler_status_t
 *
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if the context can be configured for SPM dispatch service
 * @retval ROCPROFILER_STATUS_ERROR if the context cannot be configured for SPM dispatch service
 * @retval ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED if the ROCPROFILER_SPM_BETA_ENABLED is not set
 * @retval ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED for configuration locked
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID invalid input context has not already been
 * created
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT conflicting services being enabled in the
 * context
 */
ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_status_t
rocprofiler_configure_spm_dispatch_service(
    rocprofiler_context_id_t                       context_id,
    rocprofiler_spm_dispatch_counting_service_cb_t dispatch_callback,
    void*                                          dispatch_callback_args,
    rocprofiler_spm_dispatch_counting_record_cb_t  record_callback,
    void* record_callback_args) ROCPROFILER_API ROCPROFILER_NONNULL(2, 4);
;

ROCPROFILER_EXTERN_C_FINI
