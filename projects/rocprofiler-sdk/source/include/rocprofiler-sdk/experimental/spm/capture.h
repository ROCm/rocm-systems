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

#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/experimental/spm/core.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in] agent_id GPU agent identifier
 * @param [in] cb callback to caller to get counters
 * @param [in] user_data data to pass into the callback
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if all counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND invalid agent
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 * @retbal ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED agent has no supported SPM counter
 */
rocprofiler_status_t
rocprofiler_iterate_spm_supported_counters(rocprofiler_agent_id_t              agent_id,
                                           rocprofiler_available_counters_cb_t cb,
                                           void* user_data) ROCPROFILER_API ROCPROFILER_NONNULL(2);

/**
 * @brief Configure SPM in system monitoring mode
 *
 * @param [in] context_id Context ID
 * @param [in] agent_id agent to setup SPM on. Multiple agents can be added to the same context.
 * @param [in] counters_list List of SPM counters
 * @param [in] counters_count Number of counters
 * @param [in] parameters List of rocprofiler_spm_parameter_type_t
 * @param [in] parameter_count Number of parameters
 * @param [in] data_fn Callback to receive SPM data
 * @param [in] user_data Passed back to user in data_fn
 *
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if all counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR no counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 * @retval ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED for configuration locked
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND invalid context
 * @retval ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT invalid parameter, counter or argument
 * @retval ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED agent already configured for context
 */
rocprofiler_status_t
rocprofiler_configure_spm_agent_service(rocprofiler_context_id_t        context_id,
                                        rocprofiler_agent_id_t          agent_id,
                                        rocprofiler_counter_id_t*       counters_list,
                                        size_t                          counters_count,
                                        rocprofiler_spm_parameter_t*    parameters,
                                        size_t                          parameter_count,
                                        rocprofiler_spm_data_callback_t data_fn,
                                        rocprofiler_user_data_t         user_data)
    ROCPROFILER_NONNULL(3) ROCPROFILER_API;

/**
 * @brief Configure SPM in system monitoring mode
 *
 * @param [in] context_id Context ID
 * @param [in] agent_id agent to setup SPM on. Multiple agents can be added to the same context.
 * @param [in] counters_list List of SPM counters
 * @param [in] counters_count Number of counters
 * @param [in] parameters List of rocprofiler_spm_parameter_type_t
 * @param [in] parameter_count Number of parameters
 * @param [in] dispatch_fn Callback to query whether user wants to profile dispatch
 * @param [in] data_fn Callback to receive SPM data
 * @param [in] config_userdata Passed back to user in dispatch_fn
 *
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if all counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR no counters found for agent
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI incompatible aqlprofile version is used
 * @retval ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED for configuration locked
 * @retval ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND invalid context
 * @retval ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT invalid parameter, counter or argument
 * @retval ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED agent already configured for context
 */
rocprofiler_status_t
rocprofiler_configure_spm_dispatch_service(rocprofiler_context_id_t            context_id,
                                           rocprofiler_agent_id_t              agent_id,
                                           rocprofiler_counter_id_t*           counters_list,
                                           size_t                              counters_count,
                                           rocprofiler_spm_parameter_t*        parameters,
                                           size_t                              parameter_count,
                                           rocprofiler_spm_dispatch_callback_t dispatch_fn,
                                           rocprofiler_spm_data_callback_t     data_fn,
                                           void*                               config_userdata)
    ROCPROFILER_NONNULL(3) ROCPROFILER_API;

ROCPROFILER_EXTERN_C_FINI
