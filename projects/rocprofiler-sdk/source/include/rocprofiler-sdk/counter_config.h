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
 * @defgroup COUNTER_CONFIG HW Counter Configurations
 * @brief Group one or more hardware counters into a unique handle
 *
 * @{
 */

/**
 * @brief (experimental) Create Counter Configuration. A config is bound to an agent but can
 *        be used across many contexts. The config has a fixed set of counters
 *        that are collected (and specified by counter_list). The available
 *        counters for an agent can be queried using
 *        ::rocprofiler_iterate_agent_supported_counters. An existing config
 *        may be supplied via config_id to use as a base for the new config.
 *        All counters in the existing config will be copied over to the new
 *        config. The existing config will remain unmodified and usable with
 *        the new config id being returned in config_id.
 *
 * @param [in] agent_id Agent identifier
 * @param [in] counters_list List of GPU counters
 * @param [in] counters_count Size of counters list
 * @param [in,out] config_id Identifier for GPU counters group. If an existing
                   config is supplied, that profiles counters will be copied
                   over to a new config (returned via this id)
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if config created
 * @retval ROCPROFILER_STATUS_ERROR if config could not be created
 *
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_create_counter_config(rocprofiler_agent_id_t           agent_id,
                                  rocprofiler_counter_id_t*        counters_list,
                                  size_t                           counters_count,
                                  rocprofiler_counter_config_id_t* config_id) ROCPROFILER_API
    ROCPROFILER_NONNULL(4);

/**
 * @brief (experimental) Destroy Profile Configuration.
 *
 * @param [in] config_id
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if config destroyed
 * @retval ROCPROFILER_STATUS_ERROR if config could not be destroyed
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_destroy_counter_config(rocprofiler_counter_config_id_t config_id) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
