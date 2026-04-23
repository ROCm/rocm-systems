// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"

#include <cstddef>
#include <memory>

#include <rocprofiler-sdk/context.h>
#if ROCPROFSYS_ROCM_VERSION >= 70000
#    include <rocprofiler-sdk/counter_config.h>
#else
#    include <rocprofiler-sdk/profile_config.h>
using rocprofiler_counter_config_id_t        = rocprofiler_profile_config_id_t;
using rocprofiler_counter_record_t           = rocprofiler_record_counter_t;
using rocprofiler_device_counting_agent_cb_t = rocprofiler_agent_set_profile_callback_t;
using rocprofiler_device_counting_service_cb_t =
    rocprofiler_device_counting_service_callback_t;
#endif
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace rocprofsys::pmc::drivers::rocprofiler_sdk
{

/**
 * @brief Thin wrapper around rocprofiler-sdk C APIs used by the PMC subsystem.
 *
 * Wraps both runtime APIs (sampling, context start/stop) and setup APIs
 * (counter enumeration, profile config, device counting configuration).
 * The provider calls setup APIs during construction (within tool_init),
 * and the device calls runtime APIs during sampling.
 */
struct driver
{
    static rocprofiler_status_t create_context(rocprofiler_context_id_t* context)
    {
        return rocprofiler_create_context(context);
    }

    // --- Context lifecycle ---

    static rocprofiler_status_t start_context(rocprofiler_context_id_t context)
    {
        return rocprofiler_start_context(context);
    }

    static rocprofiler_status_t stop_context(rocprofiler_context_id_t context)
    {
        return rocprofiler_stop_context(context);
    }

    // --- Runtime (hot path) ---

    static rocprofiler_status_t sample_device_counting_service(
        rocprofiler_context_id_t context, rocprofiler_user_data_t user_data,
        rocprofiler_counter_flag_t flags, rocprofiler_counter_record_t* output_records,
        size_t* record_count)
    {
        return rocprofiler_sample_device_counting_service(context, user_data, flags,
                                                          output_records, record_count);
    }

    static rocprofiler_status_t query_record_counter_id(
        rocprofiler_counter_instance_id_t record_id, rocprofiler_counter_id_t* counter_id)
    {
        return rocprofiler_query_record_counter_id(record_id, counter_id);
    }

    static rocprofiler_status_t query_counter_info(
        rocprofiler_counter_id_t counter, rocprofiler_counter_info_version_id_t version,
        void* info)
    {
        return rocprofiler_query_counter_info(counter, version, info);
    }

    // --- Setup (called by provider during construction) ---

    static rocprofiler_status_t iterate_agent_supported_counters(
        rocprofiler_agent_id_t agent_id, rocprofiler_available_counters_cb_t callback,
        void* user_data)
    {
        return rocprofiler_iterate_agent_supported_counters(agent_id, callback,
                                                            user_data);
    }

    static rocprofiler_status_t create_counter_config(
        rocprofiler_agent_id_t agent_id, rocprofiler_counter_id_t* counters_list,
        size_t counters_count, rocprofiler_counter_config_id_t* config_id)
    {
#if ROCPROFSYS_ROCM_VERSION >= 70000
        return rocprofiler_create_counter_config(agent_id, counters_list, counters_count,
                                                 config_id);
#else
        return rocprofiler_create_profile_config(agent_id, counters_list, counters_count,
                                                 config_id);
#endif
    }

    static rocprofiler_status_t configure_device_counting_service(
        rocprofiler_context_id_t context_id, rocprofiler_buffer_id_t buffer_id,
        rocprofiler_agent_id_t                   agent_id,
        rocprofiler_device_counting_service_cb_t callback, void* user_data)
    {
        return rocprofiler_configure_device_counting_service(
            context_id, buffer_id, agent_id, callback, user_data);
    }
};

/**
 * @brief Factory for creating driver instances.
 */
struct driver_factory
{
    using driver_t = driver;

    static std::shared_ptr<driver_t> create_driver()
    {
        return std::make_shared<driver_t>();
    }
};

}  // namespace rocprofsys::pmc::drivers::rocprofiler_sdk
