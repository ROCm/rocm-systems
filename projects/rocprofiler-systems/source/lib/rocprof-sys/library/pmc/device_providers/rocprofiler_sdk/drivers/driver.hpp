// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>

#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace rocprofsys::pmc::drivers::rocprofiler_sdk
{

/**
 * @brief Thin wrapper around rocprofiler-sdk C APIs used by the PMC subsystem.
 *
 * Only wraps APIs that the SDK PMC device and provider actually call.
 * Setup/configure APIs (create_context, create_buffer, create_profile_config, etc.)
 * are called directly by rocprofiler-sdk.cpp and are not part of this interface.
 *
 * This keeps the mockable surface minimal and the driver easy to extend —
 * add a new method here when a new SDK feature needs to be polled from the device.
 */
struct driver
{
    static rocprofiler_status_t start_context(rocprofiler_context_id_t context)
    {
        return rocprofiler_start_context(context);
    }

    static rocprofiler_status_t stop_context(rocprofiler_context_id_t context)
    {
        return rocprofiler_stop_context(context);
    }

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
