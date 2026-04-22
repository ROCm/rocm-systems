// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <utility>

#include <gmock/gmock.h>

#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstddef>

namespace rocprofsys::pmc::drivers::rocprofiler_sdk::testing
{

/**
 * @brief Mock implementation of rocprofiler-sdk driver for unit testing.
 *
 * Mirrors the real driver interface exactly — only the methods that the
 * PMC device and provider actually call.
 */
class mock_driver
{
public:
    MOCK_METHOD(rocprofiler_status_t, create_context,
                (rocprofiler_context_id_t * context));

    MOCK_METHOD(rocprofiler_status_t, start_context, (rocprofiler_context_id_t context));

    MOCK_METHOD(rocprofiler_status_t, stop_context, (rocprofiler_context_id_t context));

    MOCK_METHOD(rocprofiler_status_t, sample_device_counting_service,
                (rocprofiler_context_id_t context, rocprofiler_user_data_t user_data,
                 rocprofiler_counter_flag_t    flags,
                 rocprofiler_counter_record_t* output_records, size_t* record_count));

    MOCK_METHOD(rocprofiler_status_t, query_record_counter_id,
                (rocprofiler_counter_instance_id_t record_id,
                 rocprofiler_counter_id_t*         counter_id));

    MOCK_METHOD(rocprofiler_status_t, query_counter_info,
                (rocprofiler_counter_id_t              counter,
                 rocprofiler_counter_info_version_id_t version, void* info));

    MOCK_METHOD(rocprofiler_status_t, iterate_agent_supported_counters,
                (rocprofiler_agent_id_t              agent_id,
                 rocprofiler_available_counters_cb_t callback, void* user_data));

    MOCK_METHOD(rocprofiler_status_t, create_counter_config,
                (rocprofiler_agent_id_t agent_id, rocprofiler_counter_id_t* counters_list,
                 size_t counters_count, rocprofiler_counter_config_id_t* config_id));

    MOCK_METHOD(rocprofiler_status_t, configure_device_counting_service,
                (rocprofiler_context_id_t context_id, rocprofiler_buffer_id_t buffer_id,
                 rocprofiler_agent_id_t                   agent_id,
                 rocprofiler_device_counting_service_cb_t callback, void* user_data));
};

/**
 * @brief Factory for creating mock driver instances in tests.
 *
 * Supports two modes:
 * - Default: creates a new mock_driver each time (for device-level unit tests)
 * - Injectable: set a shared mock via set_mock() before constructing a provider,
 *   so the test can set expectations on the same instance the provider uses.
 */
struct mock_driver_factory
{
    using driver_t = mock_driver;

    static inline std::shared_ptr<driver_t> s_mock{};

    static void set_mock(std::shared_ptr<driver_t> mock) { s_mock = std::move(mock); }

    static std::shared_ptr<driver_t> create_driver()
    {
        if(s_mock) return s_mock;
        return std::make_shared<driver_t>();
    }

    static void reset() { s_mock.reset(); }
};

}  // namespace rocprofsys::pmc::drivers::rocprofiler_sdk::testing
