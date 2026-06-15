// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/backend.hpp"
#include <cstdint>

#include <gmock/gmock.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace rocprofsys::backends::rocprofiler_sdk::testing
{

class mock_backend
{
public:
    using counter_config_id_t          = rocprofiler_counter_config_id_t;
    using counter_id_t                 = rocprofiler_counter_id_t;
    using counter_record_t             = rocprofiler_counter_record_t;
    using device_counting_agent_cb_t   = rocprofiler_device_counting_agent_cb_t;
    using device_counting_service_cb_t = rocprofiler_device_counting_service_cb_t;
    using available_counters_cb_t      = rocprofiler_available_counters_cb_t;
    using context_id_t                 = rocprofiler_context_id_t;
    using agent_id_t                   = rocprofiler_agent_id_t;
    using buffer_id_t                  = rocprofiler_buffer_id_t;
    using status_t                     = rocprofiler_status_t;
    using user_data_t                  = rocprofiler_user_data_t;
    using counter_flag_t               = rocprofiler_counter_flag_t;

    static constexpr counter_flag_t flag_none      = ROCPROFILER_COUNTER_FLAG_NONE;
    static constexpr status_t       status_success = ROCPROFILER_STATUS_SUCCESS;
    static constexpr status_t       status_error   = ROCPROFILER_STATUS_ERROR;
    static constexpr status_t       status_hsa_not_loaded =
        ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;

    static agent_id_t make_agent_id(std::uint64_t handle) { return agent_id_t{ handle }; }

    MOCK_METHOD(rocprofiler_status_t, create_context,
                (rocprofiler_context_id_t * context));

    MOCK_METHOD(rocprofiler_status_t, start_context, (rocprofiler_context_id_t context));

    MOCK_METHOD(rocprofiler_status_t, stop_context, (rocprofiler_context_id_t context));

    MOCK_METHOD(rocprofiler_status_t, sample_device_counting_service,
                (rocprofiler_context_id_t context, rocprofiler_user_data_t user_data,
                 rocprofiler_counter_flag_t    flags,
                 rocprofiler_counter_record_t* output_records, size_t* record_count));

    MOCK_METHOD(rocprofiler_status_t, query_record_counter_id,
                (rocprofiler_counter_record_t record,
                 rocprofiler_counter_id_t*    counter_id));

    MOCK_METHOD((std::vector<counter_metadata>), query_counter_details,
                (rocprofiler_counter_id_t counter_id));

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

struct mock_backend_factory
{
    using backend_t = mock_backend;

    static inline std::shared_ptr<backend_t> s_mock{};

    static void set_mock(std::shared_ptr<backend_t> mock) { s_mock = std::move(mock); }

    static std::shared_ptr<backend_t> create_backend()
    {
        if(s_mock) return s_mock;
        return std::make_shared<backend_t>();
    }

    static void reset() { s_mock.reset(); }
};

}  // namespace rocprofsys::backends::rocprofiler_sdk::testing
