// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/rocprofiler_sdk_backend.hpp"
#include "backends/rocprofiler_sdk/types.hpp"

#include <gmock/gmock.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace rocprofsys::backends::rocprofiler_sdk::testing
{

class mock_backend
{
public:
    // Type aliases mirror backend<Sdk> — pulled from the real SDK backend so this
    // mock satisfies the same backend_contract concept.
    using status_t            = ::rocprofsys::rocprofiler_sdk::backend::status_t;
    using context_id_t        = ::rocprofsys::rocprofiler_sdk::backend::context_id;
    using agent_id_t          = ::rocprofsys::rocprofiler_sdk::backend::agent_id;
    using buffer_id_t         = ::rocprofsys::rocprofiler_sdk::backend::buffer_id;
    using counter_id_t        = ::rocprofsys::rocprofiler_sdk::backend::counter_id;
    using counter_config_id_t = ::rocprofsys::rocprofiler_sdk::backend::counter_config_id;
    using counter_record_t    = ::rocprofsys::rocprofiler_sdk::backend::counter_record;
    using counter_flag_t      = ::rocprofsys::rocprofiler_sdk::backend::counter_flag_t;
    using user_data_t         = ::rocprofsys::rocprofiler_sdk::backend::user_data_t;
    using available_counters_cb_t =
        ::rocprofsys::rocprofiler_sdk::backend::available_counters_cb_t;
    using device_counting_agent_cb_t =
        ::rocprofsys::rocprofiler_sdk::backend::device_counting_agent_cb_t;
    using device_counting_service_cb_t =
        ::rocprofsys::rocprofiler_sdk::backend::device_counting_service_cb_t;

    static constexpr counter_flag_t flag_none =
        ::rocprofsys::rocprofiler_sdk::backend::COUNTER_FLAG_NONE;
    static constexpr status_t status_success =
        ::rocprofsys::rocprofiler_sdk::backend::STATUS_SUCCESS;
    static constexpr status_t status_error =
        ::rocprofsys::rocprofiler_sdk::backend::STATUS_ERROR;
    static constexpr status_t status_hsa_not_loaded =
        ::rocprofsys::rocprofiler_sdk::backend::STATUS_ERROR_HSA_NOT_LOADED;

    static agent_id_t make_agent_id(std::uint64_t handle) { return agent_id_t{ handle }; }

    MOCK_METHOD(status_t, create_context, (context_id_t * context));
    MOCK_METHOD(status_t, start_context, (context_id_t context));
    MOCK_METHOD(status_t, stop_context, (context_id_t context));

    MOCK_METHOD(status_t, sample_device_counting_service,
                (context_id_t ctx, user_data_t user_data, counter_flag_t flags,
                 counter_record_t* output_records, size_t* record_count));

    MOCK_METHOD(status_t, query_record_counter_id,
                (counter_record_t record, counter_id_t* counter_id));

    MOCK_METHOD((std::vector<counter_metadata>), query_counter_details,
                (counter_id_t counter_id));

    MOCK_METHOD(status_t, iterate_agent_supported_counters,
                (agent_id_t agent_id, available_counters_cb_t callback, void* user_data));

    MOCK_METHOD(status_t, create_counter_config,
                (agent_id_t agent_id, counter_id_t* counters_list, size_t counters_count,
                 counter_config_id_t* config_id));

    MOCK_METHOD(status_t, configure_device_counting_service,
                (context_id_t ctx, buffer_id_t buf, agent_id_t agent,
                 device_counting_service_cb_t callback, void* user_data));
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
