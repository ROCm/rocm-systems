// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/types.hpp"

#include <gmock/gmock.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace rocprofsys::backends::rocprofiler_sdk::testing
{

// Self-contained mock types — no SDK or production includes.

enum class mock_status : int
{
    success        = 0,
    error          = 1,
    hsa_not_loaded = 2,
};

enum class mock_flag : int
{
    none = 0,
};

struct mock_context_id
{
    std::uint64_t handle{};
};
struct mock_agent_id
{
    std::uint64_t handle{};
};
struct mock_buffer_id
{
    std::uint64_t handle{};
};
struct mock_counter_id
{
    std::uint64_t handle{};
};
struct mock_config_id
{
    std::uint64_t handle{};
};
struct mock_counter_record
{
    std::uint64_t id{};
    double        counter_value{};
};
struct mock_user_data
{};

// Stateless lambdas used as callbacks are convertible to these function-pointer types.
using mock_available_counters_cb_t      = mock_status (*)(mock_agent_id, mock_counter_id*,
                                                     std::size_t, void*);
using mock_device_counting_agent_cb_t   = void (*)(mock_context_id, mock_config_id);
using mock_device_counting_service_cb_t = void (*)(mock_context_id, mock_agent_id,
                                                   mock_device_counting_agent_cb_t,
                                                   void*);

class mock_backend
{
public:
    using status_t                     = mock_status;
    using context_id_t                 = mock_context_id;
    using agent_id_t                   = mock_agent_id;
    using buffer_id_t                  = mock_buffer_id;
    using counter_id_t                 = mock_counter_id;
    using counter_config_id_t          = mock_config_id;
    using counter_record_t             = mock_counter_record;
    using counter_flag_t               = mock_flag;
    using user_data_t                  = mock_user_data;
    using available_counters_cb_t      = mock_available_counters_cb_t;
    using device_counting_agent_cb_t   = mock_device_counting_agent_cb_t;
    using device_counting_service_cb_t = mock_device_counting_service_cb_t;

    static constexpr counter_flag_t flag_none             = mock_flag::none;
    static constexpr status_t       status_success        = mock_status::success;
    static constexpr status_t       status_error          = mock_status::error;
    static constexpr status_t       status_hsa_not_loaded = mock_status::hsa_not_loaded;

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
        assert(s_mock != nullptr && "mock_backend_factory: call set_mock() before use");
        return s_mock;
    }

    static void reset() { s_mock.reset(); }
};

}  // namespace rocprofsys::backends::rocprofiler_sdk::testing
