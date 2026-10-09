// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent_manager.hpp"
#include "core/perfetto.hpp"
#include "core/timemory.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include "logger/debug.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofsys::rocprofiler_sdk
{
using hardware_counter_info = ::tim::hardware_counters::info;

using callback_arg_array_t = std::vector<std::pair<std::string, std::string>>;

using rocprofsys_agent_t = agent;

struct rocprofiler_tool_counter_info_t : rocprofiler_counter_info_v0_t
{
    using this_type            = rocprofiler_tool_counter_info_t;
    using parent_type          = rocprofiler_counter_info_v0_t;
    using dimension_info_vec_t = std::vector<rocprofiler_record_dimension_info_t>;

    rocprofiler_tool_counter_info_t(rocprofiler_agent_id_t _agent_id, parent_type _info,
                                    dimension_info_vec_t&& _dim_info);

    rocprofiler_tool_counter_info_t()                                           = default;
    ~rocprofiler_tool_counter_info_t()                                          = default;
    rocprofiler_tool_counter_info_t(const rocprofiler_tool_counter_info_t&)     = default;
    rocprofiler_tool_counter_info_t(rocprofiler_tool_counter_info_t&&) noexcept = default;
    rocprofiler_tool_counter_info_t& operator=(const rocprofiler_tool_counter_info_t&) =
        default;
    rocprofiler_tool_counter_info_t& operator=(
        rocprofiler_tool_counter_info_t&&) noexcept = default;

    rocprofiler_agent_id_t                           agent_id = {};
    std::vector<rocprofiler_record_dimension_info_t> dimension_info;
};

struct tool_agent
{
    std::uint64_t             device_id = 0;
    const rocprofsys_agent_t* agent     = nullptr;
};

struct timing_interval
{
    rocprofiler_timestamp_t start = 0;
    rocprofiler_timestamp_t end   = 0;
};

using agent_counter_info_map_t =
    std::unordered_map<rocprofiler_agent_id_t,
                       std::vector<rocprofiler_tool_counter_info_t>>;

using agent_counter_profile_map_t =
    std::unordered_map<rocprofiler_agent_id_t,
                       std::optional<rocprofiler_profile_config_id_t>>;

using counter_id_vec_t = std::vector<rocprofiler_counter_id_t>;

using agent_counter_id_map_t =
    std::unordered_map<rocprofiler_agent_id_t, counter_id_vec_t>;

using backtrace_operation_map_t =
    std::unordered_map<rocprofiler_callback_tracing_kind_t,
                       std::unordered_set<rocprofiler_tracing_operation_t>>;

struct client_data
{
    static constexpr size_t num_buffers  = 1;
    static constexpr size_t num_contexts = 1;

    using buffer_name_info_t   = rocprofiler::sdk::buffer_name_info_t<std::string_view>;
    using callback_name_info_t = rocprofiler::sdk::callback_name_info_t<std::string_view>;
    using buffer_id_vec_t      = std::array<rocprofiler_buffer_id_t, num_buffers>;
    using context_id_vec_t     = std::array<rocprofiler_context_id_t, num_contexts>;
    using agent_vec_t          = std::vector<rocprofiler_agent_v0_t>;

    rocprofiler_client_id_t*           client_id                 = nullptr;
    rocprofiler_client_finalize_t      client_fini               = nullptr;
    rocprofiler_context_id_t           counter_ctx               = { 0 };
    rocprofiler_buffer_id_t            counter_collection_buffer = { 0 };
    std::vector<tool_agent>            cpu_agents;
    std::vector<tool_agent>            gpu_agents;
    std::vector<hardware_counter_info> events_info;
    agent_counter_id_map_t             agent_events;
    agent_counter_info_map_t           agent_counter_info;
    agent_counter_profile_map_t        agent_counter_profiles;
    buffer_name_info_t                 buffered_tracing_info = {};
    callback_name_info_t               callback_tracing_info = {};
    backtrace_operation_map_t          backtrace_operations;

    void                      initialize();
    void                      initialize_event_info();
    void                      set_agents();
    context_id_vec_t          get_all_contexts() const;
    context_id_vec_t          get_main_contexts() const;
    buffer_id_vec_t           get_buffers() const;
    const rocprofsys_agent_t* get_agent(rocprofiler_agent_id_t _id) const;
    const tool_agent*         get_gpu_tool_agent(rocprofiler_agent_id_t id) const;
    const rocprofiler_tool_counter_info_t* get_tool_counter_info(
        rocprofiler_agent_id_t _agent_id, rocprofiler_counter_id_t _counter_id) const;
};

inline client_data::context_id_vec_t
client_data::get_all_contexts() const
{
    return context_id_vec_t{ counter_ctx };
}

inline client_data::context_id_vec_t
client_data::get_main_contexts() const
{
    return context_id_vec_t{
        counter_ctx,
    };
}

inline client_data::buffer_id_vec_t
client_data::get_buffers() const
{
    return buffer_id_vec_t{
        counter_collection_buffer,
    };
}

inline const rocprofsys_agent_t*
client_data::get_agent(rocprofiler_agent_id_t _id) const
{
    const auto& agent = get_agent_manager_instance().get_agent_by_handle(_id.handle);
    return &agent;
}

inline const tool_agent*
client_data::get_gpu_tool_agent(rocprofiler_agent_id_t id) const
{
    for(const auto& itr : gpu_agents)
    {
        if(id.handle == itr.agent->handle)
        {
            return &itr;
        }
    }
    return nullptr;
}

inline const rocprofiler_tool_counter_info_t*
client_data::get_tool_counter_info(rocprofiler_agent_id_t   _agent_id,
                                   rocprofiler_counter_id_t _counter_id) const
{
    for(const auto& itr : agent_counter_info.at(_agent_id))
    {
        if(itr.id == _counter_id)
        {
            return &itr;
        }
    }
    return nullptr;
}

constexpr client_data*
as_client_data(void* _ptr)
{
    return static_cast<client_data*>(_ptr);
}
}  // namespace rocprofsys::rocprofiler_sdk

#if !defined(ROCPROFILER_CALL)
#    define ROCPROFILER_CALL(result)                                                     \
        {                                                                                \
            const rocprofiler_status_t ROCPROFSYS_VARIABLE(_rocp_status_, __LINE__) =    \
                (result);                                                                \
            if(ROCPROFSYS_VARIABLE(_rocp_status_, __LINE__) !=                           \
               ROCPROFILER_STATUS_SUCCESS)                                               \
            {                                                                            \
                auto              msg        = std::stringstream{};                      \
                const std::string status_msg = rocprofiler_get_status_string(            \
                    ROCPROFSYS_VARIABLE(_rocp_status_, __LINE__));                       \
                msg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] "           \
                    << "rocprofiler-sdk call [" << #result                               \
                    << "] failed with error code "                                       \
                    << ROCPROFSYS_VARIABLE(_rocp_status_, __LINE__)                      \
                    << " :: " << status_msg;                                             \
                LOG_WARNING("{}", msg.str());                                            \
            }                                                                            \
        }
#endif
