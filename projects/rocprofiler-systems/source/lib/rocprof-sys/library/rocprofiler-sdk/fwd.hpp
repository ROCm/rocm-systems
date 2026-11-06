// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include "logger/debug.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofsys::rocprofiler_sdk
{

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

using backtrace_operation_map_t =
    std::unordered_map<rocprofiler_callback_tracing_kind_t,
                       std::unordered_set<rocprofiler_tracing_operation_t>>;

struct client_data
{
    rocprofiler_client_id_t*      client_id   = nullptr;
    rocprofiler_client_finalize_t client_fini = nullptr;
    rocprofiler_context_id_t      counter_ctx = { 0 };
    agent_counter_info_map_t      agent_counter_info;
    agent_counter_profile_map_t   agent_counter_profiles;
    backtrace_operation_map_t     backtrace_operations;

    void                                   initialize_counter_info();
    const rocprofiler_tool_counter_info_t* get_tool_counter_info(
        rocprofiler_agent_id_t _agent_id, rocprofiler_counter_id_t _counter_id) const;
};

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
