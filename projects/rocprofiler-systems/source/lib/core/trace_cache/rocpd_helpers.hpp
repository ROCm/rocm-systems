// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"

#include <profiler-hub/writer_types.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace rocprofsys
{
namespace trace_cache
{
namespace rocpd_helpers
{

inline profiler_hub::writer_types::agent_unique_id_t
make_agent_uid(const agent& a)
{
    const auto type_to_string = [](agent_type type) -> std::optional<std::string_view> {
        switch(type)
        {
            case agent_type::GPU: return "GPU";
            case agent_type::CPU: return "CPU";
            case agent_type::NIC: return "NIC";
            default: return std::nullopt;
        }
    };

    profiler_hub::writer_types::agent_unique_id_t uid;
    uid.agent_type = type_to_string(a.type);
    uid.type_index = a.device_type_index;
    return uid;
}

inline profiler_hub::writer_types::trace_environment_t
make_trace_env(size_t node_id, size_t process_id, size_t thread_id)
{
    profiler_hub::writer_types::trace_environment_t env;
    env.node_id    = node_id;
    env.process_id = process_id;
    env.thread_id  = thread_id;
    return env;
}

inline profiler_hub::writer_types::trace_environment_t
make_trace_env_with_agent(size_t node_id, size_t process_id, size_t thread_id,
                          const agent& a)
{
    auto env     = make_trace_env(node_id, process_id, thread_id);
    env.agent_id = make_agent_uid(a);
    return env;
}

inline profiler_hub::writer_types::trace_environment_t
make_trace_env_with_agent_queue_stream(size_t node_id, size_t process_id,
                                       size_t thread_id, const agent& a, size_t queue_id,
                                       size_t stream_id)
{
    auto env      = make_trace_env_with_agent(node_id, process_id, thread_id, a);
    env.queue_id  = queue_id;
    env.stream_id = stream_id;
    return env;
}

inline profiler_hub::writer_types::event_data_t
make_event(size_t stack_id, size_t parent_stack_id, size_t correlation_id,
           const char* category)
{
    profiler_hub::writer_types::event_data_t ev;
    ev.stack_id        = stack_id;
    ev.parent_stack_id = parent_stack_id;
    ev.correlation_id  = correlation_id;
    ev.event_category  = category;
    return ev;
}

using memory_operation = std::string;
using memory_type      = std::string;

inline std::pair<memory_operation, memory_type>
parse_memory_operation_name(std::string_view memory_operation_name)
{
    static const std::unordered_map<std::string_view,
                                    std::pair<memory_operation, memory_type>>
        parsing_map{
            { "MEMORY_ALLOCATION_NONE", { "NONE", "REAL" } },
            { "MEMORY_ALLOCATION_ALLOCATE", { "ALLOC", "REAL" } },
            { "MEMORY_ALLOCATION_VMEM_ALLOCATE", { "ALLOC", "VIRTUAL" } },
            { "MEMORY_ALLOCATION_FREE", { "FREE", "REAL" } },
            { "MEMORY_ALLOCATION_VMEM_FREE", { "FREE", "VIRTUAL" } },
            { "SCRATCH_MEMORY_NONE", { "NONE", "SCRATCH" } },
            { "SCRATCH_MEMORY_ALLOC", { "ALLOC", "SCRATCH" } },
            { "SCRATCH_MEMORY_FREE", { "FREE", "SCRATCH" } },
            { "SCRATCH_MEMORY_ASYNC_RECLAIM", { "ASYNC_RECLAIM", "SCRATCH" } },
        };

    auto item = parsing_map.find(memory_operation_name);
    if(item == parsing_map.end())
    {
        return { "UNKNOWN", "UNKNOWN" };
    }

    return item->second;
}

}  // namespace rocpd_helpers
}  // namespace trace_cache
}  // namespace rocprofsys
