// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "logger/debug.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <exception>
#include <string>
#include <string_view>

namespace rocprofsys::domains::buffered
{
namespace kfd
{

template <policies::domain_service::externals Externals>
inline std::string
agent_label(const typename Externals::agent_t* agent_ptr)
{
    if(!agent_ptr)
    {
        return std::string{ "?" };
    }

    const bool is_gpu = (agent_ptr->type == Externals::k_agent_type_gpu);
    return fmt::format("{} {}", is_gpu ? "GPU" : "CPU", agent_ptr->device_type_index);
}

template <policies::domain_service::externals Externals>
inline std::string
agent_node_id_string(const typename Externals::agent_t* agent_ptr)
{
    return agent_ptr ? std::to_string(agent_ptr->node_id) : std::string{ "null" };
}

// Wraps agent_manager::get_agent_by_handle, converting its std::out_of_range on an
// unrecognized handle into a logged miss + nullptr instead of letting it escape
// across the SDK's C-callback boundary (see PR #11496 finding C1).
template <policies::domain_service::externals Externals>
inline const typename Externals::agent_t*
try_get_agent(std::uint64_t handle, std::string_view domain_name, std::string_view role)
{
    try
    {
        return &Externals::get_agent_manager().get_agent_by_handle(handle);
    } catch(const std::exception& e)
    {
        LOG_DEBUG("{}: {} lookup failed for handle {} ({})", domain_name, role, handle,
                  e.what());
        return nullptr;
    }
}

template <policies::domain_service::externals Externals>
inline void
record_thread(std::uint64_t tid)
{
    Externals::add_thread_info(typename Externals::thread_info_t{
        Externals::get_ppid(), Externals::get_pid(), tid, 0, 0, "{}" });
}

}  // namespace kfd

}  // namespace rocprofsys::domains::buffered
