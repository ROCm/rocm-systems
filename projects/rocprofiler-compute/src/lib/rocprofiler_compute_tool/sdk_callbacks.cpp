// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "sdk_callbacks.h"

#include <algorithm>
#include <iostream>
#include <map>

bool is_targetted_dispatch(const tool_data_t* tool, uint64_t kernel_id, uint64_t kernel_iteration)
{
    if (!tool->target_kernel_ids.empty() && !tool->target_kernel_ids.count(kernel_id))
        return false;

    if (!tool->kernel_filter_ranges.empty())
        return std::any_of(tool->kernel_filter_ranges.begin(),
                           tool->kernel_filter_ranges.end(),
                           [kernel_iteration](const auto& range) {
                               return kernel_iteration >= range.first && kernel_iteration <= range.second;
                           });

    return true;
}

void create_counter_collection_profile(
    tool_data_t*                                                                tool,
    rocprofiler_agent_id_t                                                      agent_id,
    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_config_id_t>>& profile_cache)
{
    // get counters to collect
    std::set<std::set<std::string>> counters_to_collect;
    for (const std::string& counters_str : helper_utils::split_by_regex(tool->requested_counters, "[,]"))
    {
        if (!counters_str.empty())
        {
            auto pos = counters_str.find(':');
            if (pos != std::string::npos)
            {
                std::istringstream    ss(counters_str.substr(pos + 1));
                std::set<std::string> counters;
                for (std::string token; ss >> token;)
                {
                    counters.insert(token);
                }
                counters_to_collect.insert(counters);
            }
        }
    }

    // Get available counters for this agent
    std::vector<rocprofiler_counter_id_t> gpu_counters;
    ROCPROFILER_CALL(
        rocprofiler_iterate_agent_supported_counters(
            agent_id,
            [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters, size_t num_counters, void* user_data)
            {
                std::vector<rocprofiler_counter_id_t>* vec =
                    static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                for (size_t i = 0; i < num_counters; i++)
                {
                    vec->push_back(counters[i]);
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            static_cast<void*>(&gpu_counters)),
        "fetch supported counters");

    std::vector<std::string>                        gpu_counter_names;
    std::map<std::string, rocprofiler_counter_id_t> gpu_counter_map;
    for (auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(rocprofiler_query_counter_info(counter,
                                                        ROCPROFILER_COUNTER_INFO_VERSION_0,
                                                        static_cast<void*>(&info)),
                         "query counter info");
        gpu_counter_names.push_back(std::string(info.name));
        gpu_counter_map.insert({std::string(info.name), counter});
    }

    // Identify counters requested to collect which are available
    std::vector<std::vector<std::string>>              collect_counter_names;
    std::vector<std::vector<rocprofiler_counter_id_t>> collect_counters;
    std::vector<std::string>                           unsupported_counters;
    for (const auto& counters : counters_to_collect)
    {
        std::vector<std::string>              counter_names;
        std::vector<rocprofiler_counter_id_t> counter_ids;
        for (const auto& counter_name : counters)
        {
            if (std::find(gpu_counter_names.begin(), gpu_counter_names.end(), counter_name) !=
                gpu_counter_names.end())
            {
                counter_names.push_back(counter_name);
                counter_ids.push_back(gpu_counter_map[counter_name]);
                tool->counter_id_name_map[gpu_counter_map[counter_name].handle] = counter_name;
            }
            else
            {
                unsupported_counters.push_back(counter_name);
            }
        }
        collect_counter_names.push_back(counter_names);
        collect_counters.push_back(counter_ids);
    }

    if (!unsupported_counters.empty())
    {
        std::clog << "\033[33m[rocprofiler-compute] [" << __FUNCTION__
                  << "] WARNING: Requested counters not available: ";
        for (size_t i = 0; i < unsupported_counters.size(); ++i)
        {
            std::clog << unsupported_counters[i];
            if (i + 1 < unsupported_counters.size())
                std::clog << ", ";
        }
        std::clog << "\033[0m" << std::endl;
    }

    // Create a profile cache for the agent
    std::vector<rocprofiler_counter_config_id_t> profiles{};
    // Create a collection profile for the counters
    for (auto& collect_counters_one_iter : collect_counters)
    {
        rocprofiler_counter_config_id_t profile = {.handle = 0};
        ROCPROFILER_CALL(rocprofiler_create_counter_config(agent_id,
                                                           collect_counters_one_iter.data(),
                                                           collect_counters_one_iter.size(),
                                                           &profile),
                         "construct profile cfg");
        profiles.push_back(profile);
        profile_cache[agent_id.handle] = profiles;
    }
}

void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                       rocprofiler_counter_config_id_t*             config,
                       rocprofiler_user_data_t* /*user_data*/,
                       void* callback_data_args)
{
    auto kernel_id = dispatch_data.dispatch_info.kernel_id;
    auto agent_id  = dispatch_data.dispatch_info.agent_id.handle;

    // create static map of kernel_id to number of dispatches (zero indexed) and
    // update it
    static std::unordered_map<uint64_t, uint64_t> kernel_id_iteration_map{};
    static std::shared_mutex                      kernel_id_iteration_mutex;
    uint64_t                                      kernel_iteration = 0;
    {
        // Acquire unique lock for update and ensure map is updated correctly
        std::unique_lock<std::shared_mutex> lock(kernel_id_iteration_mutex);
        auto&                               iter = kernel_id_iteration_map[kernel_id];
        iter += 1;
        kernel_iteration = iter;
    }

    // static cast tool
    auto*        tool_data_ptr = static_cast<std::unique_ptr<tool_data_t>*>(callback_data_args);
    tool_data_t* tool;
    {
        std::lock_guard<std::mutex> lock(tool_data_ptr->get()->mut);
        tool = tool_data_ptr->get();
    }

    // kernel filtering
    if (!is_targetted_dispatch(tool, kernel_id, kernel_iteration))
    {
        return;
    }

    static std::shared_mutex m_mutex = {};
    static std::unordered_map<uint64_t, std::vector<rocprofiler_counter_config_id_t>> profile_cache = {};
    static std::unordered_map<uint64_t, iteration_multiplexing_dispatch_record_t> iteration_multiplexing_data = {};

    // check cache for existing profile for this agent
    auto search_profile_cache = [&]()
    {
        if (auto pos = profile_cache.find(agent_id); pos != profile_cache.end())
            return true;
        return false;
    };

    auto set_config_from_cache = [&]()
    {
        if (tool->iteration_multiplexing_mode != iteration_multiplexing_mode_t::DISABLED &&
            iteration_multiplexing_data.find(agent_id) == iteration_multiplexing_data.end())
        {
            // First time setting up iteration multiplexing data for this agent
            iteration_multiplexing_data[agent_id] = iteration_multiplexing_dispatch_record_t{};
            if (tool->iteration_multiplexing_mode == iteration_multiplexing_mode_t::SIMPLE)
            {
                iteration_multiplexing_data[agent_id].config = -1;  // so first increment sets to 0
            }
        }

        kernel_dispatch_info_t dispatch_info{dispatch_data.dispatch_info.kernel_id,
                                             dispatch_data.dispatch_info.queue_id.handle,
                                             dispatch_data.dispatch_info.workgroup_size,
                                             dispatch_data.dispatch_info.grid_size,
                                             dispatch_data.dispatch_info.group_segment_size};
        switch (tool->iteration_multiplexing_mode)
        {
        case iteration_multiplexing_mode_t::DISABLED:
            *config = profile_cache[agent_id][0];
            return;

        case iteration_multiplexing_mode_t::SIMPLE:
            iteration_multiplexing_data[agent_id].config =
                (iteration_multiplexing_data[agent_id].config + 1) % profile_cache[agent_id].size();
            *config = profile_cache[agent_id][iteration_multiplexing_data[agent_id].config];
            return;

        case iteration_multiplexing_mode_t::KERNEL:
            if (iteration_multiplexing_data[agent_id].kernel_config.find(kernel_id) ==
                iteration_multiplexing_data[agent_id].kernel_config.end())
            {
                // First time seeing this kernel_id for this agent
                iteration_multiplexing_data[agent_id].kernel_config[kernel_id] = -1;  // so first increment sets to 0
            }
            iteration_multiplexing_data[agent_id].kernel_config[kernel_id] =
                (iteration_multiplexing_data[agent_id].kernel_config[kernel_id] + 1) %
                profile_cache[agent_id].size();
            *config = profile_cache[agent_id][iteration_multiplexing_data[agent_id].kernel_config[kernel_id]];
            return;

        case iteration_multiplexing_mode_t::LAUNCH:
            if (iteration_multiplexing_data[agent_id].dispatch_config.find(dispatch_info) ==
                iteration_multiplexing_data[agent_id].dispatch_config.end())
            {
                // First time seeing this dispatch_info for this agent
                iteration_multiplexing_data[agent_id].dispatch_config[dispatch_info] =
                    -1;  // so first increment sets to 0
            }
            iteration_multiplexing_data[agent_id].dispatch_config[dispatch_info] =
                (iteration_multiplexing_data[agent_id].dispatch_config[dispatch_info] + 1) %
                profile_cache[agent_id].size();
            *config = profile_cache[agent_id][iteration_multiplexing_data[agent_id].dispatch_config[dispatch_info]];
            return;

        default:
            throw std::runtime_error("[" + std::string(__FUNCTION__) +
                                     "] Unsupported iteration multiplexing mode");
        }
    };

    {
        auto rlock = std::shared_lock{m_mutex};
        if ((tool->iteration_multiplexing_mode == iteration_multiplexing_mode_t::DISABLED) &&
            search_profile_cache())
        {
            *config = profile_cache[agent_id][0];
            return;
        }
    }

    // get write lock to update cache
    auto wlock = std::unique_lock{m_mutex};
    if (search_profile_cache())
    {
        set_config_from_cache();
        return;
    }

    create_counter_collection_profile(tool, dispatch_data.dispatch_info.agent_id, profile_cache);

    // Return the profile to collect those counters for this dispatch
    set_config_from_cache();
}