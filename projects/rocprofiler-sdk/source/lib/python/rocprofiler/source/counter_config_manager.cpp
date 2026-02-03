// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "counter_config_manager.hpp"

#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>

namespace rocprofiler
{
namespace python
{
CounterConfigManager::CounterConfigManager(const std::vector<std::string>& metric_names)
: metric_names_(metric_names)
{}

CounterConfigManager::~CounterConfigManager() { destroy_all_configs(); }

rocprofiler_counter_config_id_t
CounterConfigManager::get_config_for_agent(rocprofiler_agent_id_t agent_id)
{
    // Check cache with read lock
    {
        std::shared_lock lock(cache_mutex_);
        auto             it = config_cache_.find(agent_id.handle);
        if(it != config_cache_.end())
        {
            return it->second;
        }
    }

    // Not found, need to create with write lock
    std::unique_lock lock(cache_mutex_);

    // Double-check after acquiring write lock
    auto it = config_cache_.find(agent_id.handle);
    if(it != config_cache_.end())
    {
        return it->second;
    }

    // Resolve counters for this agent
    auto counters = resolve_counters_for_agent(agent_id, metric_names_);

    if(counters.empty())
    {
        throw std::runtime_error("No matching counters found for the specified metrics");
    }

    // Create the counter config
    rocprofiler_counter_config_id_t config = {.handle = 0};
    auto                            status =
        rocprofiler_create_counter_config(agent_id, counters.data(), counters.size(), &config);

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to create counter config: " +
                                 std::string(rocprofiler_get_status_string(status)));
    }

    // Cache the config
    config_cache_[agent_id.handle] = config;

    return config;
}

std::vector<rocprofiler_counter_id_t>
CounterConfigManager::resolve_counters_for_agent(rocprofiler_agent_id_t          agent_id,
                                                 const std::vector<std::string>& counter_names)
{
    std::vector<rocprofiler_counter_id_t> available_counters;

    // Get all counters for this agent
    auto status = rocprofiler_iterate_agent_supported_counters(
        agent_id,
        [](rocprofiler_agent_id_t,
           rocprofiler_counter_id_t* counters,
           size_t                    num_counters,
           void*                     user_data) {
            auto* vec = static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
            for(size_t i = 0; i < num_counters; i++)
            {
                vec->push_back(counters[i]);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        static_cast<void*>(&available_counters));

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to iterate agent counters: " +
                                 std::string(rocprofiler_get_status_string(status)));
    }

    // Filter to only the counters we want
    std::vector<rocprofiler_counter_id_t> matched_counters;

    for(const auto& counter : available_counters)
    {
        rocprofiler_counter_info_v0_t info;
        status = rocprofiler_query_counter_info(
            counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info));

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            continue;
        }

        for(const auto& wanted_name : counter_names)
        {
            if(wanted_name == info.name)
            {
                matched_counters.push_back(counter);

                // Also cache dimension info for this counter
                rocprofiler_counter_info_v1_t info_v1;
                auto                          dim_status = rocprofiler_query_counter_info(
                    counter, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info_v1));

                if(dim_status == ROCPROFILER_STATUS_SUCCESS && info_v1.dimensions_count > 0)
                {
                    dimension_cache_[counter.handle] =
                        std::vector<rocprofiler_counter_record_dimension_info_t>(
                            info_v1.dimensions, info_v1.dimensions + info_v1.dimensions_count);
                }
                break;
            }
        }
    }

    return matched_counters;
}

CounterInfo
CounterConfigManager::get_counter_info(rocprofiler_counter_id_t counter_id)
{
    CounterInfo                   result;
    rocprofiler_counter_info_v0_t info;

    auto status = rocprofiler_query_counter_info(
        counter_id, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info));

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to query counter info: " +
                                 std::string(rocprofiler_get_status_string(status)));
    }

    result.id          = counter_id.handle;
    result.name        = info.name ? info.name : "";
    result.description = info.description ? info.description : "";
    result.block       = info.block ? info.block : "";
    result.expression  = info.expression ? info.expression : "";
    result.is_constant = info.is_constant != 0;
    result.is_derived  = info.is_derived != 0;

    return result;
}

std::vector<rocprofiler_counter_record_dimension_info_t>
CounterConfigManager::get_dimensions(rocprofiler_counter_id_t counter_id)
{
    std::shared_lock lock(cache_mutex_);
    auto             it = dimension_cache_.find(counter_id.handle);
    if(it != dimension_cache_.end())
    {
        return it->second;
    }
    return {};
}

void
CounterConfigManager::destroy_all_configs()
{
    std::unique_lock lock(cache_mutex_);

    for(auto& [agent_handle, config] : config_cache_)
    {
        if(config.handle != 0)
        {
            rocprofiler_destroy_counter_config(config);
        }
    }

    config_cache_.clear();
    dimension_cache_.clear();
}

}  // namespace python
}  // namespace rocprofiler
